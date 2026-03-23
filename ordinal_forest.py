from __future__ import annotations

"""Unified Ordinal Forest implementation with diagnostics and R parity helpers.

This module provides a scikit-learn compatible Ordinal Forest classifier inspired by
Hornung (2020) and the CRAN package ``ordinalForest``.

Key features
------------
- Candidate score-system search with out-of-bag objective evaluation.
- Support for ``equal``, ``proportional``, ``oneclass``, ``custom``, and
  ``probability`` objectives.
- Sample-weight aware fitting and OOB evaluation.
- Approximate ``always.split.variables`` support through tree-level mandatory-feature
  subspaces.
- Detailed OOB diagnostics and visualization helpers.
- Optional R-package parity test helper for side-by-side comparison with CRAN
  ``ordinalForest`` when ``Rscript`` and the R package are installed locally.

Notes
-----
The original R package modifies ranger internals to enforce
``always.split.variables`` at every split. scikit-learn's ``DecisionTreeRegressor``
does not expose an equivalent hook. This module therefore uses a principled
approximation: each tree is fit on a subspace that always contains the mandatory
features plus a random subset of the remaining features, and the tree then has access
to all features in that subspace at every split.
"""

from dataclasses import dataclass
import inspect
from math import factorial
from pathlib import Path
from typing import Callable, Optional, Sequence, Union
import csv
import json
import os
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import norm

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import Bunch, check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


ArrayLike = Union[np.ndarray, Sequence]

_EPS = 1e-9
_INT32_MAX = np.iinfo(np.int32).max


@dataclass(frozen=True)
class ScoreSpec:
    """Container describing the numeric score system used by the forest."""

    scores: np.ndarray
    thresholds: np.ndarray
    borders: Optional[np.ndarray]
    latent_cutpoints: Optional[np.ndarray]
    mode: str


@dataclass(frozen=True)
class TreeModel:
    """Single-tree container storing the fitted regressor and feature subset."""

    regressor: DecisionTreeRegressor
    feature_indices: np.ndarray


# -----------------------------------------------------------------------------
# Label handling and weighted ordinal metrics
# -----------------------------------------------------------------------------

def _encode_labels_with_classes(y: ArrayLike, classes: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    mapping = {label: idx for idx, label in enumerate(classes)}
    try:
        return np.asarray([mapping[val] for val in y], dtype=int)
    except KeyError as exc:
        raise ValueError(
            f"Unknown label {exc.args[0]!r}; it is not present in classes_."
        ) from exc


def _infer_class_order(y: ArrayLike, class_order: Optional[Sequence] = None) -> np.ndarray:
    """Infer a stable ordinal class order."""

    if class_order is not None:
        classes = np.asarray(list(class_order))
        if len(np.unique(classes)) != len(classes):
            raise ValueError("class_order must not contain duplicated labels.")
        return classes

    try:
        import pandas as pd  # type: ignore

        if isinstance(y, pd.Categorical) and y.ordered:
            return np.asarray(list(y.categories))
        if isinstance(y, pd.Series) and pd.api.types.is_categorical_dtype(y.dtype):
            if y.cat.ordered:
                return np.asarray(list(y.cat.categories))
    except Exception:
        pass

    classes = np.unique(np.asarray(y))
    try:
        return np.sort(classes)
    except TypeError:
        return classes


def _normalize_weights(sample_weight: Optional[np.ndarray], n_samples: int) -> Optional[np.ndarray]:
    if sample_weight is None:
        return None
    weights = np.asarray(sample_weight, dtype=float)
    if weights.shape != (n_samples,):
        raise ValueError(f"sample_weight must have shape ({n_samples},), got {weights.shape}.")
    if np.any(weights < 0):
        raise ValueError("sample_weight must be non-negative.")
    if not np.any(weights > 0):
        raise ValueError("sample_weight must contain at least one positive value.")
    return weights


def _weighted_mean(values: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=float)
    if sample_weight is None:
        return float(np.mean(values))
    return float(np.average(values, weights=np.asarray(sample_weight, dtype=float)))


def _weighted_accuracy_from_indices(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    return _weighted_mean((y_true_idx == y_pred_idx).astype(float), sample_weight)


def _weighted_std(values: np.ndarray, sample_weight: Optional[np.ndarray] = None) -> float:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 1.0
    if sample_weight is None:
        sigma = float(np.std(values))
        return sigma if np.isfinite(sigma) and sigma >= 1e-6 else 1.0
    weights = np.asarray(sample_weight, dtype=float)
    avg = float(np.average(values, weights=weights))
    var = float(np.average((values - avg) ** 2, weights=weights))
    sigma = np.sqrt(max(var, 0.0))
    return float(sigma) if np.isfinite(sigma) and sigma >= 1e-6 else 1.0


def ranked_probability_score_from_proba(
    y_true: ArrayLike,
    proba: np.ndarray,
    classes: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Compute the multiclass ranked probability score (smaller is better)."""

    y_idx = _encode_labels_with_classes(y_true, classes)
    return ranked_probability_score_from_indices(y_idx, proba, sample_weight=sample_weight)


def ranked_probability_score_from_indices(
    y_idx: np.ndarray,
    proba: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Compute the multiclass ranked probability score from encoded class indices."""

    n_classes = proba.shape[1]
    one_hot = np.eye(n_classes, dtype=float)[y_idx]
    cum_true = np.cumsum(one_hot, axis=1)
    cum_pred = np.cumsum(proba, axis=1)
    per_sample = np.sum((cum_true[:, :-1] - cum_pred[:, :-1]) ** 2, axis=1)
    return _weighted_mean(per_sample, sample_weight)


def ordinal_rank_mae_from_labels(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    classes: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    y_true_idx = _encode_labels_with_classes(y_true, classes)
    y_pred_idx = _encode_labels_with_classes(y_pred, classes)
    return _weighted_mean(np.abs(y_true_idx - y_pred_idx), sample_weight)


def ordinal_rank_mse_from_labels(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    classes: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    y_true_idx = _encode_labels_with_classes(y_true, classes)
    y_pred_idx = _encode_labels_with_classes(y_pred, classes)
    return _weighted_mean((y_true_idx - y_pred_idx) ** 2, sample_weight)


def _youden_j_binary(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Weighted Youden's J = sensitivity + specificity - 1."""

    positives = y_true_binary == 1
    negatives = ~positives
    if not np.any(positives) or not np.any(negatives):
        return 0.0

    if sample_weight is None:
        sensitivity = float(np.mean(y_pred_binary[positives] == 1))
        specificity = float(np.mean(y_pred_binary[negatives] == 0))
        return sensitivity + specificity - 1.0

    weights = np.asarray(sample_weight, dtype=float)
    sensitivity = float(
        np.average((y_pred_binary[positives] == 1).astype(float), weights=weights[positives])
    )
    specificity = float(
        np.average((y_pred_binary[negatives] == 0).astype(float), weights=weights[negatives])
    )
    return sensitivity + specificity - 1.0


def youden_j_per_class(
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    n_classes: int,
    sample_weight: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Compute class-wise one-vs-rest Youden's J values."""

    values = np.empty(n_classes, dtype=float)
    for cls_idx in range(n_classes):
        true_bin = (y_true_idx == cls_idx).astype(int)
        pred_bin = (y_pred_idx == cls_idx).astype(int)
        values[cls_idx] = _youden_j_binary(true_bin, pred_bin, sample_weight=sample_weight)
    return values


def ordinal_performance_from_indices(
    y_true_idx: np.ndarray,
    *,
    y_pred_idx: Optional[np.ndarray],
    y_pred_proba: Optional[np.ndarray],
    performance_function: str,
    class_weights: Optional[np.ndarray] = None,
    prioritized_class_index: Optional[int] = None,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    """Evaluate predictions according to the ordinalForest objective."""

    if performance_function == "probability":
        if y_pred_proba is None:
            raise ValueError("y_pred_proba is required when performance_function='probability'.")
        return -ranked_probability_score_from_indices(
            y_true_idx, y_pred_proba, sample_weight=sample_weight
        )

    if y_pred_idx is None:
        raise ValueError(f"y_pred_idx is required when performance_function={performance_function!r}.")

    n_classes = int(np.max(y_true_idx)) + 1
    j_values = youden_j_per_class(
        y_true_idx, y_pred_idx, n_classes=n_classes, sample_weight=sample_weight
    )

    if performance_function == "equal":
        return float(np.mean(j_values))

    if performance_function == "proportional":
        if sample_weight is None:
            weights = np.bincount(y_true_idx, minlength=n_classes).astype(float)
        else:
            weights = np.bincount(
                y_true_idx, weights=np.asarray(sample_weight, dtype=float), minlength=n_classes
            ).astype(float)
        weights /= max(np.sum(weights), _EPS)
        return float(np.dot(weights, j_values))

    if performance_function == "oneclass":
        if prioritized_class_index is None:
            raise ValueError("prioritized_class_index is required when performance_function='oneclass'.")
        return float(j_values[prioritized_class_index])

    if performance_function == "custom":
        if class_weights is None:
            raise ValueError("class_weights is required when performance_function='custom'.")
        weights = np.asarray(class_weights, dtype=float)
        if weights.shape != (n_classes,):
            raise ValueError(
                f"class_weights must have shape ({n_classes},), got {weights.shape}."
            )
        if np.sum(weights) <= 0:
            raise ValueError("class_weights must have a strictly positive sum.")
        weights = weights / np.sum(weights)
        return float(np.dot(weights, j_values))

    raise ValueError(
        "performance_function must be one of {'equal', 'probability', 'proportional', 'oneclass', 'custom'}."
    )


# -----------------------------------------------------------------------------
# Scorers compatible with GridSearchCV and permutation importance
# -----------------------------------------------------------------------------

def ordinal_accuracy_scorer(estimator, X, y) -> float:
    return accuracy_score(y, estimator.predict(X))


def ordinal_neg_rank_mae_scorer(estimator, X, y) -> float:
    return -ordinal_rank_mae_from_labels(y, estimator.predict(X), estimator.classes_)


def ordinal_neg_rank_mse_scorer(estimator, X, y) -> float:
    return -ordinal_rank_mse_from_labels(y, estimator.predict(X), estimator.classes_)


def ordinal_neg_rps_scorer(estimator, X, y) -> float:
    return -ranked_probability_score_from_proba(y, estimator.predict_proba(X), estimator.classes_)


def ordinal_objective_scorer(estimator, X, y) -> float:
    return estimator.objective_score(X, y)


def make_ordinal_scorer(metric: str = "neg_rank_mae") -> Callable:
    """Return a callable scorer compatible with GridSearchCV."""

    scorers = {
        "accuracy": ordinal_accuracy_scorer,
        "neg_rank_mae": ordinal_neg_rank_mae_scorer,
        "neg_rank_mse": ordinal_neg_rank_mse_scorer,
        "neg_rps": ordinal_neg_rps_scorer,
        "objective": ordinal_objective_scorer,
    }
    if metric not in scorers:
        raise ValueError(f"metric must be one of {sorted(scorers)}")
    return scorers[metric]


# -----------------------------------------------------------------------------
# Score-system construction and candidate generation
# -----------------------------------------------------------------------------

def _clip_open_unit_interval(values: np.ndarray) -> np.ndarray:
    return np.clip(values, _EPS, 1.0 - _EPS)


def _borders_to_scores(borders: np.ndarray) -> np.ndarray:
    mids = 0.5 * (borders[:-1] + borders[1:])
    return norm.ppf(_clip_open_unit_interval(mids))


def _borders_to_latent_cutpoints(borders: np.ndarray) -> np.ndarray:
    cutpoints = np.empty_like(borders, dtype=float)
    cutpoints[0] = -np.inf
    cutpoints[-1] = np.inf
    if borders.shape[0] > 2:
        cutpoints[1:-1] = norm.ppf(_clip_open_unit_interval(borders[1:-1]))
    return cutpoints


def _make_score_spec_from_borders(borders: np.ndarray) -> ScoreSpec:
    borders = np.asarray(borders, dtype=float)
    scores = _borders_to_scores(borders)
    thresholds = 0.5 * (scores[:-1] + scores[1:])
    latent_cutpoints = _borders_to_latent_cutpoints(borders)
    return ScoreSpec(
        scores=scores,
        thresholds=thresholds,
        borders=borders,
        latent_cutpoints=latent_cutpoints,
        mode="optimized",
    )


def _make_naive_score_spec(n_classes: int) -> ScoreSpec:
    scores = np.arange(1, n_classes + 1, dtype=float)
    thresholds = np.arange(1.5, n_classes, dtype=float)
    latent_cutpoints = np.concatenate(([-np.inf], thresholds, [np.inf]))
    return ScoreSpec(
        scores=scores,
        thresholds=thresholds,
        borders=None,
        latent_cutpoints=latent_cutpoints,
        mode="naive",
    )


def _resolve_max_features(max_features, n_features: int) -> int:
    if max_features is None:
        return n_features
    if isinstance(max_features, str):
        if max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if max_features == "log2":
            return max(1, int(np.log2(n_features)))
        raise ValueError("Unsupported max_features string.")
    if isinstance(max_features, float):
        if not (0.0 < max_features <= 1.0):
            raise ValueError("Float max_features must lie in the interval (0, 1].")
        return max(1, int(np.ceil(max_features * n_features)))
    return max(1, min(int(max_features), n_features))


def _resolve_sample_fraction(bootstrap: bool, sample_fraction: Optional[float]) -> float:
    if sample_fraction is None:
        return 1.0 if bootstrap else 0.632
    if not (0.0 < sample_fraction <= 1.0):
        raise ValueError("sample_fraction must lie in the interval (0, 1].")
    return float(sample_fraction)


def _generate_diverse_candidate_borders(
    *,
    n_classes: int,
    n_sets: int,
    n_perm_trials: int,
    permute_per_default: bool,
    class_frequencies: np.ndarray,
    random_state,
) -> np.ndarray:
    """Generate a diverse collection of candidate class-border vectors."""

    rng = check_random_state(random_state)
    n_sets = int(n_sets)
    if n_sets < 1:
        raise ValueError("n_sets must be at least 1.")

    candidates = []

    equal_widths = np.full(n_classes, 1.0 / n_classes, dtype=float)
    candidates.append(np.concatenate(([0.0], np.cumsum(equal_widths))))

    if n_sets > 1:
        empirical_widths = class_frequencies / np.sum(class_frequencies)
        empirical_borders = np.concatenate(([0.0], np.cumsum(empirical_widths)))
        if np.max(np.abs(empirical_borders - candidates[0])) > 1e-12:
            candidates.append(empirical_borders)

    while len(candidates) < n_sets:
        widths = rng.dirichlet(np.ones(n_classes, dtype=float))
        permuted_widths = _select_diverse_width_permutation(
            widths=widths,
            previous_borders=candidates,
            n_perm_trials=n_perm_trials,
            permute_per_default=permute_per_default,
            random_state=rng,
        )
        borders = np.concatenate(([0.0], np.cumsum(permuted_widths)))
        borders[-1] = 1.0
        candidates.append(borders)

    return np.asarray(candidates[:n_sets], dtype=float)


def _select_diverse_width_permutation(
    *,
    widths: np.ndarray,
    previous_borders: Sequence[np.ndarray],
    n_perm_trials: int,
    permute_per_default: bool,
    random_state,
) -> np.ndarray:
    rng = check_random_state(random_state)
    n_classes = widths.shape[0]

    n_all = factorial(n_classes) if n_classes <= 8 else np.inf
    exhaustive = (not permute_per_default) and np.isfinite(n_all) and (n_all <= n_perm_trials)

    if exhaustive:
        try:
            import itertools

            permutations = [np.asarray(p, dtype=int) for p in itertools.permutations(np.arange(n_classes))]
        except Exception:
            permutations = [rng.permutation(n_classes) for _ in range(max(1, n_perm_trials))]
    else:
        permutations = [rng.permutation(n_classes) for _ in range(max(1, n_perm_trials))]

    if not previous_borders:
        return widths

    previous_internal = np.vstack([b[1:-1] for b in previous_borders])
    best_perm = widths
    best_distance = -np.inf
    for perm in permutations:
        candidate_widths = widths[perm]
        candidate_internal = np.cumsum(candidate_widths)[:-1]
        min_distance = float(np.min(np.linalg.norm(previous_internal - candidate_internal[None, :], axis=1)))
        if min_distance > best_distance:
            best_distance = min_distance
            best_perm = candidate_widths

    return best_perm


# -----------------------------------------------------------------------------
# Tree fitting and prediction helpers
# -----------------------------------------------------------------------------

def _draw_sample_indices(
    n_samples: int,
    *,
    bootstrap: bool,
    sample_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    draw_n = max(1, int(round(sample_fraction * n_samples)))
    draw_idx = rng.choice(n_samples, size=draw_n, replace=bootstrap)
    counts = np.bincount(draw_idx, minlength=n_samples).astype(int)
    return draw_idx, counts


def _resolve_always_feature_indices(
    always_split_features: Optional[Sequence[Union[int, str]]],
    feature_names: Optional[np.ndarray],
    n_features: int,
) -> np.ndarray:
    if always_split_features is None:
        return np.empty(0, dtype=int)

    resolved = []
    name_to_idx = None
    if feature_names is not None:
        name_to_idx = {str(name): idx for idx, name in enumerate(feature_names)}

    for value in always_split_features:
        if isinstance(value, (int, np.integer)):
            idx = int(value)
        else:
            if name_to_idx is None:
                raise ValueError(
                    "always_split_features contains names but no feature names are available. "
                    "Pass a DataFrame or integer feature indices."
                )
            key = str(value)
            if key not in name_to_idx:
                raise ValueError(f"Unknown feature name in always_split_features: {value!r}")
            idx = name_to_idx[key]
        if idx < 0 or idx >= n_features:
            raise ValueError(f"always_split_features index out of bounds: {idx}")
        resolved.append(idx)

    return np.unique(np.asarray(resolved, dtype=int))


def _select_tree_feature_subset(
    n_features: int,
    *,
    max_features: int,
    always_feature_indices: np.ndarray,
    random_state,
) -> np.ndarray:
    rng = check_random_state(random_state)
    mandatory = np.unique(np.asarray(always_feature_indices, dtype=int))
    if mandatory.size == 0:
        return np.arange(n_features, dtype=int)

    optional = np.setdiff1d(np.arange(n_features, dtype=int), mandatory, assume_unique=True)
    n_optional = max(0, min(int(max_features), optional.size))
    if n_optional == 0:
        return mandatory.copy()
    chosen_optional = rng.choice(optional, size=n_optional, replace=False)
    return np.sort(np.concatenate([mandatory, chosen_optional]).astype(int))


def _fit_candidate_tree(
    X: np.ndarray,
    y_cont: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray],
    bootstrap: bool,
    sample_fraction: float,
    max_features: int,
    min_samples_leaf: int,
    always_feature_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_features = X.shape
    _, inbag_counts = _draw_sample_indices(
        n_samples,
        bootstrap=bootstrap,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    inbag_mask = inbag_counts > 0
    inbag_idx = np.flatnonzero(inbag_mask)
    oob_idx = np.flatnonzero(~inbag_mask)

    feature_subset = _select_tree_feature_subset(
        n_features,
        max_features=max_features,
        always_feature_indices=always_feature_indices,
        random_state=seed + 17,
    )

    if always_feature_indices.size == 0:
        tree_max_features = max_features
    else:
        tree_max_features = None

    tree = DecisionTreeRegressor(
        max_features=tree_max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )

    fit_weights = inbag_counts[inbag_idx].astype(float)
    if sample_weight is not None:
        fit_weights *= sample_weight[inbag_idx]

    tree.fit(X[inbag_idx][:, feature_subset], y_cont[inbag_idx], sample_weight=fit_weights)

    if oob_idx.size == 0:
        return oob_idx, np.empty(0, dtype=float), feature_subset, inbag_counts

    pred = tree.predict(X[oob_idx][:, feature_subset])
    return oob_idx, pred, feature_subset, inbag_counts


def _fit_final_tree(
    X: np.ndarray,
    y_cont: np.ndarray,
    *,
    sample_weight: Optional[np.ndarray],
    bootstrap: bool,
    sample_fraction: float,
    max_features: int,
    min_samples_leaf: int,
    always_feature_indices: np.ndarray,
    seed: int,
    collect_oob: bool,
) -> tuple[TreeModel, np.ndarray, np.ndarray, np.ndarray]:
    n_samples, n_features = X.shape
    _, inbag_counts = _draw_sample_indices(
        n_samples,
        bootstrap=bootstrap,
        sample_fraction=sample_fraction,
        seed=seed,
    )
    inbag_mask = inbag_counts > 0
    inbag_idx = np.flatnonzero(inbag_mask)

    feature_subset = _select_tree_feature_subset(
        n_features,
        max_features=max_features,
        always_feature_indices=always_feature_indices,
        random_state=seed + 29,
    )
    tree_max_features = None if always_feature_indices.size > 0 else max_features

    tree = DecisionTreeRegressor(
        max_features=tree_max_features,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    fit_weights = inbag_counts[inbag_idx].astype(float)
    if sample_weight is not None:
        fit_weights *= sample_weight[inbag_idx]

    tree.fit(X[inbag_idx][:, feature_subset], y_cont[inbag_idx], sample_weight=fit_weights)
    model = TreeModel(regressor=tree, feature_indices=feature_subset)

    if not collect_oob:
        return model, np.empty(0, dtype=int), np.empty(0, dtype=float), inbag_counts

    oob_idx = np.flatnonzero(~inbag_mask)
    if oob_idx.size == 0:
        return model, oob_idx, np.empty(0, dtype=float), inbag_counts

    pred = tree.predict(X[oob_idx][:, feature_subset])
    return model, oob_idx, pred, inbag_counts


def _continuous_to_class_indices(pred_cont: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    return np.searchsorted(thresholds, pred_cont, side="right").astype(int)


def _continuous_to_class_proba(
    pred_cont: np.ndarray,
    *,
    latent_cutpoints: np.ndarray,
    sigma: float,
) -> np.ndarray:
    sigma = max(float(sigma), 1e-6)
    z = (latent_cutpoints[None, :] - pred_cont[:, None]) / sigma
    cdf = norm.cdf(z)
    proba = np.diff(cdf, axis=1)
    proba = np.clip(proba, 0.0, 1.0)
    row_sums = np.sum(proba, axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return proba / row_sums


def _estimate_sigma_from_oob(
    y_cont: np.ndarray,
    mean_cont: np.ndarray,
    covered_mask: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> float:
    if not np.any(covered_mask):
        return 1.0
    residuals = y_cont[covered_mask] - mean_cont[covered_mask]
    weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[covered_mask]
    return _weighted_std(residuals, sample_weight=weights)


def _aggregate_oob_predictions(
    *,
    oob_records: Sequence[tuple[np.ndarray, np.ndarray]],
    y_idx: np.ndarray,
    y_cont: np.ndarray,
    spec: ScoreSpec,
    performance_function: str,
    class_weights: Optional[np.ndarray],
    prioritized_class_index: Optional[int],
    sample_weight: Optional[np.ndarray] = None,
) -> dict:
    n_samples = y_idx.shape[0]
    n_classes = int(np.max(y_idx)) + 1

    counts = np.zeros(n_samples, dtype=int)
    cont_sum = np.zeros(n_samples, dtype=float)

    for oob_idx, pred_cont in oob_records:
        if oob_idx.size == 0:
            continue
        counts[oob_idx] += 1
        cont_sum[oob_idx] += pred_cont

    covered = counts > 0
    mean_cont = np.zeros(n_samples, dtype=float)
    mean_cont[covered] = cont_sum[covered] / counts[covered]
    sigma = _estimate_sigma_from_oob(
        y_cont=y_cont,
        mean_cont=mean_cont,
        covered_mask=covered,
        sample_weight=sample_weight,
    )

    masked_weights = None if sample_weight is None else np.asarray(sample_weight, dtype=float)[covered]

    if performance_function == "probability" and spec.latent_cutpoints is not None:
        proba_sum = np.zeros((n_samples, n_classes), dtype=float)
        for oob_idx, pred_cont in oob_records:
            if oob_idx.size == 0:
                continue
            proba_sum[oob_idx] += _continuous_to_class_proba(
                pred_cont,
                latent_cutpoints=spec.latent_cutpoints,
                sigma=sigma,
            )
        proba = np.zeros_like(proba_sum)
        proba[covered] = proba_sum[covered] / counts[covered, None]
        pred_idx = np.full(n_samples, -1, dtype=int)
        pred_idx[covered] = np.argmax(proba[covered], axis=1)
        perf = ordinal_performance_from_indices(
            y_idx[covered],
            y_pred_idx=pred_idx[covered],
            y_pred_proba=proba[covered],
            performance_function=performance_function,
            class_weights=class_weights,
            prioritized_class_index=prioritized_class_index,
            sample_weight=masked_weights,
        )
        return {
            "covered_mask": covered,
            "counts": counts,
            "mean_cont": mean_cont,
            "pred_idx": pred_idx,
            "proba": proba,
            "vote_proba": None,
            "sigma": sigma,
            "performance": perf,
        }

    vote_counts = np.zeros((n_samples, n_classes), dtype=float)
    for oob_idx, pred_cont in oob_records:
        if oob_idx.size == 0:
            continue
        cls_idx = _continuous_to_class_indices(pred_cont, spec.thresholds)
        vote_counts[oob_idx, cls_idx] += 1.0

    vote_proba = np.zeros_like(vote_counts)
    vote_proba[covered] = vote_counts[covered] / counts[covered, None]
    pred_idx = np.full(n_samples, -1, dtype=int)
    pred_idx[covered] = np.argmax(vote_counts[covered], axis=1)
    perf = ordinal_performance_from_indices(
        y_idx[covered],
        y_pred_idx=pred_idx[covered],
        y_pred_proba=None,
        performance_function=performance_function,
        class_weights=class_weights,
        prioritized_class_index=prioritized_class_index,
        sample_weight=masked_weights,
    )
    return {
        "covered_mask": covered,
        "counts": counts,
        "mean_cont": mean_cont,
        "pred_idx": pred_idx,
        "proba": vote_proba,
        "vote_proba": vote_proba,
        "sigma": sigma,
        "performance": perf,
    }


# -----------------------------------------------------------------------------
# Main estimator
# -----------------------------------------------------------------------------

class OrdinalForestClassifier(ClassifierMixin, BaseEstimator):
    """Ordinal Forest classifier with a scikit-learn compatible API.

    Parameters
    ----------
    n_sets : int, default=100
        Number of candidate score systems evaluated during score optimization.
    n_estimators_per_set : int, default=50
        Number of trees used in each small candidate-evaluation forest.
    n_estimators : int, default=500
        Number of trees in the final forest.
    performance_function : {"equal", "probability", "proportional", "oneclass", "custom"}, default="probability"
        Optimization objective used to rank candidate score systems.
    class_weight_vector : array-like of shape (n_classes,), default=None
        User-defined class weights for ``performance_function='custom'``.
    prioritized_class : object, default=None
        Class label emphasized when ``performance_function='oneclass'``.
    n_best : int, default=10
        Number of top candidate score systems averaged to form the final score system.
    naive : bool, default=False
        If True, skip score optimization and use the naive score vector 1, 2, ..., J.
    max_features : {"sqrt", "log2"} or int or float or None, default="sqrt"
        Number of non-mandatory features sampled for each tree. If no mandatory
        features are specified, this is also the per-split ``max_features`` value used
        inside each ``DecisionTreeRegressor``.
    min_samples_leaf : int or None, default=None
        Minimum number of observations per leaf. If None, defaults to 10 for
        ``performance_function='probability'`` and 5 otherwise.
    bootstrap : bool, default=True
        Whether each tree is fit on a bootstrap sample.
    sample_fraction : float or None, default=None
        Fraction of observations drawn for each tree. If None, defaults to 1.0 when
        ``bootstrap=True`` and 0.632 otherwise.
    class_order : sequence or None, default=None
        Explicit ordinal class order.
    always_split_features : sequence of int or str or None, default=None
        Approximate counterpart of CRAN ``always.split.variables``. Each tree is fit on a
        feature subspace that always contains these features plus randomly sampled
        additional features.
    n_perm_trials : int, default=500
        Number of width-order permutations evaluated when generating each random
        candidate border set.
    permute_per_default : bool, default=False
        If True, always try ``n_perm_trials`` random permutations.
    n_jobs : int or None, default=None
        Number of parallel jobs used in tree fitting and permutation importance.
    random_state : int, RandomState instance, or None, default=None
        Random seed.
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        *,
        n_sets: int = 100,
        n_estimators_per_set: int = 50,
        n_estimators: int = 500,
        performance_function: str = "probability",
        class_weight_vector: Optional[ArrayLike] = None,
        prioritized_class=None,
        n_best: int = 10,
        naive: bool = False,
        max_features: Union[str, int, float, None] = "sqrt",
        min_samples_leaf: Optional[int] = None,
        bootstrap: bool = True,
        sample_fraction: Optional[float] = None,
        class_order: Optional[Sequence] = None,
        always_split_features: Optional[Sequence[Union[int, str]]] = None,
        n_perm_trials: int = 500,
        permute_per_default: bool = False,
        n_jobs: Optional[int] = None,
        random_state=None,
        verbose: int = 0,
    ):
        self.n_sets = n_sets
        self.n_estimators_per_set = n_estimators_per_set
        self.n_estimators = n_estimators
        self.performance_function = performance_function
        self.class_weight_vector = class_weight_vector
        self.prioritized_class = prioritized_class
        self.n_best = n_best
        self.naive = naive
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.sample_fraction = sample_fraction
        self.class_order = class_order
        self.always_split_features = always_split_features
        self.n_perm_trials = n_perm_trials
        self.permute_per_default = permute_per_default
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None):
        """Fit the Ordinal Forest classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Ordered class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Optional non-negative observation weights. The weights affect tree fitting,
            score-set evaluation, OOB metrics, and the OOB-derived sigma estimate.
        """

        raw_X = X
        X, y = check_X_y(
            X,
            y,
            accept_sparse=False,
            dtype=np.float64,
            ensure_2d=True,
            ensure_all_finite=True,
        )
        check_classification_targets(y)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = self._extract_feature_names_in(raw_X)
        self.always_split_feature_indices_ = _resolve_always_feature_indices(
            self.always_split_features, self.feature_names_in_, self.n_features_in_
        )
        self.sample_weight_ = _normalize_weights(sample_weight, X.shape[0])

        self.classes_ = _infer_class_order(y, class_order=self.class_order)
        y_idx = _encode_labels_with_classes(y, self.classes_)
        self.n_classes_ = len(self.classes_)
        if self.n_classes_ < 2:
            raise ValueError("OrdinalForestClassifier requires at least two ordered classes.")

        if self.sample_weight_ is None:
            self.class_counts_ = np.bincount(y_idx, minlength=self.n_classes_).astype(int)
            self.class_weighted_counts_ = self.class_counts_.astype(float)
        else:
            self.class_counts_ = np.bincount(y_idx, minlength=self.n_classes_).astype(int)
            self.class_weighted_counts_ = np.bincount(
                y_idx, weights=self.sample_weight_, minlength=self.n_classes_
            ).astype(float)
        self.class_frequencies_ = self.class_weighted_counts_ / np.sum(self.class_weighted_counts_)

        self.max_features_ = _resolve_max_features(self.max_features, X.shape[1])
        self.sample_fraction_ = _resolve_sample_fraction(self.bootstrap, self.sample_fraction)
        self.min_samples_leaf_ = self._resolve_min_samples_leaf()
        self.class_weight_vector_ = self._resolve_class_weight_vector()
        self.prioritized_class_index_ = self._resolve_prioritized_class_index()

        rng = check_random_state(self.random_state)
        self.random_state_ = rng

        if self.naive:
            self.score_spec_ = _make_naive_score_spec(self.n_classes_)
            self.candidate_borders_ = None
            self.candidate_scores_ = None
            self.candidate_performances_ = None
            self.candidate_sigmas_ = None
            self.best_candidate_indices_ = None
        else:
            self._fit_score_optimization(X, y_idx, rng)

        self.optimized_scores_ = self.score_spec_.scores.copy()
        self.optimized_thresholds_ = self.score_spec_.thresholds.copy()
        self.optimized_borders_ = None if self.score_spec_.borders is None else self.score_spec_.borders.copy()
        self.latent_cutpoints_ = None if self.score_spec_.latent_cutpoints is None else self.score_spec_.latent_cutpoints.copy()

        self._fit_final_forest(X, y_idx, rng)
        return self

    def _extract_feature_names_in(self, X) -> Optional[np.ndarray]:
        try:
            columns = getattr(X, "columns", None)
            if columns is None:
                return None
            names = np.asarray(columns, dtype=object)
            if names.ndim == 1 and len(names) == X.shape[1]:
                return names
        except Exception:
            pass
        return None

    def _resolve_min_samples_leaf(self) -> int:
        if self.min_samples_leaf is not None:
            if int(self.min_samples_leaf) < 1:
                raise ValueError("min_samples_leaf must be at least 1.")
            return int(self.min_samples_leaf)
        return 10 if self.performance_function == "probability" else 5

    def _resolve_class_weight_vector(self) -> Optional[np.ndarray]:
        if self.performance_function != "custom":
            return None
        if self.class_weight_vector is None:
            raise ValueError("class_weight_vector must be provided when performance_function='custom'.")
        weights = np.asarray(self.class_weight_vector, dtype=float)
        if weights.shape != (self.n_classes_,):
            raise ValueError(
                f"class_weight_vector must have shape ({self.n_classes_},), got {weights.shape}."
            )
        if np.sum(weights) <= 0:
            raise ValueError("class_weight_vector must have a strictly positive sum.")
        return weights

    def _resolve_prioritized_class_index(self) -> Optional[int]:
        if self.performance_function != "oneclass":
            return None
        if self.prioritized_class is None:
            raise ValueError("prioritized_class must be provided when performance_function='oneclass'.")
        return int(_encode_labels_with_classes([self.prioritized_class], self.classes_)[0])

    def _fit_score_optimization(self, X: np.ndarray, y_idx: np.ndarray, rng) -> None:
        candidate_borders = _generate_diverse_candidate_borders(
            n_classes=self.n_classes_,
            n_sets=int(self.n_sets),
            n_perm_trials=int(self.n_perm_trials),
            permute_per_default=bool(self.permute_per_default),
            class_frequencies=self.class_weighted_counts_.astype(float),
            random_state=rng,
        )

        candidate_scores = np.empty((candidate_borders.shape[0], self.n_classes_), dtype=float)
        candidate_performances = np.empty(candidate_borders.shape[0], dtype=float)
        candidate_sigmas = np.empty(candidate_borders.shape[0], dtype=float)

        candidate_seeds = rng.randint(0, _INT32_MAX, size=candidate_borders.shape[0], dtype=np.int64)

        for idx, (borders, seed) in enumerate(zip(candidate_borders, candidate_seeds)):
            spec = _make_score_spec_from_borders(borders)
            candidate_scores[idx] = spec.scores
            result = self._evaluate_candidate_set(X, y_idx, spec, int(seed))
            candidate_performances[idx] = result["performance"]
            candidate_sigmas[idx] = result["sigma"]
            if self.verbose:
                print(
                    f"[OrdinalForest] Candidate {idx + 1}/{candidate_borders.shape[0]} objective={candidate_performances[idx]:.6f}"
                )

        n_best = max(1, min(int(self.n_best), candidate_borders.shape[0]))
        best_idx = np.argsort(candidate_performances)[-n_best:][::-1]
        optimized_borders = np.mean(candidate_borders[best_idx], axis=0)
        optimized_borders[0] = 0.0
        optimized_borders[-1] = 1.0

        self.candidate_borders_ = candidate_borders
        self.candidate_scores_ = candidate_scores
        self.candidate_performances_ = candidate_performances
        self.candidate_sigmas_ = candidate_sigmas
        self.best_candidate_indices_ = best_idx
        self.score_spec_ = _make_score_spec_from_borders(optimized_borders)
        self.optimization_history_ = Bunch(
            candidate_borders=candidate_borders,
            candidate_scores=candidate_scores,
            candidate_performances=candidate_performances,
            candidate_sigmas=candidate_sigmas,
            best_candidate_indices=best_idx,
            optimized_borders=optimized_borders,
            optimized_scores=self.score_spec_.scores.copy(),
        )

    def _evaluate_candidate_set(
        self,
        X: np.ndarray,
        y_idx: np.ndarray,
        spec: ScoreSpec,
        seed: int,
    ) -> dict:
        y_cont = spec.scores[y_idx]
        seeds = np.random.RandomState(seed).randint(
            0, _INT32_MAX, size=int(self.n_estimators_per_set), dtype=np.int64
        )

        records = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_candidate_tree)(
                X,
                y_cont,
                sample_weight=self.sample_weight_,
                bootstrap=self.bootstrap,
                sample_fraction=self.sample_fraction_,
                max_features=self.max_features_,
                min_samples_leaf=self.min_samples_leaf_,
                always_feature_indices=self.always_split_feature_indices_,
                seed=int(s),
            )
            for s in seeds
        )
        reduced_records = [(record[0], record[1]) for record in records]
        return _aggregate_oob_predictions(
            oob_records=reduced_records,
            y_idx=y_idx,
            y_cont=y_cont,
            spec=spec,
            performance_function=self.performance_function,
            class_weights=self.class_weight_vector_,
            prioritized_class_index=self.prioritized_class_index_,
            sample_weight=self.sample_weight_,
        )

    def _fit_final_forest(self, X: np.ndarray, y_idx: np.ndarray, rng) -> None:
        y_cont = self.score_spec_.scores[y_idx]
        seeds = rng.randint(0, _INT32_MAX, size=int(self.n_estimators), dtype=np.int64)

        final_results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_fit_final_tree)(
                X,
                y_cont,
                sample_weight=self.sample_weight_,
                bootstrap=self.bootstrap,
                sample_fraction=self.sample_fraction_,
                max_features=self.max_features_,
                min_samples_leaf=self.min_samples_leaf_,
                always_feature_indices=self.always_split_feature_indices_,
                seed=int(s),
                collect_oob=True,
            )
            for s in seeds
        )

        self.trees_ = [result[0] for result in final_results]
        oob_records = [(result[1], result[2]) for result in final_results]
        self.inbag_counts_per_tree_ = np.vstack([result[3] for result in final_results])
        oob_summary = _aggregate_oob_predictions(
            oob_records=oob_records,
            y_idx=y_idx,
            y_cont=y_cont,
            spec=self.score_spec_,
            performance_function=self.performance_function,
            class_weights=self.class_weight_vector_,
            prioritized_class_index=self.prioritized_class_index_,
            sample_weight=self.sample_weight_,
        )

        self.sigma_ = float(oob_summary["sigma"])
        self.oob_counts_ = oob_summary["counts"].astype(int)
        self.oob_decision_function_ = oob_summary["mean_cont"]
        self.oob_predicted_class_indices_ = oob_summary["pred_idx"]
        self.oob_predicted_class_proba_ = oob_summary["proba"]
        self.oob_coverage_mask_ = oob_summary["covered_mask"]
        self.oob_score_ = float(oob_summary["performance"])

        self.oob_prediction_ = np.full(y_idx.shape[0], None, dtype=object)
        covered = self.oob_coverage_mask_
        self.oob_prediction_[covered] = self.classes_[self.oob_predicted_class_indices_[covered]]

        feature_importance_matrix = np.zeros((len(self.trees_), self.n_features_in_), dtype=float)
        for tree_idx, model in enumerate(self.trees_):
            feature_importance_matrix[tree_idx, model.feature_indices] = model.regressor.feature_importances_
        self.feature_importances_ = np.mean(feature_importance_matrix, axis=0)
        self.feature_importances_std_ = np.std(feature_importance_matrix, axis=0, ddof=0)

        self.final_forest_summary_ = Bunch(
            n_trees=len(self.trees_),
            sigma=self.sigma_,
            oob_score=self.oob_score_,
            score_spec_mode=self.score_spec_.mode,
            always_split_feature_indices=self.always_split_feature_indices_.copy(),
        )

        self.oob_diagnostics_ = self._build_oob_diagnostics(y_idx)

        if self.performance_function == "probability" and self.score_spec_.latent_cutpoints is None:
            warnings.warn(
                "Probability mode was requested but no latent cutpoints are available. "
                "predict_proba will fall back to vote frequencies.",
                RuntimeWarning,
            )

    def _build_oob_diagnostics(self, y_idx: np.ndarray) -> Bunch:
        covered = self.oob_coverage_mask_
        if not np.any(covered):
            return Bunch(
                n_covered=0,
                coverage_rate=0.0,
                coverage_rate_per_class=np.zeros(self.n_classes_, dtype=float),
                weighted_coverage_rate=0.0,
                oob_counts=self.oob_counts_.copy(),
                confusion_matrix=np.zeros((self.n_classes_, self.n_classes_), dtype=int),
                per_class_youden=np.zeros(self.n_classes_, dtype=float),
                accuracy=np.nan,
                weighted_accuracy=np.nan,
                rank_mae=np.nan,
                weighted_rank_mae=np.nan,
                rank_mse=np.nan,
                weighted_rank_mse=np.nan,
                rps=np.nan,
                weighted_rps=np.nan,
                objective=np.nan,
            )

        y_cov = y_idx[covered]
        pred_cov = self.oob_predicted_class_indices_[covered]
        cov_weights = None if self.sample_weight_ is None else self.sample_weight_[covered]

        coverage_rate_per_class = np.zeros(self.n_classes_, dtype=float)
        for cls_idx in range(self.n_classes_):
            cls_mask = y_idx == cls_idx
            if np.any(cls_mask):
                coverage_rate_per_class[cls_idx] = float(np.mean(covered[cls_mask]))

        weighted_coverage_rate = (
            float(np.average(covered.astype(float), weights=self.sample_weight_))
            if self.sample_weight_ is not None
            else float(np.mean(covered))
        )

        per_class_youden = youden_j_per_class(
            y_cov,
            pred_cov,
            n_classes=self.n_classes_,
            sample_weight=cov_weights,
        )
        confusion = confusion_matrix(
            y_cov,
            pred_cov,
            labels=np.arange(self.n_classes_),
        )
        accuracy = _weighted_accuracy_from_indices(y_cov, pred_cov)
        weighted_accuracy = _weighted_accuracy_from_indices(y_cov, pred_cov, sample_weight=cov_weights)
        rank_mae = _weighted_mean(np.abs(y_cov - pred_cov))
        weighted_rank_mae = _weighted_mean(np.abs(y_cov - pred_cov), cov_weights)
        rank_mse = _weighted_mean((y_cov - pred_cov) ** 2)
        weighted_rank_mse = _weighted_mean((y_cov - pred_cov) ** 2, cov_weights)
        rps = ranked_probability_score_from_indices(y_cov, self.oob_predicted_class_proba_[covered])
        weighted_rps = ranked_probability_score_from_indices(
            y_cov,
            self.oob_predicted_class_proba_[covered],
            sample_weight=cov_weights,
        )
        objective = ordinal_performance_from_indices(
            y_cov,
            y_pred_idx=pred_cov,
            y_pred_proba=self.oob_predicted_class_proba_[covered],
            performance_function=self.performance_function,
            class_weights=self.class_weight_vector_,
            prioritized_class_index=self.prioritized_class_index_,
            sample_weight=cov_weights,
        )
        inbag_fraction = np.mean(self.inbag_counts_per_tree_ > 0, axis=0)

        return Bunch(
            n_covered=int(np.sum(covered)),
            coverage_rate=float(np.mean(covered)),
            coverage_rate_per_class=coverage_rate_per_class,
            weighted_coverage_rate=weighted_coverage_rate,
            oob_counts=self.oob_counts_.copy(),
            inbag_fraction=inbag_fraction,
            confusion_matrix=confusion,
            per_class_youden=per_class_youden,
            accuracy=accuracy,
            weighted_accuracy=weighted_accuracy,
            rank_mae=rank_mae,
            weighted_rank_mae=weighted_rank_mae,
            rank_mse=rank_mse,
            weighted_rank_mse=weighted_rank_mse,
            rps=rps,
            weighted_rps=weighted_rps,
            objective=objective,
        )

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Return the mean continuous forest prediction in score space."""

        check_is_fitted(self, "trees_")
        X = check_array(X, accept_sparse=False, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        cont = self._predict_tree_continuous_matrix(X)
        return np.mean(cont, axis=1)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities."""

        check_is_fitted(self, "trees_")
        X = check_array(X, accept_sparse=False, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        cont_matrix = self._predict_tree_continuous_matrix(X)

        if self.performance_function == "probability" and self.score_spec_.latent_cutpoints is not None:
            proba_sum = np.zeros((X.shape[0], self.n_classes_), dtype=float)
            for tree_idx in range(cont_matrix.shape[1]):
                proba_sum += _continuous_to_class_proba(
                    cont_matrix[:, tree_idx],
                    latent_cutpoints=self.score_spec_.latent_cutpoints,
                    sigma=self.sigma_,
                )
            proba = proba_sum / cont_matrix.shape[1]
            row_sums = np.sum(proba, axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            return proba / row_sums

        class_idx = _continuous_to_class_indices(cont_matrix.ravel(), self.score_spec_.thresholds)
        class_idx = class_idx.reshape(cont_matrix.shape)
        votes = np.zeros((X.shape[0], self.n_classes_), dtype=float)
        for tree_idx in range(class_idx.shape[1]):
            votes[np.arange(X.shape[0]), class_idx[:, tree_idx]] += 1.0
        return votes / class_idx.shape[1]

    def predict_log_proba(self, X: ArrayLike) -> np.ndarray:
        return np.log(np.clip(self.predict_proba(X), _EPS, 1.0))

    def predict_cumulative_proba(self, X: ArrayLike) -> np.ndarray:
        """Return cumulative class probabilities P(Y <= class_j)."""

        return np.cumsum(self.predict_proba(X), axis=1)

    def predict(self, X: ArrayLike) -> np.ndarray:
        check_is_fitted(self, "trees_")
        X = check_array(X, accept_sparse=False, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)

        if self.performance_function == "probability" and self.score_spec_.latent_cutpoints is not None:
            proba = self.predict_proba(X)
            pred_idx = np.argmax(proba, axis=1)
            return self.classes_[pred_idx]

        cont_matrix = self._predict_tree_continuous_matrix(X)
        class_idx = _continuous_to_class_indices(cont_matrix.ravel(), self.score_spec_.thresholds)
        class_idx = class_idx.reshape(cont_matrix.shape)
        vote_counts = np.zeros((X.shape[0], self.n_classes_), dtype=int)
        for tree_idx in range(class_idx.shape[1]):
            vote_counts[np.arange(X.shape[0]), class_idx[:, tree_idx]] += 1
        pred_idx = np.argmax(vote_counts, axis=1)
        return self.classes_[pred_idx]

    def _predict_tree_continuous_matrix(self, X: np.ndarray) -> np.ndarray:
        preds = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(model.regressor.predict)(X[:, model.feature_indices]) for model in self.trees_
        )
        return np.column_stack(preds)

    def score(self, X: ArrayLike, y: ArrayLike) -> float:
        """Return standard classification accuracy for scikit-learn compatibility."""

        return accuracy_score(y, self.predict(X))

    def objective_score(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: Optional[ArrayLike] = None,
    ) -> float:
        """Evaluate predictions with the same weighted objective used during fitting."""

        check_is_fitted(self, "trees_")
        X = check_array(X, accept_sparse=False, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        y_idx = _encode_labels_with_classes(y, self.classes_)
        eval_weight = _normalize_weights(sample_weight, X.shape[0])

        if self.performance_function == "probability" and self.score_spec_.latent_cutpoints is not None:
            proba = self.predict_proba(X)
            pred_idx = np.argmax(proba, axis=1)
            return ordinal_performance_from_indices(
                y_idx,
                y_pred_idx=pred_idx,
                y_pred_proba=proba,
                performance_function=self.performance_function,
                class_weights=self.class_weight_vector_,
                prioritized_class_index=self.prioritized_class_index_,
                sample_weight=eval_weight,
            )

        pred_idx = _encode_labels_with_classes(self.predict(X), self.classes_)
        return ordinal_performance_from_indices(
            y_idx,
            y_pred_idx=pred_idx,
            y_pred_proba=None,
            performance_function=self.performance_function,
            class_weights=self.class_weight_vector_,
            prioritized_class_index=self.prioritized_class_index_,
            sample_weight=eval_weight,
        )

    def permutation_importance(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        n_repeats: int = 5,
        scoring: Union[str, Callable, None] = None,
        sample_weight: Optional[ArrayLike] = None,
        n_jobs: Optional[int] = None,
        random_state=None,
    ) -> Bunch:
        """Compute permutation importance with strict sample-weight propagation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Evaluation design matrix.
        y : array-like of shape (n_samples,)
            Evaluation labels.
        n_repeats : int, default=5
            Number of random permutations per feature.
        scoring : {"objective", "accuracy", "neg_rank_mae", "neg_rank_mse", "neg_rps"} or callable, default=None
            Scoring rule. When None, the default is ``"objective"`` so that the
            same weighted ordinal objective used during fitting is also used for
            importance estimation. Custom callables may accept either
            ``(estimator, X, y)`` or ``(estimator, X, y, sample_weight)``.
        sample_weight : array-like of shape (n_samples,), default=None
            Evaluation weights used consistently for the baseline score and every
            permuted score.
        n_jobs : int or None, default=None
            Number of parallel jobs.
        random_state : int, RandomState instance, or None, default=None
            Random seed.

        Returns
        -------
        bunch : sklearn.utils.Bunch
            Dictionary-like object containing the per-repeat importance matrix,
            feature-wise means and standard deviations, the baseline score, and
            the raw permuted scores.
        """

        check_is_fitted(self, "trees_")
        X = check_array(X, accept_sparse=False, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        y = np.asarray(y)
        eval_weight = _normalize_weights(sample_weight, X.shape[0])

        if scoring is None:
            scoring = "objective"

        baseline = self._score_external(X, y, scoring, sample_weight=eval_weight)
        rng = check_random_state(random_state)
        seeds = rng.randint(0, _INT32_MAX, size=X.shape[1], dtype=np.int64)
        n_jobs = self.n_jobs if n_jobs is None else n_jobs

        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_permute_one_feature)(
                estimator=self,
                X=X,
                y=y,
                sample_weight=eval_weight,
                feature_idx=feature_idx,
                n_repeats=n_repeats,
                scoring=scoring,
                seed=int(seed),
            )
            for feature_idx, seed in enumerate(seeds)
        )

        permuted_scores = np.vstack(results)
        importances = baseline - permuted_scores
        return Bunch(
            importances=importances,
            importances_mean=np.mean(importances, axis=1),
            importances_std=np.std(importances, axis=1, ddof=0),
            baseline_score=baseline,
            permuted_scores=permuted_scores,
            sample_weight=None if eval_weight is None else eval_weight.copy(),
            scoring=scoring,
        )

    def _score_external(
        self,
        X: np.ndarray,
        y: np.ndarray,
        scoring: Union[str, Callable],
        sample_weight: Optional[np.ndarray] = None,
    ) -> float:
        if callable(scoring):
            signature = inspect.signature(scoring)
            n_params = len(signature.parameters)
            if n_params >= 4:
                return float(scoring(self, X, y, sample_weight))
            return float(scoring(self, X, y))
        if scoring == "accuracy":
            pred = self.predict(X)
            y_idx = _encode_labels_with_classes(y, self.classes_)
            pred_idx = _encode_labels_with_classes(pred, self.classes_)
            return _weighted_accuracy_from_indices(y_idx, pred_idx, sample_weight=sample_weight)
        if scoring == "neg_rank_mae":
            return -ordinal_rank_mae_from_labels(y, self.predict(X), self.classes_, sample_weight=sample_weight)
        if scoring == "neg_rank_mse":
            return -ordinal_rank_mse_from_labels(y, self.predict(X), self.classes_, sample_weight=sample_weight)
        if scoring == "neg_rps":
            return -ranked_probability_score_from_proba(
                y,
                self.predict_proba(X),
                self.classes_,
                sample_weight=sample_weight,
            )
        if scoring == "objective":
            return float(self.objective_score(X, y, sample_weight=sample_weight))
        raise ValueError(
            "scoring must be one of {'objective', 'accuracy', 'neg_rank_mae', 'neg_rank_mse', 'neg_rps'} or a callable."
        )

    def get_optimization_summary(self) -> Bunch:
        """Return a compact summary of the score-optimization stage."""

        check_is_fitted(self, "trees_")
        return Bunch(
            naive=self.naive,
            performance_function=self.performance_function,
            optimized_scores=self.optimized_scores_.copy(),
            optimized_thresholds=self.optimized_thresholds_.copy(),
            optimized_borders=None if self.optimized_borders_ is None else self.optimized_borders_.copy(),
            best_candidate_indices=None if self.best_candidate_indices_ is None else self.best_candidate_indices_.copy(),
            oob_score=self.oob_score_,
            sigma=self.sigma_,
            always_split_feature_indices=self.always_split_feature_indices_.copy(),
        )

    def get_oob_diagnostics(self) -> Bunch:
        """Return detailed out-of-bag diagnostics."""

        check_is_fitted(self, "oob_diagnostics_")
        return self.oob_diagnostics_

    def summarize_r_parity_differences(self, parity_result: Bunch) -> str:
        """Create a compact human-readable summary from ``compare_with_r_ordinalforest``."""

        if parity_result.status != "ok":
            return f"Parity test status: {parity_result.status}. Message: {parity_result.message}"

        lines = [
            f"Status: {parity_result.status}",
            f"Exact class agreement: {parity_result.class_agreement:.4f}",
            f"Mean absolute rank difference: {parity_result.mean_absolute_rank_difference:.6f}",
        ]
        if parity_result.python_proba is not None and parity_result.r_proba is not None:
            lines.append(f"Mean absolute probability difference: {parity_result.mean_absolute_probability_difference:.6f}")
            lines.append(f"Maximum absolute probability difference: {parity_result.max_absolute_probability_difference:.6f}")
        if parity_result.notes:
            lines.extend(parity_result.notes)
        return "\n".join(lines)

    def plot_optimization_history(self, ax=None, *, highlight_top: Optional[int] = None):
        """Plot candidate score-set objective values."""

        check_is_fitted(self, "trees_")
        if self.candidate_performances_ is None:
            raise ValueError("No optimization history is available when naive=True.")

        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4.5))

        x = np.arange(1, self.candidate_performances_.shape[0] + 1)
        ax.plot(x, self.candidate_performances_, marker="o", linestyle="", alpha=0.65)

        if highlight_top is None:
            highlight_top = len(self.best_candidate_indices_)
        highlight_top = max(1, min(int(highlight_top), len(self.best_candidate_indices_)))
        top_idx = self.best_candidate_indices_[:highlight_top]
        ax.plot(
            x[top_idx],
            self.candidate_performances_[top_idx],
            marker="o",
            linestyle="",
            markersize=8,
        )

        ax.set_title("Ordinal Forest Candidate Score-Set Optimization")
        ax.set_xlabel("Candidate Score-Set Index")
        ax.set_ylabel("Objective Value")
        return ax

    def plot_score_profile(self, ax=None):
        """Plot the optimized continuous score assigned to each ordinal class."""

        check_is_fitted(self, "trees_")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        x = np.arange(self.n_classes_)
        ax.plot(x, self.optimized_scores_, marker="o")
        ax.set_xticks(x)
        ax.set_xticklabels(self.classes_)
        ax.set_title("Optimized Ordinal Score Profile")
        ax.set_xlabel("Ordered Class")
        ax.set_ylabel("Assigned Continuous Score")
        return ax

    def plot_feature_importance(
        self,
        *,
        importances: Optional[np.ndarray] = None,
        top_k: Optional[int] = None,
        ax=None,
    ):
        """Plot feature importances."""

        check_is_fitted(self, "trees_")
        import matplotlib.pyplot as plt

        if importances is None:
            importances = self.feature_importances_
        values = np.asarray(importances, dtype=float)
        names = self.feature_names_in_
        if names is None:
            names = np.asarray([f"x{i}" for i in range(values.shape[0])], dtype=object)

        order = np.argsort(values)[::-1]
        if top_k is not None:
            order = order[: max(1, int(top_k))]

        ordered_values = values[order]
        ordered_names = names[order]

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 0.4 * len(order) + 2.0))

        ax.barh(np.arange(len(order)), ordered_values[::-1])
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(ordered_names[::-1])
        ax.set_title("Ordinal Forest Feature Importance")
        ax.set_xlabel("Importance")
        ax.set_ylabel("Feature")
        return ax

    def plot_oob_coverage(self, ax=None):
        """Plot the histogram of per-observation OOB counts."""

        check_is_fitted(self, "oob_diagnostics_")
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7.5, 4.2))

        ax.hist(self.oob_counts_, bins=min(30, max(5, int(np.sqrt(len(self.oob_counts_))))))
        ax.set_title("Out-of-Bag Coverage Distribution")
        ax.set_xlabel("Number of OOB Predictions per Observation")
        ax.set_ylabel("Frequency")
        return ax

    def plot_oob_confusion_matrix(self, ax=None, *, normalize: bool = False):
        """Plot the OOB confusion matrix."""

        check_is_fitted(self, "oob_diagnostics_")
        import matplotlib.pyplot as plt

        matrix = self.oob_diagnostics_.confusion_matrix.astype(float)
        if normalize:
            row_sums = matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0.0] = 1.0
            matrix = matrix / row_sums

        if ax is None:
            _, ax = plt.subplots(figsize=(5.5, 4.5))

        image = ax.imshow(matrix, aspect="auto")
        plt.colorbar(image, ax=ax)
        ax.set_title("OOB Confusion Matrix")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("True Class")
        ax.set_xticks(np.arange(self.n_classes_))
        ax.set_xticklabels(self.classes_)
        ax.set_yticks(np.arange(self.n_classes_))
        ax.set_yticklabels(self.classes_)
        return ax

    def compare_with_r_ordinalforest(
        self,
        X_train: ArrayLike,
        y_train: ArrayLike,
        X_test: ArrayLike,
        y_test: Optional[ArrayLike] = None,
        *,
        rscript_path: str = "Rscript",
        package_name: str = "ordinalForest",
        cleanup: bool = True,
    ) -> Bunch:
        """Run a parity comparison against the CRAN R package.

        This helper requires a local R installation with ``Rscript`` and the CRAN package
        ``ordinalForest`` already installed.

        Returns
        -------
        Bunch
            Contains execution status, paths, R and Python predictions, and summary
            discrepancy statistics.
        """

        check_is_fitted(self, "trees_")
        if shutil.which(rscript_path) is None:
            return Bunch(status="skipped", message=f"{rscript_path!r} was not found on PATH.")

        if self.sample_weight_ is not None:
            notes = [
                "The Python model was fitted with sample weights.",
                "The CRAN ordinalForest package does not expose a matching sample-weight interface.",
                "Expect differences attributable to weighting semantics.",
            ]
        else:
            notes = []

        X_train_arr = check_array(X_train, accept_sparse=False, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        X_test_arr = check_array(X_test, accept_sparse=False, dtype=np.float64, ensure_2d=True, ensure_all_finite=True)
        y_train_arr = np.asarray(y_train)
        y_test_arr = None if y_test is None else np.asarray(y_test)

        if X_train_arr.shape[1] != self.n_features_in_ or X_test_arr.shape[1] != self.n_features_in_:
            raise ValueError("X_train and X_test must match the fitted feature dimension.")

        feature_names = self.feature_names_in_
        if feature_names is None:
            feature_names = np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)

        workdir = tempfile.mkdtemp(prefix="ordinal_forest_r_parity_")
        train_csv = os.path.join(workdir, "train.csv")
        test_csv = os.path.join(workdir, "test.csv")
        pred_json = os.path.join(workdir, "r_predictions.json")
        script_path = os.path.join(workdir, "run_parity.R")

        _write_matrix_with_target(train_csv, X_train_arr, y_train_arr, feature_names, target_name="target")
        _write_matrix_with_target(test_csv, X_test_arr, None, feature_names, target_name="target")

        always_names = [str(feature_names[idx]) for idx in self.always_split_feature_indices_]
        class_weights = None if self.class_weight_vector_ is None else self.class_weight_vector_.tolist()
        prioritized = None if self.prioritized_class is None else str(self.prioritized_class)
        class_order = [str(x) for x in self.classes_]

        r_code = _build_r_parity_script(
            package_name=package_name,
            train_csv=train_csv,
            test_csv=test_csv,
            output_json=pred_json,
            feature_names=[str(x) for x in feature_names],
            class_order=class_order,
            depvar="target",
            performance_function=self.performance_function,
            class_weight_vector=class_weights,
            prioritized_class=prioritized,
            n_sets=int(self.n_sets),
            n_estimators_per_set=int(self.n_estimators_per_set),
            n_estimators=int(self.n_estimators),
            n_best=int(self.n_best),
            naive=bool(self.naive),
            max_features=int(self.max_features_),
            min_samples_leaf=int(self.min_samples_leaf_),
            bootstrap=bool(self.bootstrap),
            sample_fraction=float(self.sample_fraction_),
            n_perm_trials=int(self.n_perm_trials),
            permute_per_default=bool(self.permute_per_default),
            always_split_features=always_names,
            random_state=int(check_random_state(self.random_state).randint(0, _INT32_MAX)),
        )
        Path(script_path).write_text(r_code, encoding="utf-8")

        try:
            proc = subprocess.run(
                [rscript_path, script_path],
                cwd=workdir,
                text=True,
                capture_output=True,
                check=False,
            )
            if proc.returncode != 0:
                return Bunch(
                    status="failed",
                    message=f"R parity script exited with code {proc.returncode}.",
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    workdir=workdir,
                )
            if not os.path.exists(pred_json):
                return Bunch(
                    status="failed",
                    message="The R parity script completed but did not create the expected JSON output.",
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    workdir=workdir,
                )
            payload = json.loads(Path(pred_json).read_text(encoding="utf-8"))
        finally:
            if cleanup and os.path.isdir(workdir):
                shutil.rmtree(workdir, ignore_errors=True)

        python_pred = self.predict(X_test_arr)
        r_pred = np.asarray(payload["predicted_class"], dtype=object)
        if r_pred.shape[0] != X_test_arr.shape[0]:
            return Bunch(
                status="failed",
                message="R output length does not match the number of test observations.",
                payload=payload,
            )

        python_pred_idx = _encode_labels_with_classes(python_pred, self.classes_)
        r_pred_idx = _encode_labels_with_classes(r_pred, self.classes_)
        class_agreement = float(np.mean(python_pred_idx == r_pred_idx))
        mean_abs_rank_diff = float(np.mean(np.abs(python_pred_idx - r_pred_idx)))

        python_proba = None
        r_proba = None
        mean_abs_proba_diff = None
        max_abs_proba_diff = None

        if payload.get("predicted_proba") is not None:
            r_proba = np.asarray(payload["predicted_proba"], dtype=float)
            python_proba = self.predict_proba(X_test_arr)
            if r_proba.shape == python_proba.shape:
                abs_diff = np.abs(python_proba - r_proba)
                mean_abs_proba_diff = float(np.mean(abs_diff))
                max_abs_proba_diff = float(np.max(abs_diff))
            else:
                notes.append("R probability output shape differed from Python probability output shape.")

        result = Bunch(
            status="ok",
            message="Parity comparison completed.",
            class_agreement=class_agreement,
            mean_absolute_rank_difference=mean_abs_rank_diff,
            python_prediction=python_pred,
            r_prediction=r_pred,
            python_proba=python_proba,
            r_proba=r_proba,
            mean_absolute_probability_difference=mean_abs_proba_diff,
            max_absolute_probability_difference=max_abs_proba_diff,
            notes=notes,
            payload=payload,
        )

        if y_test_arr is not None:
            result.python_accuracy = float(accuracy_score(y_test_arr, python_pred))
            result.r_accuracy = float(accuracy_score(y_test_arr, r_pred))
        return result


# -----------------------------------------------------------------------------
# R parity helpers
# -----------------------------------------------------------------------------

def _write_matrix_with_target(
    filepath: str,
    X: np.ndarray,
    y: Optional[np.ndarray],
    feature_names: np.ndarray,
    *,
    target_name: str,
) -> None:
    header = [str(name) for name in feature_names]
    if y is not None:
        header = header + [target_name]
    with open(filepath, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row_idx in range(X.shape[0]):
            row = list(map(float, X[row_idx]))
            if y is not None:
                row.append(str(y[row_idx]))
            writer.writerow(row)


def _r_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, str):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("Non-finite numeric values cannot be serialized to R literals.")
        return repr(float(value))
    if isinstance(value, (list, tuple, np.ndarray)):
        elements = ", ".join(_r_literal(x) for x in value)
        return f"c({elements})"
    raise TypeError(f"Unsupported value for R serialization: {type(value)!r}")


def _build_r_parity_script(
    *,
    package_name: str,
    train_csv: str,
    test_csv: str,
    output_json: str,
    feature_names: list[str],
    class_order: list[str],
    depvar: str,
    performance_function: str,
    class_weight_vector: Optional[list[float]],
    prioritized_class: Optional[str],
    n_sets: int,
    n_estimators_per_set: int,
    n_estimators: int,
    n_best: int,
    naive: bool,
    max_features: int,
    min_samples_leaf: int,
    bootstrap: bool,
    sample_fraction: float,
    n_perm_trials: int,
    permute_per_default: bool,
    always_split_features: list[str],
    random_state: int,
) -> str:
    return f"""
options(warn = 1)
suppressPackageStartupMessages(library({package_name}))
if (!requireNamespace(\"jsonlite\", quietly = TRUE)) {{
  stop(\"The R package 'jsonlite' is required for parity testing.\")
}}

set.seed({_r_literal(random_state)})
train_df <- read.csv({_r_literal(train_csv)}, stringsAsFactors = FALSE)
test_df <- read.csv({_r_literal(test_csv)}, stringsAsFactors = FALSE)
feature_names <- {_r_literal(feature_names)}
class_order <- {_r_literal(class_order)}
depvar <- {_r_literal(depvar)}
train_df[[depvar]] <- factor(train_df[[depvar]], levels = class_order, ordered = TRUE)

res <- ordfor(
  depvar = depvar,
  data = train_df,
  nsets = {_r_literal(n_sets)},
  ntreeperdiv = {_r_literal(n_estimators_per_set)},
  ntreefinal = {_r_literal(n_estimators)},
  perffunction = {_r_literal(performance_function)},
  classweights = {_r_literal(class_weight_vector)},
  nbest = {_r_literal(n_best)},
  naive = {_r_literal(naive)},
  npermtrial = {_r_literal(n_perm_trials)},
  permperdefault = {_r_literal(permute_per_default)},
  mtry = {_r_literal(max_features)},
  min.node.size = {_r_literal(min_samples_leaf)},
  replace = {_r_literal(bootstrap)},
  sample.fraction = {_r_literal(sample_fraction)},
  always.split.variables = {_r_literal(always_split_features)},
  classind = {_r_literal(prioritized_class)}
)

pred <- predict(res, newdata = test_df)
out <- list(predicted_class = as.character(pred$ypred))
if (!is.null(pred$ypredprob)) {{
  out$predicted_proba <- unname(as.data.frame(pred$ypredprob))
}} else {{
  out$predicted_proba <- NULL
}}
jsonlite::write_json(out, path = {_r_literal(output_json)}, auto_unbox = TRUE, dataframe = \"rows\")
""".strip() + "\n"


# -----------------------------------------------------------------------------
# Permutation helper
# -----------------------------------------------------------------------------

def _permute_one_feature(
    *,
    estimator: OrdinalForestClassifier,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: Optional[np.ndarray],
    feature_idx: int,
    n_repeats: int,
    scoring: Union[str, Callable],
    seed: int,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    X_perm = X.copy()
    original = X[:, feature_idx].copy()
    scores = np.empty(n_repeats, dtype=float)

    for repeat_idx in range(n_repeats):
        X_perm[:, feature_idx] = original[rng.permutation(X.shape[0])]
        scores[repeat_idx] = estimator._score_external(X_perm, y, scoring, sample_weight=sample_weight)

    return scores
