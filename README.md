# ordinalForest (Python)

`ordinalForest` のアイデアに着想を得た、**順序付き分類 (ordinal classification)** のための Python 実装です。scikit-learn 互換の API を持ち、**順序情報を考慮したランダムフォレストモデル**を扱えます。実装の中心は `OrdinalForestClassifier` で、候補スコア系の探索、OOB による内部評価、順序-aware な予測確率、診断可視化、Permutation Importance などを提供します。

## 特徴

- **scikit-learn 互換 API**
  - `fit`, `predict`, `predict_proba`, `score`, `decision_function` を提供します。
- **候補スコア系の最適化**
  - 複数の score system を生成し、OOB 指標で最良候補を選びます。`naive=False` のとき最適化が有効です。
- **複数の最適化目的関数**
  - `equal`, `proportional`, `oneclass`, `custom`, `probability` をサポートします。
- **サンプル重み対応**
  - 学習時、OOB 評価時、Permutation Importance 時に `sample_weight` を使えます。
- **OOB 診断**
  - カバレッジ、混同行列、rank-based 指標、RPS、objective などの詳細診断を取得できます。
- **可視化ユーティリティ**
  - 最適化履歴、スコアプロファイル、特徴量重要度、OOB カバレッジ、OOB 混同行列を描画できます。
- **R `ordinalForest` との比較補助**
  - ローカルに `Rscript` と R パッケージ `ordinalForest` がある場合、予測結果の差分比較ができます。

## この実装が向いている問題

次のような **順序があるクラス分類** を想定しています。

- 満足度: `low < medium < high`
- 重症度: `mild < moderate < severe`
- レーティング: `1 < 2 < 3 < 4 < 5`

通常の多クラス分類器は「クラス間の順序距離」を扱いません。一方でこの実装は、**隣接クラスとのズレ**と**大きな順序ジャンプ**を区別しやすい設計です。チュートリアルノートブックでも、Random Forest や Ordinal Logistic Regression と比較しながら、その違いを説明しています。

## ファイル構成

```text
.
├── ordinal_forest.py
├── ordinal_forest_tutorial.ipynb
├── README.md
└── LICENSE
```

- `ordinal_forest.py`: 本体実装
- `ordinal_forest_tutorial.ipynb`: 実験・比較・可視化を含むチュートリアルノートブック

## インストール

単一モジュール構成なので、まずはこのリポジトリを clone し、必要ライブラリをインストールしてください。

```bash
pip install numpy scipy scikit-learn joblib matplotlib
```

チュートリアルノートブックも動かす場合は、追加で以下があると便利です。

```bash
pip install pandas seaborn jupyter mord
```

### R との比較機能を使う場合（任意）

以下が必要です。

- `Rscript`
- R パッケージ `ordinalForest`
- R パッケージ `jsonlite`

## 最小使用例

```python
import numpy as np
from sklearn.model_selection import train_test_split
from ordinal_forest import OrdinalForestClassifier

# 例: 0 < 1 < 2 < 3 の順序クラス
rng = np.random.RandomState(42)
X = rng.normal(size=(500, 8))
y = np.digitize(X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2], bins=[-1.0, 0.0, 1.0])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = OrdinalForestClassifier(
    performance_function="probability",
    n_sets=30,
    n_estimators_per_set=20,
    n_estimators=200,
    max_features="sqrt",
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)
decision = clf.decision_function(X_test)

print("accuracy:", clf.score(X_test, y_test))
print("oob_score_:", clf.oob_score_)
print("pred shape:", pred.shape)
print("proba shape:", proba.shape)
print("decision shape:", decision.shape)
```

## 主要な使い方

### 1. ベーシックな学習

```python
from ordinal_forest import OrdinalForestClassifier

clf = OrdinalForestClassifier(random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
```

### 2. 順序付き確率を使う

`performance_function="probability"` を使うと、順序を意識した確率予測を中心に最適化できます。`predict_proba` と `predict_cumulative_proba` も利用できます。

```python
clf = OrdinalForestClassifier(
    performance_function="probability",
    random_state=42,
    n_jobs=-1,
)
clf.fit(X_train, y_train)

proba = clf.predict_proba(X_test)
cum_proba = clf.predict_cumulative_proba(X_test)
```

### 3. naive モード

スコア最適化を無効にして、単純な `1, 2, ..., J` スコアを使うこともできます。比較実験やアブレーションに便利です。

```python
clf = OrdinalForestClassifier(
    naive=True,
    performance_function="equal",
    random_state=42,
)
```

### 4. サンプル重み

```python
clf.fit(X_train, y_train, sample_weight=sample_weight)
```

### 5. 特定クラスを重視する / custom 重みを使う

```python
clf = OrdinalForestClassifier(
    performance_function="custom",
    class_weight_vector=[0.4, 0.2, 0.2, 0.2],
    random_state=42,
)
```

または `oneclass` で特定クラスを重視できます。

```python
clf = OrdinalForestClassifier(
    performance_function="oneclass",
    prioritized_class=2,
    random_state=42,
)
```

### 6. 重要な特徴を各木で必ず候補に入れる

`always_split_features` は、R 実装の `always.split.variables` に近い目的の近似機能です。各木が必ず含む部分空間に、指定特徴を入れます。

```python
clf = OrdinalForestClassifier(
    always_split_features=[0, 1],
    random_state=42,
)
```

pandas DataFrame を入力していれば、列名でも指定できます。

## 診断・解釈

### OOB 診断

```python
diag = clf.get_oob_diagnostics()
print(diag.accuracy)
print(diag.weighted_rank_mae)
print(diag.weighted_rps)
print(diag.confusion_matrix)
```

### 最適化サマリ

```python
summary = clf.get_optimization_summary()
print(summary.optimized_scores)
print(summary.optimized_thresholds)
print(summary.oob_score)
```

### Permutation Importance

```python
perm = clf.permutation_importance(
    X_test,
    y_test,
    scoring="objective",
    n_repeats=5,
    random_state=42,
)
print(perm.importances_mean)
```

### 可視化

```python
clf.plot_optimization_history()
clf.plot_score_profile()
clf.plot_feature_importance(top_k=20)
clf.plot_oob_coverage()
clf.plot_oob_confusion_matrix(normalize=True)
```

## 主なパラメータ

| パラメータ | 説明 |
|---|---|
| `n_sets` | 候補 score system の数 |
| `n_estimators_per_set` | 候補評価用の小Forestの木数 |
| `n_estimators` | 最終Forestの木数 |
| `performance_function` | 候補 score system を評価する目的関数 |
| `n_best` | 上位候補の平均化に使う個数 |
| `naive` | スコア最適化をスキップするか |
| `max_features` | 特徴量サブサンプリング設定 |
| `min_samples_leaf` | 葉の最小サンプル数 |
| `bootstrap` | ブートストラップ有無 |
| `sample_fraction` | 各木で使うサンプル割合 |
| `class_order` | クラス順序を明示したいときに指定 |
| `always_split_features` | 各木で必ず含めたい特徴 |
| `n_jobs` | 並列数 |
| `random_state` | 乱数シード |

## チュートリアルノートブック

`ordinal_forest_tutorial.ipynb` では、次の流れで実装の挙動を確認できます。

1. データ構造の理解
2. Accuracy 以外の ordinal-aware 指標の確認
3. Random Forest / Ordinal Logistic Regression / OrdinalForest の比較
4. `naive` や `performance_function` を切り替えた ablation study
5. OOB 診断、最適化履歴、Permutation Importance、不確実性の可視化

実装の使い方だけでなく、**なぜ ordinal classification に効くのか** を確認したい場合に便利です。

## 制約・注意点

- この実装は、R の `ordinalForest` の考え方を Python / scikit-learn 流に移植したものです。内部の木実装は `DecisionTreeRegressor` を使っています。
- `always_split_features` は、R 実装の split ごとの強制とは完全一致ではなく、**木ごとの部分空間制御による近似**です。
- `probability` モードの確率は、潜在 cutpoint と OOB 由来の `sigma` を用いた近似です。
- 公開前に、README の表現は必要に応じて「inspired by」「reference implementation ではない」などへ調整してください。

## 参考

- Hornung, R. (2020). `ordinalForest`: Ordinal Forests for ordinal regression.
- CRAN package: `ordinalForest`

## ライセンス

MIT License
