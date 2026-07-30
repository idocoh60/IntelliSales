"""
Trains the IntelliSales models straight from data/intellisales.db (produced
by etl/restore_and_export.py) and writes everything the Flask app needs:
models/*.pkl, models/segment_recommendations.json, models/metrics.json.

Design decisions, and why (see PLAN notes / README for the full story):

- Classifier target: Quantity > median(Quantity) - a binary "high-quantity
  order" flag. This matches the team's own working script
  (new_modeling_run.py / v_SuperPredict_Final), not an invented "deal
  success" label the dataset has no way to support (WideWorldImporters only
  contains completed sales).

- Regressor target: Quantity - predicts how many units this kind of
  customer/product/month combination typically orders.

- Features: CustomerCategoryID, StockItemID, ActualUnitPrice,
  DiscountPercentage, OrderMonth. CustomerCategoryID/StockItemID are used
  directly (both are already stable integer keys in the source data) rather
  than factorizing CustomerCategoryName/StockItemName as the draft script
  did - factorize()'s codes aren't stable across runs, which would silently
  break inference later. Same features, more robust encoding.

- DiscountPercentage is derived, not raw data: WideWorldImporters OrderLines
  has no discount column. It's computed exactly like v_SuperPredict_Final
  does: (RecommendedRetailPrice - UnitPrice) / RecommendedRetailPrice,
  clipped to [0, 0.9].

- "Closing probability" for the discount simulation (5/15/25% scenarios)
  is the classifier's real predict_proba with the simulated discount fed in
  as the DiscountPercentage feature - not the hand-tuned formula
  (global_base + discount*1.8 + qty/250) in the draft script. Same idea
  (higher discount -> higher predicted probability, since the model
  learned that relationship from real data), no arbitrary constants to
  defend at the oral exam.

- Recommendation engine: per CustomerCategoryID segment, StockItems ranked
  by historical purchase frequency (order-line count). At request time the
  app excludes whatever's already in the current order.
"""

import json
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, "..", "data", "intellisales.db")
MODELS_DIR = os.path.join(HERE, "models")

FEATURES = ["CustomerCategoryID", "StockItemID", "ActualUnitPrice", "DiscountPercentage", "OrderMonth"]
RANDOM_STATE = 42


def load_dataset() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        sales = pd.read_sql("SELECT * FROM sales_enriched", conn)
        customer_categories = pd.read_sql("SELECT * FROM customer_categories", conn)
        stock_pricing = pd.read_sql("SELECT * FROM stock_items_pricing", conn)

    df = sales.merge(customer_categories, on="CustomerID", how="left")
    df = df.merge(stock_pricing, on=["StockItemID", "StockItemName"], how="left")

    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    df["OrderMonth"] = df["OrderDate"].dt.month
    df["ActualUnitPrice"] = df["UnitPrice"]

    df["DiscountPercentage"] = np.where(
        df["RecommendedRetailPrice"] > 0,
        (df["RecommendedRetailPrice"] - df["UnitPrice"]) / df["RecommendedRetailPrice"],
        0.0,
    ).clip(0, 0.9)

    before = len(df)
    df = df.dropna(subset=FEATURES + ["Quantity", "CustomerCategoryName"]).copy()
    print(f"Training rows after dropping incomplete records: {len(df)} (was {before})")

    return df


def compute_permutation_importance(model, X_test, y_test, scoring, sample_size=5000):
    """
    Mean Decrease Impurity (model.feature_importances_) is biased toward
    high-cardinality features (StockItemID has 227 distinct values vs.
    CustomerCategoryID's 8), which makes it an unfair test of the signed
    Modelling report's claim that customer segment/order month are strong
    predictors. Permutation importance isn't biased by cardinality - it
    measures how much the real score drops when a feature is shuffled -
    so it's the honest way to check that claim. Subsampled to `sample_size`
    rows purely for runtime (standard practice; scikit-learn's own docs do
    the same) - not for any other reason.
    """
    if len(X_test) > sample_size:
        X_sample = X_test.sample(sample_size, random_state=RANDOM_STATE)
        y_sample = y_test.loc[X_sample.index]
    else:
        X_sample, y_sample = X_test, y_test

    result = permutation_importance(
        model, X_sample, y_sample, scoring=scoring, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1
    )
    return {
        feat: {"mean": round(float(m), 4), "std": round(float(s), 4)}
        for feat, m, s in zip(FEATURES, result.importances_mean, result.importances_std)
    }


def build_targets(df: pd.DataFrame):
    median_qty = df["Quantity"].median()
    y_class = (df["Quantity"] > median_qty).astype(int)
    y_qty = df["Quantity"]
    return y_class, y_qty, median_qty


def train_models(df: pd.DataFrame):
    X = df[FEATURES]
    y_class, y_qty, median_qty = build_targets(df)

    X_train, X_test, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_qty, test_size=0.2, random_state=RANDOM_STATE
    )

    classifier = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    classifier.fit(X_train, y_train_c)

    regressor = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1)
    regressor.fit(X_train_r, y_train_r)

    class_pred = classifier.predict(X_test)
    class_proba = classifier.predict_proba(X_test)[:, 1]

    print("Computing permutation importance (classifier)...")
    classifier_perm_importance = compute_permutation_importance(classifier, X_test, y_test_c, scoring="accuracy")
    print("Computing permutation importance (regressor)...")
    regressor_perm_importance = compute_permutation_importance(
        regressor, X_test_r, y_test_r, scoring="neg_mean_absolute_error"
    )

    metrics = {
        "classifier": {
            "target_definition": f"Quantity > median ({median_qty:.1f} units)",
            "accuracy": accuracy_score(y_test_c, class_pred),
            "precision": precision_score(y_test_c, class_pred),
            "recall": recall_score(y_test_c, class_pred),
            "auc": roc_auc_score(y_test_c, class_proba),
            "confusion_matrix": confusion_matrix(y_test_c, class_pred).tolist(),
            "feature_importance": dict(zip(FEATURES, classifier.feature_importances_.round(4).tolist())),
            "permutation_importance": classifier_perm_importance,
            "global_baseline": float(y_class.mean()),
        },
        "regressor": {
            "target_definition": "Quantity (units per order line)",
            "mae": mean_absolute_error(y_test_r, regressor.predict(X_test_r)),
            "feature_importance": dict(zip(FEATURES, regressor.feature_importances_.round(4).tolist())),
            "permutation_importance": regressor_perm_importance,
            "mean_quantity": float(y_qty.mean()),
        },
        "dataset": {
            "rows_used_for_training": len(df),
            "median_quantity": median_qty,
        },
    }
    return classifier, regressor, metrics, median_qty


def one_way_anova_eta_squared(df: pd.DataFrame, group_col: str, value_col: str = "Quantity") -> dict:
    """
    Independent check of the Modelling report's segment/seasonality
    assumptions, using plain one-way ANOVA on the raw data - no model,
    no cardinality bias, just: does group membership explain variance in
    Quantity at all? eta squared = SS_between / SS_total is the standard
    effect-size measure (0 = no effect, ~0.01 small, ~0.06 medium, ~0.14+
    large, per Cohen's conventions). With 458k rows, even a trivial true
    effect will show up as "statistically significant" (tiny p-value), so
    eta squared - not the p-value - is what actually answers the question.
    """
    groups = [g[value_col].values for _, g in df.groupby(group_col)]
    f_stat, p_value = stats.f_oneway(*groups)

    grand_mean = df[value_col].mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in (gr[value_col] for _, gr in df.groupby(group_col)))
    ss_total = ((df[value_col] - grand_mean) ** 2).sum()
    eta_squared = ss_between / ss_total

    group_means = df.groupby(group_col)[value_col].mean().round(2).to_dict()
    group_means = {str(k): float(v) for k, v in group_means.items()}

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "eta_squared": float(eta_squared),
        "group_means": group_means,
    }


def build_segment_recommendations(df: pd.DataFrame, top_n: int = 10) -> dict:
    recs = {}
    for cat_id, group in df.groupby("CustomerCategoryID"):
        top_items = (
            group.groupby(["StockItemID", "StockItemName"])
            .size()
            .sort_values(ascending=False)
            .head(top_n)
        )
        recs[str(int(cat_id))] = [
            {"StockItemID": int(item_id), "StockItemName": name, "purchase_count": int(count)}
            for (item_id, name), count in top_items.items()
        ]
    return recs


def build_reference_data(df: pd.DataFrame) -> dict:
    categories = (
        df[["CustomerCategoryID", "CustomerCategoryName"]]
        .drop_duplicates()
        .sort_values("CustomerCategoryID")
        .to_dict(orient="records")
    )
    products = (
        df[["StockItemID", "StockItemName", "Category", "RecommendedRetailPrice"]]
        .drop_duplicates(subset=["StockItemID"])
        .sort_values("StockItemName")
        .to_dict(orient="records")
    )
    return {"customer_categories": categories, "products": products}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_dataset()

    classifier, regressor, metrics, median_qty = train_models(df)
    recs = build_segment_recommendations(df)
    reference = build_reference_data(df)

    # Independent, model-free check of the Modelling report's segmentation
    # and seasonality assumptions (see one_way_anova_eta_squared docstring).
    df_month = df.copy()
    df_month["OrderMonth"] = pd.to_datetime(df_month["OrderDate"]).dt.month
    metrics["bivariate_check"] = {
        "customer_segment": one_way_anova_eta_squared(df, "CustomerCategoryName"),
        "order_month": one_way_anova_eta_squared(df_month, "OrderMonth"),
    }

    joblib.dump(classifier, os.path.join(MODELS_DIR, "classifier.pkl"))
    joblib.dump(regressor, os.path.join(MODELS_DIR, "regressor.pkl"))

    with open(os.path.join(MODELS_DIR, "segment_recommendations.json"), "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)

    with open(os.path.join(MODELS_DIR, "reference_data.json"), "w", encoding="utf-8") as f:
        json.dump(reference, f, ensure_ascii=False, indent=2)

    with open(os.path.join(MODELS_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved models + metrics to {MODELS_DIR}")


if __name__ == "__main__":
    main()
