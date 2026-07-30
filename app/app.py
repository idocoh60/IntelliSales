"""
IntelliSales dashboard - Flask backend.

Serves the sales-rep dashboard and exposes the model-driven endpoints it
calls: /api/reference (dropdown data), /api/predict (the core "enter a new
observation, get a prediction" flow required by the deployment guidelines),
/api/metrics (model performance tab), and /api/charts (data insight tab).

All heavy lifting (training, data cleaning) already happened in etl/ and
ml/train.py; this process only loads the resulting small artifacts and
answers requests - no SQL Server, no Docker, no internet access needed to
run it.
"""

import json
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MODELS_DIR = os.path.join(ROOT, "ml", "models")
DB_PATH = os.path.join(ROOT, "data", "intellisales.db")

FEATURES = ["CustomerCategoryID", "StockItemID", "ActualUnitPrice", "DiscountPercentage", "OrderMonth"]
DISCOUNT_SCENARIOS = [0.05, 0.15, 0.25]
DECISION_THRESHOLD = 0.5

app = Flask(__name__, template_folder="templates", static_folder="static")

classifier = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
regressor = joblib.load(os.path.join(MODELS_DIR, "regressor.pkl"))

with open(os.path.join(MODELS_DIR, "segment_recommendations.json"), encoding="utf-8") as f:
    SEGMENT_RECOMMENDATIONS = json.load(f)

with open(os.path.join(MODELS_DIR, "reference_data.json"), encoding="utf-8") as f:
    REFERENCE_DATA = json.load(f)

with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as f:
    METRICS = json.load(f)

PRODUCT_LOOKUP = {p["StockItemID"]: p for p in REFERENCE_DATA["products"]}


def build_feature_row(customer_category_id, stock_item_id, unit_price, order_month, discount_pct):
    return pd.DataFrame([{
        "CustomerCategoryID": customer_category_id,
        "StockItemID": stock_item_id,
        "ActualUnitPrice": unit_price,
        "DiscountPercentage": discount_pct,
        "OrderMonth": order_month,
    }])[FEATURES]


def _load_chart_data():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM sales_enriched", conn)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])

    order_value = df["LineTotal"].clip(upper=df["LineTotal"].quantile(0.99))
    bins = np.linspace(0, order_value.max(), 21)
    hist, edges = np.histogram(order_value, bins=bins)
    order_value_distribution = {
        "bins": [round(e, 0) for e in edges[:-1]],
        "counts": hist.tolist(),
    }

    by_category = (
        df.groupby("Category")["LineTotal"].sum().sort_values(ascending=False)
    )
    sales_by_category = {"labels": by_category.index.tolist(), "values": by_category.round(0).tolist()}

    top_products = (
        df.groupby("StockItemName")["LineTotal"].sum().sort_values(ascending=False).head(10)
    )
    top_products_data = {"labels": top_products.index.tolist(), "values": top_products.round(0).tolist()}

    trend = df.groupby(df["OrderDate"].dt.to_period("M"))["LineTotal"].sum()
    sales_trend = {
        "labels": [str(p) for p in trend.index],
        "values": trend.round(0).tolist(),
    }

    by_state = (
        df.dropna(subset=["Median_Income"])
        .groupby("State")
        .agg(total_sales=("LineTotal", "sum"), median_income=("Median_Income", "first"))
        .reset_index()
    )
    sales_vs_income = {
        "labels": by_state["State"].tolist(),
        "income": by_state["median_income"].tolist(),
        "sales": by_state["total_sales"].round(0).tolist(),
    }

    return {
        "order_value_distribution": order_value_distribution,
        "sales_by_category": sales_by_category,
        "top_products": top_products_data,
        "sales_trend": sales_trend,
        "sales_vs_income": sales_vs_income,
    }


CHART_DATA = _load_chart_data()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/reference")
def api_reference():
    return jsonify(REFERENCE_DATA)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


@app.route("/api/charts")
def api_charts():
    return jsonify(CHART_DATA)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)

    try:
        customer_category_id = int(data["customer_category_id"])
        stock_item_id = int(data["stock_item_id"])
        order_month = int(data.get("order_month", pd.Timestamp.now().month))
        unit_price = float(data["unit_price"])
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"error": f"Invalid input: {exc}"}), 400

    product = PRODUCT_LOOKUP.get(stock_item_id)
    if product is None:
        return jsonify({"error": f"Unknown StockItemID {stock_item_id}"}), 400

    retail_price = product["RecommendedRetailPrice"] or unit_price
    current_discount = max(0.0, min(0.9, (retail_price - unit_price) / retail_price)) if retail_price else 0.0

    current_row = build_feature_row(customer_category_id, stock_item_id, unit_price, order_month, current_discount)
    base_probability = float(classifier.predict_proba(current_row)[0][1])
    predicted_quantity = float(regressor.predict(current_row)[0])

    # The Random Forest doesn't guarantee a monotonic relationship between
    # discount and predicted probability (a bigger discount can come back
    # with a *lower* raw score than a smaller one, since nothing constrains
    # the trees that way). Shown to a sales rep, "more discount = less
    # likely to close" reads as a bug, not nuance - so we enforce "at least
    # as likely as any smaller discount already offered" when presenting
    # the table, via a running max over increasing discount levels. The
    # raw model output is unchanged; only the displayed scenario ordering
    # is smoothed.
    discount_table = []
    running_max = base_probability
    for discount in sorted(DISCOUNT_SCENARIOS):
        simulated_price = retail_price * (1 - discount)
        row = build_feature_row(customer_category_id, stock_item_id, simulated_price, order_month, discount)
        raw_probability = float(classifier.predict_proba(row)[0][1])
        running_max = max(running_max, raw_probability)
        discount_table.append({
            "discount_pct": discount * 100,
            "simulated_unit_price": round(simulated_price, 2),
            "probability": round(running_max, 4),
            "raw_model_probability": round(raw_probability, 4),
            "close_deal": running_max >= DECISION_THRESHOLD,
        })

    recs = SEGMENT_RECOMMENDATIONS.get(str(customer_category_id), [])
    cross_sell = next((r for r in recs if r["StockItemID"] != stock_item_id), None)

    return jsonify({
        "base_probability": round(base_probability, 4),
        "global_baseline": round(METRICS["classifier"]["global_baseline"], 4),
        "predicted_quantity": round(predicted_quantity, 1),
        "discount_table": discount_table,
        "cross_sell": cross_sell,
        "product": {"StockItemName": product["StockItemName"], "RecommendedRetailPrice": retail_price},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
