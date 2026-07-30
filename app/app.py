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

# Matches the team's real working script (new_modeling_run.py /
# run_smart_simulation) exactly: search for the minimum discount that
# crosses an 80% success-probability target, capped at a 30% discount
# ceiling. The only change from that script is *how* probability is
# computed - the classifier's real predict_proba with the candidate
# discount fed in as a feature, rather than the hand-tuned
# (global_base + discount*1.8 + qty/250) formula - see README.md.
TARGET_PROBABILITY = 0.80
MAX_DISCOUNT_PCT = 30

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

    # Optimal-discount search: walk discount 0% -> MAX_DISCOUNT_PCT in 1%
    # steps and stop at the first one whose predicted probability crosses
    # TARGET_PROBABILITY. This is a direct port of run_smart_simulation()
    # in new_modeling_run.py, kept faithful to its threshold/ceiling
    # constants; only the probability source changed (real predict_proba,
    # not the hand-tuned formula). The full curve is returned too so the
    # UI can plot probability vs. discount, not just the single answer.
    curve = []
    optimal_discount_pct = None
    achieved_probability = base_probability
    for d in range(0, MAX_DISCOUNT_PCT + 1):
        discount = d / 100
        simulated_price = retail_price * (1 - discount)
        row = build_feature_row(customer_category_id, stock_item_id, simulated_price, order_month, discount)
        probability = float(classifier.predict_proba(row)[0][1])
        curve.append({"discount_pct": d, "probability": round(probability, 4)})
        achieved_probability = probability
        if optimal_discount_pct is None and probability >= TARGET_PROBABILITY:
            optimal_discount_pct = d
            achieved_probability = probability
            break

    close_deal = optimal_discount_pct is not None
    discount_recommendation = {
        "target_probability": TARGET_PROBABILITY,
        "max_discount_pct": MAX_DISCOUNT_PCT,
        "optimal_discount_pct": optimal_discount_pct,
        "achieved_probability": round(achieved_probability, 4),
        "simulated_unit_price": round(retail_price * (1 - (optimal_discount_pct or MAX_DISCOUNT_PCT) / 100), 2),
        "close_deal": close_deal,
        "curve": curve,
    }

    recs = SEGMENT_RECOMMENDATIONS.get(str(customer_category_id), [])
    cross_sell = next((r for r in recs if r["StockItemID"] != stock_item_id), None)

    return jsonify({
        "base_probability": round(base_probability, 4),
        "global_baseline": round(METRICS["classifier"]["global_baseline"], 4),
        "predicted_quantity": round(predicted_quantity, 1),
        "discount_recommendation": discount_recommendation,
        "cross_sell": cross_sell,
        "product": {"StockItemName": product["StockItemName"], "RecommendedRetailPrice": retail_price},
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
