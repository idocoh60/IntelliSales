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

# זה קובץ השרת (Flask) של כל האפליקציה - הוא זה שמריץ את הדשבורד ומחשוף
# את כל ה-‎API-‏ים שהעמוד קורא להם (predict, metrics, charts, sales-pitch).
# הוא לא מאמן שום דבר בעצמו - רק טוען את המודלים והקבצים המוכנים שנוצרו
# על ידי ml/train.py ועונה על בקשות בזמן אמת.

import json
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
import openai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from openai import OpenAI

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MODELS_DIR = os.path.join(ROOT, "ml", "models")
DB_PATH = os.path.join(ROOT, "data", "intellisales.db")

load_dotenv(os.path.join(ROOT, ".env"))

# פיצ'ר משפט המכירה (AI) הוא היחיד באפליקציה שדורש אינטרנט ומפתח API -
# אם אין מפתח מוגדר ב-.env פשוט openai_client יישאר None וה-‎endpoint
# בהמשך הקובץ יחזיר הודעה שהפיצ'ר לא זמין, במקום שהשרת יקרוס.
# The sales-pitch feature is optional: everything else in the app is fully
# local and free to run. This is the one feature that needs internet access
# and an OpenAI API key (OPENAI_API_KEY in .env). If it's not set, the
# endpoint below reports itself as unavailable instead of crashing the app.
OPENAI_MODEL = "gpt-4o-mini"
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) if os.environ.get("OPENAI_API_KEY") else None

# חמשת הפיצ'רים שהמודלים אומנו עליהם (ml/train.py) - חייב להיות אותו
# סדר ואותם שמות בדיוק פה כמו שם, אחרת ה-‎DataFrame שנבנה לפני predict
# לא יתאים למה שהמודל מצפה לו.
FEATURES = ["CustomerCategoryID", "StockItemID", "ActualUnitPrice", "DiscountPercentage", "OrderMonth"]

# Matches the team's real working script (new_modeling_run.py /
# run_smart_simulation) exactly: search for the minimum discount that
# crosses an 80% success-probability target, capped at a 30% discount
# ceiling. The only change from that script is *how* probability is
# computed - the classifier's real predict_proba with the candidate
# discount fed in as a feature, rather than the hand-tuned
# (global_base + discount*1.8 + qty/250) formula - see README.md.
#
# חשוב להסביר בהגנה: בדוח ה-‎Modelling שהוגש למנחה תוארה סימולציה על 3
# רמות הנחה קבועות בלבד (5%, 15%, 25%) עם "Price Sensitivity Factor"
# אחד. כאן שדרגנו את זה לחיפוש רציף על כל אחוז הנחה מ-‎0% עד 30%, ועוצרים
# ברגע שההסתברות האמיתית של המודל (לא נוסחה ידנית) עוברת את הסף של 80%.
# זה לא סותר את הדוח - זו הרחבה טבעית של אותו רעיון, פשוט מדויקת יותר.
TARGET_PROBABILITY = 0.80
MAX_DISCOUNT_PCT = 30

app = Flask(__name__, template_folder="templates", static_folder="static")

# טוענים את שני המודלים המאומנים (נוצרו מראש ע"י ml/train.py) פעם אחת
# כשהשרת עולה - לא בכל בקשה, כדי שה-‎predict יהיה מהיר.
classifier = joblib.load(os.path.join(MODELS_DIR, "classifier.pkl"))
regressor = joblib.load(os.path.join(MODELS_DIR, "regressor.pkl"))

with open(os.path.join(MODELS_DIR, "segment_recommendations.json"), encoding="utf-8") as f:
    SEGMENT_RECOMMENDATIONS = json.load(f)

with open(os.path.join(MODELS_DIR, "reference_data.json"), encoding="utf-8") as f:
    REFERENCE_DATA = json.load(f)

with open(os.path.join(MODELS_DIR, "metrics.json"), encoding="utf-8") as f:
    METRICS = json.load(f)

# דיקשנרי מהיר לפי StockItemID/CustomerCategoryID, כדי לא לחפש בלולאה
# בכל בקשה - reference_data.json כבר מגיע כרשימות, פה רק ממירים לצורת
# חיפוש O(1).
PRODUCT_LOOKUP = {p["StockItemID"]: p for p in REFERENCE_DATA["products"]}
CATEGORY_LOOKUP = {c["CustomerCategoryID"]: c["CustomerCategoryName"] for c in REFERENCE_DATA["customer_categories"]}


# בונה שורת DataFrame יחידה בדיוק באותו מבנה שהמודלים אומנו עליו (FEATURES),
# כדי שאפשר יהיה להעביר אותה ל-‎predict/predict_proba. משתמשים בפונקציה
# הזו גם לחישוב ההסתברות הנוכחית וגם בתוך לולאת חיפוש ההנחה למטה.
def build_feature_row(customer_category_id, stock_item_id, unit_price, order_month, discount_pct):
    return pd.DataFrame([{
        "CustomerCategoryID": customer_category_id,
        "StockItemID": stock_item_id,
        "ActualUnitPrice": unit_price,
        "DiscountPercentage": discount_pct,
        "OrderMonth": order_month,
    }])[FEATURES]


# מכין מראש (פעם אחת, בעליית השרת) את כל הנתונים לטאב "תובנות נתונים",
# כדי שכל בקשה ל-/api/charts רק תחזיר dict מוכן במקום לשאול את ה-‎DB בכל פעם.
def _load_chart_data():
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql("SELECT * FROM sales_enriched", conn)
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])

    # קיצוץ ל-‎quantile 99% לפני בניית ההיסטוגרמה - ככה כמה שורות הזמנה
    # ענקיות בודדות לא מותחות את כל ציר ה-‎X ומשאירות את שאר ההתפלגות דחוסה.
    order_value = df["LineTotal"].clip(upper=df["LineTotal"].quantile(0.99))
    bins = np.linspace(0, order_value.max(), 21)
    hist, edges = np.histogram(order_value, bins=bins)
    order_value_distribution = {
        "bins": [round(e, 0) for e in edges[:-1]],
        "counts": hist.tolist(),
    }

    # סך מכירות לפי קטגוריית מוצר, ממוין מהגבוה לנמוך - זה מה שהגרף
    # "מכירות לפי קטגוריה" בטאב תובנות מציג.
    by_category = (
        df.groupby("Category")["LineTotal"].sum().sort_values(ascending=False)
    )
    sales_by_category = {"labels": by_category.index.tolist(), "values": by_category.round(0).tolist()}

    # 10 המוצרים המובילים לפי סך מכירות בכסף (לא לפי כמות יחידות).
    top_products = (
        df.groupby("StockItemName")["LineTotal"].sum().sort_values(ascending=False).head(10)
    )
    top_products_data = {"labels": top_products.index.tolist(), "values": top_products.round(0).tolist()}

    # מכירות מקובצות לפי חודש-שנה, לגרף המגמה לאורך זמן.
    trend = df.groupby(df["OrderDate"].dt.to_period("M"))["LineTotal"].sum()
    sales_trend = {
        "labels": [str(p) for p in trend.index],
        "values": trend.round(0).tolist(),
    }

    # מכירות מול הכנסה חציונית לפי מדינה - dropna כאן מוריד שורות בלי
    # התאמה דמוגרפית (ראו etl/restore_and_export.py), כדי שהפיזור לא
    # ייראה מוטעה עם ערכי חסר.
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


# מחשבים את כל נתוני הגרפים פעם אחת כשהשרת עולה (לא בכל בקשה) ושומרים
# בזיכרון - הנתונים לא משתנים תוך כדי ריצה, אז אין טעם לחשב מחדש בכל פעם.
CHART_DATA = _load_chart_data()


# עמוד הבית - פשוט מגיש את ה-‎HTML של הדשבורד, כל הלוגיקה בפועל ב-‎JS/API.
@app.route("/")
def home():
    return render_template("index.html")


# מחזיר את רשימות הסגמנטים והמוצרים שהטופס בפרונט צריך למלא (תפריט
# הסגמנט וחלון החיפוש של המוצרים).
@app.route("/api/reference")
def api_reference():
    return jsonify(REFERENCE_DATA)


# מחזיר את כל מדדי הביצועים של המודלים (accuracy, precision, MAE, חשיבות
# פיצ'רים וכו') - מוצג בטאב "ביצועי המודל".
@app.route("/api/metrics")
def api_metrics():
    return jsonify(METRICS)


# מחזיר את הנתונים המוכנים לגרפים של טאב "תובנות נתונים".
@app.route("/api/charts")
def api_charts():
    return jsonify(CHART_DATA)


# ה-‎endpoint המרכזי של המערכת: מקבל תצפית חדשה (סגמנט לקוח + מוצר + מחיר
# + חודש) ומחזיר תחזית מלאה - הסתברות סגירה, כמות מומלצת, הנחה אופטימלית
# והמלצת cross-sell. זה בדיוק הזרימה "להזין תצפית חדשה ולקבל תחזית"
# שנדרשה בדרישות הפרויקט.
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)

    # ולידציה בסיסית של הקלט - כל שדה חייב להתקיים ולהיות מהסוג הנכון,
    # אחרת מחזירים 400 עם הסבר במקום לקרוס באמצע החישוב.
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

    # ה-‎DiscountPercentage מחושב מהפער בין המחיר המומלץ למחיר שהוזן -
    # בדיוק כמו שהוא מחושב באימון (ml/train.py), כי אין לנו עמודת הנחה
    # גולמית בנתונים - היא נגזרת.
    retail_price = product["RecommendedRetailPrice"] or unit_price
    current_discount = max(0.0, min(0.9, (retail_price - unit_price) / retail_price)) if retail_price else 0.0

    # תחזית "רגילה" - לפי המחיר שהמשתמש הזין בפועל, בלי סימולציית הנחה.
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
    #
    # פה קורה הלב של פיצ'ר "ההנחה האופטימלית": מריצים predict_proba שוב
    # ושוב על אותה שורה, כל פעם עם אחוז הנחה גבוה יותר (0%,1%,2%...),
    # עד שההסתברות המחושבת עוברת את הסף (80%) או שמגיעים לתקרה (30%).
    # ה-‎curve נשמר כדי שאפשר יהיה לצייר בפרונט את כל הגרף (הסתברות מול
    # הנחה), לא רק להציג את התוצאה הסופית.
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

    # close_deal=True רק אם נמצאה הנחה בטווח שבאמת מגיעה ל-‎80% הסתברות -
    # אם לא נמצאה כזו עד התקרה, זה סימן שהעסקה דורשת אישור מיוחד (המחיר
    # לא כדאי אפילו בהנחה המקסימלית).
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

    # רשימת ההמלצות של הסגמנט הזה (מדורגת לפי תדירות רכישה, נבנתה מראש
    # ב-‎ml/train.py) - שולפים ממנה את הראשון שהוא לא אותו מוצר שכבר נבחר.
    recs = SEGMENT_RECOMMENDATIONS.get(str(customer_category_id), [])
    cross_sell = next((r for r in recs if r["StockItemID"] != stock_item_id), None)

    return jsonify({
        "base_probability": round(base_probability, 4),
        "global_baseline": round(METRICS["classifier"]["global_baseline"], 4),
        "predicted_quantity": round(predicted_quantity, 1),
        "discount_recommendation": discount_recommendation,
        "cross_sell": cross_sell,
        "product": {
            "StockItemName": product["StockItemName"],
            "RecommendedRetailPrice": retail_price,
            "Category": product["Category"],
        },
        "customer_category_name": CATEGORY_LOOKUP.get(customer_category_id, ""),
    })


# הפרומפט הקבוע (system prompt) שנשלח ל-‎OpenAI בכל בקשת "משפט מכירה" -
# מגדיר לו את הכללים: לכתוב משפטים מוכנים לצטט, לא נקודות תדריך, ובלי
# להזכיר בכלל את המספרים הפנימיים של המערכת (אחוזים/תחזיות) בתוך הטקסט
# שהנציג יגיד ללקוח.
SALES_PITCH_SYSTEM_PROMPT = (
    "אתה כותב לנציג מכירות בחברה סיטונאית 4-5 משפטים קצרים ומוכנים-לשימוש, שהוא יכול "
    "**להגיד מילה במילה ללקוח** במהלך השיחה כדי לנהל אותה בביטחון ולהגדיל את הסיכוי לסגירה. "
    "אלה לא נקודות תדריך והכנה - אלה המשפטים עצמם, בגוף ראשון/פנייה ישירה ללקוח, מוכנים לצטט. "
    "הנציג כבר רואה על המסך שלו את כל המספרים הפנימיים של המערכת (אחוזי סיכוי, ממוצעים, תחזיות) - "
    "אסור להזכיר אותם בשום ניסוח (לא 'הסיכוי גבוה', לא 'לפי התחזית', לא אחוזים או נתוני מודל). "
    "כל מידע כזה שתקבל משמש רק אותך, כדי לכייל את הטון (נחוץ ללחוץ או שהעסקה כמעט סגורה) - "
    "לא כתוכן למשפטים. אל תמציא עובדות, מבצעים או תנאים שלא ניתנו לך. "
    "עברית טבעית, כאילו הנציג עצמו מדבר."
)


# מקבל את התוצאה של תחזית קודמת (מה-/api/predict) ומייצר לפי זה נקודות
# שיחה מוכנות לצטט לנציג המכירות. הפרונט שולח לכאן בדיוק את מה שהוא
# כבר קיבל מהתחזית, אז אין פה שום קריאה נוספת למודלים.
@app.route("/api/sales-pitch", methods=["POST"])
def api_sales_pitch():
    if openai_client is None:
        return jsonify({"error": "sales pitch not configured (OPENAI_API_KEY missing)"}), 503

    data = request.get_json(force=True)
    required = [
        "customer_category_name", "product_name", "predicted_quantity",
        "base_probability", "discount_recommendation",
    ]
    if any(k not in data for k in required):
        return jsonify({"error": "missing prediction context"}), 400

    # מתרגמים את הנתונים המספריים (הסתברות, הנחה) לרמזי טון בעברית -
    # אלה לא יופיעו במשפטים עצמם, הם רק עוזרים למודל השפה לכייל אם
    # לכתוב בביטחון או להתאמץ יותר לשכנע.
    rec = data["discount_recommendation"]
    tone = (
        "העסקה כבר קרובה לסגירה גם בלי לחץ - אפשר טון רגוע ובטוח."
        if data["base_probability"] >= 0.7
        else "יש התנגדות סבירה בדרך - הנציג צריך לעבוד קצת יותר כדי לשכנע."
    )
    discount_hint = (
        f'אם הלקוח מהסס, יש מרווח לתת הנחה של עד {rec["optimal_discount_pct"]}% - זה עדיין משתלם לחברה.'
        if rec.get("close_deal")
        else f'אין מרווח משמעותי להנחה מעבר לסטנדרט - עדיף להתמקד בערך של המוצר, לא במחיר.'
    )
    category_line = f" (קטגוריית {data['product_category']})" if data.get("product_category") else ""
    cross_sell_hint = (
        f'מוצר שמשתלב טבעי כתוספת: {data["cross_sell"]["StockItemName"]} - לקוחות דומים כמעט תמיד '
        f'לוקחים אותו יחד עם ההזמנה.'
        if data.get("cross_sell") else ""
    )

    # user_prompt מפריד בין "רקע פנימי" (רק לכיול הטון, אסור לצטט) לבין
    # "עובדות מותרות" (אלה כן יכולות להופיע במשפטים) - ההפרדה הזו היא
    # מה שמונע מהמודל להדליף אחוזי הצלחה/תחזיות ללקוח בטעות.
    user_prompt = f"""רקע פנימי לכיול הטון בלבד (אסור לצטט את המספרים האלה במשפטים עצמם):
- {tone}
- {discount_hint}

עובדות שמותר להשתמש בהן בתוכן המשפטים:
- סגמנט לקוח: {data["customer_category_name"]}
- מוצר: {data["product_name"]}{category_line}
- כמות מוצעת להזמנה: {data["predicted_quantity"]:.0f} יחידות בערך
- {cross_sell_hint}

כתוב 4-5 משפטים קצרים שהנציג יכול לצטט מילה במילה ללקוח כדי לנהל את השיחה ולהגדיל את הסיכוי לסגירה -
לא נקודות מידע, אלא דברים שממש אומרים בטלפון."""

    # קריאה בפועל למודל השפה. כל ה-‎except-‏ים למטה תופסים תרחישי שגיאה
    # שכיחים של OpenAI (אין קרדיט, מפתח לא תקף, בעיית רשת) ומחזירים
    # הודעה ברורה בעברית למשתמש במקום לזרוק שגיאת שרת גנרית.
    try:
        response = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SALES_PITCH_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=350,
            temperature=0.7,
        )
        pitch = response.choices[0].message.content.strip()
        return jsonify({"pitch": pitch})
    except openai.RateLimitError:
        return jsonify({"error": "אין יתרה זמינה בחשבון ה-OpenAI. יש להוסיף קרדיט ב-platform.openai.com."}), 502
    except openai.AuthenticationError:
        return jsonify({"error": "מפתח ה-OpenAI API לא תקף."}), 502
    except openai.APIConnectionError:
        return jsonify({"error": "לא ניתן להתחבר ל-OpenAI. בדקו את חיבור האינטרנט."}), 502
    except Exception as exc:  # noqa: BLE001 - unexpected error: log full detail, keep the UI message short
        app.logger.error("sales pitch request failed: %s", exc)
        return jsonify({"error": "יצירת משפט המכירה נכשלה. נסו שוב."}), 502


# מריצים ישירות (python app/app.py) בזמן פיתוח - debug=True נותן reload
# אוטומטי ומסך שגיאות מפורט. run.sh/run.bat קוראים לקובץ הזה בעצם.
if __name__ == "__main__":
    app.run(debug=True, port=5000)
