"""
One-time ETL: pulls the real, already-built IntelliSales dataset out of the
WideWorldImporters SQL Server database (restored from the team's own .bak
backup) and writes a small, portable SQLite file that the rest of the app
depends on. This script is only ever run by whoever (re)generates the data
from the raw SQL Server backup - the shipped app and its examiners never
need SQL Server or Docker, only the resulting data/intellisales.db.

Prerequisites (see ../DATA_SETUP.md for full step-by-step instructions):
  1. Docker running locally.
  2. `WideWorldImporters_enriched.bak` present in ./data/raw/ (the team's own
     backup, which already contains the `dbo.demographics_state` census
     table and the `dbo.v_ModelDataset` view merging it into sales data -
     see DATA_SETUP.md for exactly how that view was verified).
  3. SQL Server container up and the database restored (DATA_SETUP.md has
     the docker run + RESTORE DATABASE commands).
  4. `pip install -r ../requirements-etl.txt`

What it exports, and why these three tables specifically:
  - sales_enriched: a 1:1 copy of dbo.v_ModelDataset (458,270 rows, 13
    columns) - this is *the exact artifact* described in the signed Data
    Understanding report (same row count, same column list), so we don't
    reinvent the merge logic, we just materialize what was already built
    and validated in SQL Server.
  - customer_categories: CustomerID -> CustomerCategoryName, needed for the
    segment-based recommendation engine and the Quantity>median classifier
    (CustomerCategoryName is one of its trained features, per
    v_SuperPredict_Final / new_modeling_run.py).
  - stock_items_pricing: StockItemID -> RecommendedRetailPrice, needed to
    derive DiscountPercentage the same way v_SuperPredict_Final does:
    (RecommendedRetailPrice - UnitPrice) / RecommendedRetailPrice. Real
    WideWorldImporters OrderLines has no discount column, so this is a
    genuine derived feature, not raw historical data.

Cleaning rules applied here, matching the signed Data Understanding report:
  - Quantity < 0 rows are dropped (report calls these out as data errors;
    none exist in this restored backup, but the check stays so the ETL is
    correct if re-run against a dirtier copy of the data).
  - Rows with no demographic match (State/Median_Income/Population missing
    from the LEFT JOIN in v_ModelDataset) get State filled to 'Unknown'
    rather than dropped, per the report's documented policy.
"""

# זה סקריפט ה-‎ETL - מריצים אותו פעם אחת (לא בזמן ריצת האתר בכלל) כדי
# למשוך את הנתונים מ-‎SQL Server (מה-.bak של הצוות) ולהמיר אותם ל-‎SQLite
# קטן וניתן להעברה (data/intellisales.db) שכל שאר הפרויקט (ml/train.py,
# app/app.py) עובד מולו. מי שפותח את הריפו ורק רוצה להריץ את האתר לא
# צריך בכלל את הקובץ הזה - ה-‎DB כבר מצורף.
#
# הלוגיקה כאן בנויה בדיוק לפי מה שתועד בדוחות Data Understanding/
# Data Preparation שהוגשו למנחה: אותם מקורות מידע (WideWorldImporters +
# טבלת census חיצונית), אותה שרשרת JOIN דרך Customers->Cities->
# StateProvinces->demographics_state, ואותם כללי ניקוי (Quantity<0
# נזרק, ואי-התאמות שמות מדינה כמו "Massachusetts[E]" מנורמלות ולא
# נזרקות).

import os
import re
import sqlite3

import pandas as pd
import pymssql


# מנקה סיומות כמו "[E]" או "(US Territory)" משם המדינה, כדי שאותה מדינה
# תיחשב זהה גם אם היא כתובה קצת אחרת בכל טבלה - בלי זה שורות רבות
# "מפספסות" את ההתאמה הדמוגרפית סתם בגלל הבדל בסימון, לא בגלל שהמדינה
# באמת שונה.
def normalize_state(name: str) -> str:
    """Strip annotations like '[E]' or '(US Territory)' so the same state
    spelled differently across the two source systems joins correctly.
    Concretely fixes 'Massachusetts[E]' -> 'Massachusetts' and
    'Puerto Rico (US Territory)' -> 'Puerto Rico', the exact mismatches the
    signed Data Understanding report calls out as a standardization risk.
    """
    if pd.isna(name):
        return name
    return re.sub(r"\s*[\[(].*?[\])]\s*$", "", name).strip()

SQL_SERVER = os.environ.get("INTELLISALES_SQL_HOST", "localhost")
SQL_PORT = int(os.environ.get("INTELLISALES_SQL_PORT", "1433"))
SQL_USER = os.environ.get("INTELLISALES_SQL_USER", "sa")
SQL_PASSWORD = os.environ.get("INTELLISALES_SQL_PASSWORD", "IntelliSales!2026")
SQL_DATABASE = os.environ.get("INTELLISALES_SQL_DATABASE", "WideWorldImporters")

HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DB = os.path.join(HERE, "..", "data", "intellisales.db")


# פותח חיבור ל-‎SQL Server לפי משתני סביבה (עם ברירות מחדל שמתאימות
# ל-‎docker-compose.yml המצורף) - כל שלוש פונקציות ה-‎extract למטה מקבלות
# את החיבור הזה.
def connect():
    return pymssql.connect(
        server=SQL_SERVER,
        port=SQL_PORT,
        user=SQL_USER,
        password=SQL_PASSWORD,
        database=SQL_DATABASE,
    )


# שולף את הטבלה הראשית (458,270 שורות, בדיוק כמו בדוח Data Understanding)
# מה-‎view המוכן v_ModelDataset, מנקה שורות עם כמות שלילית, ומשלים נתונים
# דמוגרפיים חסרים במקום לזרוק אותם.
def extract_sales_enriched(conn) -> pd.DataFrame:
    df = pd.read_sql("SELECT * FROM dbo.v_ModelDataset", conn)

    # Quantity < 0 = שגיאת נתונים (כנראה החזרה/ביטול) - הדוח מציין את זה
    # ככלל ניקוי, גם אם בפועל בגיבוי הזה אין שורות כאלה בכלל.
    before = len(df)
    df = df[df["Quantity"] >= 0].copy()
    dropped_negative_qty = before - len(df)
    if dropped_negative_qty:
        print(f"Dropped {dropped_negative_qty} rows with Quantity < 0")

    # v_ModelDataset already LEFT JOINs demographics_state, but a handful of
    # rows come back with State present and Median_Income/Population NULL
    # because the state name is spelled differently between WWI's
    # StateProvinces and the census table (see normalize_state). Re-fetch
    # the census table standalone and re-match on a normalized key instead
    # of just labeling these "Unknown".
    # מזהים אילו שורות "נפלו" מה-‎JOIN המקורי בגלל הבדל באיות שם המדינה,
    # ומנסים לתקן אותן: מנרמלים את השם משני הצדדים (State כאן, ו-‎State
    # בטבלת ה-‎census) ומצמידים לפי המפתח המנורמל במקום המקורי.
    missing = df["Median_Income"].isna() & df["State"].notna()
    if missing.any():
        demo = pd.read_sql("SELECT State, Median_Income, Population FROM dbo.demographics_state", conn)
        demo["_key"] = demo["State"].map(normalize_state).str.lower()
        demo_lookup = demo.set_index("_key")[["Median_Income", "Population"]]

        df["_key"] = df["State"].map(normalize_state).str.lower()
        recovered = df.loc[missing, "_key"].map(demo_lookup["Median_Income"])
        recovered_pop = df.loc[missing, "_key"].map(demo_lookup["Population"])
        df.loc[missing, "Median_Income"] = df.loc[missing, "Median_Income"].fillna(recovered)
        df.loc[missing, "Population"] = df.loc[missing, "Population"].fillna(recovered_pop)
        df.drop(columns="_key", inplace=True)

        still_missing = df["Median_Income"].isna() & df["State"].notna()
        print(
            f"Recovered {missing.sum() - still_missing.sum()} of {missing.sum()} "
            f"demographic mismatches via state-name normalization "
            f"({still_missing.sum()} genuinely unmatched remain)"
        )

    # מה שבכל זאת נשאר בלי התאמה (State ריק לגמרי, לא רק כתיב שונה) -
    # ממלאים כ-'Unknown' במקום לזרוק את השורה, לפי המדיניות שתועדה בדוח.
    unmatched = df["Median_Income"].isna()
    if unmatched.any():
        df.loc[unmatched, "State"] = df.loc[unmatched, "State"].fillna("Unknown")
        print(f"{unmatched.sum()} rows still lack demographic data (State/Median_Income left as-is / Unknown)")

    return df


# טבלת עזר קטנה: לכל לקוח, לאיזה סגמנט (CustomerCategoryID/Name) הוא
# שייך - זה מה שמאפשר ל-‎train.py לצרף שם סגמנט לכל שורת מכירה.
def extract_customer_categories(conn) -> pd.DataFrame:
    query = """
        SELECT c.CustomerID, cc.CustomerCategoryID, cc.CustomerCategoryName
        FROM Sales.Customers c
        JOIN Sales.CustomerCategories cc ON c.CustomerCategoryID = cc.CustomerCategoryID
    """
    return pd.read_sql(query, conn)


# טבלת עזר שנייה: מחיר המחירון של כל מוצר - נדרש כדי לחשב את
# DiscountPercentage (הפער בין מחיר המחירון למחיר שבו נמכר בפועל).
def extract_stock_items_pricing(conn) -> pd.DataFrame:
    query = """
        SELECT StockItemID, StockItemName, RecommendedRetailPrice
        FROM Warehouse.StockItems
    """
    return pd.read_sql(query, conn)


# מריץ את כל תהליך ה-‎ETL: מתחבר, שולף את שלוש הטבלאות, וכותב אותן
# ל-‎SQLite אחד. זה מה שרץ כשקוראים python3 etl/restore_and_export.py -
# לפי DATA_SETUP.md, רק אחרי שה-‎SQL Server רץ ב-‎Docker וה-.bak שוחזר.
def main():
    print(f"Connecting to {SQL_SERVER}:{SQL_PORT}/{SQL_DATABASE} ...")
    conn = connect()
    try:
        sales_enriched = extract_sales_enriched(conn)
        customer_categories = extract_customer_categories(conn)
        stock_items_pricing = extract_stock_items_pricing(conn)
    finally:
        conn.close()

    print(f"sales_enriched: {len(sales_enriched)} rows, {len(sales_enriched.columns)} cols")
    print(f"customer_categories: {len(customer_categories)} rows")
    print(f"stock_items_pricing: {len(stock_items_pricing)} rows")

    # כותבים את שלוש הטבלאות לקובץ SQLite יחיד - if_exists="replace" כדי
    # שאפשר יהיה להריץ את הסקריפט שוב ולרענן את הנתונים בלי למחוק ידנית.
    os.makedirs(os.path.dirname(OUTPUT_DB), exist_ok=True)
    with sqlite3.connect(OUTPUT_DB) as out:
        sales_enriched.to_sql("sales_enriched", out, if_exists="replace", index=False)
        customer_categories.to_sql("customer_categories", out, if_exists="replace", index=False)
        stock_items_pricing.to_sql("stock_items_pricing", out, if_exists="replace", index=False)

    print(f"Wrote {OUTPUT_DB}")


if __name__ == "__main__":
    main()
