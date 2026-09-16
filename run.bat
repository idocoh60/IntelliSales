@echo off
REM זהה ל-‎run.sh, רק בשביל Windows - venv, התקנת תלויות, אימון אם צריך,
REM והרצת השרת.
cd /d "%~dp0"

if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -q -r requirements.txt

if not exist ml\models\classifier.pkl (
    echo No trained models found - training now (uses data\intellisales.db, already included in the repo)...
    python ml\train.py
)

echo Starting IntelliSales at http://127.0.0.1:5000
python app\app.py
