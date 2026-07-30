@echo off
REM Sets up (if needed) and runs the IntelliSales dashboard.
REM Requires only Python 3.11+ - no SQL Server, no Docker, no internet access.
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
