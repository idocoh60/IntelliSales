#!/usr/bin/env bash
# Sets up (if needed) and runs the IntelliSales dashboard.
# Requires only Python 3.11+ - no SQL Server, no Docker, no internet access.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  python3 -m venv venv
fi

source venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f ml/models/classifier.pkl ]; then
  echo "No trained models found - training now (uses data/intellisales.db, already included in the repo)..."
  python3 ml/train.py
fi

echo "Starting IntelliSales at http://127.0.0.1:5000"
python3 app/app.py
