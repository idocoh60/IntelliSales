# IntelliSales

A decision-support dashboard for sales reps: enter a customer segment and a
product, and get predicted order quantity, closing likelihood, an optimal
discount recommendation, and a cross-sell suggestion - alongside live model
performance metrics and data visualizations.

## Prerequisites

Python 3.11 or later. Nothing else - no SQL Server, no Docker, no internet
connection required to run the app.

## Getting Started

**macOS / Linux:**

```bash
git clone https://github.com/idocoh60/IntelliSales.git
cd IntelliSales
./run.sh
```

**Windows:**

```bat
git clone https://github.com/idocoh60/IntelliSales.git
cd IntelliSales
run.bat
```

Either script sets up a virtual environment, installs dependencies, and
starts the server (the cleaned dataset and trained models are already
included, so this takes only a few seconds). Then open your browser at
**http://127.0.0.1:5000**

To stop the server: `Ctrl+C` in the terminal.

## Using the Dashboard

**New Prediction** - select a customer segment and search for a product,
then click "Calculate Prediction" for:
- Success probability, with the historical average shown for comparison
- Recommended order quantity
- Optimal discount - the smallest discount (0-30%) that reaches the target
  closing probability, with the full probability-vs-discount curve
- Cross-sell suggestion - the top product in the same segment not yet
  selected

**Model Performance** - accuracy, precision, recall, AUC, and a confusion
matrix for the classifier; MAE for the regressor; feature importance for
both, measured two independent ways.

**Data Insights** - order-value distribution, sales by category, top
products, sales trend over time, and sales vs. median income by state.

## How the Models Work

- **Classifier**: Random Forest predicting whether an order is above the
  median quantity.
- **Regressor**: Random Forest predicting the exact order quantity.
- **Shared features**: customer segment, product, unit price, discount
  percentage, order month. Discount percentage is derived from the gap
  between an item's recommended retail price and its actual sale price.
- **Discount recommendation**: searches for the smallest discount that
  reaches a target closing probability, rather than checking a few fixed
  levels.
- **Cross-sell**: ranks products by purchase frequency within each segment.

**On customer segment and seasonality**: our initial assumption was that
segment and order month were strong predictors. Testing this three
independent ways (feature importance, permutation importance, and a direct
statistical test) shows both add very little once product and price are
already known - product and price already capture most of that signal.
Shown transparently in the Model Performance tab.

## Project Structure

```
IntelliSales/
  data/intellisales.db      # cleaned, merged dataset (included)
  etl/restore_and_export.py # rebuilds data/intellisales.db from the source database
  ml/train.py               # trains the models
  ml/models/                # trained models and metrics (included)
  app/                      # Flask backend and dashboard frontend
  run.sh / run.bat          # one-command setup and launch
  DATA_SETUP.md             # how the dataset was built, and how to regenerate it
```

## Retraining

```bash
source venv/bin/activate   # venv\Scripts\activate on Windows
python3 ml/train.py
```

## Tech Stack

Python, Flask, scikit-learn, pandas, SQLite, Chart.js.
