# IntelliSales

Sales-rep decision-support dashboard for the IntelliSales capstone project
(Emek Yezreel College, Dept. of Information Systems). A rep enters a new
customer/product observation and gets back a model-driven read on the deal:
predicted quantity, closing likelihood, a 3-point discount simulation, and a
cross-sell suggestion - plus a live view of the underlying data and model
performance, per the project's Deployment stage requirements.

## Running it (any machine, no SQL Server, no Docker, no internet needed)

Requires only Python 3.11+.

```bash
./run.sh        # macOS/Linux
run.bat         # Windows
```

This creates a virtual environment, installs dependencies, trains the models
if `ml/models/` is empty (a few seconds - the cleaned dataset ships in the
repo at `data/intellisales.db`), and starts the app at
**http://127.0.0.1:5000**.

## Using the dashboard

**תחזית חדשה (New prediction)** - pick a customer segment and a product (unit
price auto-fills from the recommended retail price, editable) and an order
month, then "חשב תחזית". You get back:
- **סיכוי לעסקה משמעותית** - the classifier's predicted probability, with
  the historical baseline shown alongside for context.
- **כמות מומלצת להזמנה** - the regressor's predicted quantity.
- **סימולציית הנחות** - the same prediction re-run at 5% / 15% / 25%
  simulated discount, so a rep can see whether offering a discount is
  likely to move the deal.
- **המלצת Cross-sell** - the most popular product in the same customer
  segment that isn't the one just selected.

**ביצועי המודל (Model performance)** - accuracy/precision/recall/AUC and a
confusion matrix for the classifier, MAE for the regressor, and feature
importance charts for both - live from the last training run.

**תובנות נתונים (Data insights)** - the same exploratory views validated in
the Data Understanding report (order-value distribution, sales by category,
top products, sales trend, sales vs. median income by state), rebuilt as
interactive charts.

## What's actually being modeled, and why (for the oral exam)

The signed Modelling report describes predicting "deal success," but
WideWorldImporters only contains *completed* sales - there's no natural
success/failure label in the data. The team's own working script
(`new_modeling_run.py`, and the SQL view `v_SuperPredict_Final` it reads
from) defines the real, already-tested targets, and this app keeps them:

- **Classifier**: `Quantity > median(Quantity)` - a binary "high-quantity
  order" flag, standing in for deal strength.
- **Regressor**: `Quantity` itself.
- **Features**: `CustomerCategoryID`, `StockItemID`, `ActualUnitPrice`,
  `DiscountPercentage`, `OrderMonth`. `CustomerCategoryID`/`StockItemID` are
  used directly (both are already stable integer keys) rather than
  factorizing the name columns as the draft script did - `pandas.factorize`
  codes aren't stable across runs, which would silently break inference the
  next time the model is retrained.
- **DiscountPercentage is derived, not raw data**: real
  `Sales.OrderLines` has no discount column. It's computed exactly like
  `v_SuperPredict_Final` does: `(RecommendedRetailPrice - UnitPrice) /
  RecommendedRetailPrice`.
- **The discount-simulation probabilities are the classifier's real
  `predict_proba`** with the simulated discount fed in as the
  `DiscountPercentage` feature - not a hand-tuned formula. One adjustment:
  Random Forests don't guarantee the probability rises monotonically with
  a bigger discount, which reads as a bug to a sales rep, so the displayed
  (not the raw model) probability is smoothed to never decrease as the
  discount increases - see the comment in `app/app.py`'s `/api/predict`.
- **Recommendation engine**: within a customer's segment, the most
  frequently purchased product, excluding whatever's already selected.

## Where the data comes from

`data/intellisales.db` (committed to the repo) is a cleaned, merged export
from the team's own SQL Server backup - see `DATA_SETUP.md` for exactly how
it was produced and how to regenerate it if the source data ever changes.
Nobody running the app needs to touch SQL Server; the exported SQLite file
is all `app/app.py` and `ml/train.py` read from.

## Project layout

```
IntelliSales/
  data/intellisales.db     # cleaned, merged dataset (committed)
  data/raw/                 # gitignored - raw SQL Server backup, dev-time only
  etl/restore_and_export.py # one-time: SQL Server -> data/intellisales.db
  ml/train.py                # trains the models, writes ml/models/*
  ml/models/                 # classifier.pkl, regressor.pkl, metrics.json, ...
  app/app.py                 # Flask backend
  app/templates/, app/static # dashboard frontend
  requirements.txt            # what the app needs to run
  requirements-etl.txt         # what etl/restore_and_export.py additionally needs
  run.sh / run.bat
  DATA_SETUP.md               # how to regenerate data/intellisales.db
```

## Retraining

```bash
source venv/bin/activate   # venv\Scripts\activate on Windows
python3 ml/train.py
```

Reads `data/intellisales.db`, rewrites everything in `ml/models/`.
