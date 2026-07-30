# Data setup (one-time, only for regenerating `data/intellisales.db`)

You only need this if you don't already have `data/intellisales.db` (it's
committed to the repo, so most people never run this). It's for whoever
needs to regenerate it from the raw SQL Server backup - e.g. if the source
data changes.

The shipped app (`app/app.py`) never needs SQL Server, Docker, or this
process at run time - only the resulting `data/intellisales.db`.

## What you need

- Docker Desktop running.
- The team's SQL Server backup file, placed at `data/raw/WideWorldImporters_enriched.bak`.
  This is **not** the public Microsoft WideWorldImporters sample - it's the
  team's own backup that already contains a `dbo.demographics_state` table
  (U.S. Census median household income + population by state) and several
  SQL views merging it into the sales data. Ask the team for this file if
  you don't have it (it's too large for git and isn't committed - see
  `.gitignore`).
- Python 3.11+ and `pip install -r requirements-etl.txt`.

## Apple Silicon (M1/M2/M3) note

SQL Server's Docker image crashes on startup under Docker Desktop's default
QEMU-based emulation, with an error like:

```
Invalid mapping of address 0x... in reserved address space below 0x400000000000
```

Fix: Docker Desktop → Settings → General → enable **"Use Rosetta for
x86_64/amd64 emulation on Apple Silicon"** → Apply & Restart. (Requires
Rosetta installed: `softwareupdate --install-rosetta`.) Not needed on
Intel Macs, Windows, or Linux.

## Steps

1. Start SQL Server and mount the backup folder:

   ```bash
   docker compose up -d
   ```

2. Restore the database (first check the logical file names, they can
   differ if the backup was taken on a different SQL Server instance):

   ```bash
   docker exec wwi-sql /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P 'IntelliSales!2026' -C \
     -Q "RESTORE FILELISTONLY FROM DISK = N'/var/opt/mssql/backup/WideWorldImporters_enriched.bak'"
   ```

   Then restore, adjusting the `MOVE` targets to match the logical names
   from the previous step:

   ```bash
   docker exec wwi-sql /opt/mssql-tools18/bin/sqlcmd -S localhost -U sa -P 'IntelliSales!2026' -C -Q "
   RESTORE DATABASE WideWorldImporters
   FROM DISK = N'/var/opt/mssql/backup/WideWorldImporters_enriched.bak'
   WITH MOVE 'WWI_Primary' TO '/var/opt/mssql/data/WideWorldImporters.mdf',
   MOVE 'WWI_UserData' TO '/var/opt/mssql/data/WideWorldImporters_UserData.ndf',
   MOVE 'WWI_Log' TO '/var/opt/mssql/data/WideWorldImporters.ldf',
   MOVE 'WWI_InMemory_Data_1' TO '/var/opt/mssql/data/WideWorldImporters_InMemory_Data_1',
   REPLACE, STATS = 10;
   "
   ```

3. Run the ETL:

   ```bash
   python3 -m venv venv && source venv/bin/activate   # if not already set up
   pip install -r requirements-etl.txt
   python3 etl/restore_and_export.py
   ```

   This writes `data/intellisales.db` (SQLite, ~60MB) - a `sales_enriched`
   table (1:1 copy of `dbo.v_ModelDataset`, the exact 458,270-row / 13-column
   dataset described in the signed Data Understanding report), plus small
   `customer_categories` and `stock_items_pricing` lookup tables.

4. Retrain the models against the new data:

   ```bash
   python3 ml/train.py
   ```

5. Tear down the SQL Server container - it's no longer needed:

   ```bash
   docker compose down
   ```

## What we found already in the backup, for context

Querying `INFORMATION_SCHEMA` on the restored database turned up these
existing objects (all in the `dbo` schema unless noted), which the ETL
script builds on instead of re-deriving from scratch:

- `demographics_state` - the real census table (`State`, `Median_Income`,
  `Population`, `State_Code`), 53 rows.
- `v_ModelDataset` - 458,270 rows / 13 columns, matching the Data
  Understanding report exactly. This is what `sales_enriched` is copied
  from.
- `v_SuperPredict_Final` - the view behind `new_modeling_run.py` /
  the Modelling report's screenshot; its `DiscountPercentage` computation
  `(RecommendedRetailPrice - UnitPrice) / RecommendedRetailPrice` is
  reused as-is in `ml/train.py`.
- `v_FinalDataset`, `v_ModelDataset_Final` - other iterations, not used
  directly (231,412 rows each, due to stricter INNER JOINs that drop
  unmatched-geography rows `v_ModelDataset` keeps).

One real data-quality issue found and fixed in `etl/restore_and_export.py`:
`v_ModelDataset`'s `State` column had two spellings that don't match
`demographics_state` verbatim - `"Massachusetts[E]"` and `"Puerto Rico (US
Territory)"` - causing 13,087 rows to come back with no demographic match.
The ETL normalizes both sides (stripping bracket/paren annotations) before
re-matching; all 13,087 are recovered, so no rows need an "Unknown" fallback
in practice.
