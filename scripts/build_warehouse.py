"""Create the DuckDB star schema and flat Power BI import tables."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "claims_clean.csv"
DB_PATH = ROOT / "data" / "warehouse" / "claims_intelligence.duckdb"
POWERBI_DIR = ROOT / "data" / "processed" / "powerbi"


def main() -> None:
    if not INPUT.exists():
        raise SystemExit("Cleaned data not found. Run `python scripts/prepare_data.py` first.")
    clean = pd.read_csv(INPUT)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    POWERBI_DIR.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(DB_PATH))
    try:
        connection.register("clean_frame", clean)
        connection.execute("CREATE OR REPLACE TABLE stg_claims AS SELECT * FROM clean_frame")
        schema = (ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
        for statement in schema.split(";"):
            if statement.strip():
                connection.execute(statement)
        for table in ["DimIncident", "DimPolicy", "DimVehicle", "DimClaimant", "FactClaim"]:
            data = connection.execute(f"SELECT * FROM {table}").df()
            data.to_csv(POWERBI_DIR / f"{table}.csv", index=False)
        print(f"Built DuckDB warehouse: {DB_PATH}")
        print(f"Exported star tables: {POWERBI_DIR}")
    finally:
        connection.close()


if __name__ == "__main__":
    main()

