"""Clean the public source CSV and write a row-level audit summary."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from insurance_claims.cleaning import clean_claims


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_CANDIDATES = [RAW_DIR / "carclaims.csv", RAW_DIR / "Carclaims.csv", RAW_DIR / "fraud_oracle.csv"]
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    raw_path = next((candidate for candidate in RAW_CANDIDATES if candidate.exists()), None)
    if raw_path is None:
        csv_files = sorted(RAW_DIR.glob("*.csv")) if RAW_DIR.exists() else []
        if len(csv_files) == 1:
            raw_path = csv_files[0]
        else:
            expected = ", ".join(path.name for path in RAW_CANDIDATES)
            raise SystemExit(f"Dataset not found in {RAW_DIR}. Expected one of: {expected}. See README.md for download steps.")
    raw = pd.read_csv(raw_path, low_memory=False)
    clean, audit = clean_claims(raw)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DIR / "claims_clean.csv", index=False)
    (PROCESSED_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"Read {raw_path.name}; cleaned {audit['rows']:,} rows; recorded-positive rate {audit['positive_label_rate']:.2%}.")
    print(f"Wrote {PROCESSED_DIR / 'claims_clean.csv'} and data_audit.json")


if __name__ == "__main__":
    main()

