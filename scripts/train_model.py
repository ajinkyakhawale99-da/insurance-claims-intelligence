"""Train the leakage-aware benchmark-label model and export evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from insurance_claims.modeling import train_and_evaluate


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-recall", type=float, default=0.60, help="Validation recall floor used for threshold selection.")
    args = parser.parse_args()
    metrics = train_and_evaluate(
        ROOT / "data" / "processed" / "claims_clean.csv",
        ROOT / "models",
        ROOT / "data" / "processed" / "powerbi",
        min_recall=args.min_recall,
    )
    print(json.dumps({"selected_model": metrics["selected_model"], "threshold_selection": metrics["threshold_selection"], "test_metrics": metrics["test_metrics"]}, indent=2))


if __name__ == "__main__":
    main()

