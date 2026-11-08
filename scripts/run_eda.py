"""Produce repeatable descriptive outputs from the cleaned public data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from insurance_claims.cleaning import TARGET_COLUMN


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data" / "processed" / "claims_clean.csv"
FIGURES = ROOT / "reports" / "figures"
GENERATED = ROOT / "reports" / "generated"


def main() -> None:
    if not INPUT.exists():
        raise SystemExit("Cleaned data not found. Run `python scripts/prepare_data.py` first.")
    frame = pd.read_csv(INPUT)
    FIGURES.mkdir(parents=True, exist_ok=True)
    GENERATED.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    label_counts = frame[TARGET_COLUMN].value_counts().sort_index()
    label_table = pd.DataFrame(
        {
            "recorded_label": label_counts.index.astype(int),
            "claim_count": label_counts.values.astype(int),
            "share": (label_counts / len(frame)).values,
        }
    )
    label_table.to_csv(GENERATED / "label_balance.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=label_table, x="recorded_label", y="claim_count", color="#2675a8", ax=ax)
    ax.set(title="Claims by recorded benchmark label", xlabel="Recorded label (0 / 1)", ylabel="Claim records")
    fig.tight_layout()
    fig.savefig(FIGURES / "label_balance.png", dpi=160)
    plt.close(fig)

    missing = frame.isna().sum().rename("missing_count").to_frame()
    missing["missing_share"] = missing["missing_count"] / len(frame)
    missing.sort_values("missing_share", ascending=False).to_csv(GENERATED / "missingness.csv")
    top_missing = missing[missing["missing_count"] > 0].head(15).reset_index(names="column")
    if not top_missing.empty:
        fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(top_missing))))
        sns.barplot(data=top_missing, x="missing_share", y="column", color="#bf6b3a", ax=ax)
        ax.set(title="Fields with the most missing values", xlabel="Share missing", ylabel="")
        fig.tight_layout()
        fig.savefig(FIGURES / "missingness.png", dpi=160)
        plt.close(fig)

    segment_columns = [
        "vehicle_category", "policy_type", "fault", "accident_area",
        "police_report_filed", "witness_present",
    ]
    for segment_column in segment_columns:
        if segment_column not in frame:
            continue
        segment = (
            frame.groupby(segment_column, dropna=False)[TARGET_COLUMN]
            .agg(claim_count="size", recorded_positive_count="sum", recorded_positive_rate="mean")
            .reset_index()
            .sort_values("claim_count", ascending=False)
        )
        segment.to_csv(GENERATED / f"recorded_positive_rate_by_{segment_column}.csv", index=False)
        plotted = segment[segment["claim_count"] >= 30]
        if not plotted.empty:
            fig, ax = plt.subplots(figsize=(8, max(4.5, 0.35 * len(plotted))))
            sns.barplot(data=plotted, x=segment_column, y="recorded_positive_rate", color="#2675a8", ax=ax)
            ax.set(title=f"Recorded positive-label rate by {segment_column.replace('_', ' ')} (n ≥ 30)", xlabel=segment_column.replace("_", " ").title(), ylabel="Recorded positive-label rate")
            if len(plotted) > 5:
                ax.tick_params(axis="x", rotation=30)
            fig.tight_layout()
            fig.savefig(FIGURES / f"label_rate_by_{segment_column}.png", dpi=160)
            plt.close(fig)

    numeric = frame.select_dtypes(include="number").drop(columns=["source_row_id", TARGET_COLUMN], errors="ignore")
    if not numeric.empty:
        numeric.describe().T.to_csv(GENERATED / "numeric_summary.csv")

    summary = {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "recorded_positive_count": int(frame[TARGET_COLUMN].sum()),
        "recorded_positive_rate": float(frame[TARGET_COLUMN].mean()),
        "exact_duplicate_rows_including_target": int(frame.drop(columns=["source_row_id"], errors="ignore").duplicated().sum()),
    }
    (GENERATED / "eda_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

