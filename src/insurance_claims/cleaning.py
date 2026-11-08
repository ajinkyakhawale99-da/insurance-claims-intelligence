"""Source normalization and lightweight data-quality auditing."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


TARGET_COLUMN = "fraud_found_p"
NUMERIC_COLUMNS = {
    "week_of_month",
    "week_of_month_claimed",
    "age",
    "policy_number",
    "rep_number",
    "deductible",
    "driver_rating",
    "year",
}
# `none` is a meaningful source category for prior-claim/timing fields, so keep it.
MISSING_TOKENS = {"", "?", "na", "n/a", "null", "unknown"}
TARGET_ALIASES = {
    "fraud_reported": TARGET_COLUMN,
    "fraud_found": TARGET_COLUMN,
    "fraud_found_p": TARGET_COLUMN,
}
COLUMN_ALIASES = {
    "number_of_suppliments": "number_of_supplements",
    "number_of_supplement": "number_of_supplements",
}


def normalize_column_name(name: Any) -> str:
    """Convert source headers to stable snake_case names."""
    value = str(name).strip()
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value)
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return COLUMN_ALIASES.get(value, TARGET_ALIASES.get(value, value))


def clean_claims(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Normalize a source frame and return cleaned rows plus an audit summary.

    Exact duplicate rows are retained because there is no documented source claim
    key. The audit reports them so a reviewer can decide whether to investigate.
    """
    if raw.empty:
        raise ValueError("Input claims file contains no rows.")

    df = raw.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    unnamed = [c for c in df.columns if c.startswith("unnamed") or c == "index"]
    df = df.drop(columns=unnamed, errors="ignore")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            "Expected a target column named FraudFound, FraudFound_P, or fraud_reported; "
            f"found: {', '.join(map(str, df.columns))}"
        )

    before_rows = len(df)
    original_columns = list(df.columns)
    for column in df.columns:
        if pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_string_dtype(df[column]):
            values = df[column].astype("string").str.strip()
            df[column] = values.mask(values.str.lower().isin(MISSING_TOKENS), pd.NA)

    labels = df[TARGET_COLUMN].astype("string").str.strip().str.lower()
    label_map = {
        "y": 1,
        "yes": 1,
        "1": 1,
        "true": 1,
        "fraud": 1,
        "n": 0,
        "no": 0,
        "0": 0,
        "false": 0,
        "not fraud": 0,
        "non-fraud": 0,
    }
    df[TARGET_COLUMN] = labels.map(label_map).astype("Int64")
    if df[TARGET_COLUMN].isna().any():
        bad = sorted(labels[df[TARGET_COLUMN].isna()].dropna().unique().tolist())
        raise ValueError(f"Target contains missing or unsupported labels: {bad}")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")

    for column in NUMERIC_COLUMNS.intersection(df.columns):
        numeric = pd.to_numeric(df[column], errors="coerce")
        df[column] = numeric.astype("float64")

    for column in df.select_dtypes(include=["object", "string"]).columns:
        df[column] = df[column].astype("object").where(df[column].notna(), np.nan)

    df.insert(0, "source_row_id", np.arange(1, len(df) + 1, dtype=np.int64))
    missing = {
        str(column): int(count)
        for column, count in df.isna().sum().items()
        if int(count) > 0
    }
    positives = int(df[TARGET_COLUMN].sum())
    audit = {
        "rows": int(len(df)),
        "columns_before_source_row_id": len(original_columns),
        "rows_dropped": before_rows - len(df),
        "exact_duplicate_rows_retained": int(df.drop(columns=["source_row_id"]).duplicated().sum()),
        "positive_label_count": positives,
        "negative_label_count": int(len(df) - positives),
        "positive_label_rate": float(positives / len(df)),
        "missing_by_column": missing,
        "normalized_columns": list(df.columns),
        "source_header_artifacts_removed": unnamed,
    }
    return df, audit

