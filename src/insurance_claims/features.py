"""Deterministic feature creation shared by training and the demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from insurance_claims.cleaning import TARGET_COLUMN


EXCLUDED_FROM_MODEL = {
    TARGET_COLUMN,
    "source_row_id",
    "policy_number",
    "rep_number",
    "sex",
    "marital_status",
    "age",
    "age_of_policy_holder",
}
MONTH_LOOKUP = {
    name: index
    for index, name in enumerate(
        ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        start=1,
    )
}


def _month_number(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip().str.lower()
    numeric = pd.to_numeric(text, errors="coerce")
    named = text.map(MONTH_LOOKUP)
    result = numeric.fillna(named)
    return result.where(result.between(1, 12))


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add safe, deterministic calendar encodings without using the label."""
    features = frame.copy()
    accident_month = _month_number(features["month"]) if "month" in features else None
    claim_month = _month_number(features["month_claimed"]) if "month_claimed" in features else None

    if accident_month is not None:
        angle = 2 * np.pi * (accident_month - 1) / 12
        features["accident_month_sin"] = np.sin(angle)
        features["accident_month_cos"] = np.cos(angle)
    if claim_month is not None:
        angle = 2 * np.pi * (claim_month - 1) / 12
        features["claim_month_sin"] = np.sin(angle)
        features["claim_month_cos"] = np.cos(angle)
    if accident_month is not None and claim_month is not None:
        features["reporting_month_gap_mod12"] = (claim_month - accident_month) % 12

    return features


def model_feature_columns(frame: pd.DataFrame) -> list[str]:
    engineered = engineer_features(frame)
    return [column for column in engineered.columns if column not in EXCLUDED_FROM_MODEL]


def make_model_matrix(frame: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    engineered = engineer_features(frame)
    if columns is None:
        columns = model_feature_columns(engineered)
    aligned = engineered.reindex(columns=columns)
    for column in aligned.columns:
        if pd.api.types.is_object_dtype(aligned[column]) or pd.api.types.is_string_dtype(aligned[column]):
            aligned[column] = aligned[column].astype("object").where(aligned[column].notna(), np.nan)
    return aligned

