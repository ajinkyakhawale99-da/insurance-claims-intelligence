"""Leakage-aware model training, calibration, thresholding, and evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone

from insurance_claims.cleaning import TARGET_COLUMN
from insurance_claims.features import make_model_matrix, model_feature_columns
from insurance_claims.thresholds import choose_threshold_for_recall


RANDOM_STATE = 2026


def build_preprocessor(matrix: pd.DataFrame) -> ColumnTransformer:
    categorical = list(matrix.select_dtypes(include=["object", "category", "string"]).columns)
    numeric = [column for column in matrix.columns if column not in categorical]
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
    )


def candidate_estimators(matrix: pd.DataFrame) -> dict[str, Pipeline]:
    preprocessor = build_preprocessor(matrix)
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", LogisticRegression(class_weight="balanced", max_iter=3000, random_state=RANDOM_STATE)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=350,
                        min_samples_leaf=3,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def _evaluate_at_threshold(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    precision_curve, recall_curve_values, _ = precision_recall_curve(y_true, probability)
    brier = brier_score_loss(y_true, probability)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "true_positive": int(tp),
        "false_positive": int(fp),
        "true_negative": int(tn),
        "false_negative": int(fn),
        "positive_support": int(np.sum(y_true == 1)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "pr_auc_trapezoidal": float(auc(recall_curve_values[::-1], precision_curve[::-1])),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "brier_score": float(brier),
    }


def _field_specs(training_frame: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for column in feature_columns:
        series = training_frame[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            minimum = float(numeric.min()) if numeric.notna().any() else median - 1
            maximum = float(numeric.max()) if numeric.notna().any() else median + 1
            if minimum == maximum:
                minimum, maximum = minimum - 1, maximum + 1
            specs[column] = {"kind": "numeric", "median": median, "min": minimum, "max": maximum}
        else:
            values = series.dropna().astype(str).value_counts().head(30).index.tolist()
            specs[column] = {"kind": "categorical", "options": values}
    return specs


def _fairness_audit(frame: pd.DataFrame, row_indices: np.ndarray, y_true: np.ndarray, flagged: np.ndarray) -> pd.DataFrame:
    """Descriptive test-split performance by excluded demographic attributes.

    Cells with fewer than 30 records or 10 recorded positive labels have their
    precision/recall withheld and are marked as small. This is an exploratory
    check, not a fairness certification.
    """
    audit_rows: list[dict[str, Any]] = []
    test_rows = frame.iloc[row_indices].reset_index(drop=True)
    for group_column in ["sex", "age_of_policy_holder"]:
        if group_column not in test_rows.columns:
            continue
        groups = test_rows[group_column].astype("object").where(test_rows[group_column].notna(), "Missing").astype(str)
        for group_value in sorted(groups.unique()):
            mask = groups.to_numpy() == group_value
            support = int(mask.sum())
            positives = int(y_true[mask].sum())
            predicted_positive = int(flagged[mask].sum())
            true_positive = int(((y_true == 1) & flagged & mask).sum())
            small = support < 30 or positives < 10
            audit_rows.append(
                {
                    "group_field": group_column,
                    "group_value": group_value,
                    "support": support,
                    "recorded_positive_count": positives,
                    "review_queue_count": predicted_positive,
                    "true_positive_count": true_positive,
                    "precision": None if small or predicted_positive == 0 else true_positive / predicted_positive,
                    "recall": None if small or positives == 0 else true_positive / positives,
                    "small_cell_suppressed": small,
                    "scope": "holdout descriptive audit only; not a fairness certification",
                }
            )
    return pd.DataFrame(audit_rows)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def train_and_evaluate(
    clean_csv: str | Path,
    model_dir: str | Path,
    powerbi_dir: str | Path,
    min_recall: float = 0.60,
) -> dict[str, Any]:
    clean_csv, model_dir, powerbi_dir = Path(clean_csv), Path(model_dir), Path(powerbi_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    powerbi_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(clean_csv)
    y = frame[TARGET_COLUMN].astype(int).to_numpy()
    if np.unique(y).size != 2 or min(np.bincount(y)) < 10:
        raise ValueError("Training requires at least 10 rows in each target class.")

    feature_columns = model_feature_columns(frame)
    X = make_model_matrix(frame, feature_columns)
    X_trainval, X_test, y_trainval, y_test, idx_trainval, idx_test = train_test_split(
        X,
        y,
        frame.index.to_numpy(),
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    X_train, X_validation, y_train, y_validation, idx_train, idx_validation = train_test_split(
        X_trainval,
        y_trainval,
        idx_trainval,
        test_size=0.25,
        stratify=y_trainval,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results: dict[str, float] = {}
    candidates = candidate_estimators(X_train)
    for name, estimator in candidates.items():
        scores = cross_validate(
            estimator,
            X_train,
            y_train,
            scoring="average_precision",
            cv=cv,
            n_jobs=1,
            error_score="raise",
        )["test_score"]
        cv_results[name] = float(np.mean(scores))

    selected_name = max(cv_results, key=cv_results.get)
    calibrated = CalibratedClassifierCV(
        estimator=clone(candidates[selected_name]),
        method="sigmoid",
        cv=5,
        n_jobs=1,
    )
    calibrated.fit(X_train, y_train)

    validation_probability = calibrated.predict_proba(X_validation)[:, 1]
    operating_point = choose_threshold_for_recall(y_validation, validation_probability, min_recall)
    threshold = float(operating_point["threshold"])

    test_probability = calibrated.predict_proba(X_test)[:, 1]
    test_metrics = _evaluate_at_threshold(y_test, test_probability, threshold)
    probabilities = test_probability
    actual = y_test
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(actual, probabilities)
    pr_frame = pd.DataFrame(
        {
            "precision": pr_precision[:-1],
            "recall": pr_recall[:-1],
            "threshold": pr_thresholds,
        }
    )
    pr_frame.to_csv(powerbi_dir / "PrecisionRecallCurve.csv", index=False)
    fraction_positive, mean_probability = calibration_curve(actual, probabilities, n_bins=10, strategy="quantile")
    pd.DataFrame(
        {"mean_predicted_probability": mean_probability, "observed_positive_rate": fraction_positive}
    ).to_csv(powerbi_dir / "CalibrationCurve.csv", index=False)

    scored = frame.iloc[idx_test][["source_row_id"]].copy()
    scored["actual_label"] = actual
    scored["recorded_label_probability"] = probabilities
    scored["selected_threshold"] = threshold
    scored["flagged_for_manual_review"] = probabilities >= threshold
    scored["split"] = "holdout_test"
    scored.to_csv(powerbi_dir / "ModelScores.csv", index=False)
    _fairness_audit(frame, idx_test, actual, probabilities >= threshold).to_csv(
        powerbi_dir / "FairnessAudit.csv", index=False
    )

    metrics: dict[str, Any] = {
        "dataset_rows": int(len(frame)),
        "dataset_positive_label_rate": float(np.mean(y)),
        "random_state": RANDOM_STATE,
        "split_sizes": {"train": int(len(y_train)), "validation": int(len(y_validation)), "test": int(len(y_test))},
        "model_selection_metric": "5-fold train Average Precision",
        "cross_validation_average_precision": cv_results,
        "selected_model": selected_name,
        "calibration": "sigmoid / 5-fold training-only calibration",
        "threshold_selection": {"minimum_recall_target": min_recall, **operating_point},
        "test_metrics": test_metrics,
        "model_feature_columns": feature_columns,
        "excluded_predictor_columns": sorted(
            {TARGET_COLUMN, "policy_number", "rep_number", "sex", "marital_status", "age", "age_of_policy_holder"}
        ),
    }
    metadata = {
        "threshold": threshold,
        "threshold_selection": metrics["threshold_selection"],
        "feature_columns": feature_columns,
        "field_specs": _field_specs(X_train, feature_columns),
        "selected_model": selected_name,
        "label_name": "probability of the dataset's recorded positive label",
        "disclaimer": "Advisory portfolio demo only. A score is not a finding of fraud.",
    }
    joblib.dump({"estimator": calibrated, "feature_columns": feature_columns}, model_dir / "model.joblib")
    _write_json(model_dir / "metadata.json", metadata)
    _write_json(model_dir / "metrics.json", metrics)
    pd.DataFrame(
        [{"metric": key, "value": value} for key, value in test_metrics.items() if isinstance(value, (int, float))]
        + [
            {"metric": "validation_minimum_recall_target", "value": min_recall},
            {"metric": "selected_threshold", "value": threshold},
            {"metric": "positive_label_rate", "value": float(np.mean(y))},
        ]
    ).to_csv(powerbi_dir / "ModelMetrics.csv", index=False)
    _write_model_card(model_dir / "MODEL_CARD.md", metrics)
    return metrics


def _write_model_card(path: Path, metrics: dict[str, Any]) -> None:
    test = metrics["test_metrics"]
    selection = metrics["threshold_selection"]
    lines = [
        "# Model card",
        "",
        "## Intended use",
        "Retrospective portfolio demonstration that scores the probability of the benchmark's recorded positive label. For human-review queue illustration only.",
        "",
        "## Prohibited interpretation",
        "A positive score is not a finding of fraud and must not be used to deny, delay, or reduce a claim payment.",
        "",
        "## Data and labels",
        "Public historical vehicle-claims benchmark; source label provenance and adjudication criteria are incomplete. See `docs/data_provenance.md`.",
        "",
        "## Evaluation protocol",
        f"Selected model: `{metrics['selected_model']}`. Threshold chosen on validation data using `{selection['selection']}`. The test partition was held out until final evaluation.",
        "",
        "## Holdout metrics against recorded labels",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in test.items():
        if isinstance(value, (int, float)):
            lines.append(f"| {key.replace('_', ' ').title()} | {value:.4f} |" if isinstance(value, float) else f"| {key.replace('_', ' ').title()} | {value} |")
    lines.extend(["", "## Limitations", "Small dated sample, class prevalence and labels may not match current business reality; no future-time or out-of-entity validation; some features may be available only after initial notice; calibration is sample-specific.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")

