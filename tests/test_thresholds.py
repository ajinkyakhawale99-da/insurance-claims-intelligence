import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from insurance_claims.modeling import build_preprocessor
from insurance_claims.thresholds import choose_threshold_for_recall


def test_threshold_satisfies_validation_recall_floor_when_possible():
    labels = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    probabilities = np.array([0.95, 0.72, 0.48, 0.80, 0.39, 0.21, 0.10, 0.02])
    result = choose_threshold_for_recall(labels, probabilities, min_recall=2 / 3)
    assert result["recall"] >= 2 / 3
    assert 0 <= result["threshold"] <= 1


def test_threshold_rejects_invalid_recall_target():
    try:
        choose_threshold_for_recall(np.array([0, 1]), np.array([0.2, 0.8]), min_recall=0)
    except ValueError as error:
        assert "min_recall" in str(error)
    else:
        raise AssertionError("Expected invalid recall target to raise ValueError")


def test_preprocessing_pipeline_accepts_missing_and_unseen_categories():
    train = pd.DataFrame({"amount": [1.0, 2.0, 3.0, 4.0], "vehicle": ["car", "car", "van", "van"]})
    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor(train)),
            ("model", LogisticRegression()),
        ]
    )
    pipeline.fit(train, np.array([0, 0, 1, 1]))
    predictions = pipeline.predict_proba(pd.DataFrame({"amount": [None], "vehicle": ["new category"]}))

    assert predictions.shape == (1, 2)
    assert np.isfinite(predictions).all()

