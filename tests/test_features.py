import pandas as pd

from insurance_claims.cleaning import clean_claims
from insurance_claims.features import engineer_features, model_feature_columns


def test_calendar_engineering_and_predictor_exclusions():
    raw = pd.DataFrame(
        {
            "FraudFound_P": ["Y", "N"],
            "PolicyNumber": [10, 11],
            "RepNumber": [2, 3],
            "Sex": ["Male", "Female"],
            "Age": [30, 45],
            "MaritalStatus": ["Single", "Married"],
            "AgeOfPolicyHolder": ["26 to 30", "41 to 50"],
            "Month": ["Jan", "Dec"],
            "MonthClaimed": ["Mar", "Jan"],
            "VehicleCategory": ["Sedan", "Sport"],
        }
    )
    clean, _ = clean_claims(raw)
    features = engineer_features(clean)
    columns = model_feature_columns(clean)

    assert features["reporting_month_gap_mod12"].tolist() == [2.0, 1.0]
    assert "fraud_found_p" not in columns
    assert "policy_number" not in columns
    assert "rep_number" not in columns
    assert "sex" not in columns
    assert "age" not in columns
    assert "vehicle_category" in columns


def test_source_target_alias_without_suffix_is_supported():
    clean, _ = clean_claims(pd.DataFrame({"FraudFound": ["No", "Yes"], "VehicleCategory": ["Sedan", "Sport"]}))
    assert clean["fraud_found_p"].tolist() == [0, 1]

