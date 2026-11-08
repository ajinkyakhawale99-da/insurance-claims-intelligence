from pathlib import Path

import pandas as pd

from insurance_claims.cleaning import clean_claims


def test_cleaning_normalizes_target_missing_values_and_ids():
    source = pd.DataFrame(
        {
            "Unnamed: 0": [0, 1, 2],
            "FraudFound_P": ["Y", "N", "Y"],
            "PolicyNumber": [1001, 1002, 1003],
            "NumberOfSuppliments": ["1 to 2", "?", "more than 2"],
            "WitnessPresent": ["No", "Yes", "No"],
            "AddressChange-Claim": ["no change", "1 year", "no change"],
            "NumberOfCars": ["1 vehicle", "2 vehicles", "1 vehicle"],
        }
    )
    clean, audit = clean_claims(source)

    assert "unnamed_0" not in clean.columns
    assert clean["fraud_found_p"].tolist() == [1, 0, 1]
    assert clean["policy_number"].tolist() == [1001.0, 1002.0, 1003.0]
    assert clean["number_of_supplements"].isna().sum() == 1
    assert clean["witness_present"].tolist() == ["No", "Yes", "No"]
    assert clean["address_change_claim"].tolist() == ["no change", "1 year", "no change"]
    assert clean["number_of_cars"].tolist() == ["1 vehicle", "2 vehicles", "1 vehicle"]
    assert clean["source_row_id"].tolist() == [1, 2, 3]
    assert audit["positive_label_count"] == 2


def test_fixture_loads_as_four_public_schema_examples():
    fixture = Path(__file__).parent / "fixtures" / "tiny_claims.csv"
    clean, audit = clean_claims(pd.read_csv(fixture))
    assert len(clean) == 4
    assert audit["positive_label_count"] == 2

