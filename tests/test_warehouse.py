from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _claim(source_row_id: int, **overrides) -> dict:
    row = {
        "source_row_id": source_row_id,
        "accident_area": "Urban",
        "year": 1995,
        "month": "Jan",
        "week_of_month": 1,
        "day_of_week": "Monday",
        "driver_rating": 1,
        "police_report_filed": "No",
        "witness_present": "No",
        "number_of_supplements": "none",
        "policy_type": "Sedan - Collision",
        "base_policy": "Collision",
        "fault": "Policy Holder",
        "deductible": 400,
        "days_policy_accident": "15 to 30",
        "days_policy_claim": "15 to 30",
        "past_number_of_claims": "1",
        "address_change_claim": "no change",
        "make": "Honda",
        "vehicle_category": "Sedan",
        "vehicle_price": "20000 to 29000",
        "age_of_vehicle": "7 years",
        "number_of_cars": "1 vehicle",
        "sex": "Male",
        "age": 35,
        "marital_status": "Married",
        "age_of_policy_holder": "31 to 35",
        "month_claimed": "Jan",
        "week_of_month_claimed": 1,
        "day_of_week_claimed": "Monday",
        "agent_type": "External",
        "fraud_found_p": 0,
    }
    row.update(overrides)
    return row


def test_star_schema_preserves_claim_rows_and_resolves_dimension_keys():
    claims = pd.DataFrame(
        [
            _claim(1),
            _claim(2, fraud_found_p=1),
            _claim(
                3,
                accident_area=None,
                witness_present=None,
                vehicle_price=None,
                sex=None,
                age=None,
            ),
        ]
    )

    connection = duckdb.connect(":memory:")
    try:
        connection.register("clean_frame", claims)
        connection.execute(
            "CREATE TABLE stg_claims AS SELECT * FROM clean_frame"
        )

        schema = (ROOT / "sql" / "schema.sql").read_text(encoding="utf-8")
        for statement in schema.split(";"):
            if statement.strip():
                connection.execute(statement)

        row_count, unique_claims = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT claim_key) FROM FactClaim"
        ).fetchone()

        assert row_count == len(claims)
        assert unique_claims == len(claims)

        missing_dimension_matches = connection.execute(
            """
            SELECT COUNT(*)
            FROM FactClaim AS f
            LEFT JOIN DimIncident AS i ON f.incident_key = i.incident_key
            LEFT JOIN DimPolicy AS p ON f.policy_key = p.policy_key
            LEFT JOIN DimVehicle AS v ON f.vehicle_key = v.vehicle_key
            LEFT JOIN DimClaimant AS c ON f.claimant_key = c.claimant_key
            WHERE i.incident_key IS NULL
               OR p.policy_key IS NULL
               OR v.vehicle_key IS NULL
               OR c.claimant_key IS NULL
            """
        ).fetchone()[0]

        assert missing_dimension_matches == 0
    finally:
        connection.close()
