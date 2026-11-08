-- Run after scripts/build_warehouse.py creates stg_claims.
-- Surrogate keys are ETL conveniences, with no business entities inferred beyond
-- distinct attribute combinations in this flat source extract.

CREATE OR REPLACE TABLE DimIncident AS
SELECT
    (row_number() OVER (ORDER BY accident_area, year, month, week_of_month, day_of_week, driver_rating, police_report_filed, witness_present, number_of_supplements))::BIGINT AS incident_key,
    accident_area,
    year AS accident_year,
    month AS accident_month,
    week_of_month AS accident_week_of_month,
    day_of_week AS accident_day_of_week,
    driver_rating,
    police_report_filed,
    witness_present,
    number_of_supplements
FROM (
    SELECT DISTINCT accident_area, year, month, week_of_month, day_of_week,
        driver_rating, police_report_filed, witness_present, number_of_supplements
    FROM stg_claims
);

CREATE OR REPLACE TABLE DimPolicy AS
SELECT
    (row_number() OVER (ORDER BY policy_type, base_policy, fault, deductible, days_policy_accident, days_policy_claim, past_number_of_claims, address_change_claim))::BIGINT AS policy_key,
    policy_type,
    base_policy,
    fault,
    deductible,
    days_policy_accident,
    days_policy_claim,
    past_number_of_claims,
    address_change_claim
FROM (
    SELECT DISTINCT policy_type, base_policy, fault, deductible,
        days_policy_accident, days_policy_claim, past_number_of_claims, address_change_claim
    FROM stg_claims
);

CREATE OR REPLACE TABLE DimVehicle AS
SELECT
    (row_number() OVER (ORDER BY make, vehicle_category, vehicle_price, age_of_vehicle, number_of_cars))::BIGINT AS vehicle_key,
    make,
    vehicle_category,
    vehicle_price,
    age_of_vehicle,
    number_of_cars
FROM (
    SELECT DISTINCT make, vehicle_category, vehicle_price, age_of_vehicle, number_of_cars
    FROM stg_claims
);

CREATE OR REPLACE TABLE DimClaimant AS
SELECT
    (row_number() OVER (ORDER BY sex, age, marital_status, age_of_policy_holder))::BIGINT AS claimant_key,
    sex,
    age,
    marital_status,
    age_of_policy_holder
FROM (
    SELECT DISTINCT sex, age, marital_status, age_of_policy_holder
    FROM stg_claims
);

CREATE OR REPLACE TABLE FactClaim AS
SELECT
    s.source_row_id AS claim_key,
    s.month_claimed AS claim_month,
    s.week_of_month_claimed AS claim_week_of_month,
    s.day_of_week_claimed AS claim_day_of_week,
    s.agent_type,
    s.fraud_found_p AS recorded_positive_label,
    i.incident_key,
    p.policy_key,
    v.vehicle_key,
    c.claimant_key
FROM stg_claims s
JOIN DimIncident i ON
    s.accident_area IS NOT DISTINCT FROM i.accident_area AND
    s.year IS NOT DISTINCT FROM i.accident_year AND
    s.month IS NOT DISTINCT FROM i.accident_month AND
    s.week_of_month IS NOT DISTINCT FROM i.accident_week_of_month AND
    s.day_of_week IS NOT DISTINCT FROM i.accident_day_of_week AND
    s.driver_rating IS NOT DISTINCT FROM i.driver_rating AND
    s.police_report_filed IS NOT DISTINCT FROM i.police_report_filed AND
    s.witness_present IS NOT DISTINCT FROM i.witness_present AND
    s.number_of_supplements IS NOT DISTINCT FROM i.number_of_supplements
JOIN DimPolicy p ON
    s.policy_type IS NOT DISTINCT FROM p.policy_type AND
    s.base_policy IS NOT DISTINCT FROM p.base_policy AND
    s.fault IS NOT DISTINCT FROM p.fault AND
    s.deductible IS NOT DISTINCT FROM p.deductible AND
    s.days_policy_accident IS NOT DISTINCT FROM p.days_policy_accident AND
    s.days_policy_claim IS NOT DISTINCT FROM p.days_policy_claim AND
    s.past_number_of_claims IS NOT DISTINCT FROM p.past_number_of_claims AND
    s.address_change_claim IS NOT DISTINCT FROM p.address_change_claim
JOIN DimVehicle v ON
    s.make IS NOT DISTINCT FROM v.make AND
    s.vehicle_category IS NOT DISTINCT FROM v.vehicle_category AND
    s.vehicle_price IS NOT DISTINCT FROM v.vehicle_price AND
    s.age_of_vehicle IS NOT DISTINCT FROM v.age_of_vehicle AND
    s.number_of_cars IS NOT DISTINCT FROM v.number_of_cars
JOIN DimClaimant c ON
    s.sex IS NOT DISTINCT FROM c.sex AND
    s.age IS NOT DISTINCT FROM c.age AND
    s.marital_status IS NOT DISTINCT FROM c.marital_status AND
    s.age_of_policy_holder IS NOT DISTINCT FROM c.age_of_policy_holder;

