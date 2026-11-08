-- Dataset-level recorded-label rate. Interpret the positive label as source data,
-- not confirmed fraud or a current insurer prevalence estimate.
SELECT
    COUNT(*) AS claim_records,
    SUM(fraud_found_p) AS recorded_positive_labels,
    AVG(fraud_found_p::DOUBLE) AS recorded_positive_label_rate
FROM stg_claims;

-- Segment comparison with a minimum denominator. Rates are descriptive, not causal.
SELECT
    vehicle_category,
    COUNT(*) AS claim_records,
    SUM(fraud_found_p) AS recorded_positive_labels,
    AVG(fraud_found_p::DOUBLE) AS recorded_positive_label_rate
FROM stg_claims
GROUP BY vehicle_category
HAVING COUNT(*) >= 30
ORDER BY recorded_positive_label_rate DESC;

-- Recorded label rate by fault and policy type.
SELECT
    fault,
    policy_type,
    COUNT(*) AS claim_records,
    SUM(fraud_found_p) AS recorded_positive_labels,
    AVG(fraud_found_p::DOUBLE) AS recorded_positive_label_rate
FROM stg_claims
GROUP BY fault, policy_type
HAVING COUNT(*) >= 30
ORDER BY claim_records DESC;

-- Claim and accident seasonality using the source's coarse calendar attributes.
SELECT
    year AS accident_year,
    month AS accident_month,
    month_claimed AS claim_month,
    COUNT(*) AS claim_records,
    AVG(fraud_found_p::DOUBLE) AS recorded_positive_label_rate
FROM stg_claims
GROUP BY year, month, month_claimed
ORDER BY accident_year, accident_month, claim_month;

-- Missingness report, generated in Python for all columns; this query checks the
-- key fields most often needed for segment and model analysis.
SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE vehicle_category IS NULL) AS missing_vehicle_category,
    COUNT(*) FILTER (WHERE policy_type IS NULL) AS missing_policy_type,
    COUNT(*) FILTER (WHERE fault IS NULL) AS missing_fault,
    COUNT(*) FILTER (WHERE fraud_found_p IS NULL) AS missing_recorded_label
FROM stg_claims;

