# Data dictionary overview

The downloadable source has a mix of categorical and numeric fields. Header spelling varies across copies; cleaning normalizes headers to snake case and renames `number_of_suppliments` to `number_of_supplements`. `?`, blank, and common null strings become missing values. The original target is standardized to integer `fraud_found_p` (1 = source positive, 0 = source negative).

| Standardized field | Source meaning / use | Predictive use |
|---|---|---|
| `source_row_id` | Stable row order within the downloaded file; generated during cleaning | Excluded |
| `fraud_found_p` | Source's `FraudFound` or `FraudFound_P` recorded positive/negative label | Target only |
| `policy_number` | Source policy identifier | Excluded as identifier |
| `rep_number` | Representative/adjuster code | Excluded as operational proxy |
| `sex`, `age`, `marital_status`, `age_of_policy_holder` | Demographic fields | Excluded from predictors; aggregate review only |
| `month`, `week_of_month`, `day_of_week`, `year` | Accident period attributes | Calendar fields; month sin/cos derived |
| `month_claimed`, `week_of_month_claimed`, `day_of_week_claimed` | Claim-report period attributes | Calendar fields; cyclical month features derived |
| `reporting_month_gap_mod12` | Difference between claim and accident months wrapped into 0–11 | Derived feature only; not elapsed reporting time because the source has no claim year |
| `make`, `vehicle_category`, `vehicle_price`, `age_of_vehicle`, `number_of_cars` | Vehicle attributes | Candidate predictors |
| `accident_area` | Broad accident-area category | Candidate predictor |
| `fault` | Policyholder/third-party fault category | Candidate predictor; source definition should be checked before operational use |
| `policy_type`, `base_policy`, `deductible` | Policy attributes | Candidate predictors |
| `days_policy_accident`, `days_policy_claim` | Timing categories | Candidate predictors; remain categories if source values are ranges |
| `past_number_of_claims` | Prior-claim count/range category | Candidate predictor; values are source categories |
| `driver_rating` | Source rating | Candidate predictor |
| `police_report_filed`, `witness_present`, `number_of_supplements` | Claim evidence/process attributes | Candidate predictors, with timing caveat |
| `address_change_claim` | Address-change category | Candidate predictor |
| `agent_type` | Agent/claim-channel category | Candidate predictor |

This overview is not a substitute for a source-maintained dictionary. Some names are ambiguous, and feature definitions should be independently confirmed before any non-educational use.

