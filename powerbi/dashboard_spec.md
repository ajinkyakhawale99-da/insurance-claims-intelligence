# Power BI dashboard build specification

## Data import and relationships

In Power BI Desktop, import the CSV files from `data/processed/powerbi/` after running the Python pipeline:

- `FactClaim.csv`
- `DimIncident.csv`
- `DimPolicy.csv`
- `DimVehicle.csv`
- `DimClaimant.csv`
- `ModelScores.csv`, `ModelMetrics.csv`, `PrecisionRecallCurve.csv`, and `CalibrationCurve.csv` after model training

Create one-to-many, single-direction relationships from each dimension's `*_key` to the matching foreign key in `FactClaim`. Relate `ModelScores[source_row_id]` to `FactClaim[claim_key]` as one-to-one (or many-to-one from scores) when Power BI accepts it. Use `FactClaim[claim_key]` as the row grain; it is a reproducible ETL surrogate for this file order, not an original business claim ID.

## Audience and decision

Audience: analytics and claims operations interview audience. The dashboard shows recorded-label patterns and an offline model's holdout behavior. It does not tell adjusters to reject claims. The source does not contain claim payments, investigation costs, or cycle times, so omit financial savings and service-level cards.

## Page 1 — Portfolio and data context

**Cards:** claim records, positive labels recorded in source, recorded-positive-label rate, and missing-row count if imported. Include a visible note: “Historical public benchmark; source labels are not independently adjudicated here.”

**Visuals:** claim count and recorded-positive count by accident year/month; record counts by vehicle category and policy type; missingness table (use `reports/generated/missingness.csv` as a supplemental import if desired).

**Slicers:** accident year, accident month, claim month, vehicle category, policy type, fault, accident area.

## Page 2 — Descriptive segments

Use clustered bars or a matrix for claim volume and recorded-positive-label rate by vehicle category, policy type, fault, accident area, police report, and witness presence. Always show denominator beside rate and suppress/signal very small groups (the starter SQL uses n ≥ 30). Add a text note that these are associations in one old benchmark extract, not causal explanations or current business rates.

Use claimant attributes only for a fairness/descriptive review. They are excluded from the model inputs; avoid ranking or targeting demographic groups. Apply privacy and small-cell policy before sharing any demographic view.

## Page 3 — Model review

Show the untouched holdout's precision, recall, Average Precision, trapezoidal PR-AUC, ROC-AUC, Brier score, and confusion counts from `ModelMetrics.csv`. Add:

- Precision-recall scatter/line using `PrecisionRecallCurve.csv`, with the validation-selected threshold annotated where practical.
- Calibration chart using `CalibrationCurve.csv` with a 45-degree reference line.
- A score distribution and review-queue count using `ModelScores.csv`.
- A cautious descriptive comparison of test-split precision/recall by `sex` and `age_of_policy_holder` from `FairnessAudit.csv`. Respect `small_cell_suppressed`; this small sample cannot certify fairness or support individual decisions.

Title every metric as performance against the **recorded benchmark label**. Show the selected threshold and the configured validation recall floor. Do not compare the holdout chart to the validation threshold as if the holdout had influenced threshold choice.

## Design and accessibility

- Use a calm navy/teal palette, one accent color for the positive label, and neutral gray for the majority label.
- Use descriptive titles, readable labels, high contrast, and accessible alt text.
- Display counts with percentages; use consistent rounding and avoid truncated axes for rates.
- Put the scope caveat on every page, not only in a tooltip.
- Add a report tooltip for each rate showing the numerator, denominator, and label definition.

