# Implementation walkthrough

## 1. Acquire and inspect the public file

Download `Carclaims.csv` from the Kaggle link or the GitHub mirror in the data card. Keep it local under `data/raw/`; do not add it to Git. The prep script accepts this name, `fraud_oracle.csv`, or a single CSV under another name. It supports the source target headers `FraudFound`, `FraudFound_P`, and `fraud_reported`.

## 2. Clean and audit

Run `python scripts/prepare_data.py`. The cleaner standardizes camel-case headers, handles source missing-value markers, converts the recorded target to 0/1, removes an unlabelled export index, and preserves duplicates while reporting them. Review `data/processed/data_audit.json` before modeling, especially source row count, class balance, missingness, and duplicates.

## 3. Explore the data

Run `python scripts/run_eda.py`. It writes label balance, missingness, numeric summaries, and recorded-positive rates by selected categorical fields. Segment plots use a minimum count of 30 to reduce noisy tiny-group comparisons. Treat the rates as descriptive associations in this file only.

## 4. Build the SQL model

Run `python scripts/build_warehouse.py`. The generated DuckDB file contains the cleaned staging table, four descriptive dimensions, and one fact table at one source-row per claim-record grain. Use `sql/analytics.sql` for workload, label-rate, segment, seasonality, and null checks. The surrogate `claim_key` is assigned from file row order; it is not an original claim identifier.

## 5. Train, calibrate, and set a threshold

Run `python scripts/train_model.py --min-recall 0.60`. Model selection uses 5-fold Average Precision on the training partition. Missing-value handling, numeric scaling, and one-hot encoding live inside the estimator pipeline. Calibration uses training folds. Threshold choice happens on the separate validation set; the untouched test set is scored once afterward.

Review `models/metrics.json`, `models/MODEL_CARD.md`, `data/processed/powerbi/PrecisionRecallCurve.csv`, and `CalibrationCurve.csv`. A higher recall threshold usually routes more claims for review and can reduce precision. Select an operating point with a review-capacity and error-cost discussion; the 60% recall floor is only a portfolio example.

## 6. Inspect subgroup behavior

Review `data/processed/powerbi/FairnessAudit.csv` for a small-cell-aware descriptive comparison on the test split. Sex and age-band fields are absent from model predictors. Small sample sizes, incomplete labels, and possible proxy effects mean this table is an exploratory screen, not a fairness certification.

## 7. Build the Power BI report

Import the fact/dimension CSVs from `data/processed/powerbi/` and create the relationships described in `powerbi/dashboard_spec.md`. After training, import the model score, metric, calibration, precision-recall, and subgroup-audit exports. Keep source-label summaries separate from holdout model metrics.

## 8. Demonstrate a score

Run `streamlit run streamlit_app.py`. The form edits a subset of the benchmark-style input fields and uses sample medians/common categories for defaults. The output is a review-priority illustration with an explicit warning; it does not assert fraud or make a payment decision.

