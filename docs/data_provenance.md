# Data provenance and responsible use

## Selected source

- Dataset: Vehicle Insurance Claim Fraud Detection, distributed as `Carclaims.csv` (some mirrors use `fraud_oracle.csv`).
- Public download page: [Kaggle dataset](https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection).
- Direct mirror: [Rashmi-77 GitHub CSV](https://github.com/Rashmi-77/Vehicle-Insurance-Fraud-Detection/blob/main/carclaims.csv). Select **Download raw file** on the page.
- Reported shape: approximately 15,420 claims and 33 source fields; published copies report about 923 positive rows and vary by one row in total counts.
- Label: source copies may call the binary target `FraudFound` or `FraudFound_P`. In this project the standardized name is `fraud_found_p`.
- Provenance trail: a systematic review cites the Kaggle `Carclaims.csv` file as an Angoss Knowledge Seeker dataset; related research identifies a similarly sized “Vehicle Insurance” source as Oracle-origin. The exact chain of custody from that Oracle-origin record to this Kaggle file is not documented. The original collection and adjudication documents are not part of the downloadable Kaggle page.
- License: check the current Kaggle page/uploader terms before use or redistribution. The project stores only the source URL and does not bundle the source data.

## Why this dataset

It contains a reasonably sized mixed-type motor-insurance claim table and an imbalanced binary recorded target, allowing the project to demonstrate SQL modeling, descriptive analysis, preprocessing, ranking, probability calibration, and threshold tradeoffs on public data. The benchmark supports an educational supervised-classification exercise; it does not support claims about current carrier performance.

## Limitations and risks

1. The positive label's definition, investigation process, adjudication consistency, and label timing are not documented well enough to call it confirmed fraud ground truth.
2. Several copies have 15,419 versus 15,420 rows, and CSV exports may include a synthetic index column. The cleaning step records row counts, removed index artifacts, duplicates, and class balance rather than silently claiming a single authoritative count.
3. Historical records are not representative of current regulations, products, repair costs, claims channels, fraud tactics, or population mix.
4. The data has one claim-level flat record, no durable customer/entity graph, and no reliable finalized-investigation timestamp. Random split evaluation can overstate future and new-entity performance.
5. Feature availability is unclear. Police-report and supplement fields may be observed later in the claim workflow, so the scored form is not positioned as an FNOL model.
6. The source has no claim payment amount, investigation cost, or operational outcome. Do not calculate savings or business ROI.
7. Sensitive demographic fields are kept out of predictors. They can be used only for carefully bounded aggregate fairness review; low counts make subgroup conclusions unstable.

## Use statement

The model score is the estimated probability of the public file's recorded positive label, conditional on the included fields and this training sample. It is a triage demonstration for a human reviewer. It is not a determination, allegation, or proof of fraud and must not be used to deny or delay coverage or payment.

## References

- [Kaggle distribution](https://www.kaggle.com/datasets/khusheekapoor/vehicle-insurance-fraud-detection)
- [Chawda, Grimm & Kloft, “Unsupervised Anomaly Detection for Auditing Data and Impact of Categorical Encodings” (2022)](https://arxiv.org/abs/2210.14056)
- [Systematic literature review citation of Angoss Knowledge Seeker `Carclaims.csv`](https://doi.org/10.1016/j.iswa.2024.200340)
- [Applied study reporting the benchmark class count](https://nhsjs.com/2025/detecting-fraud-insurance-claims-using-machine-learning/)

