"""Local human-review scoring demo for the recorded benchmark label."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from insurance_claims.features import make_model_matrix


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"
FORM_GROUPS = {
    "Incident and timing": [
        "accident_area", "month", "week_of_month", "day_of_week", "year",
        "month_claimed", "week_of_month_claimed", "day_of_week_claimed",
        "police_report_filed", "witness_present", "number_of_supplements",
    ],
    "Policy and vehicle": [
        "fault", "policy_type", "base_policy", "deductible", "days_policy_accident",
        "days_policy_claim", "past_number_of_claims", "address_change_claim",
        "agent_type", "make", "vehicle_category", "vehicle_price", "age_of_vehicle", "number_of_cars",
        "driver_rating",
    ],
}
INTEGER_FIELDS = {
    "week_of_month", "week_of_month_claimed", "year", "witness_present",
    "address_change_claim", "number_of_cars", "driver_rating", "deductible",
}


@st.cache_resource
def load_artifacts() -> tuple[dict, dict]:
    return joblib.load(MODEL_PATH), json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def render_input(field: str, spec: dict, key: str):
    label = field.replace("_", " ").title()
    if spec["kind"] == "numeric":
        return st.number_input(
            label,
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(spec["median"]),
            step=1.0 if field in INTEGER_FIELDS else 0.1,
            key=key,
        )
    options = spec.get("options", [])
    if not options:
        return None
    return st.selectbox(label, options, key=key)


st.set_page_config(page_title="Claims label-risk demo", page_icon="📄", layout="wide")
st.title("Insurance claims review-priority demo")
st.caption("A portfolio demonstration using a public historical dataset. This is not a fraud determination.")
st.warning(
    "The score estimates the public file's recorded positive label. It is not proof of fraud and must not be used to deny, delay, or reduce a claim payment."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model files are missing. Follow the README setup, download the dataset, and run the preparation and training steps first.")
    st.stop()

bundle, metadata = load_artifacts()
estimator = bundle["estimator"]
feature_columns = bundle["feature_columns"]
specs = metadata["field_specs"]
form_fields = {
    group: [field for field in fields if field in specs]
    for group, fields in FORM_GROUPS.items()
}

st.markdown("Enter the available claim attributes. Defaults are training-sample medians or common categories; change them to explore the demo.")
user_values = {}
with st.form("claim_score_form"):
    tabs = st.tabs(list(form_fields))
    for tab, (group, fields) in zip(tabs, form_fields.items()):
        with tab:
            left, right = st.columns(2)
            for index, field in enumerate(fields):
                target = left if index % 2 == 0 else right
                with target:
                    user_values[field] = render_input(field, specs[field], f"{group}_{field}")
    submitted = st.form_submit_button("Calculate advisory score", type="primary")

if submitted:
    row = {column: np.nan for column in feature_columns}
    row.update(user_values)
    matrix = make_model_matrix(pd.DataFrame([row]), feature_columns)
    probability = float(estimator.predict_proba(matrix)[:, 1][0])
    threshold = float(metadata["threshold"])
    st.subheader("Result")
    first, second = st.columns(2)
    first.metric("Estimated probability of recorded positive label", f"{probability:.1%}")
    second.metric("Validation-selected review threshold", f"{threshold:.1%}")
    if probability >= threshold:
        st.info("This example is at or above the configured review-priority threshold. A human would need to review the record.")
    else:
        st.info("This example is below the configured review-priority threshold. The score does not establish that the claim is legitimate.")
    st.caption("Features omitted from the demo form are handled by the trained preprocessing pipeline. The score is calibrated only to this historical sample and may not transfer to another population.")

