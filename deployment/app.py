from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The persisted pipeline uses this importable scoring function during unpickling.
from bpd_feature_utils import variance_scores  # noqa: F401

MODEL_PATH = ROOT / "v2" / "outputs" / "shareable_two_step_pipeline" / "two_step_bpd_model.joblib"
DEFAULT_THRESHOLD = 0.5


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


def predict(artifact, frame: pd.DataFrame) -> pd.DataFrame:
    features = artifact["features"]
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(f"Missing {len(missing)} required feature columns: {preview}{suffix}")

    model_input = frame[features]
    stage1 = artifact["stage1_no_vs_any_bpd"]
    stage2 = artifact["stage2_mild_moderate_severe"]
    threshold = artifact.get("stage1_threshold", DEFAULT_THRESHOLD)
    bpd_probability = stage1.predict_proba(model_input)[:, 1]
    predictions = pd.Series(0, index=frame.index, dtype=int)
    routed = bpd_probability >= threshold
    if routed.any():
        predictions.loc[routed] = stage2.predict(model_input.loc[routed])

    labels = artifact["label_map"]
    result = frame.copy()
    result["bpd_probability"] = bpd_probability
    result["predicted_bpd_group"] = predictions
    result["predicted_label"] = predictions.map({int(key): value for key, value in labels.items()})
    return result


def main() -> None:
    st.set_page_config(page_title="BPD Severity Predictor", page_icon="🫁", layout="wide")
    st.title("BPD Severity Predictor")
    st.write("Upload one sample per row with clinical measurements and mapped gene-symbol expression values.")

    if not MODEL_PATH.exists():
        st.error(f"Original model artifact was not found at {MODEL_PATH}")
        st.stop()

    artifact = load_artifact()
    features = artifact["features"]
    st.caption(
        f"v2 model view: {artifact.get('view', 'unknown')} | "
        f"Required features: {len(features)} | "
        f"Gene symbols: {artifact.get('candidate_gene_count', 'unknown')}"
    )
    metrics = artifact.get("grouped_cv_metrics", {})
    if metrics:
        st.caption(
            "Grouped five-fold CV on the training cohort: "
            f"accuracy {metrics.get('accuracy', float('nan')):.1%}, "
            f"macro-F1 {metrics.get('macro_f1', float('nan')):.1%}."
        )

    template = pd.DataFrame([{feature: 0.0 for feature in features}])
    st.download_button(
        "Download input template",
        template.to_csv(index=False),
        "bpd_input_template.csv",
        "text/csv",
    )
    st.info(
        "Template values are zero placeholders. Prepare real inputs with "
        "`python src/prepare_prediction_input.py` using probe-level expression, "
        "clinical metadata, and the matching GPL6244 platform annotation."
    )

    uploaded = st.file_uploader("Upload prediction CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to generate predictions.")
        return

    try:
        input_frame = pd.read_csv(uploaded)
        predictions = predict(artifact, input_frame)
    except Exception as error:
        st.error(str(error))
        st.stop()

    st.success(f"Generated predictions for {len(predictions)} row(s).")
    counts = predictions["predicted_label"].value_counts().rename_axis("prediction").reset_index(name="count")
    st.dataframe(counts, use_container_width=True, hide_index=True)
    st.dataframe(predictions[["bpd_probability", "predicted_bpd_group", "predicted_label"]], use_container_width=True)
    st.download_button(
        "Download predictions",
        predictions.to_csv(index=False),
        "bpd_predictions.csv",
        "text/csv",
    )


if __name__ == "__main__":
    main()
