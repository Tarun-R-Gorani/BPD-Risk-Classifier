from __future__ import annotations

import base64
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The persisted pipeline imports this helper when Joblib restores the model.
from bpd_feature_utils import variance_scores  # noqa: F401,E402

MODEL_PATH = ROOT / "v2" / "outputs" / "shareable_two_step_pipeline" / "two_step_bpd_model.joblib"
SYNTHETIC_TEMPLATE_PATH = ROOT / "deployment" / "assets" / "synthetic_bpd_input_examples.csv"
BACKGROUND_IMAGE_PATH = ROOT / "deployment" / "assets" / "medical_lung_background.svg"
DEFAULT_THRESHOLD = 0.5
CLINICAL_FEATURES = ["days of life", "gestational age (weeks)", "birth weight (grams)"]
PAGES = ["Home", "Predict", "Results", "Timeline Explorer", "Model & Methodology"]
LABELS = {0: "No BPD", 1: "Mild BPD", 2: "Moderate BPD", 3: "Severe BPD"}
TIMELINE_PATH = ROOT / "data" / "processed" / "gbtm_input_long_format(500_genes).csv"
TIMEPOINT_LABELS = {1: "4–6 days", 2: "12–17 days", 3: "24–33 days"}


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_timeline_data():
    frame = pd.read_csv(TIMELINE_PATH)
    frame["timepoint_code"] = pd.to_numeric(frame["timepoint_code"], errors="coerce")
    frame["bpd_group"] = pd.to_numeric(frame["bpd_group"], errors="coerce")
    frame["timepoint_label"] = frame["timepoint_code"].map(TIMEPOINT_LABELS)
    frame["bpd_label"] = frame["bpd_group"].map(LABELS)
    excluded = {
        "infant_id", "timepoint_code", "timepoint_label", "bpd_group", "bpd_label",
        *CLINICAL_FEATURES,
    }
    genes = [column for column in frame.columns if column not in excluded]
    return frame, genes


def predict(artifact, frame: pd.DataFrame) -> pd.DataFrame:
    features = artifact["features"]
    missing = [feature for feature in features if feature not in frame.columns]
    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""
        raise ValueError(f"Missing {len(missing)} required feature columns: {preview}{suffix}")
    if frame.empty:
        raise ValueError("The uploaded CSV contains no sample rows.")

    raw_input = frame[features]
    model_input = raw_input.apply(pd.to_numeric, errors="coerce")
    invalid = raw_input.notna() & model_input.isna()
    if invalid.to_numpy().any():
        bad_columns = invalid.columns[invalid.any(axis=0)].tolist()
        raise ValueError(
            "Some model inputs are not numeric. Check these columns: "
            + ", ".join(bad_columns[:8])
            + (" ..." if len(bad_columns) > 8 else "")
        )

    stage1 = artifact["stage1_no_vs_any_bpd"]
    stage2 = artifact["stage2_mild_moderate_severe"]
    threshold = artifact.get("stage1_threshold", DEFAULT_THRESHOLD)
    class_one = list(stage1.classes_).index(1)
    any_bpd_score = stage1.predict_proba(model_input)[:, class_one]
    predictions = pd.Series(0, index=frame.index, dtype=int)
    routed = any_bpd_score >= threshold
    result = frame.copy()
    if not any(column in result for column in ["sample_id", "geo_accession", "title", "sample_number"]):
        result.insert(0, "sample_number", [f"Sample {i + 1}" for i in range(len(result))])

    result["bpd_probability"] = any_bpd_score
    result["stage1_decision"] = np.where(routed, "Any BPD", "No BPD")
    if routed.any():
        routed_input = model_input.loc[routed]
        predictions.loc[routed] = stage2.predict(routed_input)
        conditional_scores = stage2.predict_proba(routed_input)
        stage2_classes = list(stage2.classes_)
        for class_id, label in ((1, "mild"), (2, "moderate"), (3, "severe")):
            result.loc[routed, f"stage2_{label}_score"] = conditional_scores[
                :, stage2_classes.index(class_id)
            ]
    else:
        for label in ("mild", "moderate", "severe"):
            result[f"stage2_{label}_score"] = np.nan

    labels = artifact["label_map"]
    result["predicted_bpd_group"] = predictions
    result["predicted_label"] = predictions.map({int(key): value for key, value in labels.items()})
    return result


def go_to(page: str) -> None:
    st.session_state["next_page"] = page
    st.rerun()


def inject_styles() -> None:
    background_image = "none"
    if BACKGROUND_IMAGE_PATH.is_file():
        encoded_image = base64.b64encode(BACKGROUND_IMAGE_PATH.read_bytes()).decode("ascii")
        background_image = f"url('data:image/svg+xml;base64,{encoded_image}')"
    st.markdown(
        """
        <style>
        html, body, [class*="css"], button, input, textarea, select {
            font-family: Inter, "Aptos", "Segoe UI", system-ui, sans-serif !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #f7fbfd;
            background-image:
                linear-gradient(rgba(247,251,253,.75), rgba(245,250,252,.84)),
                __BACKGROUND_IMAGE__,
                radial-gradient(ellipse at 5% 0%, rgba(12,124,134,.085), transparent 32%),
                radial-gradient(ellipse at 96% 4%, rgba(49,101,137,.07), transparent 30%),
                linear-gradient(145deg, #f7fbfd 0%, #eff6f9 56%, #f5fafb 100%);
            background-position: center top;
            background-size: cover;
            background-attachment: fixed;
        }
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stHeader"] { background: transparent; }
        .main .block-container { max-width: 1380px; padding: 1.3rem 2rem 3rem; }
        h1, h2, h3 { color: #163b56; letter-spacing: -.025em; }
        p, li, label { color: #3c5669; }
        a { color: #087f89 !important; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,.9); border: 1px solid #dce9ef;
            border-radius: 18px; box-shadow: 0 10px 28px rgba(25,64,88,.055);
        }
        [data-testid="stMetric"] {
            background: #fff; border: 1px solid #dce9ef;
            border-radius: 16px; padding: 14px 18px;
        }
        [data-testid="stMetricValue"] { color: #087f89; }
        [data-testid="stFileUploaderDropzone"] {
            background: rgba(244,250,252,.96); border: 1px dashed #a9cbd6;
            border-radius: 14px; color: #244b63;
        }
        [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"],
        [data-testid="stFileUploaderDropzone"] small { color: #496779 !important; }
        [data-testid="stFileUploaderDropzone"] button {
            background: #fff !important; color: #176b78 !important;
            border: 1px solid #b7d4dd !important; border-radius: 9px !important;
        }
        [data-baseweb="select"] > div {
            background: #fff !important; border-color: #c9dce4 !important;
            color: #244b63 !important; border-radius: 10px !important;
        }
        [data-baseweb="popover"] [role="listbox"],
        [data-baseweb="popover"] ul {
            background: #fff !important; color: #244b63 !important;
        }
        [data-baseweb="popover"] [role="option"] { color: #244b63 !important; }
        [data-baseweb="tag"] {
            background: #e4f3f4 !important; color: #176b78 !important;
        }
        [data-testid="stRadio"] label, [data-testid="stCheckbox"] label {
            color: #34566b !important;
        }
        [data-testid="stCode"] pre, [data-testid="stCode"] code {
            background: #f2f7f9 !important; color: #244b63 !important;
            border: 1px solid #dce9ef; border-radius: 10px;
        }
        [data-testid="stExpander"] details {
            background: rgba(255,255,255,.93); border: 1px solid #dce9ef;
            border-radius: 14px;
        }
        [data-testid="stDataFrame"] {
            background: #fff; border: 1px solid #dce9ef; border-radius: 12px;
            --gdg-bg-cell: #fff; --gdg-bg-header: #edf5f7;
            --gdg-text-dark: #244b63; --gdg-text-medium: #496779;
        }
        [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
            background: rgba(255,255,255,.86); border: 1px solid #dce9ef;
            border-radius: 16px; padding: .5rem;
        }
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 12px; min-height: 2.7rem; font-weight: 650;
            transition: transform .15s ease, box-shadow .15s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            transform: translateY(-1px); box-shadow: 0 8px 20px rgba(8,91,106,.14);
        }
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"] {
            background: linear-gradient(120deg, #087f89, #12657c);
            color: #fff; border: 1px solid #087f89;
        }
        div.stButton > button[kind="secondary"], div.stDownloadButton > button[kind="secondary"] {
            background: rgba(255,255,255,.9); color: #244b63; border: 1px solid #d4e3ea;
        }
        .site-brand { color: #163b56; font-size: 1.15rem; font-weight: 800; padding: .55rem 0; }
        .hero {
            background: linear-gradient(118deg, #123a58 0%, #0c6577 58%, #0b8588 100%);
            color: #fff; border-radius: 24px; padding: 2.2rem 2.5rem;
            box-shadow: 0 18px 42px rgba(18,58,88,.18); margin: .5rem 0 1.5rem;
        }
        .hero h1 { color: #fff; font-size: clamp(2.2rem,4vw,3.6rem); margin: .35rem 0 .65rem; }
        .hero p { color: rgba(255,255,255,.86); font-size: 1.08rem; max-width: 760px; }
        .hero-kicker { color: #b8e9e8; font-size: .78rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
        @media (max-width: 750px) {
            .main .block-container { padding-left: 1rem; padding-right: 1rem; }
            .hero { padding: 1.5rem; }
        }
        </style>
        """.replace("__BACKGROUND_IMAGE__", background_image),
        unsafe_allow_html=True,
    )


def render_top_navigation() -> None:
    active_page = st.session_state.get("active_page", "Home")
    columns = st.columns([1.35, .78, .78, .78, 1.22, 1.48])
    with columns[0]:
        st.markdown('<div class="site-brand">BPD <span style="color:#0b8588">/</span> Insight</div>', unsafe_allow_html=True)
    for column, page in zip(columns[1:], PAGES):
        with column:
            if st.button(page, key=f"top_nav_{page}",
                         type="primary" if page == active_page else "secondary",
                         use_container_width=True):
                go_to(page)
    st.markdown("<hr style='border:0;border-top:1px solid #d9e7ed;margin:.35rem 0 1.6rem'>", unsafe_allow_html=True)


def render_home(artifact) -> None:
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Neonatal respiratory research</div>'
        '<h1>Understanding BPD through data</h1>'
        '<p>Explore bronchopulmonary dysplasia, follow gene-expression patterns across visits, '
        'and try a two-stage research classifier.</p></div>',
        unsafe_allow_html=True,
    )

    about, why = st.columns([1.15, 1])
    with about:
        with st.container(border=True):
            st.subheader("What is BPD?")
            st.write(
                "Bronchopulmonary dysplasia (BPD) is a chronic lung condition that most often "
                "affects babies born prematurely. Their lungs may still be developing, and some "
                "infants need breathing support or supplemental oxygen for an extended period."
            )
            st.markdown(
                "[Learn more from the U.S. National Heart, Lung, and Blood Institute]("
                "https://www.nhlbi.nih.gov/health/bronchopulmonary-dysplasia)."
            )
    with why:
        with st.container(border=True):
            st.subheader("Why study it?")
            st.write(
                "BPD can have a complex course, and respiratory needs may continue after the "
                "newborn period. Studying clinical and molecular patterns can help researchers "
                "describe how the condition varies across infants and time."
            )
            st.caption("This project explores patterns in a public gene-expression cohort; it does not guide care.")

    st.subheader("About this project")
    project, model = st.columns([1.25, 1])
    with project:
        with st.container(border=True):
            st.write(
                "The classifier combines three clinical measurements with gene-expression "
                "features from the 24-33 day timepoint. Stage 1 separates No BPD from Any BPD; "
                "Stage 2 predicts severity for samples routed to Any BPD."
            )
            if st.button("Explore the model", type="secondary"):
                go_to("Model & Methodology")
    with model:
        with st.container(border=True):
            st.metric("Model inputs", f"{len(artifact['features']):,}")
            st.caption(
                f"{artifact.get('candidate_gene_count', 0):,} gene symbols + "
                f"{len(CLINICAL_FEATURES)} clinical features | Timepoint 3"
            )
            if st.button("Try the prediction demo"):
                go_to("Predict")


def render_predict(artifact) -> None:
    st.title("Predict BPD severity")
    st.write(
        "Upload one sample per row. Required inputs are three clinical fields and "
        f"{artifact.get('candidate_gene_count', 0):,} mapped gene-symbol measurements."
    )
    if SYNTHETIC_TEMPLATE_PATH.exists():
        st.download_button(
            "Download synthetic sample template",
            SYNTHETIC_TEMPLATE_PATH.read_bytes(),
            "synthetic_bpd_input_examples.csv",
            "text/csv",
        )
        st.caption(
            "This file contains fabricated demo values generated from mixtures "
            "of training-feature values with small noise. Use it to explore the "
            "website only; predictions on synthetic rows are not meaningful model results."
        )
    else:
        st.warning("Synthetic template not found. Run `python src/create_synthetic_template.py`.")

    uploaded = st.file_uploader("Upload prediction CSV", type=["csv"], key="prediction_upload")
    if uploaded is None:
        st.caption("The prepared input CSV should use the same GPL6244 probe-to-symbol mapping as the training data.")
        return

    try:
        input_frame = pd.read_csv(uploaded)
    except Exception as error:
        st.error(f"Could not read the CSV: {error}")
        return

    st.write(f"**Rows:** {len(input_frame):,} · **Columns:** {len(input_frame.columns):,}")
    preview_columns = [
        column for column in ["sample_id", "geo_accession", *CLINICAL_FEATURES]
        if column in input_frame.columns
    ]
    preview_columns += [
        column for column in artifact.get("gene_features", [])
        if column in input_frame.columns
    ][:5]
    if preview_columns:
        st.dataframe(input_frame[preview_columns].head(5), use_container_width=True, hide_index=True)

    if st.button("Generate predictions", type="primary", disabled=input_frame.empty):
        with st.spinner("Running the two-stage model..."):
            try:
                output = predict(artifact, input_frame)
            except Exception as error:
                st.error(str(error))
                return
        st.session_state["prediction_results"] = output
        st.session_state["prediction_source_name"] = uploaded.name
        st.success(f"Generated predictions for {len(output):,} sample(s). Open Results to review them.")
        if st.button("View results", key="view_results_button"):
            go_to("Results")


def render_results() -> None:
    st.title("Prediction results")
    results = st.session_state.get("prediction_results")
    if results is None:
        st.info("No predictions yet. Go to Predict and upload a prepared CSV.")
        if st.button("Go to Predict", type="primary"):
            go_to("Predict")
        return

    st.caption(f"Source file: {st.session_state.get('prediction_source_name', 'Uploaded CSV')}")
    counts = results["predicted_label"].value_counts().reindex(list(LABELS.values()), fill_value=0)
    metrics = st.columns(4)
    for column, (label, count) in zip(metrics, counts.items()):
        column.metric(label, int(count))

    st.subheader("Per-sample results")
    display_columns = [
        column for column in [
            "sample_id", "geo_accession", "title", "sample_number",
            "predicted_label", "stage1_decision", "bpd_probability",
            "stage2_mild_score", "stage2_moderate_score", "stage2_severe_score",
        ] if column in results.columns
    ]
    st.dataframe(results[display_columns], use_container_width=True, hide_index=True)
    st.caption(
        "Stage 2 scores are conditional on being routed to Any BPD. "
        "The Stage 1 score is a model score and has not been presented as a calibrated clinical risk."
    )
    st.download_button(
        "Download results CSV",
        results.to_csv(index=False),
        "bpd_predictions.csv",
        "text/csv",
        type="primary",
    )


def render_methodology(artifact) -> None:
    st.title("Model & Methodology")
    st.write(
        "The deployed model uses a timepoint-3 sample view and a two-stage "
        "classification hierarchy."
    )

    st.subheader("Inputs")
    st.write(
        f"Each sample has **{len(artifact['features']):,} input features**: "
        f"{artifact.get('candidate_gene_count', 0):,} gene-symbol expression values and "
        f"{len(CLINICAL_FEATURES)} clinical measurements."
    )
    st.code(", ".join(CLINICAL_FEATURES), language=None)

    st.subheader("Prediction flow")
    st.markdown(
        "**Stage 1 — No BPD vs Any BPD**  \n"
        "A balanced logistic regression model produces an Any-BPD score. "
        f"Scores at or above the saved threshold ({artifact.get('stage1_threshold', DEFAULT_THRESHOLD):.2f}) "
        "are routed to Stage 2."
    )
    st.markdown(
        "**Stage 2 — BPD severity**  \n"
        "A balanced multiclass logistic regression model predicts Mild, Moderate, or Severe BPD "
        "for samples routed from Stage 1."
    )

    st.subheader("Feature processing")
    st.markdown(
        "1. Map microarray probes to gene symbols and average probes assigned to the same symbol.\n"
        "2. Impute missing values and scale clinical and gene-expression features.\n"
        "3. Each stage selects the 5,000 highest-variance genes, then uses an ANOVA F-test "
        "to retain its stage-specific gene set.\n"
        "4. Apply the fitted transformations and logistic-regression models to new rows."
    )

    stage1_genes = artifact.get("stage1_selected_genes", [])
    stage2_genes = artifact.get("stage2_selected_genes", [])
    with st.expander(f"Stage 1 selected genes ({len(stage1_genes)})"):
        st.write(", ".join(stage1_genes) if stage1_genes else "Gene list unavailable.")
    with st.expander(f"Stage 2 selected genes ({len(stage2_genes)})"):
        st.write(", ".join(stage2_genes) if stage2_genes else "Gene list unavailable.")

    st.subheader("Reading a model result")
    result_steps = st.columns(3)
    with result_steps[0]:
        st.markdown("**1 · Initial screen**")
        st.write(
            "Stage 1 assigns an Any-BPD score. A score at or above "
            f"**{artifact.get('stage1_threshold', DEFAULT_THRESHOLD):.2f}** sends the sample to Stage 2."
        )
    with result_steps[1]:
        st.markdown("**2 · Severity estimate**")
        st.write(
            "For samples sent forward, Stage 2 compares the Mild, Moderate, "
            "and Severe BPD scores and selects the highest scoring class."
        )
    with result_steps[2]:
        st.markdown("**3 · Combined label**")
        st.write(
            "Samples below the Stage 1 threshold are labeled No BPD. "
            "Samples routed to Stage 2 receive its predicted severity label."
        )


def render_timeline() -> None:
    st.title("Timeline Explorer")
    st.write(
        "Explore aggregate gene-expression trends across the cohort’s three "
        "observed timepoints, grouped by recorded BPD category."
    )
    st.info(
        "This is a descriptive cohort view, not an individual prediction. The "
        "deployed classifier uses timepoint-3 inputs."
    )
    if not TIMELINE_PATH.exists():
        st.error(f"Longitudinal dataset was not found at {TIMELINE_PATH}")
        return

    frame, genes = load_timeline_data()
    gene = st.selectbox("Gene symbol", genes)
    all_groups = list(LABELS.values())
    groups = st.multiselect("BPD groups", all_groups, default=all_groups)
    statistic = st.radio("Expression summary", ["Median", "Mean"], horizontal=True)
    if not groups:
        st.warning("Select at least one BPD group to display the timeline.")
        return

    group_codes = [code for code, label in LABELS.items() if label in groups]
    filtered = frame[frame["bpd_group"].isin(group_codes)]
    aggfunc = "median" if statistic == "Median" else "mean"
    summary = (
        filtered.groupby(["timepoint_code", "timepoint_label", "bpd_label"], observed=True)
        .agg(expression=(gene, aggfunc), samples=("infant_id", "nunique"))
        .reset_index()
        .sort_values("timepoint_code")
    )
    ordered_timepoints = [TIMEPOINT_LABELS[code] for code in sorted(TIMEPOINT_LABELS)]
    chart = summary.pivot(index="timepoint_label", columns="bpd_label", values="expression")
    chart = chart.reindex(ordered_timepoints)
    chart.index.name = "Timepoint"
    chart.columns.name = None

    st.subheader(f"{statistic} expression for {gene}")
    st.line_chart(chart, use_container_width=True)
    st.caption("Each point summarizes available samples at that visit; sample counts are shown below.")
    st.dataframe(
        summary[["bpd_label", "timepoint_label", "expression", "samples"]],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(page_title="BPD Risk Classification", page_icon="BPD", layout="wide")
    inject_styles()
    if not MODEL_PATH.exists():
        st.error(f"Model artifact was not found at {MODEL_PATH}")
        st.stop()

    if "next_page" in st.session_state:
        st.session_state["active_page"] = st.session_state.pop("next_page")
    artifact = load_artifact()
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Home"
    render_top_navigation()

    page = st.session_state["active_page"]
    if page == "Home":
        render_home(artifact)
    elif page == "Predict":
        render_predict(artifact)
    elif page == "Results":
        render_results()
    elif page == "Timeline Explorer":
        render_timeline()
    else:
        render_methodology(artifact)


if __name__ == "__main__":
    main()
