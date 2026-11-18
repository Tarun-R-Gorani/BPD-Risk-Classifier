from __future__ import annotations

import base64
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import altair as alt
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


def is_git_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    with path.open("rb") as file:
        return file.read(80).startswith(b"version https://git-lfs.github.com/spec/v1")


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_timeline_data():
    if is_git_lfs_pointer(TIMELINE_PATH):
        raise RuntimeError(
            "The timeline dataset is a Git LFS pointer, not the actual CSV. "
            "Run `git lfs install` and `git lfs pull` from the repository root."
        )
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
    result["stage1_threshold"] = threshold
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
        :root {
            --ink: #173f55;
            --ink-soft: #496779;
            --teal: #087f89;
            --teal-deep: #10566b;
            --teal-pale: #e0f2f3;
            --orange: #e99a62;
            --orange-deep: #bd642b;
            --peach: #fff0e4;
            --line: #d8e7ec;
            --surface: rgba(255,255,255,.88);
        }
        html, body, [class*="css"], button, input, textarea, select {
            font-family: Inter, "Aptos", "Segoe UI", system-ui, sans-serif !important;
        }
        [data-testid="stAppViewContainer"] {
            background-color: #f4fafb;
            background-image:
                radial-gradient(ellipse at 0% 8%, rgba(102,207,211,.20), transparent 34%),
                radial-gradient(ellipse at 100% 8%, rgba(247,173,124,.24), transparent 31%),
                radial-gradient(ellipse at 2% 100%, rgba(247,173,124,.20), transparent 29%),
                radial-gradient(ellipse at 100% 92%, rgba(255,213,181,.24), transparent 32%),
                __BACKGROUND_IMAGE__,
                linear-gradient(135deg, #f1f8fa 0%, #f5fafb 55%, #fff7f0 100%);
            background-position: center top;
            background-size: auto, auto, auto, auto, cover, cover;
            background-attachment: fixed;
        }
        [data-testid="stSidebar"] { display: none; }
        [data-testid="stHeader"] { background: transparent; }
        .main .block-container { max-width: 1500px; padding: .8rem 2rem 3.2rem; }
        h1, h2, h3, h4 { color: var(--ink); letter-spacing: -.025em; }
        h2 { margin-top: 1.4rem; }
        p, li, label { color: #3c5669; }
        .main .block-container p { line-height:1.58; }
        .main .block-container [data-testid="stCaptionContainer"] { color:#78909d; }
        a { color: var(--teal) !important; text-decoration-color: rgba(217,119,50,.55); }
        a:hover { color: var(--orange-deep) !important; text-decoration-color: var(--orange-deep); }
        hr { border-color: var(--line) !important; }

        /* Shared surfaces */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(255,255,255,.94), rgba(248,252,252,.88)) !important;
            border: 1px solid rgba(196,219,226,.95) !important;
            border-radius: 18px !important;
            overflow: hidden;
            box-shadow: 0 10px 28px rgba(25,64,88,.075) !important;
            transition: border-color .18s ease, box-shadow .18s ease, transform .18s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: #c2dce2 !important;
            box-shadow: 0 12px 30px rgba(25,64,88,.085) !important;
        }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, #fff 0%, #fffaf6 100%);
            border: 1px solid #eadfd6;
            border-radius: 16px;
            padding: 14px 18px;
        }
        [data-testid="stMetricLabel"] { color: var(--ink-soft); }
        [data-testid="stMetricValue"] { color: var(--orange-deep); font-weight: 750; }
        [data-testid="stMetricDelta"] { color: var(--teal); }

        /* Inputs and controls */
        [data-testid="stFileUploaderDropzone"] {
            background: linear-gradient(120deg, rgba(240,249,250,.98), rgba(255,247,240,.92));
            border: 1px dashed #b7d1d9;
            border-radius: 15px;
            color: #244b63;
        }
        [data-testid="stFileUploaderDropzone"] [data-testid="stMarkdownContainer"],
        [data-testid="stFileUploaderDropzone"] small { color: var(--ink-soft) !important; }
        [data-testid="stFileUploaderDropzone"] button {
            background: #fff !important; color: var(--teal-deep) !important;
            border: 1px solid #b7d4dd !important; border-radius: 10px !important;
        }
        [data-testid="stFileUploaderDropzone"] button:hover {
            color: #984b19 !important; border-color: var(--orange) !important;
            background: #fff8f1 !important;
        }
        [data-baseweb="select"] > div {
            background: rgba(255,255,255,.96) !important;
            border-color: #c9dce4 !important; color: #244b63 !important;
            border-radius: 11px !important;
        }
        [data-baseweb="select"] > div:focus-within { border-color: var(--orange) !important; }
        [data-baseweb="popover"] [role="listbox"], [data-baseweb="popover"] ul {
            background: #fff !important; color: #244b63 !important;
        }
        [data-baseweb="popover"] [role="option"] { color: #244b63 !important; }
        [data-baseweb="tag"] {
            background: var(--peach) !important; color: #9f501c !important;
            border: 1px solid #f0d1b8 !important; border-radius: 8px !important;
        }
        [data-testid="stRadio"] label, [data-testid="stCheckbox"] label { color: #34566b !important; }
        [data-testid="stRadio"] [role="radio"]:checked > div:first-child { border-color: var(--orange) !important; }
        [data-testid="stRadio"] [role="radio"]:checked > div:first-child > div { background: var(--orange) !important; }

        /* Buttons */
        div.stButton > button, div.stDownloadButton > button {
            border-radius: 12px; min-height: 2.75rem; padding: .45rem 1.1rem;
            font-weight: 650; transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
        }
        div.stButton > button:hover, div.stDownloadButton > button:hover {
            transform: translateY(-1px); box-shadow: 0 8px 20px rgba(8,91,106,.13);
            border-color: var(--orange) !important;
        }
        div.stButton > button[kind="primary"], div.stDownloadButton > button[kind="primary"],
        div.stButton > button[data-testid="stBaseButton-primary"],
        div.stDownloadButton > button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(115deg, #087f89 0%, #12657c 100%) !important;
            color: #fff !important; -webkit-text-fill-color:#fff !important;
            border: 1px solid #087f89 !important;
        }
        div.stButton > button[kind="primary"] *,
        div.stButton > button[data-testid="stBaseButton-primary"] * {
            color:#fff !important; -webkit-text-fill-color:#fff !important;
        }
        div.stButton > button[kind="primary"]:hover, div.stDownloadButton > button[kind="primary"]:hover,
        div.stButton > button[data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(115deg, #0a737d 0%, #15566d 100%);
            box-shadow: 0 9px 22px rgba(8,91,106,.2);
        }
        div.stButton > button[kind="secondary"], div.stDownloadButton > button[kind="secondary"],
        div.stButton > button[data-testid="stBaseButton-secondary"] {
            background: rgba(255,255,255,.96) !important; color: #244b63 !important; border: 1px solid #d4e3ea !important;
        }
        div.stButton > button[kind="secondary"]:hover, div.stDownloadButton > button[kind="secondary"]:hover {
            background: #fff7ef !important; color: #984b19 !important;
        }

        /* Tables, code and expanders */
        [data-testid="stCode"] pre, [data-testid="stCode"] code {
            background: #f1f7f8 !important; color: #244b63 !important;
            border: 1px solid var(--line); border-radius: 11px;
        }
        [data-testid="stExpander"] details {
            background: rgba(255,255,255,.94); border: 1px solid var(--line);
            border-radius: 13px; overflow: hidden;
        }
        [data-testid="stExpander"] summary:hover { color: var(--orange-deep); }
        [data-testid="stDataFrame"] {
            background: #fff; border: 1px solid var(--line); border-radius: 13px;
            --gdg-bg-cell: #fff; --gdg-bg-header: #edf5f7;
            --gdg-text-dark: #244b63; --gdg-text-medium: #496779;
        }
        [data-testid="stAlert"] { border-radius: 13px; }
        [data-testid="stAlert"] [data-testid="stMarkdownContainer"] p { color: inherit; }
        [data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {
            background: rgba(255,255,255,.9); border: 1px solid var(--line);
            border-radius: 17px; padding: .6rem;
            box-shadow: 0 8px 22px rgba(25,64,88,.045);
        }

        /* Brand and navigation */
        .top-nav-row { align-items: center; }
        .site-brand { color: var(--ink); font-size: 1.24rem; font-weight: 800; padding: .55rem 0; white-space: nowrap; }
        div.stButton > button { font-size:.9rem; }
        .top-nav-row [data-testid="stBaseButton-primary"] { box-shadow:0 5px 14px rgba(8,127,137,.16); }
        .site-brand .brand-orange { color: #c7611c !important; }
        .site-brand span:not(.brand-orange) { color: var(--teal) !important; }
        .main .block-container h2 { border-left: 5px solid var(--orange); padding-left: .65rem; }

        /* Hero banners */
        .hero, .page-hero {
            position: relative; overflow: hidden; isolation: isolate;
        }
        .hero {
            background: linear-gradient(118deg, #123a58 0%, #0c6577 54%, #0b8588 84%, #c8753a 125%);
            padding: 2.35rem 2.5rem; margin: .5rem 0 1.5rem;
            border:1px solid rgba(255,255,255,.25); border-radius:24px;
            box-shadow:0 18px 38px rgba(16,72,91,.16);
        }
        .hero h1 { color: #fff; font-size: clamp(2.2rem,4vw,3.6rem); margin: .35rem 0 .65rem; }
        .hero p { color: rgba(255,255,255,.88); font-size: 1.08rem; max-width: 760px; }
        .hero-kicker, .page-kicker {
            color: #ffd2ad; font-size: .76rem; font-weight: 750;
            letter-spacing: .13em; text-transform: uppercase;
        }
        .hero-kicker::after {
            content: ""; display: block; width: 54px; height: 3px;
            margin-top: .55rem; background: #f18b40; border-radius: 4px;
        }
        .hero::after {
            content: ""; position: absolute; z-index: -1; inset: 0 0 0 auto;
            width: 43%; opacity: .38; pointer-events: none;
            background-image: linear-gradient(90deg, #0c6577 0%, rgba(12,101,119,.15) 58%, transparent 100%), __BACKGROUND_IMAGE__;
            background-repeat: no-repeat; background-size: cover, contain;
            background-position: center, right center;
        }
        .hero::before, .page-hero::before {
            content: ""; position: absolute; z-index: -1; right: -2%; bottom: 19px;
            width: 38%; height: 42px; border-top: 3px solid #f18b40;
            border-radius: 52% 48% 0 0; transform: rotate(-8deg);
            filter: drop-shadow(0 0 8px rgba(255,170,92,.48));
        }
        .hero h1 .hero-orange { color: #ff9a4f; }
        .page-hero {
            margin: .35rem 0 1.2rem; padding: 1.05rem 1.35rem 1.1rem;
            border-left: 4px solid var(--orange); border-bottom: 1px solid rgba(190,215,222,.8);
            border-radius: 0 16px 16px 0;
            background:linear-gradient(100deg,rgba(255,255,255,.82),rgba(255,247,240,.48));
            box-shadow:0 8px 22px rgba(25,64,88,.055);
        }
        .page-hero::before, .page-hero::after { display: none; }
        .page-hero h1 { color: var(--ink); margin: 0; font-size: clamp(1.85rem,2.7vw,2.45rem); }
        .page-hero p { color: var(--ink-soft); margin: .35rem 0 0; max-width: 900px; font-size: .98rem; }
        .page-kicker { display: none; }
        .section-kicker { color: var(--orange-deep); font-size: .76rem; font-weight: 750; letter-spacing: .12em; text-transform: uppercase; }
        .soft-note { background: linear-gradient(100deg, rgba(225,245,246,.9), rgba(255,241,229,.86)); border: 1px solid #d9e9e9; border-radius: 12px; padding: .7rem .9rem; }
        .empty-state { text-align: center; padding: 2.5rem 1.25rem; border: 1px dashed #e8cdb8; border-radius: 19px; background: linear-gradient(120deg,rgba(255,255,255,.96),rgba(255,241,229,.88)); box-shadow:0 10px 26px rgba(25,64,88,.055); }
        .empty-state h3 { margin: .25rem 0 .4rem; color: var(--ink); }
        .empty-state p { color: var(--ink-soft); margin-bottom: .8rem; }
        .step-number { display:inline-grid; place-items:center; width:1.7rem; height:1.7rem; border-radius:50%; background:#fff0e3; color:#b85d21; font-weight:800; margin-right:.45rem; }
        .method-card-title {
            display:flex !important; flex-direction:row !important; flex-wrap:nowrap !important;
            align-items:center !important; justify-content:flex-start !important;
            gap:.75rem !important; width:100% !important; min-width:0 !important;
            margin:0 0 1rem !important; padding:.1rem 0 .8rem !important;
            border-bottom:1px solid rgba(211,229,233,.9);
        }
        .method-card-title .method-icon {
            display:flex !important; flex:0 0 2.55rem !important;
            align-items:center !important; justify-content:center !important;
            width:2.55rem !important; min-width:2.55rem !important; height:2.55rem !important;
            border-radius:14px; background:linear-gradient(145deg,#fff0e3,#ffe5d0);
            color:#b85d21; font-size:1.15rem; line-height:1;
        }
        .method-card-title .method-heading {
            display:block !important; flex:1 1 auto !important;
            width:auto !important; min-width:0 !important; max-width:none !important;
            margin:0 !important; color:#10475c; font-size:1.22rem !important;
            font-weight:750 !important; line-height:1.25 !important;
            white-space:normal !important; overflow-wrap:normal !important;
            word-break:keep-all !important;
        }
        .st-key-method_input_card [data-testid="stVerticalBlockBorderWrapper"] {
            background:linear-gradient(145deg,#fff 0%,#f1fafb 100%) !important;
            border-top:3px solid #72c6c8 !important;
        }
        .st-key-method_stage_card [data-testid="stVerticalBlockBorderWrapper"] {
            background:linear-gradient(145deg,#fff 0%,#fff7f0 100%) !important;
            border-top:3px solid #e99a62 !important;
        }
        .st-key-method_process_card [data-testid="stVerticalBlockBorderWrapper"] {
            background:linear-gradient(145deg,#fff 0%,#fff8f2 100%) !important;
            border-top:3px solid #e99a62 !important;
        }
        .st-key-method_reading_card [data-testid="stVerticalBlockBorderWrapper"] {
            background:linear-gradient(110deg,rgba(255,255,255,.97),rgba(255,247,240,.9)) !important;
        }
        .st-key-method_input_card, .st-key-method_stage_card, .st-key-method_process_card,
        .st-key-method_reading_card { min-width:0; }
        .st-key-method_input_card [data-testid="stCode"] { margin-top:.7rem; }
        .st-key-method_input_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-method_stage_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-method_process_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-method_reading_card [data-testid="stVerticalBlockBorderWrapper"] {
            padding: .35rem !important;
        }
        .method-copy { line-height:1.65; }
        .step-number { flex:0 0 1.7rem; }
        .method-copy { color:#496779; font-size:.93rem; }
        .result-stat-label { color:var(--ink-soft); font-size:.82rem; }


        /* Home page components */
        .home-card-title { display:flex; align-items:center; gap:.8rem; margin-bottom:.35rem; }
        .home-card-title h3 { margin: 0; color: #10475c; font-size: 1.3rem; }
        .home-icon {
            display:grid; place-items:center; width:3.1rem; height:3.1rem;
            border-radius:13px; background:#fff0e3; color:#bd5d1c; font-size:1.5rem;
        }
        .home-icon-teal { background:#def3f4; color:#087f89; }
        .st-key-home_about_card, .st-key-home_why_card,
        .st-key-home_project_card, .st-key-home_model_card { min-height: 185px; }
        .st-key-home_about_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-home_why_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-home_project_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-home_model_card [data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px !important; }
        .st-key-home_about_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-home_why_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-home_project_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-home_model_card [data-testid="stVerticalBlockBorderWrapper"] { min-height: 185px; height: 100%; }
        .st-key-home_about_card [data-testid="stVerticalBlockBorderWrapper"] { border-left: 3px solid #e99a62; }
        .st-key-home_why_card [data-testid="stVerticalBlockBorderWrapper"] { border-left: 3px solid #70c6c8; }
        .st-key-home_project_card [data-testid="stVerticalBlockBorderWrapper"] { border-top: 2px solid #e99a62; }
        .st-key-home_model_card [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(145deg, rgba(255,255,255,.94), rgba(255,242,230,.9));
            border-top: 2px solid #e99a62;
        }

        /* Page-specific accents */
        .st-key-predict_upload_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-timeline_filter_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-method_input_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-method_stage_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-method_process_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-method_reading_card [data-testid="stVerticalBlockBorderWrapper"],
        .st-key-results_table_card [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255,255,255,.92); border-top: 2px solid #e99a62;
        }
        .st-key-timeline_chart_card [data-testid="stVerticalBlockBorderWrapper"] { padding:.35rem; }
        .st-key-predict_upload_card [data-testid="stVerticalBlockBorderWrapper"] { border-top:2px solid #74c6c8; }
        div.stDownloadButton > button { background:linear-gradient(110deg,#fff8f1,#fff); color:#a94f18; border:1px solid #efc8a8; }
        div.stDownloadButton > button:hover { background:#fff0e3 !important; color:#984b19 !important; }
        [data-testid="stMultiSelect"] [data-baseweb="tag"] { background:#e0f2f3 !important; color:#087f89 !important; border-color:#b8dfe1 !important; }
        [data-testid="stMetric"] { min-height:100px; }
        [data-testid="stMetricValue"] { font-size:1.8rem; }
        .st-key-home_model_card [data-testid="stMetric"] { background:linear-gradient(110deg,#fff4e9,#fff); }

        /* Refinement pass: motion, focus states and compact data surfaces */
        @keyframes bpdRiseIn {
            from { opacity: 0; transform: translateY(9px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .hero, .page-hero { animation: bpdRiseIn .42s ease-out both; }
        .main .block-container > div { animation: bpdRiseIn .28s ease-out both; }
        [data-testid="stVerticalBlockBorderWrapper"] { animation: bpdRiseIn .36s ease-out both; }
        [data-testid="stMetric"] { transition: transform .18s ease, box-shadow .18s ease; }
        [data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(25,64,88,.09); }
        .score-note { color:#78909d; font-size:.78rem; margin-top:-.45rem; }
        .overview-chip { display:inline-block; padding:.28rem .65rem; margin:.15rem .25rem .15rem 0;
            border-radius:999px; background:#e6f4f4; color:#087f89; font-size:.76rem; font-weight:700; }
        .research-note { border-left:3px solid #e99a62; background:rgba(255,247,240,.88);
            border-radius:0 12px 12px 0; padding:.75rem .95rem; color:#526e7e; }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 15px 32px rgba(25,64,88,.10) !important;
        }
        div.stButton > button:focus-visible, div.stDownloadButton > button:focus-visible,
        [data-baseweb="select"] > div:focus-within {
            outline: 3px solid rgba(233,154,98,.38) !important;
            outline-offset: 2px;
        }
        .stage-flow {
            display:grid; grid-template-columns:minmax(0,1fr) 34px minmax(0,1fr);
            align-items:stretch; gap:.65rem; margin:.25rem 0 1.35rem;
        }
        .stage-flow-card {
            border:1px solid #d7e7eb; border-radius:15px; padding:.9rem 1rem;
            background:linear-gradient(145deg,rgba(255,255,255,.96),rgba(239,249,250,.92));
        }
        .stage-flow-card.stage-two {
            background:linear-gradient(145deg,rgba(255,255,255,.97),rgba(255,243,232,.92));
            border-color:#ecd9c9;
        }
        .stage-flow-kicker { color:#8197a2; font-size:.72rem; font-weight:750; letter-spacing:.1em; text-transform:uppercase; }
        .stage-flow-title { color:#174b62; font-weight:800; font-size:1.02rem; margin:.2rem 0; }
        .stage-flow-detail { color:#597384; font-size:.84rem; line-height:1.45; }
        .stage-flow-arrow { display:grid; place-items:center; color:#d17a3c; font-size:1.45rem; font-weight:800; }
        .result-count { display:inline-block; border-radius:999px; padding:.16rem .55rem; margin-top:.45rem;
            background:#e2f3f3; color:#087f89; font-size:.75rem; font-weight:750; }
        .st-key-results_table_card [data-testid="stVerticalBlockBorderWrapper"] { border-top:3px solid #75c5c8; }
        .st-key-timeline_chart_card [data-testid="stVerticalBlockBorderWrapper"] { background:rgba(255,255,255,.94); }
        [data-testid="stDataFrame"] { box-shadow:0 7px 18px rgba(25,64,88,.035); }
        @media (prefers-reduced-motion: reduce) {
            *, *::before, *::after { animation-duration:.01ms !important; transition-duration:.01ms !important; }
        }
        @media (max-width: 620px) {
            .stage-flow { grid-template-columns:1fr; }
            .stage-flow-arrow { transform:rotate(90deg); min-height:1.2rem; }
            .hero h1 { font-size:2rem; }
            .main .block-container { padding-left:.75rem; padding-right:.75rem; }
        }

        @media (max-width: 900px) {
            .method-card-title h3 { font-size:1.05rem !important; }
        }
        @media (max-width: 750px) {
            .main .block-container { padding-left: 1rem; padding-right: 1rem; }
            .hero { padding: 1.5rem; border-radius:18px; }
            .hero::after { opacity: .18 !important; width: 70% !important; }
            .page-hero { padding: 1rem 1.1rem; }
            .top-nav-row { gap: .2rem; }
            .method-card-title { grid-template-columns:2.2rem minmax(0,1fr); }
        }
        </style>
        """.replace("__BACKGROUND_IMAGE__", background_image),
        unsafe_allow_html=True,
    )

def render_top_navigation() -> None:
    active_page = st.session_state.get("active_page", "Home")
    columns = st.columns([1.35, .78, .78, .78, 1.22, 1.48])
    with columns[0]:
        st.markdown(
            '<div class="site-brand">BPD <span style="color:#0b8588">/</span> '
            '<span class="brand-orange">Insight</span></div>',
            unsafe_allow_html=True,
        )
    for column, page in zip(columns[1:], PAGES):
        with column:
            if st.button(page, key=f"top_nav_{page}",
                         type="primary" if page == active_page else "secondary",
                         use_container_width=True):
                go_to(page)
    st.markdown("<hr style='border:0;border-top:1px solid #d9e7ed;margin:.35rem 0 1.6rem'>", unsafe_allow_html=True)


def render_page_hero(title: str, description: str) -> None:
    st.markdown(
        '<div class="page-hero">'
        f'<h1>{title}</h1><p>{description}</p></div>',
        unsafe_allow_html=True,
    )


def render_home(artifact) -> None:
    st.markdown(
        '<div class="hero"><div class="hero-kicker">Neonatal respiratory research</div>'
        '<h1>Understanding <span class="hero-orange">BPD</span> through data</h1>'
        '<p>Explore bronchopulmonary dysplasia, follow gene-expression patterns across visits, '
        'and try a two-stage research classifier.</p></div>',
        unsafe_allow_html=True,
    )

    about, why = st.columns([1.15, 1])
    with about:
        with st.container(border=True, key="home_about_card"):
            st.markdown(
                '<div class="home-card-title"><span class="home-icon">🫁</span>'
                '<h3>What is BPD?</h3></div>', unsafe_allow_html=True
            )
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
        with st.container(border=True, key="home_why_card"):
            st.markdown(
                '<div class="home-card-title"><span class="home-icon home-icon-teal">⚕</span>'
                '<h3>Why study it?</h3></div>', unsafe_allow_html=True
            )
            st.write(
                "BPD can have a complex course, and respiratory needs may continue after the "
                "newborn period. Studying clinical and molecular patterns can help researchers "
                "describe how the condition varies across infants and time."
            )
            st.caption("This project explores patterns in a public gene-expression cohort; it does not guide care.")

    st.subheader("About this project")
    project, model = st.columns([1.25, 1])
    with project:
        with st.container(border=True, key="home_project_card"):
            st.write(
                "The classifier combines three clinical measurements with gene-expression "
                "features from the 24-33 day timepoint. Stage 1 separates No BPD from Any BPD; "
                "Stage 2 predicts severity for samples routed to Any BPD."
            )
            if st.button("Explore the model", type="secondary"):
                go_to("Model & Methodology")
    with model:
        with st.container(border=True, key="home_model_card"):
            st.metric("Model inputs", f"{len(artifact['features']):,}")
            st.caption(
                f"{artifact.get('candidate_gene_count', 0):,} gene symbols + "
                f"{len(CLINICAL_FEATURES)} clinical features | Timepoint 3"
            )
            if st.button("Try the prediction demo"):
                go_to("Predict")

    st.subheader("Research snapshot")
    snapshot = st.columns(3)
    for column, value, label in zip(
        snapshot,
        ["GSE32472", "97 infants", "24–33 days"],
        ["Public gene-expression cohort", "Model development cohort", "Deployed model timepoint"],
    ):
        with column:
            with st.container(border=True):
                st.markdown(f"<div style='font-size:1.25rem;font-weight:800;color:#174b62'>{value}</div>", unsafe_allow_html=True)
                st.caption(label)
    with st.expander("Project scope and responsible use"):
        st.markdown(
            "The deployed classifier uses the third observed timepoint (24–33 days), "
            "three clinical measurements, and mapped gene-symbol expression features. "
            "The Timeline Explorer separately presents cohort-level expression summaries "
            "across all three observed visits."
        )
        st.markdown(
            '<div class="research-note"><b>Research prototype only.</b> This application is '
            'for educational and research exploration. It has not been clinically validated '
            'and must not be used to diagnose infants or guide treatment.</div>',
            unsafe_allow_html=True,
        )


def render_predict(artifact) -> None:
    render_page_hero(
        "Predict BPD severity",
        "Upload one sample per row. Required inputs are three clinical fields and "
        f"{artifact.get('candidate_gene_count', 0):,} mapped gene-symbol measurements."
    )

    template_ready = SYNTHETIC_TEMPLATE_PATH.exists() and not is_git_lfs_pointer(SYNTHETIC_TEMPLATE_PATH)
    if template_ready:
        st.download_button(
            "↓  Download synthetic sample template",
            SYNTHETIC_TEMPLATE_PATH.read_bytes(),
            "synthetic_bpd_input_examples.csv",
            "text/csv",
            help="A fabricated template for checking the required input format.",
        )
        st.markdown(
            '<div class="soft-note">ⓘ &nbsp; This template contains fabricated demonstration values. '
            'Use it to understand the required CSV format; predictions on synthetic rows are not meaningful model results.</div>',
            unsafe_allow_html=True,
        )
    elif is_git_lfs_pointer(SYNTHETIC_TEMPLATE_PATH):
        st.warning("The synthetic CSV is stored with Git LFS. Run `git lfs pull` from the repository root to download it.")
    else:
        st.warning("Synthetic template not found. Run `python src/create_synthetic_template.py`.")

    with st.container(border=True, key="predict_upload_card"):
        st.markdown('<div class="section-kicker">Prediction input</div>', unsafe_allow_html=True)
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

    if st.button("▶  Run prediction", type="primary", disabled=input_frame.empty):
        with st.spinner("Running the two-stage model..."):
            try:
                output = predict(artifact, input_frame)
            except Exception as error:
                st.error(str(error))
                return
        st.session_state["prediction_results"] = output
        st.session_state["prediction_source_name"] = uploaded.name

    saved_results = st.session_state.get("prediction_results")
    if saved_results is not None:
        st.success(
            f"Generated predictions for {len(saved_results):,} sample(s). "
            "Open Results to review them."
        )
        if st.button("View results", key="view_results_button"):
            go_to("Results")


def render_results() -> None:
    render_page_hero("Prediction results", "Review sample-level predictions and export the results table.")
    results = st.session_state.get("prediction_results")
    if results is None:
        st.markdown(
            '<div class="empty-state"><div style="font-size:2rem">📊</div>'
            '<h3>Your results will appear here</h3>'
            '<p>Upload a prepared CSV on the Predict page to run the two-stage classifier and review sample-level outputs.</p></div>',
            unsafe_allow_html=True,
        )
        if st.button("Go to prediction", type="primary"):
            go_to("Predict")
        return

    st.caption(f"Source file: {st.session_state.get('prediction_source_name', 'Uploaded CSV')}")
    counts = results["predicted_label"].value_counts().reindex(list(LABELS.values()), fill_value=0)
    metrics = st.columns(4)
    for column, (label, count) in zip(metrics, counts.items()):
        with column:
            st.metric(label, int(count))

    no_bpd_count = int((results["stage1_decision"] == "No BPD").sum()) if "stage1_decision" in results else int(counts.get("No BPD", 0))
    routed_count = len(results) - no_bpd_count
    st.markdown(
        '<div class="stage-flow">'
        '<div class="stage-flow-card">'
        '<div class="stage-flow-kicker">Stage 1 · Screening</div>'
        '<div class="stage-flow-title">No BPD vs Any BPD</div>'
        '<div class="stage-flow-detail">The first model separates samples below the threshold from samples routed for severity classification.</div>'
        f'<span class="result-count">{no_bpd_count:,} labelled No BPD</span>'
        '</div><div class="stage-flow-arrow">→</div>'
        '<div class="stage-flow-card stage-two">'
        '<div class="stage-flow-kicker">Stage 2 · Severity</div>'
        '<div class="stage-flow-title">Mild · Moderate · Severe</div>'
        '<div class="stage-flow-detail">Only samples routed from Stage 1 receive a severity estimate from the second model.</div>'
        f'<span class="result-count">{routed_count:,} routed to Stage 2</span>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    st.subheader("Prediction score breakdown")
    sample_key = next((key for key in ["sample_id", "geo_accession", "title", "sample_number"] if key in results.columns), None)
    if sample_key and len(results):
        sample_options = results[sample_key].astype(str).tolist()
        selected_sample = st.selectbox("Choose a sample to inspect", sample_options, key="result_score_sample")
        selected_row = results.loc[results[sample_key].astype(str) == selected_sample].iloc[0]
        score_left, score_right = st.columns(2, gap="large")
        with score_left:
            with st.container(border=True):
                st.markdown("**Stage 1 · No BPD vs Any BPD**")
                stage1_score = float(selected_row.get("bpd_probability", 0.0))
                st.progress(min(max(stage1_score, 0.0), 1.0))
                st.write(f"Any-BPD model score: **{stage1_score:.1%}**")
                threshold = float(selected_row.get("stage1_threshold", DEFAULT_THRESHOLD))
                st.caption(f"Routing threshold: {threshold:.0%}. This is a model score, not a calibrated clinical risk.")
        with score_right:
            with st.container(border=True):
                st.markdown("**Stage 2 · Conditional severity scores**")
                stage2_values = []
                for label, key in [("Mild BPD", "stage2_mild_score"), ("Moderate BPD", "stage2_moderate_score"), ("Severe BPD", "stage2_severe_score")]:
                    value = selected_row.get(key, np.nan)
                    if pd.notna(value):
                        stage2_values.append({"Severity": label, "Model score": float(value)})
                if stage2_values:
                    score_frame = pd.DataFrame(stage2_values)
                    score_chart = (
                        alt.Chart(score_frame)
                        .mark_bar(cornerRadiusEnd=7, height=22)
                        .encode(
                            x=alt.X("Model score:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", title=None)),
                            y=alt.Y("Severity:N", sort=["Mild BPD", "Moderate BPD", "Severe BPD"], axis=alt.Axis(title=None)),
                            color=alt.Color("Severity:N", scale=alt.Scale(domain=["Mild BPD", "Moderate BPD", "Severe BPD"], range=["#2878b8", "#e99a62", "#d45a52"]), legend=None),
                            tooltip=[alt.Tooltip("Severity:N"), alt.Tooltip("Model score:Q", format=".1%")],
                        ).properties(height=125)
                    )
                    st.altair_chart(score_chart, use_container_width=True)
                else:
                    st.info("This sample was not routed to Stage 2, so no severity scores are available.")

    st.subheader("Per-sample results")
    display_columns = [
        column for column in [
            "sample_id", "geo_accession", "title", "sample_number",
            "predicted_label", "stage1_decision", "bpd_probability",
            "stage2_mild_score", "stage2_moderate_score", "stage2_severe_score",
        ] if column in results.columns
    ]
    with st.container(border=True, key="results_table_card"):
        st.dataframe(results[display_columns], use_container_width=True, hide_index=True)
    st.caption(
        "Stage 2 scores are conditional on being routed to Any BPD. "
        "The Stage 1 score is a model score and has not been presented as a calibrated clinical risk."
    )
    st.download_button(
        "↓  Download results CSV",
        results.to_csv(index=False),
        "bpd_predictions.csv",
        "text/csv",
        type="primary",
    )


def render_methodology(artifact) -> None:
    render_page_hero(
        "Model & Methodology",
        "The deployed model uses a timepoint-3 sample view and a two-stage classification hierarchy."
    )

    inputs_col, process_col = st.columns([1, 1.08], gap="large")
    with inputs_col:
        with st.container(border=True, key="method_input_card"):
            st.markdown('<div class="method-card-title"><span class="method-icon">▤</span><span class="method-heading">Inputs</span></div>', unsafe_allow_html=True)
            st.markdown(
                f"Each sample has **{len(artifact['features']):,} input features**: "
                f"{artifact.get('candidate_gene_count', 0):,} gene-symbol expression values and "
                f"{len(CLINICAL_FEATURES)} clinical measurements."
            )
            st.code(", ".join(CLINICAL_FEATURES), language=None)

        with st.container(border=True, key="method_stage_card"):
            st.markdown('<div class="method-card-title"><span class="method-icon">⌘</span><span class="method-heading">Prediction flow</span></div>', unsafe_allow_html=True)
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

    with process_col:
        with st.container(border=True, key="method_process_card"):
            st.markdown('<div class="method-card-title"><span class="method-icon">⚙</span><span class="method-heading">Feature processing</span></div>', unsafe_allow_html=True)
            steps = [
                "Map microarray probes to gene symbols and average probes assigned to the same symbol.",
                "Impute missing values and scale clinical and gene-expression features.",
                "Each stage selects the 5,000 highest-variance genes, then uses an ANOVA F-test to retain its stage-specific gene set.",
                "Apply the fitted transformations and logistic-regression models to new rows.",
            ]
            for index, step in enumerate(steps, 1):
                st.markdown(f'<div style="display:flex;align-items:flex-start;margin:.55rem 0"><span class="step-number">{index}</span><span>{step}</span></div>', unsafe_allow_html=True)

            stage1_genes = artifact.get("stage1_selected_genes", [])
            stage2_genes = artifact.get("stage2_selected_genes", [])
            with st.expander(f"Stage 1 selected genes ({len(stage1_genes)})"):
                st.write(", ".join(stage1_genes) if stage1_genes else "Gene list unavailable.")
            with st.expander(f"Stage 2 selected genes ({len(stage2_genes)})"):
                st.write(", ".join(stage2_genes) if stage2_genes else "Gene list unavailable.")

    st.subheader("Reading a model result")
    with st.container(border=True, key="method_reading_card"):
        result_steps = st.columns(3)
        explanations = [
            ("1 · Initial screen", "Stage 1 assigns an Any-BPD score. A score at or above the saved threshold sends the sample to Stage 2."),
            ("2 · Severity estimate", "For samples sent forward, Stage 2 compares the Mild, Moderate, and Severe BPD scores and selects the highest scoring class."),
            ("3 · Combined label", "Samples below the Stage 1 threshold are labeled No BPD. Samples routed to Stage 2 receive its predicted severity label."),
        ]
        for column, (heading, explanation) in zip(result_steps, explanations):
            with column:
                st.markdown(f"**{heading}**")
                st.write(explanation)


def render_timeline() -> None:
    render_page_hero(
        "Timeline Explorer",
        "Explore aggregate gene-expression trends across the cohort's three observed timepoints, "
        "grouped by recorded BPD category."
    )
    if not TIMELINE_PATH.exists():
        st.error(f"Longitudinal dataset was not found at {TIMELINE_PATH}")
        return
    if is_git_lfs_pointer(TIMELINE_PATH):
        st.error(
            "The timeline CSV is a Git LFS pointer, not the dataset itself. "
            "Run `git lfs install` and `git lfs pull` from the repository root, then restart the app."
        )
        return

    frame, genes = load_timeline_data()
    with st.container(border=True, key="timeline_filter_card"):
        filter_gene, filter_groups, filter_stat, filter_style = st.columns([1.0, 1.55, .85, .8])
        with filter_gene:
            gene = st.selectbox("Gene symbol", genes)
        all_groups = list(LABELS.values())
        with filter_groups:
            groups = st.multiselect("BPD groups", all_groups, default=all_groups)
        with filter_stat:
            statistic = st.radio("Expression summary", ["Median", "Mean"], horizontal=True)
        with filter_style:
            chart_style = st.selectbox("Chart style", ["Line", "Area"])
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
    st.subheader(f"{statistic} expression for {gene}")
    palette = alt.Scale(
        domain=["No BPD", "Mild BPD", "Moderate BPD", "Severe BPD"],
        range=["#0b8588", "#2878b8", "#e99a62", "#d45a52"],
    )
    base_chart = alt.Chart(summary)
    if chart_style == "Area":
        timeline_mark = base_chart.mark_area(opacity=.13, line={"strokeWidth": 2.5}).encode(
            x=alt.X("timepoint_label:N", sort=ordered_timepoints, title="Observed timepoint"),
            y=alt.Y("expression:Q", title="Gene expression", scale=alt.Scale(zero=False)),
            color=alt.Color("bpd_label:N", scale=palette, title="BPD group"),
        )
        timeline_points = base_chart.mark_line(point=alt.OverlayMarkDef(filled=True, size=95), strokeWidth=2.8)
        base_chart = alt.Chart(summary)
    tooltip = [
        alt.Tooltip("bpd_label:N", title="BPD group"),
        alt.Tooltip("timepoint_label:N", title="Timepoint"),
        alt.Tooltip("expression:Q", title=f"{statistic} expression", format=".4f"),
        alt.Tooltip("samples:Q", title="Infants"),
    ]
    encodings = dict(
        x=alt.X("timepoint_label:N", sort=ordered_timepoints, title="Observed timepoint",
                axis=alt.Axis(labelAngle=0, labelPadding=10, titlePadding=15, tickSize=0)),
        y=alt.Y("expression:Q", title="Gene expression", scale=alt.Scale(zero=False),
                axis=alt.Axis(grid=True, gridColor="#e8f0f2", gridOpacity=1, tickCount=5)),
        color=alt.Color("bpd_label:N", scale=palette, title="BPD group"),
        tooltip=tooltip,
    )
    if chart_style == "Area":
        timeline_chart = base_chart.mark_area(opacity=.13, line={"strokeWidth": 2.5}).encode(**encodings)
        timeline_chart = timeline_chart + base_chart.mark_line(point=alt.OverlayMarkDef(filled=True, size=95), strokeWidth=2.8).encode(**encodings)
    else:
        timeline_chart = base_chart.mark_line(
            point=alt.OverlayMarkDef(filled=True, size=105, strokeWidth=2), strokeWidth=3.2
        ).encode(**encodings)
    timeline_chart = (
        timeline_chart.properties(height=390, padding={"left": 12, "right": 18, "top": 12, "bottom": 8})
        .configure_view(strokeOpacity=0)
        .configure_legend(orient="top", direction="horizontal", title=None, symbolSize=100, labelFontSize=12)
        .configure_axis(labelColor="#496779", titleColor="#34566b", domainColor="#cbdde3")
        .interactive()
    )
    with st.container(border=True, key="timeline_chart_card"):
        st.altair_chart(timeline_chart, use_container_width=True)
    st.caption("Each point summarizes available samples at that visit; sample counts are shown below.")
    with st.container(border=True, key="timeline_summary_card"):
        st.dataframe(
            summary[["bpd_label", "timepoint_label", "expression", "samples"]],
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.set_page_config(page_title="BPD Insight | Research Classifier", page_icon="BPD", layout="wide")
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
