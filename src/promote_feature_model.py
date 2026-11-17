"""Train and promote the best grouped-CV feature model for deployment."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compare_feature_sets import (
    choose_threshold,
    load_data,
    make_pipeline,
    selected_gene_names,
    tune,
)


MODEL_DIR = ROOT / "v2/outputs/shareable_two_step_pipeline"
MODEL_PATH = MODEL_DIR / "two_step_bpd_model.joblib"
BACKUP_PATH = MODEL_DIR / "two_step_bpd_model_before_fold_local_gene_filter.joblib"
SUMMARY_PATH = ROOT / "src/outputs/feature_set_comparison/summary_metrics.csv"
LABEL_MAP = {
    0: "No BPD",
    1: "Mild BPD",
    2: "Moderate BPD",
    3: "Severe BPD",
}


def main():
    data, genes = load_data()
    clinical = data[["days of life", "gestational age (weeks)", "birth weight (grams)"]]
    X = pd.concat([clinical, data[genes].apply(pd.to_numeric, errors="coerce")], axis=1)
    y = data["bpd_group"]
    groups = data["infant_id"]
    features, _, _ = make_pipeline("combined", genes, y)

    print(f"Tuning final Stage 1 and Stage 2 models on {len(data)} infants...", flush=True)
    stage1 = tune("combined", genes, X, (y > 0).astype(int), groups)
    bpd = y > 0
    stage2 = tune("combined", genes, X.loc[bpd], y.loc[bpd], groups.loc[bpd])
    threshold = choose_threshold(
        stage1.best_estimator_, stage2.best_estimator_, X, y, groups, features
    )

    cv_summary = pd.read_csv(SUMMARY_PATH)
    combined_metrics = cv_summary.loc[cv_summary.feature_set == "combined"].iloc[0].to_dict()
    artifact = {
        "description": (
            "Fold-local gene-variance-filtered two-step BPD classifier: "
            "No vs Any BPD, then Mild/Moderate/Severe."
        ),
        "view": "timepoint_3",
        "training_infants": int(groups.nunique()),
        "training_rows": int(len(data)),
        "candidate_gene_count": int(len(genes)),
        "variance_gene_count": 5000,
        "stage1_threshold": float(threshold),
        "stage1_no_vs_any_bpd": stage1.best_estimator_,
        "stage2_mild_moderate_severe": stage2.best_estimator_,
        "features": list(features),
        "gene_features": list(genes),
        "label_map": LABEL_MAP,
        "stage1_best_params": stage1.best_params_,
        "stage2_best_params": stage2.best_params_,
        "stage1_selected_genes": selected_gene_names(stage1.best_estimator_, "combined", genes),
        "stage2_selected_genes": selected_gene_names(stage2.best_estimator_, "combined", genes),
        "grouped_cv_metrics": {
            key: float(value) if key not in {"feature_set", "samples"} else value
            for key, value in combined_metrics.items()
        },
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and not BACKUP_PATH.exists():
        shutil.copy2(MODEL_PATH, BACKUP_PATH)
        print(f"Preserved previous model at {BACKUP_PATH}")
    joblib.dump(artifact, MODEL_PATH)
    (MODEL_DIR / "two_step_bpd_model_metadata.json").write_text(
        json.dumps({key: value for key, value in artifact.items()
                    if key not in {"stage1_no_vs_any_bpd", "stage2_mild_moderate_severe",
                                   "gene_features", "features"}},
                   indent=2, default=str),
        encoding="utf-8",
    )
    print(f"Promoted combined model to {MODEL_PATH}")
    print(f"Threshold: {threshold}; Stage 1 params: {stage1.best_params_}")
    print(f"Stage 2 params: {stage2.best_params_}")
    print(f"Grouped CV summary: {combined_metrics}")
    print(f"Expected input columns: {len(features)} ({len(genes)} genes + 3 clinical)")


if __name__ == "__main__":
    main()
