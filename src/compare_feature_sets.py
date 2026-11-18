"""Leakage-aware comparison of clinical, gene, and combined BPD models.

The probe-to-symbol mapping is prepared once, while both unsupervised variance
filtering and supervised feature selection are fitted inside training folds.

Run from the repository root:
    python src/compare_feature_sets.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bpd_feature_utils import variance_scores
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

METADATA = ROOT / "data/processed/GSE32472_full_metadata.csv"
RAW_EXPRESSION = ROOT / "data/processed/GSE32472_expression_new.csv"
PLATFORM = ROOT / "data/raw/GPL6244.txt"
OUTPUT = ROOT / "src/outputs/feature_set_comparison"
SEED = 42
CLINICAL = ["days of life", "gestational age (weeks)", "birth weight (grams)"]
LABELS = [0, 1, 2, 3]
THRESHOLDS = [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75]


def load_data(view: str = "timepoint_3"):
    metadata = pd.read_csv(METADATA)
    raw_expression = pd.read_csv(RAW_EXPRESSION, index_col=0)
    raw_expression.index = raw_expression.index.astype(str).str.strip()
    table_start = None
    with PLATFORM.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.strip() == "!platform_table_begin":
                table_start = line_number + 1
                break
    if table_start is None:
        raise ValueError(f"Platform table not found in {PLATFORM}")
    annotation = pd.read_csv(
        PLATFORM, sep="\t", skiprows=table_start, usecols=["ID", "gene_assignment"],
        dtype={"ID": str}, low_memory=False,
    )
    annotation["ID"] = annotation["ID"].astype(str).str.strip()
    annotation["gene_symbol"] = annotation["gene_assignment"].map(
        lambda value: str(value).split("//")[1].strip()
        if pd.notna(value) and len(str(value).split("//")) > 1 else None
    )
    annotation = annotation.dropna(subset=["gene_symbol"])
    annotation = annotation.loc[annotation.gene_symbol.ne("---"), ["ID", "gene_symbol"]]
    expression = raw_expression.join(annotation.set_index("ID"), how="inner")
    expression_by_sample = expression.groupby("gene_symbol").mean(numeric_only=True).T
    expression_by_sample = expression_by_sample.loc[:, expression_by_sample.notna().any()]
    metadata.columns = metadata.columns.str.strip()
    metadata["geo_accession"] = metadata["geo_accession"].astype(str).str.strip()
    metadata["infant_id"] = metadata["title"].str.extract(r"rep\s*(\d+)", flags=re.I)[0]
    source = metadata["source_name_ch1"].astype(str)
    metadata["timepoint_code"] = source.str.extract(r"(4-6|12-17|24-33)")[0].map(
        {"4-6": 1, "12-17": 2, "24-33": 3}
    )
    target_col = "bronchopulmonary dysplasia (bpd) group"
    metadata["bpd_group"] = pd.to_numeric(metadata[target_col], errors="coerce")
    for col in CLINICAL:
        metadata[col] = pd.to_numeric(metadata[col], errors="coerce")
    joined = metadata.merge(expression_by_sample, left_on="geo_accession", right_index=True, how="inner")
    joined = joined.dropna(subset=["infant_id", "timepoint_code", "bpd_group"])
    repeated_infants = joined.groupby("infant_id").size()
    joined = joined.loc[joined.infant_id.isin(repeated_infants[repeated_infants >= 2].index)]
    if view.startswith("timepoint_"):
        timepoint = int(view.rsplit("_", 1)[1])
        data = joined.loc[joined.timepoint_code == timepoint].copy()
    elif view == "latest":
        data = (joined.sort_values(["infant_id", "timepoint_code"])
                .drop_duplicates("infant_id", keep="last").copy())
    elif view == "all_rows":
        data = joined.copy()
    else:
        raise ValueError("view must be timepoint_1, timepoint_2, timepoint_3, latest, or all_rows")
    data = data.dropna(subset=["infant_id", "bpd_group"]).copy()
    data["infant_id"] = data["infant_id"].astype(str)
    data["bpd_group"] = data["bpd_group"].astype(int)
    genes = list(expression_by_sample.columns)
    return data, genes


def make_pipeline(kind: str, genes: list[str], target: pd.Series):
    clinical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    gene_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("variance_select", SelectKBest(score_func=variance_scores, k=5000)),
        ("scale", StandardScaler()),
        ("select", SelectKBest(score_func=f_classif)),
    ])
    if kind == "clinical_only":
        features = CLINICAL
        prep = clinical_pipe
        select_key = None
    elif kind == "genes_only":
        features = genes
        prep = gene_pipe
        select_key = "preprocess__select__k"
    else:
        features = CLINICAL + genes
        prep = ColumnTransformer([
            ("clinical", clinical_pipe, CLINICAL),
            ("genes", gene_pipe, genes),
        ])
        select_key = "preprocess__genes__select__k"
    classifier = LogisticRegression(max_iter=5000, random_state=SEED)
    if target.nunique() == 2:
        classifier = OneVsRestClassifier(classifier)
        c_key = "classifier__estimator__C"
        weight_key = "classifier__estimator__class_weight"
    else:
        c_key = "classifier__C"
        weight_key = "classifier__class_weight"
    model = Pipeline([("preprocess", prep), ("classifier", classifier)])
    params = {c_key: [0.1, 1.0], weight_key: ["balanced"]}
    if select_key:
        params[select_key] = [10, 25, 50]
    return features, model, params


def tune(kind, genes, X, y, groups):
    features, model, params = make_pipeline(kind, genes, y)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(model, params, scoring="f1_macro", cv=cv, n_jobs=-1, refit=True)
    search.fit(X[features], y, groups=groups)
    return search


def tune_ordinal(kind, genes, X, y, groups):
    features, model, params = make_pipeline(kind, genes, pd.Series([0, 1, 2, 3]))
    model.set_params(classifier=Ridge())
    params = {key: value for key, value in params.items()
              if not key.endswith("__C") and not key.endswith("__class_weight")}
    params["classifier__alpha"] = [0.1, 1.0, 10.0, 100.0]
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = GridSearchCV(model, params, scoring="neg_mean_absolute_error",
                          cv=cv, n_jobs=-1, refit=True)
    search.fit(X[features], y, groups=groups)
    return search


def selected_gene_names(estimator, kind, genes):
    prep = estimator.named_steps["preprocess"]
    gene_steps = (prep.named_transformers_["genes"] if kind == "combined" else prep)
    variance_mask = gene_steps.named_steps["variance_select"].get_support()
    genes_after_variance = np.asarray(genes)[variance_mask]
    selection_mask = gene_steps.named_steps["select"].get_support()
    return genes_after_variance[selection_mask].tolist()


def choose_threshold(stage1, stage2, X, y, groups, features):
    binary = (y > 0).astype(int)
    cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=SEED)
    probs, severity_preds, truth = [], [], []
    for train, valid in cv.split(X, binary, groups):
        xt, xv = X.iloc[train], X.iloc[valid]
        yt = y.iloc[train]
        one = clone(stage1).fit(xt[features], (yt > 0).astype(int))
        mask = yt > 0
        two = clone(stage2).fit(xt.loc[mask, features], yt.loc[mask])
        class_one = list(one.classes_).index(1)
        probs.extend(one.predict_proba(xv[features])[:, class_one])
        severity_preds.extend(two.predict(xv[features]))
        truth.extend(y.iloc[valid])
    scores = {
        t: f1_score(truth, [p if pr >= t else 0 for p, pr in zip(severity_preds, probs)],
                    labels=LABELS, average="macro", zero_division=0)
        for t in THRESHOLDS
    }
    return max(THRESHOLDS, key=lambda t: (scores[t], -abs(t - 0.5)))


def evaluate(kind, data, genes):
    clinical = data[CLINICAL]
    gene_frame = data[genes].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([clinical, gene_frame], axis=1)
    y = data["bpd_group"]
    groups = data["infant_id"]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    binary = (y > 0).astype(int)
    all_true, all_pred, fold_results, selections = [], [], [], []
    stage1_true, stage1_pred, oracle_true, oracle_pred = [], [], [], []
    for fold, (train, test) in enumerate(outer.split(X, binary, groups), 1):
        xt, xv = X.iloc[train], X.iloc[test]
        yt, gt = y.iloc[train], groups.iloc[train]
        s1 = tune(kind, genes, xt, (yt > 0).astype(int), gt)
        bpd = yt > 0
        s2 = tune(kind, genes, xt.loc[bpd], yt.loc[bpd], gt.loc[bpd])
        features = make_pipeline(kind, genes, yt)[0]
        if kind != "clinical_only":
            for stage, search in (("stage1", s1), ("stage2", s2)):
                for gene in selected_gene_names(search.best_estimator_, kind, genes):
                    selections.append({"feature_set": kind, "fold": fold,
                                       "stage": stage, "gene": gene})
        threshold = choose_threshold(s1.best_estimator_, s2.best_estimator_, xt, yt, gt, features)
        estimator1, estimator2 = s1.best_estimator_, s2.best_estimator_
        class_one = list(estimator1.classes_).index(1)
        probability = estimator1.predict_proba(xv[features])[:, class_one]
        stage1_prediction = (probability >= threshold).astype(int)
        stage1_true.extend((y.iloc[test] > 0).astype(int).tolist())
        stage1_pred.extend(stage1_prediction.tolist())
        prediction = pd.Series(0, index=xv.index, dtype=int)
        route = probability >= threshold
        if route.any():
            prediction.loc[route] = estimator2.predict(xv.loc[route, features])
        true_bpd = y.iloc[test] > 0
        if true_bpd.any():
            oracle_true.extend(y.iloc[test][true_bpd].tolist())
            oracle_pred.extend(estimator2.predict(xv.loc[true_bpd, features]).tolist())
        all_true.extend(y.iloc[test].tolist())
        all_pred.extend(prediction.tolist())
        fold_results.append({
            "feature_set": kind, "fold": fold, "threshold": threshold,
            "accuracy": accuracy_score(y.iloc[test], prediction),
            "macro_f1": f1_score(y.iloc[test], prediction, labels=LABELS,
                                  average="macro", zero_division=0),
            "stage1_best_params": str(s1.best_params_),
            "stage2_best_params": str(s2.best_params_),
        })
    return fold_results, selections, {
        "feature_set": kind,
        "samples": len(all_true),
        "accuracy": accuracy_score(all_true, all_pred),
        "balanced_accuracy": balanced_accuracy_score(all_true, all_pred),
        "macro_f1": f1_score(all_true, all_pred, labels=LABELS,
                             average="macro", zero_division=0),
        "weighted_f1": f1_score(all_true, all_pred, labels=LABELS,
                                average="weighted", zero_division=0),
        "stage1_accuracy": accuracy_score(stage1_true, stage1_pred),
        "stage1_balanced_accuracy": balanced_accuracy_score(stage1_true, stage1_pred),
        "stage1_macro_f1": f1_score(stage1_true, stage1_pred, average="macro", zero_division=0),
        "stage2_oracle_accuracy": accuracy_score(oracle_true, oracle_pred),
        "stage2_oracle_balanced_accuracy": balanced_accuracy_score(oracle_true, oracle_pred),
        "stage2_oracle_macro_f1": f1_score(oracle_true, oracle_pred, labels=[1, 2, 3],
                                            average="macro", zero_division=0),
    }


def evaluate_ordinal(kind, data, genes):
    clinical = data[CLINICAL]
    gene_frame = data[genes].apply(pd.to_numeric, errors="coerce")
    X = pd.concat([clinical, gene_frame], axis=1)
    y = data["bpd_group"]
    groups = data["infant_id"]
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_rows, truths, predictions = [], [], []
    for fold, (train, test) in enumerate(outer.split(X, y, groups), 1):
        xt, xv = X.iloc[train], X.iloc[test]
        yt, gt = y.iloc[train], groups.iloc[train]
        search = tune_ordinal(kind, genes, xt, yt, gt)
        features = make_pipeline(kind, genes, pd.Series([0, 1, 2, 3]))[0]
        pred = np.clip(np.rint(search.predict(xv[features])), 0, 3).astype(int)
        fold_y = y.iloc[test].to_numpy()
        truths.extend(fold_y.tolist())
        predictions.extend(pred.tolist())
        fold_rows.append({
            "feature_set": kind, "fold": fold,
            "accuracy": accuracy_score(fold_y, pred),
            "macro_f1": f1_score(fold_y, pred, labels=LABELS,
                                  average="macro", zero_division=0),
            "best_params": str(search.best_params_),
        })
    return fold_rows, {
        "feature_set": kind, "samples": len(truths),
        "accuracy": accuracy_score(truths, predictions),
        "balanced_accuracy": balanced_accuracy_score(truths, predictions),
        "macro_f1": f1_score(truths, predictions, labels=LABELS,
                             average="macro", zero_division=0),
        "weighted_f1": f1_score(truths, predictions, labels=LABELS,
                                average="weighted", zero_division=0),
    }


def print_stage_metrics(metrics: dict) -> None:
    """Print the combined model's two-stage accuracy and macro-F1 clearly."""
    print("\nCombined model - grouped five-fold cross-validation")
    if "accuracy" in metrics and "macro_f1" in metrics:
        print(
            "End-to-end (No/Mild/Moderate/Severe BPD): "
            f"accuracy {metrics['accuracy']:.1%} | macro-F1 {metrics['macro_f1']:.1%}"
        )
    print(
        "Stage 1 (No BPD vs Any BPD): "
        f"accuracy {metrics['stage1_accuracy']:.1%} | "
        f"macro-F1 {metrics['stage1_macro_f1']:.1%}"
    )
    print(
        "Stage 2 (severity among actual BPD cases): "
        f"accuracy {metrics['stage2_oracle_accuracy']:.1%} | "
        f"macro-F1 {metrics['stage2_oracle_macro_f1']:.1%}"
    )


def print_saved_stage_metrics() -> None:
    """Display saved stage scores without rerunning model training."""
    path = OUTPUT / "combined_stage_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Saved stage metrics not found at {path}. Run the comparison first."
        )
    saved = pd.read_csv(path)
    row = saved.loc[saved["feature_set"] == "combined"]
    if row.empty:
        raise ValueError(f"No combined-model row found in {path}")
    metrics = row.iloc[0].to_dict()
    summary_path = OUTPUT / "summary_metrics.csv"
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        combined = summary.loc[summary["feature_set"] == "combined"]
        if not combined.empty:
            metrics.update(combined.iloc[0].to_dict())
    print_stage_metrics(metrics)


def main():
    data, genes = load_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fold_rows, summaries, selection_rows = [], [], []
    for kind in ["clinical_only", "genes_only", "combined"]:
        print(f"Evaluating {kind} on {len(data)} timepoint-3 infants with {len(genes)} fold-filtered candidate genes...", flush=True)
        folds, selections, summary = evaluate(kind, data, genes)
        fold_rows.extend(folds)
        selection_rows.extend(selections)
        summaries.append(summary)
        pd.DataFrame(summaries).to_csv(OUTPUT / "summary_metrics.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUTPUT / "fold_metrics.csv", index=False)
        print(summary, flush=True)
        if kind == "combined":
            print_stage_metrics(summary)
    stability = pd.DataFrame(selection_rows)
    if not stability.empty:
        stability.groupby(["feature_set", "stage", "gene"]).size().rename(
            "folds_selected"
        ).reset_index().assign(selection_frequency=lambda frame: frame.folds_selected / 5).sort_values(
            ["feature_set", "stage", "folds_selected", "gene"],
            ascending=[True, True, False, True],
        ).to_csv(OUTPUT / "gene_selection_stability.csv", index=False)
    ordinal_rows, ordinal_summaries = [], []
    for kind in ["clinical_only", "combined"]:
        print(f"Evaluating ordinal Ridge model: {kind}...", flush=True)
        folds, summary = evaluate_ordinal(kind, data, genes)
        ordinal_rows.extend(folds)
        ordinal_summaries.append(summary)
        pd.DataFrame(ordinal_rows).to_csv(OUTPUT / "ordinal_fold_metrics.csv", index=False)
        pd.DataFrame(ordinal_summaries).to_csv(OUTPUT / "ordinal_summary_metrics.csv", index=False)
        print(summary, flush=True)
    print(f"Saved comparison to {OUTPUT}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare BPD feature sets and print stage metrics.")
    parser.add_argument(
        "--show-saved",
        action="store_true",
        help="Print saved combined Stage 1 and Stage 2 scores without retraining.",
    )
    arguments = parser.parse_args()
    if arguments.show_saved:
        print_saved_stage_metrics()
    else:
        main()
