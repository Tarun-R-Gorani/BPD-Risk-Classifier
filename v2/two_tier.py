"""Version 2 of the grouped two-step BPD pipeline.

This version keeps the original data flow and evaluation design, while making
the class-weight and Stage 2 search spaces tunable. It imports the shared
implementation from ``src/two_tier.py`` so the original pipeline is unchanged.

Run from the repository root or from this directory:
    python v2/two_tier.py
"""

from __future__ import annotations

import sys
import importlib.util
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif


SRC = Path(__file__).resolve().parents[1]
BASE_PATH = SRC / "src" / "two_tier.py"
BASE_SPEC = importlib.util.spec_from_file_location("base_two_tier", BASE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError(f"Could not load base pipeline from {BASE_PATH}")
base = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(base)


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = SRC / "data" / "processed" / "gbtm_input_long_format(500_genes).csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "shareable_two_step_pipeline"
REDUCED_GENE_COUNT = 25
THRESHOLDS = [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75]


def load_reduced_data(input_path: Path, view: str):
    """Load the established top-ranked genes and retain only a compact subset."""

    X, y, groups, gene_features = base.load_data(input_path, view)
    selected_genes = gene_features[:REDUCED_GENE_COUNT]
    reduced_features = base.CLINICAL_FEATURES + selected_genes
    return X[reduced_features].copy(), y, groups, selected_genes


def make_stage1_model(gene_features: list[str]) -> tuple[Pipeline, dict]:
    """Create Stage 1 with tunable class weighting for BPD recall."""

    model = Pipeline(
        [
            ("preprocess", base.make_preprocessor(gene_features)),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif)),
            (
                "classifier",
                base.OneVsRestClassifier(
                    LogisticRegression(max_iter=5000, random_state=base.RANDOM_STATE)
                ),
            ),
        ]
    )
    params = {
        "select__k": [25, 50, 100, 200],
        "classifier__estimator__C": [0.01, 0.03, 0.1, 1, 10],
        "classifier__estimator__class_weight": [
            None,
            "balanced",
            {0: 1.0, 1: 1.25},
            {0: 1.0, 1: 1.5},
        ],
    }
    return model, params


def make_stage2_model(gene_features: list[str]) -> tuple[Pipeline, dict]:
    """Create Stage 2 with a broader search for minority severity classes."""

    model = Pipeline(
        [
            ("preprocess", base.make_preprocessor(gene_features)),
            ("variance", VarianceThreshold()),
            ("pca", PCA(random_state=base.RANDOM_STATE)),
            (
                "classifier",
                LogisticRegression(max_iter=5000, random_state=base.RANDOM_STATE),
            ),
        ]
    )
    params = {
        "pca__n_components": [3, 5, 8, 10, 15, 20, 30, 40],
        "classifier__C": [0.001, 0.01, 0.03, 0.1, 1, 10, 30],
        "classifier__class_weight": [
            None,
            "balanced",
            {1: 1.0, 2: 1.5, 3: 1.5},
            {1: 0.8, 2: 1.5, 3: 2.0},
        ],
    }
    return model, params


def route_predictions(stage1, stage2, X: pd.DataFrame, threshold: float) -> pd.Series:
    """Route rows using a tunable Stage 1 probability threshold."""

    bpd_probability = stage1.predict_proba(X)[:, 1]
    predictions = pd.Series(0, index=X.index, dtype=int)
    routed = bpd_probability >= threshold
    if routed.any():
        predictions.loc[routed] = stage2.predict(X.loc[routed])
    return predictions


def tune_threshold(
    stage1,
    stage2,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> float:
    """Choose a routing threshold using grouped validation within training data."""

    stage1_y = (y > 0).astype(int)
    _, inner_splits = base.feasible_splits(stage1_y, groups)
    inner_cv = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=base.RANDOM_STATE,
    )
    probabilities = []
    stage2_predictions = []
    validation_labels = []

    for inner_train_idx, inner_valid_idx in inner_cv.split(X, stage1_y, groups):
        X_inner_train = X.iloc[inner_train_idx]
        y_inner_train = y.iloc[inner_train_idx]
        groups_inner_train = groups.iloc[inner_train_idx]
        X_inner_valid = X.iloc[inner_valid_idx]

        fitted_stage1 = clone(stage1).fit(
            X_inner_train,
            (y_inner_train > 0).astype(int),
        )
        bpd_mask = y_inner_train > 0
        fitted_stage2 = clone(stage2).fit(
            X_inner_train[bpd_mask],
            y_inner_train[bpd_mask],
        )
        probabilities.extend(fitted_stage1.predict_proba(X_inner_valid)[:, 1])
        stage2_predictions.extend(fitted_stage2.predict(X_inner_valid))
        validation_labels.extend(y.iloc[inner_valid_idx])

    probabilities = pd.Series(probabilities)
    stage2_predictions = pd.Series(stage2_predictions, dtype=int)
    validation_labels = pd.Series(validation_labels, dtype=int)
    scores = {}
    for threshold in THRESHOLDS:
        routed = stage2_predictions.where(probabilities >= threshold, 0)
        scores[threshold] = f1_score(
            validation_labels,
            routed,
            labels=list(base.LABELS),
            average="macro",
            zero_division=0,
        )
    return max(THRESHOLDS, key=lambda threshold: (scores[threshold], -abs(threshold - 0.5)))


def evaluate_two_step(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    gene_features: list[str],
    output_dir: Path,
) -> dict:
    """Evaluate v2 with grouped threshold tuning for every outer fold."""

    stage1_model, stage1_params = make_stage1_model(gene_features)
    stage2_model, stage2_params = make_stage2_model(gene_features)
    stage1_y = (y > 0).astype(int)
    outer_splits, _ = base.feasible_splits(stage1_y, groups)
    outer_cv = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=base.RANDOM_STATE,
    )

    final_true, final_pred = [], []
    stage1_true, stage1_pred = [], []
    oracle_true, oracle_pred = [], []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, stage1_y, groups), start=1):
        X_train, y_train, groups_train = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
        stage1 = base.tune_model(stage1_model, stage1_params, X_train, (y_train > 0).astype(int), groups_train)
        bpd_train_mask = y_train > 0
        stage2 = base.tune_model(stage2_model, stage2_params, X_train[bpd_train_mask], y_train[bpd_train_mask], groups_train[bpd_train_mask])
        threshold = tune_threshold(stage1.best_estimator_, stage2.best_estimator_, X_train, y_train, groups_train)

        fold_final_pred = route_predictions(stage1.best_estimator_, stage2.best_estimator_, X_test, threshold)
        fold_stage1_pred = (stage1.best_estimator_.predict_proba(X_test)[:, 1] >= threshold).astype(int)
        fold_stage1_pred = pd.Series(fold_stage1_pred, index=X_test.index)
        true_stage1 = (y_test > 0).astype(int)
        final_true.extend(y_test.tolist())
        final_pred.extend(fold_final_pred.tolist())
        stage1_true.extend(true_stage1.tolist())
        stage1_pred.extend(fold_stage1_pred.tolist())

        true_bpd = y_test > 0
        if true_bpd.any():
            oracle_true.extend(y_test[true_bpd].tolist())
            oracle_pred.extend(stage2.best_estimator_.predict(X_test[true_bpd]).tolist())

        fold_rows.append({
            "fold": fold,
            "stage1_threshold": threshold,
            "stage1_best_params": stage1.best_params_,
            "stage2_best_params": stage2.best_params_,
            "final_accuracy": accuracy_score(y_test, fold_final_pred),
            "final_balanced_accuracy": balanced_accuracy_score(y_test, fold_final_pred),
            "final_macro_f1": f1_score(y_test, fold_final_pred, labels=list(base.LABELS), average="macro", zero_division=0),
            "final_mae": mean_absolute_error(y_test, fold_final_pred),
            "stage1_accuracy": accuracy_score(true_stage1, fold_stage1_pred),
            "stage1_macro_f1": f1_score(true_stage1, fold_stage1_pred, average="macro"),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    base.save_reports(final_true, final_pred, stage1_true, stage1_pred, oracle_true, oracle_pred, output_dir)
    return {
        "stage1_accuracy": accuracy_score(stage1_true, stage1_pred),
        "stage1_balanced_accuracy": balanced_accuracy_score(stage1_true, stage1_pred),
        "stage1_macro_f1": f1_score(stage1_true, stage1_pred, average="macro"),
        "stage2_oracle_bpd_accuracy": accuracy_score(oracle_true, oracle_pred),
        "stage2_oracle_bpd_balanced_accuracy": balanced_accuracy_score(oracle_true, oracle_pred),
        "stage2_oracle_bpd_macro_f1": f1_score(oracle_true, oracle_pred, average="macro"),
        "final_routed_accuracy": accuracy_score(final_true, final_pred),
        "final_routed_balanced_accuracy": balanced_accuracy_score(final_true, final_pred),
        "final_routed_macro_f1": f1_score(final_true, final_pred, labels=list(base.LABELS), average="macro", zero_division=0),
        "final_routed_mae": mean_absolute_error(final_true, final_pred),
    }


def fit_final_model(X, y, groups, gene_features, output_dir, view) -> None:
    """Fit final v2 models and persist a threshold selected from grouped CV."""

    stage1_model, stage1_params = make_stage1_model(gene_features)
    stage2_model, stage2_params = make_stage2_model(gene_features)
    stage1 = base.tune_model(stage1_model, stage1_params, X, (y > 0).astype(int), groups)
    bpd_mask = y > 0
    stage2 = base.tune_model(stage2_model, stage2_params, X[bpd_mask], y[bpd_mask], groups[bpd_mask])
    threshold = tune_threshold(stage1.best_estimator_, stage2.best_estimator_, X, y, groups)
    artifact = {
        "description": "Two-step BPD v2 classifier with 25 genes and grouped threshold tuning.",
        "view": view,
        "gene_count": REDUCED_GENE_COUNT,
        "stage1_threshold": threshold,
        "stage1_no_vs_any_bpd": stage1.best_estimator_,
        "stage2_mild_moderate_severe": stage2.best_estimator_,
        "features": list(X.columns),
        "label_map": base.LABELS,
        "stage1_best_params": stage1.best_params_,
        "stage2_best_params": stage2.best_params_,
    }
    joblib.dump(artifact, output_dir / "two_step_bpd_model.joblib")


def main() -> None:
    """Run the v2 evaluation and save the tuned final artifact."""

    args = base.parse_args()
    if args.input == base.DEFAULT_INPUT:
        args.input = DEFAULT_INPUT
    if args.output == base.DEFAULT_OUTPUT:
        args.output = DEFAULT_OUTPUT

    X, y, groups, gene_features = load_reduced_data(args.input, args.view)
    args.output.mkdir(parents=True, exist_ok=True)

    original_stage1 = base.make_stage1_model
    original_stage2 = base.make_stage2_model
    base.make_stage1_model = make_stage1_model
    base.make_stage2_model = make_stage2_model
    try:
        metrics = evaluate_two_step(X, y, groups, gene_features, args.output)
        fit_final_model(X, y, groups, gene_features, args.output, args.view)
    finally:
        base.make_stage1_model = original_stage1
        base.make_stage2_model = original_stage2

    import pandas as pd

    pd.DataFrame([metrics]).to_csv(args.output / "summary_metrics.csv", index=False)
    print("Two-step BPD v2 pipeline complete.")
    print(f"View: {args.view}")
    print(f"Rows: {len(X)}")
    print(f"Infants: {groups.nunique()}")
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(f"Outputs written to: {args.output}")


if __name__ == "__main__":
    main()