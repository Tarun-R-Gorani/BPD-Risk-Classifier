"""Shareable two-step BPD severity pipeline.

This script implements the clinically structured model selected from the
exploratory work:

1. Stage 1: No BPD vs Any BPD.
2. Stage 2: among samples predicted as BPD, classify Mild, Moderate, Severe.

No synthetic samples are used. Evaluation uses grouped cross-validation so rows
from the same infant never appear in both train and test folds.

Run:
    python two_tier.py

Main outputs are written to:
    outputs/shareable_two_step_pipeline/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "../data/processed/gbtm_input_long_format(500_genes).csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "shareable_two_step_pipeline"
RANDOM_STATE = 42

LABELS = {
    0: "No BPD",
    1: "Mild BPD",
    2: "Moderate BPD",
    3: "Severe BPD",
}

CLINICAL_FEATURES = [
    "days of life",
    "gestational age (weeks)",
    "birth weight (grams)",
]

ID_COLUMNS = ["infant_id", "timepoint_code", "bpd_group", *CLINICAL_FEATURES]


def load_data(input_path: Path, view: str) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str]]:
    """Load the prepared 500-gene table and return one modeling view.

    Args:
        input_path: Path to `gbtm_input_long_format(500_genes).csv`.
        view: Modeling view. Supported values are `timepoint_1`, `timepoint_2`,
            `timepoint_3`, `latest`, and `all_rows`.

    Returns:
        X: Feature table.
        y: Four-class BPD labels encoded as 0, 1, 2, 3.
        groups: Infant IDs used for grouped cross-validation.
        gene_features: Gene-expression feature names.
    """

    df = pd.read_csv(input_path)
    df["infant_id"] = df["infant_id"].astype(str)
    df["timepoint_code"] = df["timepoint_code"].astype(int)
    df["bpd_group"] = df["bpd_group"].astype(int)
    gene_features = [column for column in df.columns if column not in ID_COLUMNS]

    if view.startswith("timepoint_"):
        timepoint = int(view.split("_")[-1])
        df = df[df["timepoint_code"] == timepoint].copy()
    elif view == "latest":
        df = (
            df.sort_values(["infant_id", "timepoint_code"])
            .drop_duplicates("infant_id", keep="last")
            .copy()
        )
    elif view == "all_rows":
        df = df.copy()
    else:
        raise ValueError(
            "view must be one of: timepoint_1, timepoint_2, timepoint_3, latest, all_rows"
        )

    X = df[CLINICAL_FEATURES + gene_features].copy()
    y = df["bpd_group"].copy()
    groups = df["infant_id"].copy()
    return X, y, groups, gene_features


def feasible_splits(y: pd.Series, groups: pd.Series) -> tuple[int, int]:
    """Choose outer and inner grouped CV split counts for available classes.

    Args:
        y: Labels for the current task.
        groups: Infant IDs.

    Returns:
        Tuple of `(outer_splits, inner_splits)`.
    """

    class_group_counts = (
        pd.DataFrame({"y": y, "group": groups})
        .drop_duplicates(["y", "group"])
        .groupby("y")["group"]
        .nunique()
    )
    min_groups = int(class_group_counts.min())
    if min_groups < 3:
        raise ValueError("At least 3 infant groups per class are needed for this evaluation.")
    return min(5, min_groups), min(3, max(2, min_groups - 1))


def make_preprocessor(gene_features: list[str]) -> ColumnTransformer:
    """Build common clinical and gene preprocessing."""

    return ColumnTransformer(
        [
            (
                "clinical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                CLINICAL_FEATURES,
            ),
            (
                "genes",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                gene_features,
            ),
        ]
    )


def make_stage1_model(gene_features: list[str]) -> tuple[Pipeline, dict]:
    """Create Stage 1 model: No BPD vs Any BPD."""

    model = Pipeline(
        [
            ("preprocess", make_preprocessor(gene_features)),
            ("variance", VarianceThreshold()),
            ("select", SelectKBest(score_func=f_classif)),
            (
                "classifier",
                OneVsRestClassifier(
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    )
                ),
            ),
        ]
    )
    params = {
        "select__k": [25, 50, 100, 200],
        "classifier__estimator__C": [0.01, 0.1, 1, 10],
    }
    return model, params


def make_stage2_model(gene_features: list[str]) -> tuple[Pipeline, dict]:
    """Create Stage 2 model: Mild vs Moderate vs Severe among BPD cases."""

    model = Pipeline(
        [
            ("preprocess", make_preprocessor(gene_features)),
            ("variance", VarianceThreshold()),
            ("pca", PCA(random_state=RANDOM_STATE)),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    params = {
        "pca__n_components": [5, 10, 20, 40],
        "classifier__C": [0.01, 0.1, 1, 10],
    }
    return model, params


def prune_grid(params: dict, n_samples: int, n_features: int) -> dict:
    """Remove impossible feature-selection and PCA settings."""

    pruned = {}
    for key, values in params.items():
        if key == "select__k":
            pruned[key] = [value for value in values if value <= n_features]
        elif key == "pca__n_components":
            max_components = min(max(1, n_samples // 2), n_features)
            pruned[key] = [value for value in values if value <= max_components]
        else:
            pruned[key] = values
    return pruned


def tune_model(
    model: Pipeline,
    params: dict,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> GridSearchCV:
    """Tune a model with grouped inner cross-validation."""

    _, inner_splits = feasible_splits(y, groups)
    cv = StratifiedGroupKFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    search = GridSearchCV(
        clone(model),
        prune_grid(params, len(X), X.shape[1]),
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y, groups=groups)
    return search


def evaluate_two_step(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    gene_features: list[str],
    output_dir: Path,
) -> dict:
    """Evaluate the hard-routed two-step classifier with grouped outer CV.

    Args:
        X: Feature table.
        y: Four-class labels.
        groups: Infant IDs.
        gene_features: Gene-expression feature names.
        output_dir: Directory where evaluation artifacts are saved.

    Returns:
        Summary metrics for final model, Stage 1, and oracle Stage 2.
    """

    stage1_model, stage1_params = make_stage1_model(gene_features)
    stage2_model, stage2_params = make_stage2_model(gene_features)
    stage1_y = (y > 0).astype(int)
    outer_splits, _ = feasible_splits(stage1_y, groups)
    outer_cv = StratifiedGroupKFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    final_true: list[int] = []
    final_pred: list[int] = []
    stage1_true: list[int] = []
    stage1_pred: list[int] = []
    oracle_stage2_true: list[int] = []
    oracle_stage2_pred: list[int] = []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X, stage1_y, groups), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        groups_train = groups.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        stage1 = tune_model(
            stage1_model,
            stage1_params,
            X_train,
            (y_train > 0).astype(int),
            groups_train,
        )

        bpd_train_mask = y_train > 0
        stage2 = tune_model(
            stage2_model,
            stage2_params,
            X_train[bpd_train_mask],
            y_train[bpd_train_mask],
            groups_train[bpd_train_mask],
        )

        fold_stage1_pred = pd.Series(stage1.predict(X_test), index=X_test.index)
        fold_final_pred = pd.Series(0, index=X_test.index)
        routed_bpd_mask = fold_stage1_pred == 1
        if routed_bpd_mask.any():
            fold_final_pred.loc[routed_bpd_mask] = stage2.predict(X_test[routed_bpd_mask])

        true_stage1 = (y_test > 0).astype(int)
        final_true.extend(y_test.tolist())
        final_pred.extend(fold_final_pred.tolist())
        stage1_true.extend(true_stage1.tolist())
        stage1_pred.extend(fold_stage1_pred.tolist())

        true_bpd_test_mask = y_test > 0
        if true_bpd_test_mask.any():
            oracle_predictions = stage2.predict(X_test[true_bpd_test_mask])
            oracle_stage2_true.extend(y_test[true_bpd_test_mask].tolist())
            oracle_stage2_pred.extend(oracle_predictions.tolist())

        fold_rows.append(
            {
                "fold": fold,
                "stage1_best_params": stage1.best_params_,
                "stage2_best_params": stage2.best_params_,
                "final_accuracy": accuracy_score(y_test, fold_final_pred),
                "final_balanced_accuracy": balanced_accuracy_score(y_test, fold_final_pred),
                "final_macro_f1": f1_score(y_test, fold_final_pred, average="macro"),
                "final_mae": mean_absolute_error(y_test, fold_final_pred),
                "stage1_accuracy": accuracy_score(true_stage1, fold_stage1_pred),
                "stage1_macro_f1": f1_score(true_stage1, fold_stage1_pred, average="macro"),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(fold_rows).to_csv(output_dir / "fold_metrics.csv", index=False)
    save_reports(
        final_true,
        final_pred,
        stage1_true,
        stage1_pred,
        oracle_stage2_true,
        oracle_stage2_pred,
        output_dir,
    )

    return {
        "stage1_accuracy": accuracy_score(stage1_true, stage1_pred),
        "stage1_balanced_accuracy": balanced_accuracy_score(stage1_true, stage1_pred),
        "stage1_macro_f1": f1_score(stage1_true, stage1_pred, average="macro"),
        "stage2_oracle_bpd_accuracy": accuracy_score(oracle_stage2_true, oracle_stage2_pred),
        "stage2_oracle_bpd_balanced_accuracy": balanced_accuracy_score(
            oracle_stage2_true,
            oracle_stage2_pred,
        ),
        "stage2_oracle_bpd_macro_f1": f1_score(
            oracle_stage2_true,
            oracle_stage2_pred,
            average="macro",
        ),
        "final_routed_accuracy": accuracy_score(final_true, final_pred),
        "final_routed_balanced_accuracy": balanced_accuracy_score(final_true, final_pred),
        "final_routed_macro_f1": f1_score(final_true, final_pred, average="macro"),
        "final_routed_mae": mean_absolute_error(final_true, final_pred),
    }


def save_reports(
    final_true: list[int],
    final_pred: list[int],
    stage1_true: list[int],
    stage1_pred: list[int],
    stage2_true: list[int],
    stage2_pred: list[int],
    output_dir: Path,
) -> None:
    """Save classification reports and the final confusion matrix."""

    final_report = classification_report(
        final_true,
        final_pred,
        labels=list(LABELS),
        target_names=[LABELS[label] for label in LABELS],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(final_report).transpose().to_csv(output_dir / "final_classification_report.csv")

    stage1_report = classification_report(
        stage1_true,
        stage1_pred,
        labels=[0, 1],
        target_names=["No BPD", "Any BPD"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(stage1_report).transpose().to_csv(output_dir / "stage1_report.csv")

    stage2_report = classification_report(
        stage2_true,
        stage2_pred,
        labels=[1, 2, 3],
        target_names=["Mild BPD", "Moderate BPD", "Severe BPD"],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(stage2_report).transpose().to_csv(output_dir / "stage2_oracle_bpd_report.csv")

    matrix = confusion_matrix(final_true, final_pred, labels=list(LABELS))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[LABELS[label] for label in LABELS],
    )
    display.plot(cmap="Blues", xticks_rotation=30)
    plt.title("Two-Step BPD Classifier")
    plt.tight_layout()
    plt.savefig(output_dir / "final_confusion_matrix.png", dpi=200)
    plt.close()


def fit_final_model(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    gene_features: list[str],
    output_dir: Path,
    view: str,
) -> None:
    """Train final Stage 1 and Stage 2 models on all available data."""

    stage1_model, stage1_params = make_stage1_model(gene_features)
    stage2_model, stage2_params = make_stage2_model(gene_features)

    stage1 = tune_model(stage1_model, stage1_params, X, (y > 0).astype(int), groups)
    bpd_mask = y > 0
    stage2 = tune_model(stage2_model, stage2_params, X[bpd_mask], y[bpd_mask], groups[bpd_mask])

    artifact = {
        "description": "Two-step BPD classifier: Stage 1 No vs Any BPD; Stage 2 Mild/Moderate/Severe.",
        "view": view,
        "stage1_no_vs_any_bpd": stage1.best_estimator_,
        "stage2_mild_moderate_severe": stage2.best_estimator_,
        "features": list(X.columns),
        "label_map": LABELS,
        "stage1_best_params": stage1.best_params_,
        "stage2_best_params": stage2.best_params_,
    }
    joblib.dump(artifact, output_dir / "two_step_bpd_model.joblib")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""

    parser = argparse.ArgumentParser(description="Train and evaluate a two-step BPD classifier.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to 500-gene CSV.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for reports and trained model.",
    )
    parser.add_argument(
        "--view",
        choices=["timepoint_1", "timepoint_2", "timepoint_3", "latest", "all_rows"],
        default="timepoint_3",
        help="Data view to use. Default matches the best hard-routed four-class run.",
    )
    return parser.parse_args()


def main() -> None:
    """Run evaluation and train final two-step model."""

    args = parse_args()
    X, y, groups, gene_features = load_data(args.input, args.view)
    args.output.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_two_step(X, y, groups, gene_features, args.output)
    fit_final_model(X, y, groups, gene_features, args.output, args.view)
    pd.DataFrame([metrics]).to_csv(args.output / "summary_metrics.csv", index=False)

    print("Two-step BPD pipeline complete.")
    print(f"View: {args.view}")
    print(f"Rows: {len(X)}")
    print(f"Infants: {groups.nunique()}")
    print("\nStage 1: No BPD vs Any BPD")
    print(
        pd.DataFrame(
            [
                {
                    "accuracy": metrics["stage1_accuracy"],
                    "balanced_accuracy": metrics["stage1_balanced_accuracy"],
                    "macro_f1": metrics["stage1_macro_f1"],
                }
            ]
        ).to_string(index=False)
    )
    print("\nStage 2: Mild vs Moderate vs Severe among true BPD cases")
    print(
        pd.DataFrame(
            [
                {
                    "accuracy": metrics["stage2_oracle_bpd_accuracy"],
                    "balanced_accuracy": metrics["stage2_oracle_bpd_balanced_accuracy"],
                    "macro_f1": metrics["stage2_oracle_bpd_macro_f1"],
                }
            ]
        ).to_string(index=False)
    )
    print("\nFinal hard-routed system: No/Mild/Moderate/Severe")
    print(
        pd.DataFrame(
            [
                {
                    "accuracy": metrics["final_routed_accuracy"],
                    "balanced_accuracy": metrics["final_routed_balanced_accuracy"],
                    "macro_f1": metrics["final_routed_macro_f1"],
                    "mae": metrics["final_routed_mae"],
                }
            ]
        ).to_string(index=False)
    )
    print(f"\nOutputs written to: {args.output}")
    print(f"Model artifact: {args.output / 'two_step_bpd_model.joblib'}")


if __name__ == "__main__":
    main()