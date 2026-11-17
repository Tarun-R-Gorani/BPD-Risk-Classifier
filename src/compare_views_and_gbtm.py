"""Compare longitudinal views and a fold-fitted GBTM baseline.

Run from the repository root. This is an evaluation only; it does not replace
the deployed model.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compare_feature_sets import LABELS, evaluate, load_data
from src.gbtm_model import PureGBTM


OUT = ROOT / "src/outputs/view_and_gbtm_comparison"
SEED = 42
VIEWS = ["timepoint_1", "timepoint_2", "timepoint_3", "latest", "all_rows"]
CLINICAL = ["days of life", "gestational age (weeks)", "birth weight (grams)"]


def evaluate_gbtm(data: pd.DataFrame, genes: list[str]):
    """Outer-fold evaluation; PCA and GBTM are fitted on training infants only."""
    infant = (data.sort_values(["infant_id", "timepoint_code"])
              .groupby("infant_id", as_index=False).first())
    infant_ids = infant["infant_id"].astype(str)
    y = infant["bpd_group"].astype(int)
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    y_true, predictions = {"clinical_only": [], "clinical_plus_gbtm": []}, {
        "clinical_only": [], "clinical_plus_gbtm": []
    }
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(outer.split(infant, y, infant_ids), 1):
        train_ids = set(infant.iloc[train_idx].infant_id.astype(str))
        test_ids = set(infant.iloc[test_idx].infant_id.astype(str))
        train_rows = data[data.infant_id.astype(str).isin(train_ids)].copy()
        test_rows = data[data.infant_id.astype(str).isin(test_ids)].copy()

        # Unsupervised gene variance, scaling, and PCA are estimated on this
        # outer training fold only.
        train_genes = train_rows[genes].apply(pd.to_numeric, errors="coerce")
        test_genes = test_rows[genes].apply(pd.to_numeric, errors="coerce")
        imputer = SimpleImputer(strategy="median")
        train_imputed = imputer.fit_transform(train_genes)
        test_imputed = imputer.transform(test_genes)
        variance = np.nanvar(train_imputed, axis=0)
        selected = np.argsort(variance)[-500:]
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_imputed[:, selected])
        test_scaled = scaler.transform(test_imputed[:, selected])
        pca = PCA(n_components=3, random_state=SEED)
        train_pc = pca.fit_transform(train_scaled)
        test_pc = pca.transform(test_scaled)

        train_projected = train_rows[["infant_id", "timepoint_code"]].reset_index(drop=True)
        test_projected = test_rows[["infant_id", "timepoint_code"]].reset_index(drop=True)
        for component in range(3):
            col = f"pc{component + 1}"
            train_projected[col] = train_pc[:, component]
            test_projected[col] = test_pc[:, component]

        train_trajectory_features, test_trajectory_features = [], []
        for component in range(1, 4):
            col = f"pc{component}"
            gbtm = PureGBTM(n_groups=3, degree=2)
            train_res, _ = gbtm.fit(
                train_projected, id_col="infant_id", time_col="timepoint_code", val_col=col
            )
            test_res = gbtm.predict_groups(
                test_projected, id_col="infant_id", time_col="timepoint_code", val_col=col
            )
            train_res = train_res.rename(columns={
                "gbtm_group": f"gbtm_group_pc{component}",
                "group_prob": f"group_prob_pc{component}",
            })
            test_res = test_res.rename(columns={
                "gbtm_group": f"gbtm_group_pc{component}",
                "group_prob": f"group_prob_pc{component}",
            })
            train_trajectory_features.append(train_res)
            test_trajectory_features.append(test_res)

        train_features = (train_rows.groupby("infant_id", as_index=False)[CLINICAL]
                          .first())
        test_features = (test_rows.groupby("infant_id", as_index=False)[CLINICAL]
                         .first())
        for feature_frame in train_trajectory_features:
            train_features = train_features.merge(feature_frame, on="infant_id", how="left")
        for feature_frame in test_trajectory_features:
            test_features = test_features.merge(feature_frame, on="infant_id", how="left")

        train_labels = infant.set_index("infant_id").loc[train_features.infant_id, "bpd_group"].astype(int)
        test_labels = infant.set_index("infant_id").loc[test_features.infant_id, "bpd_group"].astype(int)
        models = {
            "clinical_only": CLINICAL,
            "clinical_plus_gbtm": CLINICAL + [
                name for component in range(1, 4)
                for name in (f"gbtm_group_pc{component}", f"group_prob_pc{component}")
            ],
        }
        for name, columns in models.items():
            model = make_pipeline(
                SimpleImputer(strategy="median"), StandardScaler(),
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
            )
            model.fit(train_features[columns], train_labels)
            pred = model.predict(test_features[columns])
            y_true[name].extend(test_labels.tolist())
            predictions[name].extend(pred.tolist())
            fold_rows.append({
                "model": name, "fold": fold,
                "accuracy": accuracy_score(test_labels, pred),
                "balanced_accuracy": balanced_accuracy_score(test_labels, pred),
                "macro_f1": f1_score(test_labels, pred, labels=LABELS,
                                      average="macro", zero_division=0),
            })
        print(f"  GBTM fold {fold}/5 complete", flush=True)

    summaries = []
    for name in y_true:
        summaries.append({
            "view": "longitudinal_trajectory",
            "model": name,
            "infants": len(y_true[name]),
            "accuracy": accuracy_score(y_true[name], predictions[name]),
            "balanced_accuracy": balanced_accuracy_score(y_true[name], predictions[name]),
            "macro_f1": f1_score(y_true[name], predictions[name], labels=LABELS,
                                  average="macro", zero_division=0),
            "weighted_f1": f1_score(y_true[name], predictions[name], labels=LABELS,
                                    average="weighted", zero_division=0),
        })
    return summaries, fold_rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows, fold_rows = [], []
    for view in VIEWS:
        if view == "timepoint_3":
            # This result is the already completed, fold-local evaluation used
            # to promote the deployed candidate.
            current = pd.read_csv(ROOT / "src/outputs/feature_set_comparison/summary_metrics.csv")
            row = current.loc[current.feature_set == "combined"].iloc[0].to_dict()
            stage = pd.read_csv(OUT.parent / "feature_set_comparison/combined_stage_metrics.csv").iloc[0]
            row.update({
                "view": view, "model": "two_stage_combined",
                "rows": int(row["samples"]), "infants": int(row["samples"]),
                **{key: stage[key] for key in stage.index if key.startswith("stage")},
            })
            summary_rows.append(row)
            continue
        print(f"Evaluating combined two-stage model on {view}...", flush=True)
        data, genes = load_data(view)
        folds, _, summary = evaluate("combined", data, genes)
        summary.update({"view": view, "model": "two_stage_combined",
                        "rows": len(data), "infants": data.infant_id.nunique()})
        summary_rows.append(summary)
        for fold in folds:
            fold["view"] = view
        fold_rows.extend(folds)
        pd.DataFrame(summary_rows).to_csv(OUT / "view_summary_metrics.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(OUT / "view_fold_metrics.csv", index=False)
        print(summary, flush=True)

    print("Evaluating fold-fitted GBTM trajectory features...", flush=True)
    all_rows, genes = load_data("all_rows")
    gbtm_summary, gbtm_folds = evaluate_gbtm(all_rows, genes)
    summary_rows.extend(gbtm_summary)
    for fold in gbtm_folds:
        fold["view"] = "longitudinal_trajectory"
    fold_rows.extend(gbtm_folds)
    pd.DataFrame(summary_rows).to_csv(OUT / "view_summary_metrics.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "view_fold_metrics.csv", index=False)
    print(pd.DataFrame(gbtm_summary).to_string(index=False))
    print(f"Saved reports to {OUT}")


if __name__ == "__main__":
    main()
