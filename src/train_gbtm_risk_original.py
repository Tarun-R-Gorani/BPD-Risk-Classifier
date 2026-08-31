import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import label_binarize
import statsmodels.api as sm

sns.set_theme(style="whitegrid")


class PureGBTM:
    """Group-Based Trajectory Model (GBTM) fitted via EM with Weighted Least Squares."""

    def __init__(self, n_groups=3, degree=2, max_iter=100, tol=1e-4):
        self.n_groups = n_groups
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        self.group_params = []
        self.sigmas = []
        self.pi = None

    def _build_design_matrix(self, timepoints):
        return np.column_stack([timepoints**d for d in range(self.degree + 1)])

    def fit(self, df, id_col, time_col, val_col):
        subjects = df[id_col].unique()
        N = len(subjects)
        self.pi = np.full(self.n_groups, 1.0 / self.n_groups)

        subj_data = {}
        for s in subjects:
            s_df = df[df[id_col] == s].sort_values(time_col)
            t = s_df[time_col].values
            y = s_df[val_col].values
            X = self._build_design_matrix(t)
            subj_data[s] = (X, y)

        np.random.seed(42)
        resp = np.random.dirichlet(np.ones(self.n_groups), size=N)

        for iteration in range(self.max_iter):
            old_pi = self.pi.copy()
            self.group_params = []
            self.sigmas = []

            for k in range(self.n_groups):
                weights = resp[:, k]
                X_all, y_all, w_all = [], [], []

                for idx, s in enumerate(subjects):
                    X_s, y_s = subj_data[s]
                    X_all.append(X_s)
                    y_all.append(y_s)
                    w_all.append(np.full(len(y_s), weights[idx]))

                X_mat = np.vstack(X_all)
                y_vec = np.concatenate(y_all)
                w_vec = np.concatenate(w_all)

                wls_model = sm.WLS(y_vec, X_mat, weights=w_vec + 1e-6).fit()
                beta = wls_model.params
                resids = y_vec - X_mat @ beta
                sigma = np.sqrt(
                    np.sum(w_vec * (resids**2)) / np.sum(w_vec) + 1e-6
                )

                self.group_params.append(beta)
                self.sigmas.append(sigma)

            self.pi = resp.mean(axis=0)

            log_lik = np.zeros((N, self.n_groups))
            for idx, s in enumerate(subjects):
                X_s, y_s = subj_data[s]
                for k in range(self.n_groups):
                    pred = X_s @ self.group_params[k]
                    res = y_s - pred
                    log_lik[idx, k] = np.log(self.pi[k] + 1e-12) - np.sum(
                        0.5 * np.log(2 * np.pi * (self.sigmas[k] ** 2))
                        + (res**2) / (2 * (self.sigmas[k] ** 2))
                    )

            max_log = np.max(log_lik, axis=1, keepdims=True)
            lik = np.exp(log_lik - max_log)
            resp = lik / np.sum(lik, axis=1, keepdims=True)

            if np.max(np.abs(self.pi - old_pi)) < self.tol:
                break

        assigned_groups = np.argmax(resp, axis=1)
        group_df = pd.DataFrame(
            {
                id_col: subjects,
                "gbtm_group": assigned_groups,
                "group_prob": np.max(resp, axis=1),
            }
        )
        return group_df, resp


def main():
    print("1. Loading long-format dataset...")
    data_path = "data/processed/gbtm_input_long_format.csv"
    if not os.path.exists(data_path):
        data_path = "../data/processed/gbtm_input_long_format.csv"

    df = pd.read_csv(data_path)
    gene_cols = [
        c
        for c in df.columns
        if c
        not in [
            "infant_id",
            "timepoint_code",
            "days of life",
            "gestational age (weeks)",
            "birth weight (grams)",
            "bpd_group",
        ]
    ]

    print("2. Extracting primary trajectory expression signal...")
    pca = PCA(n_components=1, random_state=42)
    df["expr_trajectory_signal"] = pca.fit_transform(df[gene_cols])[:, 0]

    print("3. Fitting Pure GBTM Polynomial Trajectories (3 Groups)...")
    gbtm = PureGBTM(n_groups=3, degree=2)
    gbtm_results, _ = gbtm.fit(
        df,
        id_col="infant_id",
        time_col="timepoint_code",
        val_col="expr_trajectory_signal",
    )

    df = df.merge(gbtm_results, on="infant_id")

    infant_df = (
        df.groupby("infant_id")
        .first()
        .reset_index()[
            [
                "infant_id",
                "gestational age (weeks)",
                "birth weight (grams)",
                "bpd_group",
                "gbtm_group",
                "group_prob",
            ]
        ]
    )

    # Convert bpd_group to integer targets (0, 1, 2, 3)
    infant_df["bpd_target"] = infant_df["bpd_group"].astype(int)
    class_names = ["No BPD", "Mild BPD", "Moderate BPD", "Severe BPD"]
    unique_classes = sorted(infant_df["bpd_target"].unique())
    active_class_names = [class_names[c] for c in unique_classes]

    print(
        "4. Evaluating Multi-class BPD Risk Classifier using Leave-One-Out (LOOCV)..."
    )
    X = infant_df[
        [
            "gestational age (weeks)",
            "birth weight (grams)",
            "gbtm_group",
            "group_prob",
        ]
    ]
    y = infant_df["bpd_target"]

    loo = LeaveOneOut()
    y_true, y_pred, y_probs = [], [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_true.append(y_test.values[0])
        y_pred.append(clf.predict(X_test)[0])

        # Handle class probability alignment during cross-validation
        probs = clf.predict_proba(X_test)[0]
        full_prob = np.zeros(len(unique_classes))
        for idx, c in enumerate(clf.classes_):
            full_prob[c] = probs[idx]
        y_probs.append(full_prob)

    y_probs = np.array(y_probs)

    print(
        "\n================ 4-CLASS BPD GBTM EVALUATION REPORT ================"
    )
    print(
        classification_report(
            y_true,
            y_pred,
            labels=unique_classes,
            target_names=active_class_names,
        )
    )

    print("5. Retraining final Risk Classifier on full dataset...")
    final_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    final_clf.fit(X, y)

    print("6. Saving trained models for later use...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(gbtm, "models/gbtm_trajectory_model.pkl")
    joblib.dump(final_clf, "models/bpd_risk_classifier.pkl")
    joblib.dump(pca, "models/pca_transformer.pkl")
    print("   -> Saved 'models/gbtm_trajectory_model.pkl'")
    print("   -> Saved 'models/bpd_risk_classifier.pkl'")
    print("   -> Saved 'models/pca_transformer.pkl'")

    print("\n7. Generating multi-class visualization charts...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Longitudinal Trajectories
    sns.lineplot(
        data=df,
        x="timepoint_code",
        y="expr_trajectory_signal",
        hue="gbtm_group",
        palette="Set1",
        style="gbtm_group",
        markers=True,
        dashes=False,
        ax=axes[0],
    )
    axes[0].set_title("GBTM Polynomial Trajectory Classes", fontweight="bold")
    axes[0].set_xlabel("Timepoint Visit")
    axes[0].set_ylabel("Gene Trajectory Signal")

    # Plot 2: Multiclass One-vs-Rest ROC Curves
    y_bin = label_binarize(y_true, classes=unique_classes)
    colors = ["navy", "green", "orange", "darkred"]

    for i, c in enumerate(unique_classes):
        if y_bin.shape[1] > 1:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            axes[1].plot(
                fpr,
                tpr,
                color=colors[c % len(colors)],
                lw=2,
                label=f"{class_names[c]} (AUC = {roc_auc:.2f})",
            )

    axes[1].plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    axes[1].set_title("Multiclass BPD Risk ROC Curves", fontweight="bold")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(loc="lower right")

    # Plot 3: 4x4 Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=active_class_names,
        yticklabels=active_class_names,
        ax=axes[2],
    )
    axes[2].set_title(
        "4-Class Risk Classification Matrix", fontweight="bold"
    )
    axes[2].set_xlabel("Predicted BPD Grade")
    axes[2].set_ylabel("Actual BPD Grade")

    plt.tight_layout()
    plot_path = "gbtm_multiclass_bpd_risk.png"
    plt.savefig(plot_path, dpi=300)
    print(f"   -> Saved 4-class evaluation plot to '{plot_path}'!")
    plt.show()


if __name__ == "__main__":
    main()