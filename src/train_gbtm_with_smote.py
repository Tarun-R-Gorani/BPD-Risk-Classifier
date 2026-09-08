import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import label_binarize
import statsmodels.api as sm

from imblearn.over_sampling import SMOTENC, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTEENN

sns.set_theme(style="whitegrid")

# ----------------------------------------------------------------------------
# Resampling configuration
# ----------------------------------------------------------------------------
# "smotenc"    -> recommended default. Handles the mix of continuous features
#                 (gestational age, birth weight, group probabilities) and
#                 categorical features (gbtm_group_pc1/2/3) correctly: it never
#                 interpolates a fractional group label.
# "borderline" -> Borderline-SMOTE. Focuses synthetic samples near the
#                 decision boundary. Treats ALL features as continuous, so
#                 the categorical group columns get rounded/clipped back to
#                 valid categories after resampling (see resample_fold).
# "smoteenn"   -> SMOTE-ENN. Oversamples then removes noisy/ambiguous
#                 samples via Edited Nearest Neighbours. Useful when class
#                 overlap (not just imbalance) is hurting the classifier.
#                 Same continuous-feature caveat as Borderline-SMOTE.
SMOTE_METHOD = "smoteenn"  # "smotenc", "borderline", or "smoteenn"

# Column names for the 3 GBTM group-membership features. These are the
# categorical columns SMOTENC needs to be told about.
CATEGORICAL_COLS = ["gbtm_group_pc1", "gbtm_group_pc2", "gbtm_group_pc3"]


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


def build_resampler(method, categorical_indices, k_neighbors):
    """Instantiate the requested imbalanced-learn resampler."""
    if method == "smotenc":
        return SMOTENC(
            categorical_features=categorical_indices,
            k_neighbors=k_neighbors,
            random_state=42,
        )
    elif method == "borderline":
        return BorderlineSMOTE(k_neighbors=k_neighbors, random_state=42)
    elif method == "smoteenn":
        return SMOTEENN(random_state=42)
    else:
        raise ValueError(f"Unknown SMOTE_METHOD: {method}")


def resample_fold(X_train, y_train, categorical_cols, n_groups, method=SMOTE_METHOD):
    """
    Balance a single training fold.

    Applied ONLY to training data (never to the LOOCV test point) to avoid
    leaking synthetic information derived from the held-out infant.

    Falls back to plain random oversampling (duplication, no neighbor math)
    when a class has fewer than 2 members in this fold -- SMOTE variants
    can't interpolate from a single point, which is a real risk with rare
    BPD grades in small NICU cohorts.
    """
    class_counts = y_train.value_counts()
    min_count = class_counts.min()
    categorical_indices = [X_train.columns.get_loc(c) for c in categorical_cols]

    if min_count < 2:
        sampler = RandomOverSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X_train, y_train)
    else:
        k_neighbors = max(1, min(5, min_count - 1))
        try:
            sampler = build_resampler(method, categorical_indices, k_neighbors)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        except ValueError:
            # e.g. SMOTEENN's internal ENN step can wipe out a fold with too
            # few neighbors to clean sensibly -- fall back gracefully.
            sampler = RandomOverSampler(random_state=42)
            X_res, y_res = sampler.fit_resample(X_train, y_train)

    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    y_res = pd.Series(np.asarray(y_res), name=y_train.name)

    # Borderline-SMOTE / SMOTE-ENN treat every column as continuous, so the
    # categorical GBTM group labels can come out fractional. SMOTE-NC never
    # does this (it samples an existing category), so this is a no-op there.
    for col in categorical_cols:
        X_res[col] = X_res[col].round().clip(0, n_groups - 1).astype(int)

    return X_res, y_res


def _find_data_path():
    """Look in a few likely spots so this runs whether launched from repo
    root, a scripts/ subfolder, or right next to the CSV."""
    candidates = [
        #"../data/processed/gbtm_input_5genes.csv",
        "../data/processed/gbtm_input_long_format.csv",
        #"../data/processed/gbtm_input_long_format_500_genes_.csv"
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "Could not find the GBTM input CSV. Checked: " + ", ".join(candidates)
    )


def main():
    print("1. Loading long-format dataset...")
    data_path = _find_data_path()
    print(f"   -> Using: {data_path}")

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


        # 1. Standardize gene expression
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[gene_cols])

    # 2. Apply PCA to standardized expression
    pca = PCA(n_components=3, random_state=42)
    pca_features = pca.fit_transform(X_scaled)

    # 3. Add PCs to dataframe
    df["pc1"] = pca_features[:, 0]
    df["pc2"] = pca_features[:, 1]
    df["pc3"] = pca_features[:, 2]

    '''print("2. Extracting 3 Principal Components for multi-signal trajectories...")
    pca = PCA(n_components=3, random_state=42)
    pca_features = pca.fit_transform(df[gene_cols])'''
    '''pc_df = pd.DataFrame(
        pca_features, columns=["pc1", "pc2", "pc3"], index=df.index
    )
    df = pd.concat([df, pc_df], axis=1)'''

    print("3. Fitting GBTM Polynomial Trajectories for PC1, PC2, and PC3...")
    gbtm_models = {}
    n_groups = 3

    for pc in ["pc1", "pc2", "pc3",]:
        gbtm = PureGBTM(n_groups=n_groups, degree=2)
        res, _ = gbtm.fit(
            df, id_col="infant_id", time_col="timepoint_code", val_col=pc
        )
        # Rename columns so they don't overwrite each other when merged
        res.rename(
            columns={"gbtm_group": f"gbtm_group_{pc}", "group_prob": f"group_prob_{pc}"},
            inplace=True,
        )
        df = df.merge(res, on="infant_id")
        gbtm_models[pc] = gbtm

    # Aggregate to one row per infant
    infant_df = (
        df.groupby("infant_id")
        .first()
        .reset_index()[
            [
                "infant_id",
                "gestational age (weeks)",
                "birth weight (grams)",
                "bpd_group",
                "gbtm_group_pc1", "group_prob_pc1",
                "gbtm_group_pc2", "group_prob_pc2",
                "gbtm_group_pc3", "group_prob_pc3",
                
            ]
        ]
    )

    infant_df["bpd_target"] = infant_df["bpd_group"].astype(int)
    class_names = ["No BPD", "Mild BPD", "Moderate BPD", "Severe BPD"]
    unique_classes = sorted(infant_df["bpd_target"].unique())
    active_class_names = [class_names[c] for c in unique_classes]

    print("   Class distribution (full cohort):")
    print(f"   {infant_df['bpd_target'].value_counts().sort_index().to_dict()}")
    print(f"   Resampling method: {SMOTE_METHOD}")

    print("4. Evaluating Multi-class BPD Risk Classifier using stratified K fold")

    # Random Forest trains on 8 features
    feature_cols = [
        "gestational age (weeks)",
        "birth weight (grams)",
        "gbtm_group_pc1",
        "group_prob_pc1",
        "gbtm_group_pc2",
        "group_prob_pc2",
        "gbtm_group_pc3",
        "group_prob_pc3",

    ]
    X = infant_df[feature_cols]
    y = infant_df["bpd_target"]



    skf = StratifiedKFold(n_splits=7, shuffle=True, random_state=42)
    y_true, y_pred, y_probs = [], [], []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Resample the TRAINING fold only -- the held-out infant is never
        # touched, so no information from it leaks into synthetic samples.
        X_train_res, y_train_res = resample_fold(
            X_train, y_train, CATEGORICAL_COLS, n_groups, method=SMOTE_METHOD
        )


        # RF tuning
        param_grid = {
    "svm__C": [0.01, 0.1, 1, 10, 100, 1000],
    "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1]
}


        svm_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(
        kernel="rbf",
        class_weight="balanced",
        probability=True,
        random_state=42
    ))
])
        clf = RandomizedSearchCV(
            svm_pipeline,
            param_distributions=param_grid,
            n_iter=15,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42
        )

        clf.fit(X_train_res, y_train_res)
        # Use clf.best_estimator_ for predictions

        # Collect ALL predictions from this fold, not just first sample
        y_true.extend(y_test.values)
        y_pred.extend(clf.best_estimator_.predict(X_test))

        # Collect probabilities for ALL test samples
        probs = clf.best_estimator_.predict_proba(X_test)
        for pred_probs in probs:
            full_prob = np.zeros(len(unique_classes))
            for idx, c in enumerate(clf.classes_):
                full_prob[c] = pred_probs[idx]
            y_probs.append(full_prob)

    y_probs = np.array(y_probs)

    print("\n================ 4-CLASS BPD GBTM EVALUATION REPORT (3 PCs) ================")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=unique_classes,
            target_names=active_class_names,
        )
    )

    print("5. Retraining final Risk Classifier on full (resampled) dataset...")
    print(f"   Original class distribution:  {y.value_counts().sort_index().to_dict()}")
    X_final_res, y_final_res = resample_fold(
        X, y, CATEGORICAL_COLS, n_groups, method=SMOTE_METHOD
    )
    print(f"   Balanced class distribution:  {y_final_res.value_counts().sort_index().to_dict()}")

    
    final_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", CalibratedClassifierCV(SVC(
        kernel="rbf",
        C=1,
        gamma="scale",
        class_weight="balanced",
        probability=True,
        random_state=42), ensemble=False
    ))
])

    final_clf.fit(X_final_res, y_final_res)

    print("Number of gene features:", len(gene_cols))
    print(gene_cols[:10])
    print(gene_cols[-10:])
    print("Missing gene values:", df[gene_cols].isna().sum().sum())
    print("Expression shape:", df[gene_cols].shape)
    print("PCA explained variance:", pca.explained_variance_ratio_)
    print("Total variance explained:", pca.explained_variance_ratio_.sum())

    print("Best parameters:", clf.best_params_)
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print("Weighted F1:", f1_score(y_true, y_pred, average="weighted"))
    print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))

    print("6. Saving trained models for later use...")
    os.makedirs("models", exist_ok=True)
    # Save the dictionary containing all 3 GBTMs
    joblib.dump(gbtm_models, "models/gbtm_trajectory_models_dict.pkl")
    joblib.dump(final_clf, "models/bpd_risk_classifier.pkl")
    joblib.dump(pca, "models/pca_transformer.pkl")
    #print("   -> Saved 'models/gbtm_trajectory_models_dict.pkl'")
    #print("   -> Saved 'models/bpd_risk_classifier.pkl'")
    #print("   -> Saved 'models/pca_transformer.pkl'")

    print("\n7. Generating multi-class visualization charts...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Longitudinal Trajectories (Plotting PC1 as the primary visual)
    sns.lineplot(
        data=df,
        x="timepoint_code",
        y="pc1",
        hue="gbtm_group_pc1",
        palette="Set1",
        style="gbtm_group_pc1",
        markers=True,
        dashes=False,
        ax=axes[0],
    )
    axes[0].set_title("Primary GBTM Trajectory Classes (PC1)", fontweight="bold")
    axes[0].set_xlabel("Timepoint Visit")
    axes[0].set_ylabel("Gene Trajectory Signal (PC1)")

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
    axes[2].set_title("4-Class Risk Classification Matrix", fontweight="bold")
    axes[2].set_xlabel("Predicted BPD Grade")
    axes[2].set_ylabel("Actual BPD Grade")

    plt.tight_layout()
    plot_path = "gbtm_multiclass_bpd_risk_3pcs.png"
    plt.savefig(plot_path, dpi=300)
    print(f"   -> Saved 4-class evaluation plot to '{plot_path}'!")
    plt.show()


if __name__ == "__main__":
    main()