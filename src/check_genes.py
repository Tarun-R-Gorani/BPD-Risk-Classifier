import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
import statsmodels.api as sm

sns.set_theme(style="whitegrid")


class PureGBTM:
    """Group-Based Trajectory Model (GBTM) fitted via EM with Weighted Least Squares."""

    def __init__(self, n_groups=4, degree=2, max_iter=100, tol=1e-4):
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
        group_prob = np.max(resp, axis=1)
        group_prob_spread = np.std(resp, axis=1)
        group_entropy = -np.sum(resp * np.log(resp + 1e-12), axis=1)
        group_df = pd.DataFrame(
            {
                id_col: subjects,
                "gbtm_group": assigned_groups,
                "group_prob": group_prob,
                "group_prob_spread": group_prob_spread,
                "group_entropy": group_entropy,
            }
        )
        return group_df, resp

    def predict_groups(self, df, id_col, time_col, val_col):
        """
        Assign new infants to groups using fitted group_params and sigmas.
        
        Returns a DataFrame with infant_id, gbtm_group, and group_prob columns.
        """
        subjects = df[id_col].unique()
        N = len(subjects)
        
        log_lik = np.zeros((N, self.n_groups))
        
        for idx, s in enumerate(subjects):
            s_df = df[df[id_col] == s].sort_values(time_col)
            t = s_df[time_col].values
            y = s_df[val_col].values
            X = self._build_design_matrix(t)
            
            for k in range(self.n_groups):
                pred = X @ self.group_params[k]
                res = y - pred
                log_lik[idx, k] = np.log(self.pi[k] + 1e-12) - np.sum(
                    0.5 * np.log(2 * np.pi * (self.sigmas[k] ** 2))
                    + (res**2) / (2 * (self.sigmas[k] ** 2))
                )
        
        max_log = np.max(log_lik, axis=1, keepdims=True)
        lik = np.exp(log_lik - max_log)
        resp = lik / np.sum(lik, axis=1, keepdims=True)
        
        assigned_groups = np.argmax(resp, axis=1)
        group_prob = np.max(resp, axis=1)
        group_prob_spread = np.std(resp, axis=1)
        group_entropy = -np.sum(resp * np.log(resp + 1e-12), axis=1)
        group_df = pd.DataFrame(
            {
                id_col: subjects,
                "gbtm_group": assigned_groups,
                "group_prob": group_prob,
                "group_prob_spread": group_prob_spread,
                "group_entropy": group_entropy,
            }
        )
        return group_df


def _find_data_path():
    """Look in a few likely spots so this runs whether launched from repo
    root, a scripts/ subfolder, or right next to the CSV."""
    candidates = [
        "../data/processed/gbtm_input_long_format(500_genes).csv"
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

    print(f"2. Aggregating to infant-level data...")
    # Aggregate to one row per infant
    infant_df = df.groupby("infant_id").first().reset_index()[
        ["infant_id", "gestational age (weeks)", "birth weight (grams)", "bpd_group"]
    ]

    infant_df["bpd_target"] = infant_df["bpd_group"].astype(int)

    # X has only clinical features (not genes, not trajectories)
    feature_cols = ["gestational age (weeks)", "birth weight (grams)"]
    X = infant_df[feature_cols]
    y = infant_df["bpd_target"]

    class_names = ["No BPD", "Mild BPD", "Moderate BPD", "Severe BPD"]

    # Model settings
    N_PCA_COMPONENTS = 12  # Cap at 12 components (not 0.95 variance)
    N_GBTM_GROUPS = 4
    N_TOP_GENES = 20
    OUTER_FOLDS = 5
    RANDOM_SEARCH_ITER = 20  # Reduced from 40

    unique_classes = sorted(y.unique())
    active_class_names = [class_names[c] for c in unique_classes]
    
    print(f"   Class distribution: {y.value_counts().sort_index().to_dict()}")
    print(f"   Total infants: {len(infant_df)}")
    print(f"   Gene features: {len(gene_cols)}")

    gbtm_models_full = {}
    pca_full = None
    best_clf = None

    print(f"\n3. Evaluating Multi-class BPD Risk Classifier using stratified {OUTER_FOLDS}-fold...")
    skf = StratifiedKFold(n_splits=OUTER_FOLDS, shuffle=True, random_state=42)
    y_true, y_pred, y_probs = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"   Fold {fold_idx + 1}/{OUTER_FOLDS}...", end=" ", flush=True)
        
        X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Get training and test infant IDs
        train_infant_ids = infant_df.iloc[train_idx]["infant_id"].values
        test_infant_ids = infant_df.iloc[test_idx]["infant_id"].values
        
        # Filter df to training infants only
        df_train = df[df["infant_id"].isin(train_infant_ids)].copy()
        df_test = df[df["infant_id"].isin(test_infant_ids)].copy()
        
        # 1. Fit PCA on training data only
        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_train_imputed = imputer.fit_transform(df_train[gene_cols])
        X_test_imputed = imputer.transform(df_test[gene_cols])
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)

        pca = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        n_pcs = pca.n_components_
        print(f"PCs={n_pcs}; ", end="", flush=True)

        # 2. Fit GBTM on every retained PC using concat instead of loop assignment
        gbtm_train_features = {}
        gbtm_test_features = {}
        
        pc_dict_train = {f"pc{pc_idx + 1}": X_train_pca[:, pc_idx] for pc_idx in range(n_pcs)}
        pc_dict_test = {f"pc{pc_idx + 1}": X_test_pca[:, pc_idx] for pc_idx in range(n_pcs)}
        
        df_train = pd.concat([df_train.reset_index(drop=True), pd.DataFrame(pc_dict_train)], axis=1)
        df_test = pd.concat([df_test.reset_index(drop=True), pd.DataFrame(pc_dict_test)], axis=1)

        for pc_idx in range(n_pcs):
            pc = f"pc{pc_idx + 1}"

            gbtm = PureGBTM(n_groups=N_GBTM_GROUPS, degree=2)

            train_res, _ = gbtm.fit(
                df_train, id_col="infant_id",
                time_col="timepoint_code", val_col=pc
            )
            train_res.rename(columns={
                "gbtm_group": f"gbtm_group_{pc}",
                "group_prob": f"group_prob_{pc}",
                "group_prob_spread": f"group_prob_spread_{pc}",
                "group_entropy": f"group_entropy_{pc}",
            }, inplace=True)
            gbtm_train_features[pc] = train_res

            test_res = gbtm.predict_groups(
                df_test, id_col="infant_id",
                time_col="timepoint_code", val_col=pc
            )
            test_res.rename(columns={
                "gbtm_group": f"gbtm_group_{pc}",
                "group_prob": f"group_prob_{pc}",
                "group_prob_spread": f"group_prob_spread_{pc}",
                "group_entropy": f"group_entropy_{pc}",
            }, inplace=True)
            gbtm_test_features[pc] = test_res

        # 3. Merge GBTM features into X_train and X_test
        X_train = infant_df.iloc[train_idx][
            ["infant_id", "gestational age (weeks)", "birth weight (grams)"]
        ].copy()

        X_test = infant_df.iloc[test_idx][
            ["infant_id", "gestational age (weeks)", "birth weight (grams)"]
        ].copy()

        for pc_idx in range(n_pcs):
            pc = f"pc{pc_idx + 1}"
            X_train = X_train.merge(gbtm_train_features[pc], on="infant_id", how="left")
            X_test = X_test.merge(gbtm_test_features[pc], on="infant_id", how="left")

        # 4. Add top gene-level features selected using TRAINING labels only
        train_gene_by_infant = df_train.groupby("infant_id")[gene_cols].mean()
        test_gene_by_infant = df_test.groupby("infant_id")[gene_cols].mean()
        train_gene_by_infant = train_gene_by_infant.reindex(
            infant_df.iloc[train_idx]["infant_id"]
        )
        test_gene_by_infant = test_gene_by_infant.reindex(
            infant_df.iloc[test_idx]["infant_id"]
        )

        train_gene_filled = train_gene_by_infant.fillna(train_gene_by_infant.median())
        f_scores, _ = f_classif(train_gene_filled, y_train)
        f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
        n_select = min(N_TOP_GENES, len(gene_cols))
        selected_gene_indices = np.argsort(f_scores)[-n_select:]
        selected_genes = [gene_cols[i] for i in selected_gene_indices]

        # Use concat for genes too
        gene_dict_train = {f"gene_{gene}": train_gene_by_infant[gene].values for gene in selected_genes}
        gene_dict_test = {f"gene_{gene}": test_gene_by_infant[gene].values for gene in selected_genes}
        
        X_train = pd.concat([X_train, pd.DataFrame(gene_dict_train, index=X_train.index)], axis=1)
        X_test = pd.concat([X_test, pd.DataFrame(gene_dict_test, index=X_test.index)], axis=1)

        X_train = X_train.drop("infant_id", axis=1)
        X_test = X_test.drop("infant_id", axis=1)

        # 5. Random Forest (no SMOTE for medical data)
        param_grid = {
            "rf__n_estimators": [300, 500, 800],
            "rf__max_depth": [10, 15, 20],
            "rf__min_samples_split": [2, 5],
            "rf__min_samples_leaf": [1, 2],
            "rf__max_features": ["sqrt", "log2"],
            "rf__class_weight": ["balanced", "balanced_subsample"],
        }

        rf_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("rf", RandomForestClassifier(random_state=42, n_jobs=-1))
        ])

        clf = RandomizedSearchCV(
            rf_pipeline,
            param_distributions=param_grid,
            n_iter=RANDOM_SEARCH_ITER,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42
        )

        clf.fit(X_train, y_train)
        best_clf = clf  # ← Save for accessing best_params_ later

        # Collect predictions
        y_true.extend(y_test.values)
        y_pred.extend(clf.best_estimator_.predict(X_test))

        # Collect probabilities for ALL test samples
        probs = clf.best_estimator_.predict_proba(X_test)
        for pred_probs in probs:
            full_prob = np.zeros(len(unique_classes))
            for idx, c in enumerate(clf.classes_):
                full_prob[c] = pred_probs[idx]
            y_probs.append(full_prob)
        
        print("done")

    y_probs = np.array(y_probs)

    print("\n4. K-Fold Cross-Validation Results:")
    print("=" * 80)
    print(
        classification_report(
            y_true,
            y_pred,
            labels=unique_classes,
            target_names=active_class_names,
        )
    )
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print("Weighted F1:", f1_score(y_true, y_pred, average="weighted"))
    print("Balanced Accuracy:", balanced_accuracy_score(y_true, y_pred))
    print("Best hyperparameters:", best_clf.best_params_)
    print("=" * 80)

    # Fit GBTM/PCA on full data for visualization and final model
    print("\n5. Fitting GBTM/PCA on full dataset for final model...")
    
    imputer_full = SimpleImputer(strategy="median")
    scaler_full = StandardScaler()
    X_full_imputed = imputer_full.fit_transform(df[gene_cols])
    X_full_scaled = scaler_full.fit_transform(X_full_imputed)

    pca_full = PCA(n_components=N_PCA_COMPONENTS, random_state=42)
    pca_features_full = pca_full.fit_transform(X_full_scaled)
    n_pcs_full = pca_full.n_components_

    print(f"   Full-data PCA retained {n_pcs_full} PCs.")

    # Use concat for full data PCs
    pc_dict_full = {f"pc{i+1}": pca_features_full[:, i] for i in range(n_pcs_full)}
    df = pd.concat([df.reset_index(drop=True), pd.DataFrame(pc_dict_full)], axis=1)

    # Fit GBTM on full data
    X_final = infant_df[
        ["infant_id", "gestational age (weeks)", "birth weight (grams)"]
    ].copy()

    # Add top gene-level features for final model
    full_gene_by_infant = df.groupby("infant_id")[gene_cols].mean()
    full_gene_by_infant = full_gene_by_infant.reindex(infant_df["infant_id"])

    f_scores_full, _ = f_classif(
        full_gene_by_infant.fillna(full_gene_by_infant.median()), y
    )
    f_scores_full = np.nan_to_num(f_scores_full, nan=0.0, posinf=0.0, neginf=0.0)
    n_select_full = min(N_TOP_GENES, len(gene_cols))
    selected_gene_indices_full = np.argsort(f_scores_full)[-n_select_full:]
    selected_genes_full = [gene_cols[i] for i in selected_gene_indices_full]

    gene_dict_final = {f"gene_{gene}": full_gene_by_infant[gene].values for gene in selected_genes_full}
    X_final = pd.concat([X_final, pd.DataFrame(gene_dict_final, index=X_final.index)], axis=1)

    # Separate dataframe for longitudinal trajectory visualization
    df_plot = df.copy()

    for pc_idx in range(n_pcs_full):
        pc = f"pc{pc_idx + 1}"
        gbtm = PureGBTM(n_groups=N_GBTM_GROUPS, degree=2)

        res, _ = gbtm.fit(
            df,
            id_col="infant_id",
            time_col="timepoint_code",
            val_col=pc
        )

        res.rename(
            columns={
                "gbtm_group": f"gbtm_group_{pc}",
                "group_prob": f"group_prob_{pc}",
                "group_prob_spread": f"group_prob_spread_{pc}",
                "group_entropy": f"group_entropy_{pc}"
            },
            inplace=True
        )

        # Add GBTM results to classifier dataset
        X_final = X_final.merge(
            res,
            on="infant_id",
            how="left"
        )

        # Add GBTM results to longitudinal plotting dataset
        df_plot = df_plot.merge(
            res,
            on="infant_id",
            how="left"
        )

        gbtm_models_full[pc] = gbtm
    
    # Drop infant_id and reset index
    X_final = X_final.drop("infant_id", axis=1).reset_index(drop=True)
    
    print("6. Retraining final Risk Classifier on full dataset...")
    print(f"   Original class distribution: {y.value_counts().sort_index().to_dict()}")

    best_params = best_clf.best_params_
    
    final_clf = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=best_params["rf__n_estimators"],
            max_depth=best_params["rf__max_depth"],
            min_samples_split=best_params["rf__min_samples_split"],
            min_samples_leaf=best_params["rf__min_samples_leaf"],
            max_features=best_params["rf__max_features"],
            class_weight=best_params["rf__class_weight"],
            random_state=42,
            n_jobs=-1
        ))
    ])

    final_clf.fit(X_final, y)

    print("   Final model trained on full dataset.")
    
    print("\n7. Saving trained models...")
    os.makedirs("models", exist_ok=True)
    joblib.dump(gbtm_models_full, "models/gbtm_trajectory_models_dict.pkl")
    joblib.dump(final_clf, "models/bpd_risk_classifier.pkl")
    joblib.dump({"imputer": imputer_full, "scaler": scaler_full, "pca": pca_full}, "models/pca_transformer.pkl")
    joblib.dump(selected_genes_full, "models/selected_gene_features.pkl")
    print("   -> Saved gbtm_trajectory_models_dict.pkl")
    print("   -> Saved bpd_risk_classifier.pkl")
    print("   -> Saved pca_transformer.pkl")
    print("   -> Saved selected_gene_features.pkl")

    print("\n8. Generating multi-class visualization charts...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Longitudinal Trajectories (PC1)
    sns.lineplot(
        data=df_plot,
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

    # Plot 3: Confusion Matrix
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
    plot_path = "gbtm_multiclass_bpd_risk_optimized.png"
    plt.savefig(plot_path, dpi=300)
    print(f"   -> Saved 4-class evaluation plot to '{plot_path}'")
    plt.show()

    print("\n9. Model Summary:")
    print(f"   Number of gene features: {len(gene_cols)}")
    print(f"   PCA components retained: {n_pcs_full}")
    print(f"   GBTM groups per PC: {N_GBTM_GROUPS}")
    print(f"   Top genes selected: {len(selected_genes_full)}")
    print(f"   PCA explained variance: {pca_full.explained_variance_ratio_}")
    print(f"   Total variance explained: {pca_full.explained_variance_ratio_.sum():.4f}")
    print(f"   Missing gene values in full data: {df[gene_cols].isna().sum().sum()}")
    print(f"   Expression shape: {df[gene_cols].shape}")


if __name__ == "__main__":
    main()