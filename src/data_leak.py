import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    auc,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_curve,
)

sns.set_theme(style="whitegrid")


# ============================================================================
# 1. PURE GBTM
# ============================================================================

class PureGBTM:
    """
    Custom Group-Based Trajectory Model using:
        - Polynomial trajectories
        - EM-like responsibility updates
        - Weighted Least Squares

    fit()     -> learns trajectory groups from training infants
    predict() -> assigns NEW/unseen infants to the learned groups
    """

    def __init__(self, n_groups=3, degree=2, max_iter=100, tol=1e-4):
        self.n_groups = n_groups
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol

        self.group_params = []
        self.sigmas = []
        self.pi = None

    def _build_design_matrix(self, timepoints):
        return np.column_stack(
            [timepoints ** d for d in range(self.degree + 1)]
        )

    # ------------------------------------------------------------------------
    # FIT GBTM ON TRAINING DATA ONLY
    # ------------------------------------------------------------------------

    def fit(self, df, id_col, time_col, val_col):

        subjects = df[id_col].unique()
        N = len(subjects)

        self.pi = np.full(
            self.n_groups,
            1.0 / self.n_groups
        )

        # Store longitudinal data for each infant
        subj_data = {}

        for s in subjects:

            s_df = (
                df[df[id_col] == s]
                .sort_values(time_col)
            )

            t = s_df[time_col].values
            y = s_df[val_col].values

            X = self._build_design_matrix(t)

            subj_data[s] = (X, y)

        # Initial responsibilities
        np.random.seed(42)

        resp = np.random.dirichlet(
            np.ones(self.n_groups),
            size=N
        )

        # EM iterations
        for iteration in range(self.max_iter):

            old_pi = self.pi.copy()

            self.group_params = []
            self.sigmas = []

            # --------------------------------------------------------------
            # M STEP
            # --------------------------------------------------------------

            for k in range(self.n_groups):

                weights = resp[:, k]

                X_all = []
                y_all = []
                w_all = []

                for idx, s in enumerate(subjects):

                    X_s, y_s = subj_data[s]

                    X_all.append(X_s)
                    y_all.append(y_s)

                    w_all.append(
                        np.full(
                            len(y_s),
                            weights[idx]
                        )
                    )

                X_mat = np.vstack(X_all)
                y_vec = np.concatenate(y_all)
                w_vec = np.concatenate(w_all)

                # Weighted least squares
                wls_model = sm.WLS(
                    y_vec,
                    X_mat,
                    weights=w_vec + 1e-6
                ).fit()

                beta = wls_model.params

                residuals = (
                    y_vec -
                    X_mat @ beta
                )

                sigma = np.sqrt(
                    np.sum(
                        w_vec * residuals ** 2
                    )
                    /
                    np.sum(w_vec)
                    + 1e-6
                )

                self.group_params.append(beta)
                self.sigmas.append(sigma)

            # Update group proportions
            self.pi = resp.mean(axis=0)

            # --------------------------------------------------------------
            # E STEP
            # --------------------------------------------------------------

            log_lik = np.zeros(
                (N, self.n_groups)
            )

            for idx, s in enumerate(subjects):

                X_s, y_s = subj_data[s]

                for k in range(self.n_groups):

                    pred = (
                        X_s @
                        self.group_params[k]
                    )

                    residuals = (
                        y_s - pred
                    )

                    log_lik[idx, k] = (
                        np.log(
                            self.pi[k] + 1e-12
                        )
                        -
                        np.sum(
                            0.5 *
                            np.log(
                                2 *
                                np.pi *
                                (
                                    self.sigmas[k] ** 2
                                )
                            )
                            +
                            (
                                residuals ** 2
                            )
                            /
                            (
                                2 *
                                self.sigmas[k] ** 2
                            )
                        )
                    )

            # Numerical stability
            max_log = np.max(
                log_lik,
                axis=1,
                keepdims=True
            )

            likelihood = np.exp(
                log_lik - max_log
            )

            resp = (
                likelihood /
                np.sum(
                    likelihood,
                    axis=1,
                    keepdims=True
                )
            )

            # Convergence
            if (
                np.max(
                    np.abs(
                        self.pi - old_pi
                    )
                )
                <
                self.tol
            ):
                break

        # Final training assignments
        assigned_groups = np.argmax(
            resp,
            axis=1
        )

        group_df = pd.DataFrame(
            {
                id_col: subjects,
                "gbtm_group": assigned_groups,
                "group_prob": np.max(
                    resp,
                    axis=1
                ),
            }
        )

        return group_df, resp

    # ------------------------------------------------------------------------
    # PREDICT GBTM GROUP FOR NEW INFANTS
    # ------------------------------------------------------------------------

    def predict(
        self,
        df,
        id_col,
        time_col,
        val_col
    ):
        """
        Assign NEW infants to the trajectory groups learned during fit().

        IMPORTANT:
        This method does NOT refit the GBTM.
        """

        if not self.group_params:
            raise RuntimeError(
                "GBTM must be fitted before calling predict()."
            )

        subjects = df[id_col].unique()

        results = []

        for s in subjects:

            s_df = (
                df[df[id_col] == s]
                .sort_values(time_col)
            )

            t = s_df[time_col].values
            y = s_df[val_col].values

            X = self._build_design_matrix(t)

            log_lik = np.zeros(
                self.n_groups
            )

            for k in range(self.n_groups):

                pred = (
                    X @
                    self.group_params[k]
                )

                residuals = (
                    y - pred
                )

                log_lik[k] = (
                    np.log(
                        self.pi[k] + 1e-12
                    )
                    -
                    np.sum(
                        0.5 *
                        np.log(
                            2 *
                            np.pi *
                            (
                                self.sigmas[k] ** 2
                            )
                        )
                        +
                        (
                            residuals ** 2
                        )
                        /
                        (
                            2 *
                            self.sigmas[k] ** 2
                        )
                    )
                )

            # Convert log likelihoods into probabilities
            max_log = np.max(log_lik)

            probabilities = np.exp(
                log_lik - max_log
            )

            probabilities = (
                probabilities /
                probabilities.sum()
            )

            assigned_group = np.argmax(
                probabilities
            )

            results.append(
                {
                    id_col: s,
                    "gbtm_group": assigned_group,
                    "group_prob": probabilities[
                        assigned_group
                    ],
                }
            )

        return pd.DataFrame(results)


# ============================================================================
# 2. FIND DATASET
# ============================================================================

def _find_data_path():

    candidates = [
        "../data/processed/gbtm_input_long_format(500_genes).csv"
    ]

    for path in candidates:

        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Could not find the GBTM input CSV."
    )


# ============================================================================
# 3. CREATE INFANT-LEVEL CLINICAL DATA
# ============================================================================

def get_infant_clinical_data(df):

    infant_df = (
        df.groupby("infant_id")
        .first()
        .reset_index()
        [
            [
                "infant_id",
                "gestational age (weeks)",
                "birth weight (grams)",
                "bpd_group",
            ]
        ]
    )

    infant_df["bpd_target"] = (
        infant_df["bpd_group"]
        .astype(int)
    )

    return infant_df


# ============================================================================
# 4. CREATE GBTM FEATURES
# ============================================================================

def create_trajectory_features(
    df_train,
    df_test,
    pcs
):
    """
    Fit GBTM on TRAINING infants.

    Then use the fitted GBTMs to predict trajectory
    membership for both training and test infants.
    """

    train_features = pd.DataFrame(
        {
            "infant_id":
                df_train["infant_id"]
                .unique()
        }
    )

    test_features = pd.DataFrame(
        {
            "infant_id":
                df_test["infant_id"]
                .unique()
        }
    )

    gbtm_models = {}

    for pc in pcs:

        print(
            f"      Fitting GBTM for {pc}..."
        )

        gbtm = PureGBTM(
            n_groups=3,
            degree=2
        )

        # --------------------------------------------------------------
        # FIT ONLY ON TRAINING INFANTS
        # --------------------------------------------------------------

        train_res, _ = gbtm.fit(
            df_train,
            id_col="infant_id",
            time_col="timepoint_code",
            val_col=pc
        )

        # --------------------------------------------------------------
        # ASSIGN TRAINING INFANTS
        # --------------------------------------------------------------

        train_res = train_res.rename(
            columns={
                "gbtm_group":
                    f"gbtm_group_{pc}",
                "group_prob":
                    f"group_prob_{pc}",
            }
        )

        # --------------------------------------------------------------
        # ASSIGN UNSEEN TEST INFANTS
        # USING THE TRAINED GBTM
        # --------------------------------------------------------------

        test_res = gbtm.predict(
            df_test,
            id_col="infant_id",
            time_col="timepoint_code",
            val_col=pc
        )

        test_res = test_res.rename(
            columns={
                "gbtm_group":
                    f"gbtm_group_{pc}",
                "group_prob":
                    f"group_prob_{pc}",
            }
        )

        # Merge
        train_features = train_features.merge(
            train_res,
            on="infant_id",
            how="left"
        )

        test_features = test_features.merge(
            test_res,
            on="infant_id",
            how="left"
        )

        gbtm_models[pc] = gbtm

    return (
        train_features,
        test_features,
        gbtm_models
    )


# ============================================================================
# 5. MAIN
# ============================================================================

def main():

    # ------------------------------------------------------------------------
    # LOAD DATA
    # ------------------------------------------------------------------------

    print("1. Loading long-format dataset...")

    data_path = _find_data_path()

    print(
        f"   -> Using: {data_path}"
    )

    df = pd.read_csv(
        data_path
    )

    print(
        "   Expression dataset shape:",
        df.shape
    )

    # ------------------------------------------------------------------------
    # GENE COLUMNS
    # ------------------------------------------------------------------------

    metadata_cols = [
        "infant_id",
        "timepoint_code",
        "days of life",
        "gestational age (weeks)",
        "birth weight (grams)",
        "bpd_group",
    ]

    gene_cols = [
        c for c in df.columns
        if c not in metadata_cols
    ]

    print(
        "   Number of gene features:",
        len(gene_cols)
    )

    print(
        "   Missing gene values:",
        df[gene_cols]
        .isna()
        .sum()
        .sum()
    )

    # ------------------------------------------------------------------------
    # INFANT-LEVEL DATA
    # ------------------------------------------------------------------------

    infant_df = get_infant_clinical_data(
        df
    )

    class_names = [
        "No BPD",
        "Mild BPD",
        "Moderate BPD",
        "Severe BPD",
    ]

    unique_classes = sorted(
        infant_df["bpd_target"]
        .unique()
    )

    active_class_names = [
        class_names[c]
        for c in unique_classes
    ]

    print(
        "\n2. Infant-level cohort:"
    )

    print(
        "   Number of infants:",
        len(infant_df)
    )

    print(
        "   Class distribution:",
        infant_df[
            "bpd_target"
        ]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # =========================================================================
    # OUTER STRATIFIED CV
    # =========================================================================

    print(
        "\n3. Starting LEAKAGE-FREE 7-FOLD CV..."
    )

    # X and y here are only used to create infant-level splits.
    X_dummy = infant_df[
        ["infant_id"]
    ]

    y = infant_df[
        "bpd_target"
    ]

    skf = StratifiedKFold(
        n_splits=7,
        shuffle=True,
        random_state=42
    )

    # Store out-of-fold predictions
    y_true = []
    y_pred = []
    y_probs = []

    fold_number = 1

    pcs = [
        "pc1",
        "pc2",
        "pc3"
    ]

    # =========================================================================
    # OUTER CV LOOP
    # =========================================================================

    for train_idx, test_idx in skf.split(
        X_dummy,
        y
    ):

        print(
            f"\n--------------------------------------------------"
        )

        print(
            f"FOLD {fold_number}"
        )

        print(
            f"--------------------------------------------------"
        )

        # --------------------------------------------------------------------
        # 1. IDENTIFY TRAINING AND TEST INFANTS
        # --------------------------------------------------------------------

        train_infants = infant_df.iloc[
            train_idx
        ]["infant_id"].values

        test_infants = infant_df.iloc[
            test_idx
        ]["infant_id"].values

        print(
            "Training infants:",
            len(train_infants)
        )

        print(
            "Test infants:",
            len(test_infants)
        )

        # --------------------------------------------------------------------
        # 2. SPLIT LONGITUDINAL DATA BY INFANT
        # --------------------------------------------------------------------

        df_train = df[
            df["infant_id"]
            .isin(train_infants)
        ].copy()

        df_test = df[
            df["infant_id"]
            .isin(test_infants)
        ].copy()

        print(
            "Training longitudinal rows:",
            len(df_train)
        )

        print(
            "Test longitudinal rows:",
            len(df_test)
        )

        # --------------------------------------------------------------------
        # 3. FIT SCALER ONLY ON TRAINING DATA
        # --------------------------------------------------------------------

        print(
            "   Fitting StandardScaler on training data..."
        )

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(
            df_train[gene_cols]
        )

        X_test_scaled = scaler.transform(
            df_test[gene_cols]
        )

        # --------------------------------------------------------------------
        # 4. FIT PCA ONLY ON TRAINING DATA
        # --------------------------------------------------------------------

        print(
            "   Fitting PCA on training data..."
        )

        pca = PCA(
            n_components=3,
            random_state=42
        )

        train_pcs = pca.fit_transform(
            X_train_scaled
        )

        test_pcs = pca.transform(
            X_test_scaled
        )

        print(
            "   Fold PCA variance:",
            pca.explained_variance_ratio_
        )

        print(
            "   Fold total variance:",
            pca.explained_variance_ratio_.sum()
        )

        # --------------------------------------------------------------------
        # 5. ADD PCA SCORES
        # --------------------------------------------------------------------

        for i, pc in enumerate(pcs):

            df_train[pc] = (
                train_pcs[:, i]
            )

            df_test[pc] = (
                test_pcs[:, i]
            )

        # --------------------------------------------------------------------
        # 6. FIT GBTM ONLY ON TRAINING INFANTS
        # --------------------------------------------------------------------

        print(
            "   Fitting GBTM using training infants only..."
        )

        (
            train_trajectory,
            test_trajectory,
            fold_gbtm_models
        ) = create_trajectory_features(
            df_train,
            df_test,
            pcs
        )

        # --------------------------------------------------------------------
        # 7. ADD CLINICAL FEATURES
        # --------------------------------------------------------------------

        clinical_cols = [
            "gestational age (weeks)",
            "birth weight (grams)",
        ]

        train_clinical = infant_df[
            infant_df["infant_id"]
            .isin(train_infants)
        ][
            ["infant_id"] +
            clinical_cols
        ].copy()

        test_clinical = infant_df[
            infant_df["infant_id"]
            .isin(test_infants)
        ][
            ["infant_id"] +
            clinical_cols
        ].copy()

        # --------------------------------------------------------------------
        # 8. MERGE TRAJECTORY + CLINICAL FEATURES
        # --------------------------------------------------------------------

        train_features = train_trajectory.merge(
            train_clinical,
            on="infant_id",
            how="left"
        )

        test_features = test_trajectory.merge(
            test_clinical,
            on="infant_id",
            how="left"
        )

        # --------------------------------------------------------------------
        # 9. DEFINE FINAL CLASSIFIER FEATURES
        # --------------------------------------------------------------------

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

        X_train = train_features[
            feature_cols
        ]

        X_test = test_features[
            feature_cols
        ]

        y_train = infant_df[
            infant_df["infant_id"]
            .isin(train_infants)
        ]["bpd_target"].values

        y_test = infant_df[
            infant_df["infant_id"]
            .isin(test_infants)
        ]["bpd_target"].values

        # --------------------------------------------------------------------
        # 10. SVM HYPERPARAMETER SEARCH
        # --------------------------------------------------------------------

        print(
            "   Tuning polynomial SVM..."
        )

        param_distributions = {

            "svm__C": [
                0.01,
                0.1,
                1,
                10,
                100,
                1000,
            ],

            "svm__gamma": [
                "scale",
                "auto",
                0.001,
                0.01,
                0.1,
                1,
            ],

            "svm__degree": [
                2,
                3,
                4,
            ],

            "svm__coef0": [
                0,
                0.1,
                1,
            ],
        }

        svm_pipeline = Pipeline(
            [
                (
                    "scaler",
                    StandardScaler()
                ),

                (
                    "svm",
                    SVC(
                        kernel="poly",
                        class_weight="balanced",
                        probability=True,
                        random_state=42,
                    )
                ),
            ]
        )

        # --------------------------------------------------------------------
        # IMPORTANT:
        # Internal CV operates ONLY on the OUTER TRAINING DATA.
        # --------------------------------------------------------------------

        clf = RandomizedSearchCV(
            svm_pipeline,
            param_distributions=
                param_distributions,
            n_iter=15,
            cv=3,
            scoring="f1_macro",
            n_jobs=-1,
            random_state=42,
        )

        clf.fit(
            X_train,
            y_train
        )

        print(
            "   Best parameters:",
            clf.best_params_
        )

        # --------------------------------------------------------------------
        # 11. PREDICT THE HELD-OUT INFANTS
        # --------------------------------------------------------------------

        fold_predictions = (
            clf.best_estimator_
            .predict(X_test)
        )

        fold_probabilities = (
            clf.best_estimator_
            .predict_proba(X_test)
        )

        y_true.extend(
            y_test
        )

        y_pred.extend(
            fold_predictions
        )

        # Convert probability columns into
        # fixed class ordering
        for pred_probs in fold_probabilities:

            full_prob = np.zeros(
                len(unique_classes)
            )

            for idx, c in enumerate(
                clf.classes_
            ):

                full_prob[c] = (
                    pred_probs[idx]
                )

            y_probs.append(
                full_prob
            )

        print(
            "   Fold accuracy:",
            accuracy_score(
                y_test,
                fold_predictions
            )
        )

        fold_number += 1

    # =========================================================================
    # FINAL OUT-OF-FOLD RESULTS
    # =========================================================================

    y_true = np.array(
        y_true
    )

    y_pred = np.array(
        y_pred
    )

    y_probs = np.array(
        y_probs
    )

    print(
        "\n============================================================"
    )

    print(
        "LEAKAGE-FREE 4-CLASS BPD EVALUATION"
    )

    print(
        "============================================================"
    )

    print(
        classification_report(
            y_true,
            y_pred,
            labels=unique_classes,
            target_names=active_class_names,
            zero_division=0,
        )
    )

    print(
        "Accuracy:",
        accuracy_score(
            y_true,
            y_pred
        )
    )

    print(
        "Macro F1:",
        f1_score(
            y_true,
            y_pred,
            average="macro"
        )
    )

    print(
        "Weighted F1:",
        f1_score(
            y_true,
            y_pred,
            average="weighted"
        )
    )

    print(
        "Balanced Accuracy:",
        balanced_accuracy_score(
            y_true,
            y_pred
        )
    )

    # =========================================================================
    # ROC-AUC
    # =========================================================================

    print(
        "\n========== ONE-vs-REST ROC-AUC =========="
    )

    y_bin = label_binarize(
        y_true,
        classes=unique_classes
    )

    roc_auc_values = {}

    for i, c in enumerate(
        unique_classes
    ):

        fpr, tpr, _ = roc_curve(
            y_bin[:, i],
            y_probs[:, i]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        roc_auc_values[c] = roc_auc

        print(
            f"{class_names[c]} AUC: "
            f"{roc_auc:.3f}"
        )

    # =========================================================================
    # CONFUSION MATRIX
    # =========================================================================

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=unique_classes
    )

    print(
        "\nConfusion Matrix:"
    )

    print(cm)

    # =========================================================================
    # FINAL MODEL
    #
    # Now that unbiased CV is finished, we can train the final model
    # using ALL 106 infants.
    # =========================================================================

    print(
        "\n============================================================"
    )

    print(
        "FINAL MODEL TRAINING ON ALL INFANTS"
    )

    print(
        "============================================================"
    )

    # ------------------------------------------------------------------------
    # FINAL SCALER
    # ------------------------------------------------------------------------

    final_scaler = StandardScaler()

    X_all_scaled = (
        final_scaler.fit_transform(
            df[gene_cols]
        )
    )

    # ------------------------------------------------------------------------
    # FINAL PCA
    # ------------------------------------------------------------------------

    final_pca = PCA(
        n_components=3,
        random_state=42
    )

    all_pcs = (
        final_pca.fit_transform(
            X_all_scaled
        )
    )

    for i, pc in enumerate(pcs):

        df[pc] = all_pcs[:, i]

    print(
        "Final PCA explained variance:",
        final_pca.explained_variance_ratio_
    )

    print(
        "Final total variance explained:",
        final_pca.explained_variance_ratio_.sum()
    )

    # ------------------------------------------------------------------------
    # FINAL GBTM
    # ------------------------------------------------------------------------

    final_gbtm_models = {}

    final_trajectory = pd.DataFrame(
        {
            "infant_id":
                df["infant_id"]
                .unique()
        }
    )

    for pc in pcs:

        print(
            f"   Fitting final GBTM for {pc}..."
        )

        gbtm = PureGBTM(
            n_groups=3,
            degree=2
        )

        res, _ = gbtm.fit(
            df,
            id_col="infant_id",
            time_col="timepoint_code",
            val_col=pc
        )

        res = res.rename(
            columns={
                "gbtm_group":
                    f"gbtm_group_{pc}",
                "group_prob":
                    f"group_prob_{pc}",
            }
        )

        final_trajectory = (
            final_trajectory
            .merge(
                res,
                on="infant_id",
                how="left"
            )
        )

        final_gbtm_models[pc] = gbtm

    # ------------------------------------------------------------------------
    # FINAL INFANT FEATURES
    # ------------------------------------------------------------------------

    final_infant_df = (
        final_trajectory
        .merge(
            infant_df[
                [
                    "infant_id",
                    "gestational age (weeks)",
                    "birth weight (grams)",
                    "bpd_group",
                    "bpd_target",
                ]
            ],
            on="infant_id",
            how="left"
        )
    )

    X_final = final_infant_df[
        feature_cols
    ]

    y_final = final_infant_df[
        "bpd_target"
    ]

    # ------------------------------------------------------------------------
    # FINAL SVM
    #
    # Use the best polynomial configuration found during CV.
    # ------------------------------------------------------------------------

    final_clf = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),

            (
                "svm",
                CalibratedClassifierCV(
                    SVC(
                        kernel="poly",
                        C=1,
                        gamma="scale",
                        degree=2,
                        coef0=0.1,
                        class_weight="balanced",
                        random_state=42,
                    ),
                    ensemble=False,
                )
            ),
        ]
    )

    final_clf.fit(
        X_final,
        y_final
    )

    # =========================================================================
    # SAVE MODELS
    # =========================================================================

    print(
        "\n6. Saving final models..."
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        final_scaler,
        "models/gene_scaler.pkl"
    )

    joblib.dump(
        final_pca,
        "models/pca_transformer.pkl"
    )

    joblib.dump(
        final_gbtm_models,
        "models/gbtm_trajectory_models_dict.pkl"
    )

    joblib.dump(
        final_clf,
        "models/bpd_risk_classifier.pkl"
    )

    print(
        "   -> Saved gene_scaler.pkl"
    )

    print(
        "   -> Saved pca_transformer.pkl"
    )

    print(
        "   -> Saved gbtm_trajectory_models_dict.pkl"
    )

    print(
        "   -> Saved bpd_risk_classifier.pkl"
    )

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    print(
        "\n7. Generating evaluation plots..."
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    # ------------------------------------------------------------------------
    # PLOT 1: GBTM TRAJECTORIES
    # ------------------------------------------------------------------------

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

    axes[0].set_title(
        "Primary GBTM Trajectory Classes (PC1)",
        fontweight="bold"
    )

    axes[0].set_xlabel(
        "Timepoint Visit"
    )

    axes[0].set_ylabel(
        "Gene Trajectory Signal (PC1)"
    )

    # ------------------------------------------------------------------------
    # PLOT 2: ROC
    # ------------------------------------------------------------------------

    colors = [
        "navy",
        "green",
        "orange",
        "darkred",
    ]

    for i, c in enumerate(
        unique_classes
    ):

        fpr, tpr, _ = roc_curve(
            y_bin[:, i],
            y_probs[:, i]
        )

        axes[1].plot(
            fpr,
            tpr,
            color=colors[
                c % len(colors)
            ],
            lw=2,
            label=(
                f"{class_names[c]} "
                f"(AUC = "
                f"{roc_auc_values[c]:.2f})"
            ),
        )

    axes[1].plot(
        [0, 1],
        [0, 1],
        color="gray",
        lw=1,
        linestyle="--"
    )

    axes[1].set_title(
        "Multiclass BPD Risk ROC Curves",
        fontweight="bold"
    )

    axes[1].set_xlabel(
        "False Positive Rate"
    )

    axes[1].set_ylabel(
        "True Positive Rate"
    )

    axes[1].legend(
        loc="lower right"
    )

    # ------------------------------------------------------------------------
    # PLOT 3: CONFUSION MATRIX
    # ------------------------------------------------------------------------

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
        "4-Class Risk Classification Matrix",
        fontweight="bold"
    )

    axes[2].set_xlabel(
        "Predicted BPD Grade"
    )

    axes[2].set_ylabel(
        "Actual BPD Grade"
    )

    plt.tight_layout()

    plot_path = (
        "gbtm_multiclass_bpd_risk_"
        "3pcs_leakage_free.png"
    )

    plt.savefig(
        plot_path,
        dpi=300
    )

    print(
        f"   -> Saved plot to '{plot_path}'"
    )

    plt.show()


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
