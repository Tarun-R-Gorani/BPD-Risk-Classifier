import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)


class BPDDataLoader:
    """
    Loads and prepares the GSE32472 Day-5 dataset.
    """

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        """Load CSV file."""
        self.data = pd.read_csv(self.file_path)

        print(f"Dataset loaded.")
        print(f"Shape: {self.data.shape}")

        return self.data

    def inspect_data(self):
        """Display basic information about the dataset."""
        print("\nColumns:")
        print(self.data.columns[:15].tolist())

        print("\nBPD group values:")
        print(
            self.data[
                "bronchopulmonary dysplasia (bpd) group"
            ].value_counts(dropna=False)
        )

    def prepare_features_and_labels(self):
        """
        Separate metadata, gene-expression features,
        and target label.
        """

        target_column = "bronchopulmonary dysplasia (bpd) group"

        # Clinical/metadata columns
        metadata_columns = [
            "title",
            "geo_accession",
            "source_name_ch1",
            "days of life",
            "gender",
            "gestational age (weeks)",
            "birth weight (grams)",
            "oxygen 28 days",
            "oxygen 36 weeks postmenstrual age (wpma)",
            target_column
        ]

        # Keep only columns that actually exist
        metadata_columns = [
            col for col in metadata_columns
            if col in self.data.columns
        ]

        # Target
        y = self.data[target_column].copy()

        # Gene-expression columns
        gene_columns = [
            col for col in self.data.columns
            if col not in metadata_columns
        ]

        X = self.data[gene_columns].copy()

        # Make sure expression values are numeric
        X = X.apply(pd.to_numeric, errors="coerce")

        y = self.data["bronchopulmonary dysplasia (bpd) group"].copy()

        # Convert 4 BPD categories into binary classes
        y = y.apply(lambda x: 0 if x == 0 else 1)

        print("\nFeature matrix shape:", X.shape)
        print("Target shape:", y.shape)

        print("\nTarget distribution:")
        print(y.value_counts(dropna=False))

        return X, y


class BPDFeatureSelector:
    """
    Selects the most statistically relevant gene-expression
    features using ANOVA F-test.
    """

    def __init__(self, k=100):
        self.k = k
        self.selector = SelectKBest(
            score_func=f_classif,
            k=k
        )

    def fit_transform(self, X, y):
        X_selected = self.selector.fit_transform(X, y)

        selected_features = X.columns[
            self.selector.get_support()
        ]

        print(f"\nSelected {len(selected_features)} genes.")

        return pd.DataFrame(
            X_selected,
            columns=selected_features,
            index=X.index
        )

    def transform(self, X):
        X_selected = self.selector.transform(X)

        selected_features = X.columns[
            self.selector.get_support()
        ]

        return pd.DataFrame(
            X_selected,
            columns=selected_features,
            index=X.index
        )


class BPDClassifier:
    """
    SVC classifier for BPD vs No-BPD prediction.
    """

    def __init__(self):
        self.model = Pipeline([
            ("scaler", StandardScaler()),

            ("svc", SVC(
                kernel="linear",
                C=1.0,

            ))
        ])

        self.cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        self.param_grid = {
            "svc__C": [0.1, 1.0, 10.0, 100.0],
        }

    def train(self, X_train, y_train):

        self.grid = GridSearchCV(
            estimator=self.model,
            param_grid=self.param_grid,
            cv=self.cv,
            scoring="f1_macro",
            n_jobs=-1,
            verbose=1
        )

        self.grid.fit(X_train, y_train)

        # Store the best model
        self.model = self.grid.best_estimator_

        print("\nGrid Search completed.")
        print("Best parameters:")
        print(self.grid.best_params_)

        print("\nBest CV Macro F1:")
        print(self.grid.best_score_)

    def predict(self, X_test):

        return self.model.predict(X_test)

    def evaluate(self, X_test, X_train, y_test, y_train):
        predictions = self.predict(X_test)

        print("\n========== MODEL RESULTS ==========")

        print(
            "Accuracy:",
            accuracy_score(y_test, predictions)
        )

        print(
            "Macro F1:",
            f1_score(
                y_test,
                predictions,
                average="macro"
            )
        )

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                predictions
            )
        )

        print("\nConfusion Matrix:")
        print(
            confusion_matrix(
                y_test,
                predictions
            )
        )

        print("Cross val score:")
        scores = cross_val_score(
        self.model,
        X_train,
        y_train,
        cv=self.cv,
        scoring="f1_macro"
        )
        print(scores)


def main():

    # ==========================================
    # 1. LOAD DATA
    # ==========================================

    file_path = "../data/processed/filtered_genes.csv"

    loader = BPDDataLoader(file_path)

    loader.load_data()
    loader.inspect_data()

    X, y = loader.prepare_features_and_labels()


    # ==========================================
    # 2. REMOVE MISSING VALUES
    # ==========================================

    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    # Replace infinite values
    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove genes containing missing values
    valid_features = X.columns[
        X.notna().all()
    ]

    X = X[valid_features]


    # ==========================================
    # 3. TRAIN / TEST SPLIT
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])


    # ==========================================
    # 4. FEATURE SELECTION
    # ==========================================

    selector = BPDFeatureSelector(k=100)

    X_train_selected = selector.fit_transform(
        X_train,
        y_train
    )

    X_test_selected = selector.transform(
        X_test
    )


    # ==========================================
    # 5. TRAIN CLASSIFIER
    # ==========================================

    classifier = BPDClassifier()

    classifier.train(
        X_train_selected,
        y_train
    )


    # ==========================================
    # 6. EVALUATE
    # ==========================================

    classifier.evaluate(
        X_test_selected,
        X_train_selected,
        y_test,
        y_train
    )
    # Get the trained SVC inside the pipeline
    svc = classifier.model.named_steps["svc"]

    # Get coefficients for the binary classifier
    coefficients = svc.coef_[0]

    # Get selected gene names
    feature_names = X_train_selected.columns

    # Create gene importance table
    biomarkers = pd.DataFrame({
        "gene": feature_names,
        "coefficient": coefficients,
        "abs_coefficient": np.abs(coefficients)
    })

    # Rank by absolute coefficient
    biomarkers = biomarkers.sort_values(
        "abs_coefficient",
        ascending=False
    )

    print("\nTop 20 predictive genes:")
    print(biomarkers.head(20).to_string(index=False))

if __name__ == "__main__":
    main()