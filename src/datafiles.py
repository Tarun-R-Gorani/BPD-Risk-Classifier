import pandas as pd
from pathlib import Path


# -----------------------------
# File paths
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

expression_file = BASE_DIR / "data" / "processed" / "GSE32472_expression.csv"
metadata_file = BASE_DIR / "data" / "processed" / "GSE32472_full_metadata.csv"


# -----------------------------
# Load datasets
# -----------------------------

expression = pd.read_csv(expression_file)
metadata = pd.read_csv(metadata_file)


# -----------------------------
# Basic information
# -----------------------------

print("\n========== EXPRESSION DATA ==========")
print("Shape:", expression.shape)
print("First 5 columns:", expression.columns[:5].tolist())
print("First 5 rows:")
print(expression.head())


print("\n========== METADATA ==========")
print("Shape:", metadata.shape)
print("Columns:")
print(metadata.columns.tolist())

print("\nFirst 5 rows:")
print(metadata.head())


# -----------------------------
# Check GSM IDs
# -----------------------------

expression_samples = set(expression.columns[1:])
metadata_samples = set(metadata["geo_accession"].dropna())


print("\n========== GSM MATCHING ==========")

print("Number of expression samples:", len(expression_samples))
print("Number of metadata samples:", len(metadata_samples))

common_samples = expression_samples.intersection(metadata_samples)

print("Common GSM samples:", len(common_samples))

missing_from_expression = metadata_samples - expression_samples
missing_from_metadata = expression_samples - metadata_samples

print("\nMetadata samples NOT found in expression data:")
print(sorted(missing_from_expression))

print("\nExpression samples NOT found in metadata:")
print(sorted(missing_from_metadata))


# -----------------------------
# Duplicate GSM checks
# -----------------------------

print("\n========== DUPLICATE CHECK ==========")

duplicate_metadata = metadata[
    metadata["geo_accession"].duplicated(keep=False)
]

if duplicate_metadata.empty:
    print("No duplicate GSM IDs in metadata.")
else:
    print("Duplicate GSM IDs found:")
    print(duplicate_metadata[["geo_accession"]])


# -----------------------------
# Final summary
# -----------------------------

print("\n========== SUMMARY ==========")

if len(common_samples) == len(expression_samples) == len(metadata_samples):
    print("✓ All GSM samples match between the two datasets.")
else:
    print("⚠ GSM mismatch detected.")
    print("Investigate the missing samples before proceeding.")