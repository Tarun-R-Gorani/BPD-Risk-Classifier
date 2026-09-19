import pandas as pd
import numpy as np
# ============================================================
# 1. LOAD FILES
# ============================================================

expression_path = "../data/processed/GSE32472_expression_new.csv"
metadata_path = "../data/processed/GSE32472_full_metadata.csv"

expression = pd.read_csv(expression_path)
metadata = pd.read_csv(metadata_path)

# ============================================================
# 2. KEEP ONLY DAY-5 SAMPLES
# ============================================================

day5_metadata = metadata[
    metadata["days of life"].isin([4, 5, 6])
].copy()

print("Day-5 samples:", len(day5_metadata))

print(day5_metadata["bronchopulmonary dysplasia (bpd) group"].value_counts())

# ============================================================
# 3. GET GSM IDs
# ============================================================

day5_gsm = day5_metadata["geo_accession"].astype(str).tolist()

print("\nNumber of Day-5 samples:", len(day5_gsm))
print("First GSMs:", day5_gsm[:10])

# ============================================================
# 4. CHECK BPD DISTRIBUTION
# ============================================================

bpd_col = "bronchopulmonary dysplasia (bpd) group"

print("\nBPD distribution:")
print(day5_metadata[bpd_col].value_counts().sort_index())

# ============================================================
# 5. GET GSM IDs
# ============================================================

day5_gsm = day5_metadata["geo_accession"].astype(str).tolist()
print("\nFirst Day-5 GSM IDs:")
print(day5_gsm[:10])
#============================================================
# 6. CHECK THAT GSM IDs EXIST IN EXPRESSION FILE
# ============================================================

expression_gsm = set(expression.columns)

available_gsm = [
    gsm for gsm in day5_gsm
    if gsm in expression_gsm
]

missing_gsm = [
    gsm for gsm in day5_gsm
    if gsm not in expression_gsm
]

print("\nDay-5 GSMs:", len(day5_gsm))
print("Found in expression:", len(available_gsm))
print("Missing:", len(missing_gsm))

if missing_gsm:
    print("Missing GSMs:")
    print(missing_gsm)

# ============================================================
# 7. SELECT DAY-5 EXPRESSION
# ============================================================

expression_day5 = expression[
    ["probe_id"] + available_gsm
].copy()

print("\nExpression Day-5 shape:")
print(expression_day5.shape)

# ============================================================
# 8. TRANSPOSE
# ============================================================

expression_day5 = expression_day5.set_index(
    "probe_id"
).T

expression_day5.index.name = "geo_accession"

print("\nAfter transpose:")
print(expression_day5.shape)

print(expression_day5.head())

# ============================================================
# 9. MERGE WITH METADATA
# ============================================================

final_dataset = day5_metadata.merge(
    expression_day5,
    on="geo_accession",
    how="inner"
)

print("\nFINAL DATASET SHAPE:")
print(final_dataset.shape)

print("\nFinal columns:")
print(final_dataset.columns[:20].tolist())


# ============================================================
# 10. CHECK FINAL BPD DISTRIBUTION
# ============================================================

print("\nFinal BPD distribution:")
print(
    final_dataset[bpd_col]
    .value_counts()
    .sort_index()
)


# ============================================================
# 11. SAVE
# ============================================================

output_path = (
    "../data/processed/"
    "GSE32472_day5_BPD_prediction.csv"
)

final_dataset.to_csv(
    output_path,
    index=False
)

print("\nSaved:", output_path)
