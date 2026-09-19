import pandas as pd
import numpy as np

csv_path = "../data/processed/GSE32472_day5_BPD_prediction.csv"  # Change this
output_path = "../data/processed/filtered_genes.csv"

print("Loading CSV...")
df = pd.read_csv(csv_path)

metadata_cols = [
    'title', 'geo_accession', 'source_name_ch1', 'days of life', 'gender',
    'gestational age (weeks)', 'birth weight (grams)', 'oxygen 28 days',
    'oxygen 36 weeks postmenstrual age (wpma)', 'bronchopulmonary dysplasia (bpd) group'
]

gene_cols = [col for col in df.columns if col not in metadata_cols]
bpd_col = 'bronchopulmonary dysplasia (bpd) group'

print(f"Found {len(gene_cols)} genes. Filtering...")

# FAST: Keep genes with high variance
gene_data = df[gene_cols].astype(float)
variances = gene_data.var()
high_var_genes = variances.nlargest(1000).index.tolist()  # Top 1000 by variance

print(f"After variance filter: {len(high_var_genes)} genes")

# FAST: Rank by correlation with BPD
bpd_numeric = df[bpd_col].astype(float)
correlations = gene_data[high_var_genes].corrwith(bpd_numeric).abs().sort_values(ascending=False)
top_500_genes = correlations.head(500).index.tolist()

print(f"Selected {len(top_500_genes)} genes")

# Save
filtered_df = df[metadata_cols + top_500_genes]
filtered_df.to_csv(output_path, index=False)
print(f"Done! Saved to {output_path}")
print(f"New shape: {filtered_df.shape}")