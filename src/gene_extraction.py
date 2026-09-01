import pandas as pd
import numpy as np
from scipy import stats

# Load your long-format data
df = pd.read_csv('../data/processed/gbtm_input_long_format(500_genes).csv')

print(df.head(10))
print(df.shape)
print(df.columns.tolist())
# For each gene, calculate how much it changes from day 5 → day 28
gene_columns = [col for col in df.columns if col not in ['infant_id', 'timepoint_code', 'bpd_group']]

variance_scores = {}

for gene in gene_columns:
    # Get day 5 and day 28 values for each subject
    day5 = df[df['timepoint_code'] == 5].groupby('infant_id')[gene].mean()
    day28 = df[df['timepoint_code'] == 28].groupby('infant_id')[gene].mean()
    
    # Calculate % change
    pct_change = ((day28 - day5) / day5.abs()).mean()
    
    # Store score
    variance_scores[gene] = abs(pct_change)

# Rank by change
ranked_genes = sorted(variance_scores.items(), key=lambda x: x[1], reverse=True)
print("Top 20 genes by change (day 5 → 28):")
for gene, score in ranked_genes[:20]:
    print(f"{gene}: {score:.3f}")

# For each gene, compare expression in BPD=1 vs BPD=0 subjects
difference_scores = {}

for gene in gene_columns:
    bpd_yes = df[df['bpd_group'] == 1][gene]
    bpd_no = df[df['bpd_group'] == 0][gene]
    
    # T-test: do they differ?
    t_stat, p_value = stats.ttest_ind(bpd_yes, bpd_no)
    
    # Effect size: fold-change
    fold_change = bpd_yes.mean() / bpd_no.mean()
    
    # Store score (lower p-value = higher score)
    difference_scores[gene] = -np.log10(p_value) if p_value > 0 else 10

# Rank by difference
ranked_genes_bpd = sorted(difference_scores.items(), key=lambda x: x[1], reverse=True)
print("\nTop 20 genes by BPD difference:")
for gene, score in ranked_genes_bpd[:20]:
    print(f"{gene}: {score:.3f}")

# Get top 10 from each ranking
top_change = [gene for gene, score in ranked_genes[:10]]
top_bpd = [gene for gene, score in ranked_genes_bpd[:10]]

# Combine: genes that appear in BOTH lists get priority
combined = list(set(top_change) & set(top_bpd))

# If overlap too small, just take top 5 from change + top 5 from BPD
if len(combined) < 5:
    final_genes = list(dict.fromkeys(top_change[:5] + top_bpd[:5]))[:5]
else:
    final_genes = combined[:5]

print(f"\nFinal 5 genes for GBTM: {final_genes}")

# Keep only the 5 selected genes + metadata
df_gbtm = df[['infant_id', 'timepoint_code'] + final_genes + ['bpd_group']].copy()

# Save
df_gbtm.to_csv('../data/processed/gbtm_input_5genes.csv', index=False)

print(f"\nDataset ready: {df_gbtm.shape[0]} rows")
print(df_gbtm.head(10))