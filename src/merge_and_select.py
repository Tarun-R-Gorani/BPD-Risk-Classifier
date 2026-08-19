import os
import re
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


def find_col(df, keyword):
    """Dynamically finds a column name containing a specific keyword."""
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    if not matches:
        raise KeyError(
            f"Could not find any column containing '{keyword}'. Available columns: {df.columns.tolist()}"
        )
    return matches[0]


def main():
    print("1. Loading metadata and filtered expression data...")

    meta_path = "data/processed/GSE32472_full_metadata.csv"
    expr_path = "data/processed/intermediate_filtered_expression.csv"

    if not os.path.exists(meta_path):
        meta_path = "../data/processed/GSE32472_full_metadata.csv"
        expr_path = "../data/processed/intermediate_filtered_expression.csv"

    meta_df = pd.read_csv(meta_path)
    expr_filtered = pd.read_csv(expr_path, index_col=0)

    # Clean whitespace in column names and string cells
    meta_df.columns = meta_df.columns.str.strip()

    col_title = find_col(meta_df, "title")
    col_geo = find_col(meta_df, "geo_accession")
    col_source = find_col(meta_df, "source_name_ch1")
    col_days = find_col(meta_df, "days of life")
    col_ga = find_col(meta_df, "gestational age (weeks)")
    col_bw = find_col(meta_df, "birth weight (grams)")
    col_bpd = find_col(meta_df, "bronchopulmonary dysplasia (bpd) group")

    print("\n2. Cleaning metadata and mapping tracking variables...")

    # Extract infant ID from title
    meta_df["infant_id"] = meta_df[col_title].apply(
        lambda x: int(re.search(r"rep\s*(\d+)", str(x), re.IGNORECASE).group(1))
    )

    # Map timepoints flexibly handling whitespace
    def map_timepoint(val):
        val_str = str(val).strip().lower()
        if "4-6" in val_str:
            return 1
        elif "12-17" in val_str:
            return 2
        elif "24-33" in val_str:
            return 3
        return None

    meta_df["timepoint_code"] = meta_df[col_source].apply(map_timepoint)
    meta_df["bpd_group"] = pd.to_numeric(meta_df[col_bpd], errors="coerce")

    # Clean accession strings
    meta_df[col_geo] = meta_df[col_geo].astype(str).str.strip()
    expr_filtered.index = expr_filtered.index.astype(str).str.strip()

    print("3. Merging metadata with gene expression matrix...")
    cols_to_keep = [
        col_geo,
        "infant_id",
        "timepoint_code",
        col_days,
        col_ga,
        col_bw,
        "bpd_group",
    ]

    merged_df = meta_df[cols_to_keep].merge(
        expr_filtered, left_on=col_geo, right_index=True, how="inner"
    )

    # Clean rows with missing clinical tracking data
    merged_df = merged_df.dropna(subset=["timepoint_code", "bpd_group"])

    print(f"   -> Valid Merged Samples for Modeling: {len(merged_df)}")

    # Standardize column names
    merged_df.rename(
        columns={
            col_days: "days of life",
            col_ga: "gestational age (weeks)",
            col_bw: "birth weight (grams)",
        },
        inplace=True,
    )

    # Retain infants with 2 or 3 timepoints
    valid_infants = merged_df.groupby("infant_id").size()
    merged_df = merged_df[
        merged_df["infant_id"].isin(valid_infants[valid_infants >= 2].index)
    ]

    print(
        "\n4. Running Group x Time Interaction ANOVA across 5,000 genes..."
    )
    gene_cols = expr_filtered.columns.tolist()
    safe_map = {g: f"g_{i}" for i, g in enumerate(gene_cols)}
    reverse_map = {v: k for k, v in safe_map.items()}
    merged_df.rename(columns=safe_map, inplace=True)

    p_values = {}
    total_genes = len(safe_map)
    first_error_logged = False

    for idx, (original_gene, safe_gene) in enumerate(
        safe_map.items(), start=1
    ):
        if idx % 1000 == 0 or idx == total_genes:
            print(f"   Processed {idx}/{total_genes} genes...")

        try:
            formula = f"{safe_gene} ~ C(timepoint_code) * C(bpd_group)"
            model = ols(formula, data=merged_df).fit()
            anova_tbl = sm.stats.anova_lm(model, typ=2)

            # Locate interaction row dynamically
            interaction_row = [
                r for r in anova_tbl.index if ":" in str(r)
            ]
            if interaction_row:
                p_values[safe_gene] = anova_tbl.loc[
                    interaction_row[0], "PR(>F)"
                ]
            else:
                p_values[safe_gene] = anova_tbl.iloc[2]["PR(>F)"]
        except Exception as e:
            if not first_error_logged:
                print(f"   [Diagnostic Warning] ANOVA failed on sample gene: {e}")
                first_error_logged = True
            continue

    if not p_values:
        raise ValueError(
            "ANOVA failed to calculate p-values across all genes. Check dataset variables."
        )

    print("\n5. Selecting top 100 trajectory-informative genes...")
    top_100_safe = sorted(p_values, key=p_values.get)[:100]
    top_100_genes = [reverse_map[g] for g in top_100_safe]

    merged_df.rename(columns=reverse_map, inplace=True)

    clinical_cols = [
        "infant_id",
        "timepoint_code",
        "days of life",
        "gestational age (weeks)",
        "birth weight (grams)",
        "bpd_group",
    ]
    final_df = merged_df[clinical_cols + top_100_genes]

    output_path = (
        "data/processed/gbtm_input_long_format.csv"
        if os.path.exists("data/processed")
        else "../data/processed/gbtm_input_long_format.csv"
    )

    print(f"6. Saving final dataset to {output_path}...")
    final_df.to_csv(output_path, index=False)
    print(
        f"\nSUCCESS: Saved 'gbtm_input_long_format.csv' containing {final_df.shape[1]} columns!"
    )


if __name__ == "__main__":
    main()