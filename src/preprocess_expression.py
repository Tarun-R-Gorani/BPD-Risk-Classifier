import GEOparse
import pandas as pd


def extract_gene_symbol(val):
    if pd.isna(val) or val == "---":
        return None
    parts = str(val).split("//")
    if len(parts) >= 2:
        symbol = parts[1].strip()
        return symbol if symbol and symbol != "---" else None
    return None


def main():
    print("Loading expression matrix...")
    # Direct path to your processed expression file
    expr_df = pd.read_csv("data/processed/GSE32472_expression_new.csv", index_col=0)

    print("Fetching GPL6244 annotation table...")
    gpl = GEOparse.get_GEO(geo="GPL6244", destdir="./")
    gpl_table = gpl.table

    print("Mapping probes to gene symbols...")
    id_col = "ID" if "ID" in gpl_table.columns else gpl_table.columns[0]

    # Extract gene symbols from GPL6244 'gene_assignment' column
    if "gene_assignment" in gpl_table.columns:
        ann_df = gpl_table[[id_col, "gene_assignment"]].copy()
        ann_df["gene_symbol"] = ann_df["gene_assignment"].apply(extract_gene_symbol)
    elif "Gene Symbol" in gpl_table.columns:
        ann_df = gpl_table[[id_col, "Gene Symbol"]].copy()
        ann_df.rename(columns={"Gene Symbol": "gene_symbol"}, inplace=True)
    else:
        symbol_cols = [c for c in gpl_table.columns if "symbol" in c.lower() or "assignment" in c.lower()]
        ann_df = gpl_table[[id_col, symbol_cols[0]]].copy()
        ann_df["gene_symbol"] = ann_df[symbol_cols[0]].apply(extract_gene_symbol)

    ann_df = ann_df.dropna(subset=["gene_symbol"])[[id_col, "gene_symbol"]]
    ann_df[id_col] = ann_df[id_col].astype(str)
    expr_df.index = expr_df.index.astype(str)

    # Join probes with gene symbols
    mapped_expr = expr_df.merge(ann_df, left_index=True, right_on=id_col, how="inner")

    print("Collapsing duplicate probes by mean gene expression...")
    mapped_expr = mapped_expr.groupby("gene_symbol").mean(numeric_only=True)

    # Transpose so Rows = Samples (GSMs), Columns = Gene Symbols
    expr_transposed = mapped_expr.T

    print("Applying variance filter (keeping top 5,000 genes)...")
    top_5000_genes = expr_transposed.var().nlargest(5000).index
    expr_filtered = expr_transposed[top_5000_genes]

    print("Saving intermediate filtered expression file...")
    expr_filtered.to_csv("data/processed/intermediate_filtered_expression.csv")
    print("Done! Proceed to run merge_and_select.py.")


if __name__ == "__main__":
    main()