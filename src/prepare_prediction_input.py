"""Convert probe-level expression plus clinical metadata to app input CSV.

Run from the repository root. The expression CSV must have probe IDs in its
first column and one sample per remaining column. The clinical CSV must have
one row per sample, a `geo_accession` column, and the three model clinical
features. Rows are restricted to timepoint 3 (24-33 days of life).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_MODEL = ROOT / "v2/outputs/shareable_two_step_pipeline/two_step_bpd_model.joblib"
DEFAULT_EXPRESSION = ROOT / "data/processed/GSE32472_expression_new.csv"
DEFAULT_CLINICAL = ROOT / "data/processed/GSE32472_full_metadata.csv"
DEFAULT_PLATFORM = ROOT / "data/raw/GPL6244.txt"
DEFAULT_OUTPUT = ROOT / "data/processed/bpd_model_input_timepoint3.csv"
CLINICAL_FEATURES = ["days of life", "gestational age (weeks)", "birth weight (grams)"]


def read_platform_annotation(path: Path) -> pd.DataFrame:
    table_start = None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle):
            if line.strip() == "!platform_table_begin":
                table_start = line_number + 1
                break
    if table_start is None:
        raise ValueError(f"Platform table not found in {path}")
    annotation = pd.read_csv(
        path, sep="\t", skiprows=table_start,
        usecols=["ID", "gene_assignment"], dtype={"ID": str}, low_memory=False,
    )
    annotation["ID"] = annotation["ID"].astype(str).str.strip()
    annotation["gene_symbol"] = annotation["gene_assignment"].map(
        lambda value: str(value).split("//")[1].strip()
        if pd.notna(value) and len(str(value).split("//")) > 1 else None
    )
    annotation = annotation.dropna(subset=["gene_symbol"])
    return annotation.loc[annotation.gene_symbol.ne("---"), ["ID", "gene_symbol"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expression", type=Path, default=DEFAULT_EXPRESSION)
    parser.add_argument("--clinical", type=Path, default=DEFAULT_CLINICAL)
    parser.add_argument("--platform", type=Path, default=DEFAULT_PLATFORM)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    artifact = joblib.load(args.model)
    raw = pd.read_csv(args.expression, index_col=0)
    raw.index = raw.index.astype(str).str.strip()
    mapped = raw.join(read_platform_annotation(args.platform).set_index("ID"), how="inner")
    expression_by_sample = mapped.groupby("gene_symbol").mean(numeric_only=True).T
    expression_by_sample.index = expression_by_sample.index.astype(str).str.strip()

    metadata = pd.read_csv(args.clinical)
    metadata.columns = metadata.columns.str.strip()
    metadata["geo_accession"] = metadata["geo_accession"].astype(str).str.strip()
    if "source_name_ch1" in metadata:
        metadata = metadata[metadata["source_name_ch1"].astype(str).str.contains("24-33", regex=False)]
    elif "timepoint_code" in metadata:
        metadata = metadata[pd.to_numeric(metadata["timepoint_code"], errors="coerce") == 3]
    else:
        raise ValueError("Clinical metadata must include source_name_ch1 or timepoint_code to select timepoint 3.")
    metadata = metadata[["geo_accession", *CLINICAL_FEATURES]].copy()
    for feature in CLINICAL_FEATURES:
        metadata[feature] = pd.to_numeric(metadata[feature], errors="coerce")

    result = metadata.merge(expression_by_sample, left_on="geo_accession", right_index=True, how="inner")
    genes = artifact["gene_features"]
    missing = [gene for gene in genes if gene not in result.columns]
    if missing:
        raise ValueError(
            f"Expression mapping is missing {len(missing)} model genes; "
            f"first missing symbols: {', '.join(missing[:10])}"
        )
    result = result[[*CLINICAL_FEATURES, *genes]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Prepared {len(result)} timepoint-3 row(s) with {result.shape[1]} model features.")
    print(f"Saved app-ready CSV to {args.output}")


if __name__ == "__main__":
    main()
