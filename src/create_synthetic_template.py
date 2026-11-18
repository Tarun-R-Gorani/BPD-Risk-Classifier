"""Generate clearly labeled synthetic rows for the prediction-page template."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.compare_feature_sets import CLINICAL, load_data


OUTPUT = ROOT / "deployment/assets/synthetic_bpd_input_examples.csv"
SEED = 2026
N_EXAMPLES = 3


def main():
    data, genes = load_data("timepoint_3")
    features = CLINICAL + genes
    values = data[features].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    spread = np.nanstd(values, axis=0)
    lower = np.nanpercentile(values, 1, axis=0)
    upper = np.nanpercentile(values, 99, axis=0)

    rows = []
    for index in range(N_EXAMPLES):
        source_rows = rng.choice(len(values), size=4, replace=False)
        weights = rng.dirichlet(np.full(4, 4.0))
        synthetic = weights @ values[source_rows]
        synthetic += rng.normal(0.0, spread * 0.01)
        synthetic = np.clip(synthetic, lower, upper)
        # Keep the three clinical demo fields readable as whole-number values.
        for clinical_feature in CLINICAL:
            synthetic[features.index(clinical_feature)] = round(
                synthetic[features.index(clinical_feature)]
            )
        rows.append({"sample_id": f"SYNTHETIC-DEMO-{index + 1:02d}", **dict(zip(features, synthetic))})

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT, index=False)
    print(f"Wrote {len(rows)} fabricated demo rows with {len(features)} model features to {OUTPUT}")


if __name__ == "__main__":
    main()
