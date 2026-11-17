# Deployed BPD Model

The active artifact in `outputs/shareable_two_step_pipeline/` is the combined
clinical and gene-expression two-stage model. It uses the timepoint-3 view,
3 clinical inputs, and 23,307 gene-symbol expression inputs. Gene variance
filtering and supervised selection are part of the persisted pipelines and are
fit on training data only.

The model was selected from grouped five-fold comparisons on 97 infants. Its
saved comparison estimate is 61.9% accuracy and 54.0% macro-F1. The ordinal
Ridge comparison scored lower. See `src/outputs/feature_set_comparison/` for
fold metrics and gene-selection stability.

To rebuild the comparison and retrain/promote the combined model, run from the
repository root:

```text
python src/compare_feature_sets.py
python src/promote_feature_model.py
```

The previous 28-feature artifact is retained as
`two_step_bpd_model_before_fold_local_gene_filter.joblib` for rollback. The
previous model's reports are archived under
`v2/outputs/reduced_input_v2_results/`. The
Streamlit app loads the active artifact at the same path and validates uploaded
CSV columns against its saved feature list.
