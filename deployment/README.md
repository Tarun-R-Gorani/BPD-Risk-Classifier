# BPD Model v2 Deployment

This folder deploys the fold-local gene-filtered two-step BPD model.

The Streamlit interface has five pages: Home, Predict, Results, Timeline
Explorer, and Model & Methodology. It has no separate evaluation page.
Navigation is in a top row, with a gradient background and updated typography.
The Home page includes a short BPD overview linked to the NHLBI source.

The app loads:

```text
../v2/outputs/shareable_two_step_pipeline/two_step_bpd_model.joblib
```

The promoted model uses the timepoint-3 view and expects 23,310 columns: three
clinical features plus 23,307 gene-symbol expression features. It performs its
5,000-gene variance filter and supervised gene selection inside the fitted
pipelines. Upload one infant per row, with gene symbols as columns and values
prepared using the same probe-to-symbol mapping and expression scale as the
training data. The app validates required columns and ignores additional ones.
The app offers a synthetic sample CSV for exploring the workflow. Its rows are
fabricated demo values and must not be used for model evaluation. To prepare a
CSV from probe-level expression and clinical metadata, run from the repository
root:

```text
python src/prepare_prediction_input.py --expression <probe_matrix.csv> --clinical <clinical_metadata.csv> --output <app_input.csv>
```

The helper uses the local GPL6244 annotation, collapses probes to symbols, and
keeps timepoint-3 samples. The expression and metadata tables must share sample
accession IDs.

The saved grouped five-fold cross-validation estimates for this cohort are
61.9% accuracy and 54.0% macro-F1. These are evaluation results, not a promise
of performance on a different cohort.


## Git LFS data files

The timeline dataset and synthetic prediction template are tracked with Git LFS. A GitHub-generated source ZIP can contain LFS pointer text files instead of the underlying CSV data. For a complete local checkout, run `git lfs install` before cloning and `git lfs pull` from the repository root. The app detects missing LFS content and displays an explanatory message rather than treating the pointer as a CSV.

## Run locally

From the repository root:

```text
python -m pip install -r deployment/requirements.txt
streamlit run deployment/app.py
```

## Deploy with Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit app from that repository.
3. Set the main file to `deployment/app.py`.
4. Streamlit will install `deployment/requirements.txt` only if it is in the app directory, so set the app working directory to `deployment` if the platform provides that option. Otherwise copy the requirements file to the repository root or configure the platform to install it explicitly.
5. The model artifact and `bpd_feature_utils.py` must remain in the repository
   because the persisted preprocessing pipeline imports that helper.

The app provides an input CSV template, validates required columns, predicts BPD severity, and allows downloading the prediction results.
