# Neonatal BPD Temporal Trajectory Risk Modelling

A research prototype for exploring bronchopulmonary dysplasia (BPD), gene-expression trends, and a two-stage BPD classifier.

## Important: download the Git LFS assets

Some CSV datasets and templates are stored with Git Large File Storage (Git LFS). Downloading the repository using GitHub's **Download ZIP** option may provide small pointer files instead of the actual CSV contents.

Clone the repository with Git LFS enabled:

```bash
git lfs install
git clone https://github.com/Tarun-R-Gorani/BPD-Risk-Classifier.git
cd BPD-Risk-Classifier
git lfs pull
```

If you already cloned the repository, run `git lfs pull` from its root directory.

## Run the Streamlit application

```bash
python -m pip install -r deployment/requirements.txt
streamlit run deployment/app.py
```

The deployed model artifact is stored under `v2/outputs/shareable_two_step_pipeline/`. The app uses the timepoint-3 classifier for predictions and the longitudinal CSV for the Timeline Explorer.

See [`deployment/README.md`](deployment/README.md) for input format and deployment details.


## BPD Insight V6 interface refinements

- Preserves the V5 teal-and-peach visual system and existing two-stage prediction pipeline.
- Adds a research snapshot and responsible-use note to the Home page.
- Adds an interactive per-sample score breakdown on the Results page. Stage 1 scores are explicitly described as model scores, not calibrated clinical risk.
- Adds Line/Area chart style selection to the Timeline Explorer.
- Retains responsive layout and reduced-motion accessibility behavior.

Run from the repository root with `python -m streamlit run deployment/app.py`.
