# Breast Cancer Classification — CI/CD with MLflow

A continuous-integration workflow that **retrains, tracks, and packages** a breast-cancer
classifier on every push. Training is defined as a reproducible **MLflow Project** and
executed automatically by **GitHub Actions**, with experiment tracking on DagsHub and a
Docker image for serving.

## Workflow

```
Push → GitHub Actions → MLflow Project run → log metrics & model → package Docker image
```

- **Training** (`MLProject/modelling.py`) — Random Forest with `GridSearchCV`
  (`n_estimators`, `max_depth`), an 80/20 split, and weighted precision/recall/F1
- **Experiment tracking** — MLflow logs params, metrics, the model, a confusion matrix,
  and feature-importance artifacts to a remote DagsHub tracking server
- **Reproducibility** — environment pinned via `conda.yaml`; entry point defined in the
  `MLProject` file
- **Packaging** — `Dockerfile` builds a container around the trained model (pushed to
  Docker Hub)

## Metrics Tracked

`accuracy` · `precision` · `recall` · `f1_score` — plus a logged confusion matrix and
ranked feature importances.

## Tech Stack

`MLflow` · `scikit-learn` · `GitHub Actions` · `Docker` · `DagsHub` · `conda`

## Repository Structure

```
.
├── MLProject/
│   ├── MLProject                    # MLflow project definition
│   ├── modelling.py                 # Training + tracking script
│   ├── conda.yaml                   # Pinned environment
│   ├── Dockerfile                   # Serving image
│   └── breastcancer_preprocessing.csv
├── .github/workflows/
│   └── ci_workflow.yml              # CI pipeline
└── requirements.txt
```

## Run Locally

```bash
pip install -r requirements.txt
mlflow run MLProject
```
