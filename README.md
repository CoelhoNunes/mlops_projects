# MLOps Project

## Project Purpose — What this repo is for

Build a **production-grade, MLflow-centric MLOps template** that turns raw CSVs into a **trained, registered, and served model** with solid engineering around it.  
It proves you can: ingest & validate data, engineer features, train & tune multiple models (including **deep learning**), log & compare experiments in **MLflow**, **register** the best model, and **serve** it via a FastAPI microservice that runs locally, in Docker, and on Kubernetes—with CI, tests, and clear docs.

**Elevator pitch (1-liner):**  
"An end-to-end, **MLflow-only** MLOps stack that trains, registers, and serves tabular ML models (classic + deep learning) with reproducibility, observability, and K8s-ready deployment."

---

## What it creates (concrete outputs)

1) **ML artifacts**
   - MLflow experiment runs (params, metrics, plots, artifacts)
   - A **registered model** in the MLflow Model Registry (Staging/Production)
   - A documented **model card** plus validation & drift reports

2) **Services & runtime**
   - **FastAPI** prediction service (`/healthz`, `/predict`), containerized
   - **Docker images** for training and serving
   - **Kubernetes** manifests (Job for training, Deployment/Service for serving, HPA-ready)
   - Optional **local MLflow server** script

3) **Engineering scaffolding**
   - Structured **repo layout**, typed Python 3.11+, docstrings & clean comments
   - **Typer CLI**: `data-validate`, `train`, `evaluate`, `register`, `promote`, `serve`, `batch-predict`
   - **CI** (ruff, black, mypy, pytest ~80% coverage), **Makefile**, and developer docs

---

## What it demonstrates (skills aligned to the role)

- **Kubernetes**: batch training as a Job with retries/backoff; serving as a Deployment with liveness/readiness; resources & optional GPU hints
- **Python scripting**: composable pipelines (data → features → train → evaluate → register → serve)
- **Model serving**: FastAPI REST (optionally gRPC), Pydantic validation, Prometheus-friendly metrics
- **ML workflows**: train/eval/registry flows, model versioning, promotion gates, signed/model-signatured artifacts
- **ML metadata**: MLflow autolog + custom logging of validations, plots, model card, signature & input example
- **Linux/containers**: slim multi-stage Dockerfiles (non-root), healthchecks
- **CI/CD**: formatting, linting, typing, tests, and image build/push on tags
- **Reliability**: deterministic seeds, config-first design, graceful fallbacks, clear logs
- **Deep Learning**: PyTorch tabular MLP with AMP, early stopping, scheduler, class-imbalance handling, TorchScript/ONNX export

---

## Core ML pipeline (now with deep learning)

- **Data**: CSV loader, schema & sanity checks, class imbalance report
- **Features**: sklearn `ColumnTransformer` (numeric: impute+scale; categorical: impute+OHE)
- **Modeling (classic)**: LogReg, RandomForest (+ gated XGBoost/LightGBM if installed)
- **Modeling (deep learning)**: **PyTorch MLP** for tabular data (sklearn-compatible estimator) with:
  - Mixed precision (AMP) toggle, early stopping, ReduceLROnPlateau, gradient clipping
  - Class weights for imbalance; GPU if available; TorchScript + ONNX export (if onnx present)
- **Tuning**: Optuna-based search per model; stratified K-fold for classification; time-aware split toggle
- **Evaluation**: holdout metrics (classification & regression), confusion matrix/residuals, SHAP summary (config-gated)
- **Registry & Promotion**: rules-based stage transitions (e.g., F1 ≥ threshold; or RMSE ≤ threshold)
- **Serving**: Production model loaded from MLflow registry; batch prediction support via CLI

---

## Add-on ML enhancements (extend to show more depth)

### Modeling & generalization
- **Ensembles** (stacking/blending of top-k candidates)
- **Calibration** (Platt/Isotonic) + Brier score & reliability curves
- **Threshold optimization** (F1/Fβ or cost-sensitive utility)
- **Cost-sensitive metrics** (custom utility matrices)
- **Feature selection** (mutual information / permutation importance)
- **Class imbalance** (class weights vs SMOTE; compare uplift)
- **Fairness checks** (if sensitive groups exist): DP/EO gaps, per-group metrics

### Time & drift robustness
- **Time-aware CV** (rolling splits)
- **Data/prediction drift**: PSI/JS divergence logged as drift dashboard artifacts
- **Adversarial validation**: train-vs-holdout classifier to detect leakage/mismatch

### Explainability & transparency
- **Global**: SHAP summary/bar; permutation importance
- **Local**: SHAP force (small sample); optional per-prediction reason codes in serving
- **Model card++**: ethical risks, failure modes, data coverage, monitoring checklist

### Performance & deployability
- **Latency benchmarking**: offline timings + p95 in serving
- **Model compression**: ONNX export + simple quantization (when compatible)
- **Batch vs online**: `batch-predict` writes parquet with predictions + confidences

### Monitoring & operations
- **Canary/shadow** serving: compare responses & log deltas (no user impact)
- **Simple alarms**: input drift / error-rate thresholds; emit metrics/logs
- **Data contracts**: strict schema (types, ranges, enums); hard-fail violations

(Each add-on is config-gated and logged to MLflow with metrics & artifacts.)

---

## Repo hygiene & legacy cleanup (HARD)

This project enforces **no dead code** and **MLflow-only** tracking.

- Remove any legacy trackers/frameworks (ZenML, W&B, Neptune, DVC, Airflow/Prefect/Argo), unused notebooks/scripts/configs.
- Keep only files used by the new CLI, training, evaluation, registry, and serving paths.
- CI runs `ruff/black/mypy/pytest` and a prune step (e.g., `scripts/prune_repo.sh`) that:
  - Greps for forbidden trackers and fails if found
  - Uses `vulture` / `ruff` rules to catch unused code/imports
  - Deletes/flags orphaned dirs (`old/`, `legacy/`, `experiments/`) not referenced by Make/CLI

---

## Quickstart

```bash
# 1) Setup (Python 3.11+)
make setup

# 2) Point MLflow to local store (default is ./mlruns)
export MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI:-"./mlruns"}

# 3) Configure target column and task in config if needed
#    mlops/config/settings.yaml  (or via env: MLOPS_TARGET, MLOPS_TASK)

# 4) Validate data, train models (classic + PyTorch), register best
make train

# 5) Evaluate on holdout, log artifacts (plots, model card), apply promotion rules
make evaluate

# 6) Serve the Production model (FastAPI)
make serve
# then in another shell:
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"rows":[{"feature1":1.2,"feature2":"A", "...": "..."}]}'
```

## Available Commands

The project provides a comprehensive CLI with the following commands:

- **`data-validate`**: Validate data quality and generate reports
- **`train`**: Train models with hyperparameter optimization
- **`evaluate`**: Evaluate trained models on test set
- **`register`**: Register best model to MLflow Model Registry
- **`promote`**: Promote model to specified stage in MLflow Model Registry
- **`serve`**: Start model serving server
- **`batch-predict`**: Run batch predictions on input data
- **`info`**: Display project information and configuration

## Project Structure

```
mlops/
├── config/          # Configuration files
├── data/           # Data loading, validation, and feature engineering
├── modeling/       # Model training, evaluation, and promotion
├── serving/        # Model serving and API
├── orchestration/  # CLI and workflow orchestration
└── utils/          # Utility functions and helpers
```

## Requirements

- Python 3.11+
- MLflow for experiment tracking and model registry
- PyTorch for deep learning models
- FastAPI for model serving
- Docker and Kubernetes for deployment
- Comprehensive testing and CI/CD setup
