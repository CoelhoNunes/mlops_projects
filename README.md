# MLOps Project

A **production-ready, MLflow-centric MLOps template** that demonstrates end-to-end machine learning workflows with Docker containerization and CI/CD.

**Elevator pitch:**  
"An end-to-end, **MLflow-only** MLOps stack that trains models, logs experiments, and deploys via Docker with reproducibility and observability."

---

## What it creates

1) **ML artifacts**
   - MLflow experiment runs (params, metrics, plots, artifacts)
   - Trained models with preprocessing pipelines
   - Evaluation reports and visualizations

2) **Services & runtime**
   - **Docker images** for training (CPU and GPU variants)
   - **Kubernetes** manifests for deployment
   - **CI/CD** workflows for automated testing and building

3) **Engineering scaffolding**
   - Structured **repo layout**, Python 3.11+, clean code
   - **Simple training pipeline** with sklearn digits dataset
   - **Docker multi-stage builds** with CPU-first approach

---

## Core ML pipeline

- **Data**: Load sklearn digits dataset (handwritten digit classification)
- **Preprocessing**: StandardScaler for feature normalization
- **Modeling**: RandomForestClassifier with configurable hyperparameters
- **Evaluation**: Accuracy metrics, confusion matrix, classification report
- **Logging**: Complete MLflow integration (params, metrics, artifacts, models)
- **Serving**: Docker images ready for deployment

---

## Quickstart

```bash
# 1) Setup (Python 3.11+)
pip install -r config/requirements.txt

# 2) Run training locally
python -m src.train

# 3) Run smoke test (fast, small dataset)
python -m src.train --smoke

# 4) Build Docker image
docker build -f deployment/docker/Dockerfile.train --target cpu-trainer --platform linux/amd64 .

# 5) Run training in Docker
docker run --rm -v $(pwd)/mlruns:/app/mlruns your-image:latest
```

## Available Commands

The project provides a simple training script with the following options:

- **`python -m src.train`**: Run full training pipeline
- **`python -m src.train --smoke`**: Run fast smoke test
- **`python -m src.train --mlflow-tracking-uri <uri>`**: Specify MLflow tracking URI

## Project Structure

```
├── README.md                    # This file - project overview
├── LICENSE                      # Project license
├── Makefile                     # Build and automation commands
├── .gitignore                   # Git ignore patterns
├── .github/workflows/           # CI/CD workflows
├── src/                         # Source code
│   ├── __init__.py
│   ├── train.py                 # Main training script
│   └── train_customer.py        # Customer classification training
├── tests/                       # Test files
├── scripts/                     # Utility scripts
├── docs/                        # Documentation
│   ├── guides/                  # Setup and usage guides
│   ├── architecture/            # Architecture documentation
│   ├── ADRs/                    # Architecture Decision Records
│   ├── IMPLEMENTATION_SUMMARY.md
│   ├── REFACTORING_SUMMARY.md
│   ├── ARCHITECTURE.md
│   └── CHANGELOG.md
├── config/                      # Configuration files
│   ├── requirements.txt         # Python dependencies
│   ├── pyproject.toml          # Project configuration
│   └── pytest.ini              # Test configuration
├── data/                        # Datasets
│   └── customer_dataset.csv     # Customer classification dataset
├── validation/                  # Data validation results
│   ├── validation_issues_classification.txt
│   ├── validation_results_classification.json
│   └── validation_summary_dataset.yaml
├── assets/                      # Media and visual assets
│   ├── images/                  # Images and plots
│   │   └── confusion_matrix.png
│   └── videos/                  # Video demonstrations
│       └── MLflow.mp4
├── deployment/                  # Deployment configurations
│   ├── docker/                  # Docker configurations
│   │   ├── Dockerfile.train     # Training container
│   │   └── Dockerfile.serve     # Serving container
│   └── k8s/                     # Kubernetes manifests
├── mlruns/                      # MLflow experiment tracking
└── venv/                        # Virtual environment (gitignored)
```

## Requirements

- Python 3.11+
- MLflow for experiment tracking
- scikit-learn for ML algorithms
- Docker for containerization
- GitHub Actions for CI/CD

## Docker Builds

The project provides two Docker targets:

- **`cpu-trainer`** (default): CPU-only training image
- **`gpu-trainer`**: GPU-enabled training image (CUDA 12.1, Ubuntu 22.04)

```bash
# Build CPU image (default)
docker build -f deployment/docker/Dockerfile.train --target cpu-trainer .

# Build GPU image
docker build -f deployment/docker/Dockerfile.train --target gpu-trainer .
```

## CI/CD

GitHub Actions automatically:
1. Tests the training pipeline locally
2. Builds and pushes CPU training images on main/tags
3. Optionally builds GPU images via manual workflow dispatch
4. Runs security scans and sends optional Slack notifications

## Local Development

```bash
# Install dependencies
pip install -r config/requirements.txt

# Run tests
pytest tests/

# Format code
ruff check src/
black src/

# Run training
python -m src.train --smoke
```

## MLflow Integration

The training pipeline automatically:
- Creates experiments and runs
- Logs hyperparameters and metrics
- Saves trained models and preprocessing pipelines
- Stores evaluation artifacts (confusion matrix, feature importance)
- Tracks all experiments in local `mlruns/` directory

## Kubernetes Deployment

Use the provided manifests in `k8s/` to deploy:
- Training jobs for batch model training
- Model serving deployments
- MLflow tracking server
- ConfigMaps and secrets for configuration
