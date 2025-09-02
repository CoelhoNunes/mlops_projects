# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-01-XX

### Added
- **Complete MLOps stack** with MLflow-only tracking
- **PyTorch MLP estimator** with sklearn-compatible API
- **Automatic Mixed Precision (AMP)** support for GPU training
- **Comprehensive CLI orchestration** using Typer
- **Data validation and quality checks** with automated reporting
- **Feature engineering pipeline** with sklearn ColumnTransformer
- **Hyperparameter optimization** using Optuna for all models
- **Model registry and promotion** with MLflow Model Registry
- **FastAPI serving** with health checks and monitoring
- **Kubernetes manifests** for production deployment
- **Docker multi-stage builds** for training and serving
- **CI/CD pipeline** with GitHub Actions
- **Comprehensive testing** with pytest and coverage
- **Code quality tools** (ruff, black, mypy)
- **Makefile** with all common operations
- **Repository pruning script** for cleanup

### Changed
- **Replaced legacy stack** (ZenML, W&B, etc.) with MLflow-only approach
- **Modernized project structure** using pyproject.toml
- **Consolidated configuration** into single settings.yaml
- **Standardized logging** with structured logging support
- **Improved reproducibility** with deterministic seeds and MLflow tracking

### Removed
- **ZenML framework** and all related code
- **Weights & Biases** integration
- **Neptune tracking** references
- **DVC** data versioning
- **Airflow/Prefect/Argo** orchestration
- **Legacy notebooks** and unused scripts
- **Old configuration files** and duplicate settings
- **Unused dependencies** and dead code

### Fixed
- **Repository bloat** from legacy frameworks
- **Configuration duplication** across multiple files
- **Dependency conflicts** from unused packages
- **Code quality issues** with modern linting tools

### Technical Details
- **Python 3.11+** support with type hints
- **MLflow 2.8+** for experiment tracking and model registry
- **PyTorch 2.0+** with CUDA support and AMP
- **scikit-learn** for classical ML algorithms
- **FastAPI** for high-performance model serving
- **Kubernetes** manifests with HPA and health checks
- **Docker** images with security best practices

### Migration Notes
- **Data paths**: Update to use `./data/*.csv` pattern
- **Configuration**: All settings now in `mlops/config/settings.yaml`
- **CLI**: Use `python -m mlops.orchestration.cli` instead of old scripts
- **MLflow**: Tracking URI defaults to `./mlruns` (local)
- **Models**: PyTorch MLP now available as `torch_mlp` in model roster

### Breaking Changes
- **Legacy CLI commands** removed
- **Old configuration format** no longer supported
- **ZenML pipelines** replaced with MLflow tracking
- **Streamlit app** replaced with FastAPI serving

### Next Steps
1. **Setup**: Run `make setup` to install dependencies
2. **Data**: Place CSV files in `./data/` directory
3. **Training**: Run `make train` to train models
4. **Serving**: Run `make serve` to start API
5. **MLflow UI**: Run `make mlflow-ui` to view experiments

---

## [Legacy] - Pre-2024

### Previous Stack
- ZenML for pipeline orchestration
- Weights & Biases for experiment tracking
- Streamlit for model serving
- Basic ML pipeline without deep learning
- Limited automation and testing

### Known Issues
- Repository bloat from multiple frameworks
- Configuration scattered across files
- Limited reproducibility and tracking
- No production deployment support
- Minimal testing and code quality tools
