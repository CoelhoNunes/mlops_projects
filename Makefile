# MLOps Project Makefile
# Production-ready MLOps project with MLflow tracking and PyTorch support

.PHONY: help setup install dev-install lint format type-check test test-unit test-e2e clean train evaluate register promote serve docker-build docker-run k8s-apply prune

# Default target
help:
	@echo "MLOps Project - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  setup          Install project dependencies"
	@echo "  install        Install production dependencies"
	@echo "  dev-install    Install development dependencies"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint           Run linting with ruff"
	@echo "  format         Format code with black"
	@echo "  type-check     Run type checking with mypy"
	@echo ""
	@echo "Testing:"
	@echo "  test           Run all tests with coverage"
	@echo "  test-unit      Run unit tests only"
	@echo "  test-e2e       Run end-to-end tests only"
	@echo ""
	@echo "MLOps Operations:"
	@echo "  train          Train models with hyperparameter optimization"
	@echo "  evaluate       Evaluate trained models"
	@echo "  register       Register best model to MLflow"
	@echo "  promote        Promote model to Production stage"
	@echo "  serve          Start model serving server"
	@echo ""
	@echo "Docker & Kubernetes:"
	@echo "  docker-build   Build Docker images"
	@echo "  docker-run     Run services with Docker Compose"
	@echo "  k8s-apply      Apply Kubernetes manifests"
	@echo ""
	@echo "Maintenance:"
	@echo "  clean          Clean up generated files and artifacts"
	@echo "  prune          Remove legacy code and unused dependencies"

# Setup and Installation
setup: install dev-install
	@echo "✅ Project setup completed!"

install:
	@echo "📦 Installing production dependencies..."
	pip install -e .

dev-install:
	@echo "🔧 Installing development dependencies..."
	pip install -e ".[dev]"
	@echo "📋 Installing pre-commit hooks..."
	pre-commit install

# Code Quality
lint:
	@echo "🔍 Running linting with ruff..."
	ruff check mlops/ tests/
	@echo "✅ Linting completed!"

format:
	@echo "🎨 Formatting code with black..."
	black mlops/ tests/
	@echo "✅ Code formatting completed!"

type-check:
	@echo "🔍 Running type checking with mypy..."
	mypy mlops/ --ignore-missing-imports
	@echo "✅ Type checking completed!"

# Testing
test: test-unit test-e2e
	@echo "✅ All tests completed!"

test-unit:
	@echo "🧪 Running unit tests..."
	pytest tests/unit/ -v --cov=mlops --cov-report=term-missing --cov-report=html
	@echo "✅ Unit tests completed!"

test-e2e:
	@echo "🧪 Running end-to-end tests..."
	pytest tests/e2e/ -v --cov=mlops --cov-report=term-missing
	@echo "✅ End-to-end tests completed!"

# MLOps Operations
train:
	@echo "🚀 Starting model training..."
	python -m mlops.orchestration.cli train
	@echo "✅ Training completed!"

evaluate:
	@echo "📊 Starting model evaluation..."
	python -m mlops.orchestration.cli evaluate
	@echo "✅ Evaluation completed!"

register:
	@echo "📝 Registering model to MLflow..."
	python -m mlops.orchestration.cli register
	@echo "✅ Model registration completed!"

promote:
	@echo "🚀 Promoting model to Production..."
	python -m mlops.orchestration.cli promote
	@echo "✅ Model promotion completed!"

serve:
	@echo "🌐 Starting model serving server..."
	python -m mlops.orchestration.cli serve
	@echo "✅ Model serving started!"

# Data Operations
data-validate:
	@echo "🔍 Validating data quality..."
	python -m mlops.orchestration.cli data-validate
	@echo "✅ Data validation completed!"

# Docker Operations
docker-build:
	@echo "🐳 Building Docker images..."
	docker build -f Dockerfile.train -t mlops-trainer:latest .
	docker build -f Dockerfile.serve -t mlops-server:latest .
	@echo "✅ Docker images built!"

docker-run:
	@echo "🐳 Starting services with Docker Compose..."
	docker-compose up -d
	@echo "✅ Services started! Check http://localhost:8000 for API and http://localhost:5000 for MLflow UI"

docker-stop:
	@echo "🐳 Stopping Docker services..."
	docker-compose down
	@echo "✅ Services stopped!"

# Kubernetes Operations
k8s-apply:
	@echo "☸️  Applying Kubernetes manifests..."
	kubectl apply -f k8s/
	@echo "✅ Kubernetes manifests applied!"

k8s-delete:
	@echo "☸️  Deleting Kubernetes resources..."
	kubectl delete -f k8s/
	@echo "✅ Kubernetes resources deleted!"

# MLflow Operations
mlflow-ui:
	@echo "📊 Starting MLflow UI..."
	mlflow ui --backend-store-uri file:./mlruns --port 5000
	@echo "✅ MLflow UI started at http://localhost:5000"

# Maintenance
clean:
	@echo "🧹 Cleaning up generated files and artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf temp/
	rm -rf logs/
	@echo "✅ Cleanup completed!"

clean-artifacts:
	@echo "🧹 Cleaning up ML artifacts..."
	rm -rf artifacts/
	rm -rf mlruns/
	@echo "✅ Artifacts cleanup completed!"

prune:
	@echo "✂️  Pruning repository..."
	@if [ -f "scripts/prune_repo.sh" ]; then \
		chmod +x scripts/prune_repo.sh; \
		./scripts/prune_repo.sh; \
	else \
		echo "⚠️  Prune script not found. Running basic cleanup..."; \
		find . -type f -name "*.pyc" -delete; \
		find . -type d -name "__pycache__" -delete; \
		find . -type d -name "*.egg-info" -exec rm -rf {} +; \
	fi
	@echo "✅ Repository pruning completed!"

# Development Workflow
dev-setup: setup
	@echo "🔧 Setting up development environment..."
	@echo "📋 Installing pre-commit hooks..."
	pre-commit install
	@echo "✅ Development environment setup completed!"

dev-check: lint format type-check test
	@echo "✅ All development checks completed!"

# Quick Start
quickstart: setup data-validate train evaluate register
	@echo "🎉 Quick start completed! Your model is ready for production!"
	@echo "📊 View results: make mlflow-ui"
	@echo "🌐 Start serving: make serve"

# Production Deployment
deploy: train evaluate register promote
	@echo "🚀 Production deployment completed!"
	@echo "🌐 Start serving: make serve"

# Monitoring and Health Checks
health-check:
	@echo "🏥 Running health checks..."
	@echo "📊 Checking MLflow tracking server..."
	@if curl -s http://localhost:5000 > /dev/null 2>&1; then \
		echo "✅ MLflow UI is running"; \
	else \
		echo "❌ MLflow UI is not running"; \
	fi
	@echo "🌐 Checking model serving API..."
	@if curl -s http://localhost:8000/healthz > /dev/null 2>&1; then \
		echo "✅ Model serving API is running"; \
	else \
		echo "❌ Model serving API is not running"; \
	fi
	@echo "✅ Health checks completed!"

# Documentation
docs-serve:
	@echo "📚 Starting documentation server..."
	@if command -v mkdocs > /dev/null; then \
		mkdocs serve; \
	else \
		echo "⚠️  MkDocs not installed. Install with: pip install mkdocs"; \
	fi

# Backup and Recovery
backup:
	@echo "💾 Creating backup..."
	tar -czf "mlops-backup-$(date +%Y%m%d-%H%M%S).tar.gz" \
		--exclude='.git' \
		--exclude='.venv' \
		--exclude='__pycache__' \
		--exclude='*.pyc' \
		mlruns/ artifacts/ logs/
	@echo "✅ Backup created!"

# Performance Testing
benchmark:
	@echo "⚡ Running performance benchmarks..."
	@echo "📊 Training performance..."
	time python -m mlops.orchestration.cli train --model torch_mlp
	@echo "🌐 Serving performance..."
	@echo "✅ Benchmarking completed!"

# Security Checks
security-check:
	@echo "🔒 Running security checks..."
	@echo "📦 Checking for known vulnerabilities..."
	@if command -v safety > /dev/null; then \
		safety check; \
	else \
		echo "⚠️  Safety not installed. Install with: pip install safety"; \
	fi
	@echo "✅ Security checks completed!"

# Environment Management
env-create:
	@echo "🌍 Creating virtual environment..."
	python -m venv .venv
	@echo "✅ Virtual environment created!"
	@echo "💡 Activate with: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)"

env-activate:
	@echo "🌍 Activating virtual environment..."
	@echo "💡 Run: source .venv/bin/activate (Linux/Mac) or .venv\\Scripts\\activate (Windows)"

env-delete:
	@echo "🗑️  Deleting virtual environment..."
	rm -rf .venv/
	@echo "✅ Virtual environment deleted!"

# Helpers
check-deps:
	@echo "🔍 Checking dependencies..."
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import mlflow; print(f'MLflow: {mlflow.__version__}')"
	@python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
	@echo "✅ Dependency check completed!"

version:
	@echo "📋 Project version information:"
	@python -c "import mlops; print(f'MLOps version: {mlops.__version__ if hasattr(mlops, \"__version__\") else \"dev\"}')"
	@echo "Python version: $(shell python --version)"
	@echo "Pip version: $(shell pip --version)"

# Default target
all: help
