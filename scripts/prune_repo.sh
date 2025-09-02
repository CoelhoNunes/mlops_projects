#!/bin/bash

# MLOps Project Repository Pruning Script
# Removes legacy trackers, dead code, and unused dependencies

set -e

echo "🧹 Starting repository pruning..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for forbidden trackers
print_status "Checking for forbidden trackers..."

FORBIDDEN_TRACKERS=(
    "zenml"
    "wandb"
    "neptune"
    "dvc"
    "airflow"
    "prefect"
    "argo"
    "hydra"
)

FOUND_FORBIDDEN=false

    for tracker in "${FORBIDDEN_TRACKERS[@]}"; do
        # Use find with multiple exclude patterns and more specific exclusions
        # Exclude documentation, CI files, and other legitimate references
        if find . -type f -not -path "./.git/*" -not -path "./.github/*" -not -path "./node_modules/*" -not -path "./.venv/*" -not -path "./venv/*" -not -path "./__pycache__/*" -not -name "*.pyc" -not -name "*.pyo" -not -name "prune_repo.sh" -not -name "CHANGELOG.md" -not -name "*.csv" -not -path "./docs/*" -not -path "./.github/*" -not -path "./k8s/*" -not -name "*.md" -exec grep -l -i "$tracker" {} \; 2>/dev/null | grep -q .; then
            print_error "Forbidden tracker found: $tracker"
            FOUND_FORBIDDEN=true
        fi
    done

if [ "$FOUND_FORBIDDEN" = true ]; then
    print_error "Repository contains forbidden trackers. CI will fail."
    exit 1
fi

print_status "✅ No forbidden trackers found"

# Remove legacy directories
print_status "Removing legacy directories..."

LEGACY_DIRS=(
    "old"
    "legacy"
    "experiments"
    "notebooks"
    "jupyter"
    "ipynb"
    ".zen"
    "zenml"
    "wandb"
    "neptune"
    "dvc"
)

for dir in "${LEGACY_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        print_warning "Removing legacy directory: $dir"
        rm -rf "$dir"
    fi
done

# Remove unused Python files
print_status "Checking for unused Python files..."

# Find Python files that might be unused
find . -name "*.py" -not -path "./mlops/*" -not -path "./tests/*" -not -path "./.venv/*" -not -path "./venv/*" | while read -r file; do
    # Skip if it's a main file or has imports
    if [[ "$file" == *"__main__.py" ]] || [[ "$file" == *"main.py" ]]; then
        continue
    fi
    
    # Check if file is imported anywhere
    filename=$(basename "$file" .py)
    if ! grep -r --exclude-dir={.git,node_modules,.venv,venv,__pycache__} --exclude={*.pyc,*.pyo} -l "import.*$filename\|from.*$filename" . > /dev/null 2>&1; then
        print_warning "Potentially unused file: $file"
        # Don't delete automatically, just warn
    fi
done

# Remove __pycache__ directories
print_status "Removing Python cache directories..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove egg-info directories
print_status "Removing egg-info directories..."
find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

# Remove build artifacts
print_status "Removing build artifacts..."
rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

# Remove test artifacts
print_status "Removing test artifacts..."
rm -rf .pytest_cache/ .coverage htmlcov/ .mypy_cache/ .ruff_cache/ 2>/dev/null || true

# Remove temporary files
print_status "Removing temporary files..."
rm -rf temp/ logs/ 2>/dev/null || true

# Check for large files that might be unnecessary
print_status "Checking for large files..."
find . -type f -size +10M -not -path "./.git/*" -not -path "./.venv/*" -not -path "./venv/*" | while read -r file; do
    print_warning "Large file found: $file ($(du -h "$file" | cut -f1))"
done

# Remove empty directories
print_status "Removing empty directories..."
find . -type d -empty -not -path "./.git/*" -not -path "./.venv/*" -not -path "./venv/*" -delete 2>/dev/null || true

# Check for unused imports in Python files
print_status "Checking for unused imports..."
if command -v ruff > /dev/null; then
    print_status "Running ruff to find unused imports..."
    ruff check --select=F401,F841 mlops/ tests/ 2>/dev/null || print_warning "Ruff not available or failed"
else
    print_warning "Ruff not available. Install with: pip install ruff"
fi

# Check for dead code
print_status "Checking for dead code..."
if command -v vulture > /dev/null; then
    print_status "Running vulture to find dead code..."
    vulture mlops/ tests/ --min-confidence 80 2>/dev/null || print_warning "Vulture not available or failed"
else
    print_warning "Vulture not available. Install with: pip install vulture"
fi

# Clean up requirements.txt if it exists and we have pyproject.toml
if [ -f "pyproject.toml" ] && [ -f "requirements.txt" ]; then
    print_warning "Found both pyproject.toml and requirements.txt"
    print_warning "Consider removing requirements.txt as pyproject.toml is the modern standard"
fi

# Remove old configuration files
OLD_CONFIGS=(
    "config.yaml"
    "config.yml"
    "settings.yaml"
    "settings.yml"
)

for config in "${OLD_CONFIGS[@]}"; do
    if [ -f "$config" ] && [ "$config" != "mlops/config/settings.yaml" ]; then
        print_warning "Found old config file: $config"
        # Don't delete automatically, just warn
    fi
done

# Check for duplicate configuration
if [ -f "src/config.yaml" ] && [ -f "mlops/config/settings.yaml" ]; then
    print_warning "Found duplicate config files in src/ and mlops/"
    print_warning "Consider consolidating to mlops/config/settings.yaml"
fi

# Remove old MLflow runs if they're very old (older than 30 days)
if [ -d "mlruns" ]; then
    print_status "Checking for old MLflow runs..."
    find mlruns -type f -mtime +30 -name "*.yaml" | while read -r file; do
        print_warning "Old MLflow run found: $file"
        # Don't delete automatically, just warn
    done
fi

# Check for large MLflow artifacts
if [ -d "mlruns" ]; then
    print_status "Checking MLflow artifact sizes..."
    find mlruns -type f -size +100M | while read -r file; do
        print_warning "Large MLflow artifact: $file ($(du -h "$file" | cut -f1))"
    done
fi

# Remove old log files
print_status "Cleaning up old log files..."
find . -name "*.log" -mtime +7 -delete 2>/dev/null || true

# Check for environment-specific files
ENV_FILES=(
    ".env.local"
    ".env.production"
    ".env.staging"
    "local_settings.py"
)

for env_file in "${ENV_FILES[@]}"; do
    if [ -f "$env_file" ]; then
        print_warning "Found environment-specific file: $env_file"
        print_warning "Consider adding to .gitignore if not needed in repo"
    fi
done

# Final cleanup summary
print_status "Repository pruning completed!"
print_status "Summary of actions:"
echo "  ✅ Removed legacy directories"
echo "  ✅ Cleaned Python cache files"
echo "  ✅ Removed build artifacts"
echo "  ✅ Removed test artifacts"
echo "  ✅ Removed temporary files"
echo "  ✅ Checked for unused code"
echo "  ✅ Checked for large files"

# Check repository size
REPO_SIZE=$(du -sh . --exclude=.git --exclude=.venv --exclude=venv 2>/dev/null | cut -f1)
print_status "Repository size after pruning: $REPO_SIZE"

# Recommendations
print_status "Recommendations:"
echo "  📋 Run 'make lint' to check code quality"
echo "  🧪 Run 'make test' to ensure everything works"
echo "  🔍 Review warnings above for manual cleanup"
echo "  📊 Consider using 'git clean -fd' for additional cleanup"

print_status "✅ Repository pruning completed successfully!"
