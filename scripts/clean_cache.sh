#!/bin/bash

# Cache cleanup script for Python project
echo "🧹 Cleaning cache files from repository..."

# Remove Python cache files
echo "Removing Python bytecode cache..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true

# Remove testing and coverage cache
echo "Removing test and coverage cache..."
rm -rf .pytest_cache/ 2>/dev/null || true
rm -rf .coverage 2>/dev/null || true
rm -rf htmlcov/ 2>/dev/null || true

# Remove type checking cache
echo "Removing type checking cache..."
rm -rf .mypy_cache/ 2>/dev/null || true

# Remove linting cache
echo "Removing linting cache..."
rm -rf .ruff_cache/ 2>/dev/null || true

# Remove benchmarking cache
echo "Removing benchmarking cache..."
rm -rf .benchmarks/ 2>/dev/null || true

# Remove build artifacts
echo "Removing build artifacts..."
rm -rf dist/ 2>/dev/null || true
rm -rf build/ 2>/dev/null || true
rm -rf ./*.egg-info/ 2>/dev/null || true
rm -rf site/ 2>/dev/null || true

# Remove temporary files
echo "Removing temporary files..."
rm -rf temp/ 2>/dev/null || true
rm -rf test_artifacts/ 2>/dev/null || true

# Remove IDE cache (but keep configuration)
echo "Removing IDE cache..."
find .vscode/ -name "*.log" -delete 2>/dev/null || true
find .cursor/ -name "*.log" -delete 2>/dev/null || true

echo "✅ Cache cleanup completed!"
echo ""
echo "Kept the following (as they should be preserved):"
echo "  - .venv/ (virtual environment)"
echo "  - uv.lock (dependency lock file)"
echo "  - .git/ (git repository)"
echo "  - Configuration files (.vscode/, .cursor/ settings)"
