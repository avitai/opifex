#!/bin/bash
# SciML Development Environment Configuration
# UV Configuration for optimal performance across filesystems
export UV_LINK_MODE=copy

# JAX Configuration (existing)
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK=1

# Alias for convenient pre-commit execution
alias precommit="uv run pre-commit run --all-files"
alias precommit-fix="uv run pre-commit run --all-files && uv run ruff format && uv run ruff check --fix"
