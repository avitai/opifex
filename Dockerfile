# =============================================================================
# Opifex — Development & GPU Runtime Image
# =============================================================================
# Unified image for running opifex training, evaluation, and tests.
# Uses uv for deterministic, fast dependency resolution.
#
# Build:  docker build -t opifex:latest .
# Run:    docker run --rm --gpus all opifex:latest python -c "import opifex; print('OK')"
# Test:   docker run --rm -e JAX_PLATFORMS=cpu opifex:latest python -m pytest tests/ -x -q
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# JAX runtime defaults — prevent full GPU memory preallocation
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# System dependencies + Python 3.12
RUN apt-get update && apt-get install -y --no-install-recommends \
  python3.12 \
  python3.12-venv \
  python3.12-dev \
  python3-pip \
  git \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv — single-layer binary copy from official OCI image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# --- Layer 1: Dependencies (cached unless pyproject.toml or uv.lock change) ---
COPY pyproject.toml uv.lock README.md LICENSE ./

RUN uv venv /app/.venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install main + dev/gpu/test extras (not docs)
RUN uv pip install -e ".[dev,gpu,test]"

# --- Layer 2: Source code (changes frequently, invalidates only this layer) ---
COPY src ./src
COPY tests ./tests
COPY scripts ./scripts
COPY examples ./examples

# Reinstall opifex in editable mode now that source is present
RUN uv pip install -e ".[dev,gpu,test]"

# Verify JAX can import (allow failure on CPU-only build hosts)
RUN python -c "import jax; print(f'JAX {jax.__version__}, devices: {jax.devices()}')" || true

# Default command — overridable at runtime
CMD ["python", "-m", "pytest", "tests/", "-x", "-q"]
