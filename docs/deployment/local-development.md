# Local Development Setup

This guide covers setting up a local development environment for the Opifex framework, including native installation, Docker-based workflows, and the model serving API.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Native Installation](#native-installation)
3. [Docker Setup](#docker-setup)
4. [Model Serving API](#model-serving-api)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 50GB free space
- **GPU**: Optional NVIDIA GPU for CUDA acceleration

### Required Software

- **Python 3.12+**
- **[uv](https://github.com/astral-sh/uv)** -- the sole package manager for this project
- **Git**
- **Docker** and **Docker Compose** (for container workflows)

## Native Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/avitai/opifex.git
cd opifex
```

### Step 2: Run Setup

The `setup.sh` script auto-detects your hardware, creates a `.venv` virtual environment, and installs all dependencies via `uv`.

```bash
./setup.sh
```

**Flags:**

| Flag | Description |
|------|-------------|
| `--backend auto\|cpu\|cuda12\|metal` | Choose compute backend. Default: `auto` (detects GPU) |
| `--python <version>` | Create the `.venv` with a specific Python version |
| `--recreate` | Remove existing `.venv` before syncing |
| `--force-clean` | Remove `.venv`, generated `.opifex.env`, and test artifacts |
| `--dry-run` | Print resolved backend and `uv` commands without making changes |

The script writes a `.opifex.env` file with the resolved backend settings. If you have a `.env` file, it acts as a user-owned override layer loaded after `.opifex.env`.

### Step 3: Activate Environment

```bash
source ./activate.sh
```

You must run `source activate.sh` before any `uv run` command (tests, linting, etc.).

### Step 4: Install Pre-commit Hooks

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

### Step 5: Verify Installation

```bash
# Verify JAX
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"

# Run quick tests
source activate.sh && uv run pytest tests/ -x -q

# Code quality checks
uv run ruff check src/
uv run ruff format --check src/
uv run pyright src/
```

## Docker Setup

The project provides a single `Dockerfile` (based on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`) and a `docker-compose.yml` with two services.

### Services

| Service | Description | GPU |
|---------|-------------|-----|
| `opifex-gpu` | Full GPU runtime with NVIDIA device passthrough | Yes |
| `opifex-cpu` | CPU-only runtime (`JAX_PLATFORMS=cpu`) | No |

Both services mount `./data` and `./checkpoints` as volumes.

### Build and Run

```bash
# Build the image
docker build -t opifex:latest .

# Run with GPU support
docker compose up opifex-gpu

# Run CPU-only
docker compose up opifex-cpu

# Run tests inside the container
docker compose run opifex-cpu pytest tests/ -x -q

# Quick smoke test
docker run --rm --gpus all opifex:latest python -c "import opifex; print('OK')"
```

### Dockerfile Details

The `Dockerfile` uses a two-layer caching strategy:

1. **Layer 1 (dependencies)**: Copies `pyproject.toml`, `uv.lock`, and installs `.[dev,gpu,test]` extras via `uv`. Cached unless lock files change.
2. **Layer 2 (source)**: Copies `src/`, `tests/`, `scripts/`, `examples/` and reinstalls in editable mode.

Key environment variables set in the image:

- `XLA_PYTHON_CLIENT_PREALLOCATE=false` -- prevents full GPU memory preallocation
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` -- limits JAX GPU memory usage

## Model Serving API

Opifex includes a FastAPI-based model serving server at `opifex.deployment.server`.

### Running the Server

```bash
# Via the module entry point
source activate.sh
python -m opifex.deployment.server
```

The server reads configuration from environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPIFEX_HOST` | `127.0.0.1` | Bind address |
| `OPIFEX_PORT` | `8080` | Bind port |
| `OPIFEX_WORKERS` | `1` | Uvicorn worker count |
| `OPIFEX_LOG_LEVEL` | `info` | Log level |
| `OPIFEX_MODEL_NAME` | `default` | Model name |
| `OPIFEX_MODEL_TYPE` | `neural_operator` | Model type |
| `OPIFEX_BATCH_SIZE` | `32` | Batch size |
| `OPIFEX_PRECISION` | `float32` | Precision (`float16`, `float32`, `float64`) |
| `OPIFEX_MODEL_REGISTRY` | `./models` | Path for model registry storage |
| `JAX_PLATFORM_NAME` | `cpu` | JAX platform (set to `gpu` for GPU inference) |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API information and links |
| `GET` | `/health` | Health check (component status, uptime) |
| `GET` | `/models` | List registered models |
| `POST` | `/predict` | Run inference (JSON body: `{"data": [...]}`) |
| `GET` | `/metrics` | Performance metrics (latency, throughput) |
| `GET` | `/docs` | Swagger UI (auto-generated) |
| `GET` | `/redoc` | ReDoc documentation |

### Core Serving Components

The serving infrastructure lives in `src/opifex/deployment/core_serving.py`:

- **`DeploymentConfig`** -- dataclass for model deployment configuration (port, batch size, precision, GPU toggle)
- **`InferenceEngine`** -- JIT-compiled inference with performance tracking. Load a Flax NNX model via `engine.load_model(model, metadata)`, then call `engine.predict(input_data)`
- **`ModelRegistry`** -- file-based registry for storing and versioning models with metadata
- **`ModelServer`** -- programmatic server wrapper around `InferenceEngine`
- **`ModelMetadata`** -- dataclass for model metadata (name, version, shapes, accuracy metrics)

Source: [`src/opifex/deployment/core_serving.py`](../../src/opifex/deployment/core_serving.py)

### Health Monitoring

The `HealthChecker` class (`src/opifex/deployment/monitoring/health.py`) provides production health monitoring:

- System resource checks (CPU, memory, disk via `psutil`)
- GPU availability and memory usage checks (via JAX)
- External dependency health checks
- Custom health check registration
- Periodic async health check loops

Source: [`src/opifex/deployment/monitoring/health.py`](../../src/opifex/deployment/monitoring/health.py)

## Development Workflow

### Running Examples

```bash
source activate.sh

# Run a neural operator example
python examples/neural-operators/operator_tour.py

# Run with specific backend
JAX_PLATFORMS=cpu python examples/neural-operators/operator_tour.py
```

### Code Quality

```bash
source activate.sh

# Format
uv run ruff format src/

# Lint with autofix
uv run ruff check src/ --fix

# Type check
uv run pyright src/

# All pre-commit hooks
uv run pre-commit run --all-files
```

### IDE Configuration

#### VS Code

```json
{
    "python.defaultInterpreterPath": "./.venv/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"]
}
```

#### PyCharm

1. Set interpreter to `./.venv/bin/python`
2. Enable pytest as test runner
3. Configure ruff as external tool

## Testing

### Test Directory Structure

The test suite is organized by domain area:

```
tests/
  core/           # Core framework tests
  neural/         # Neural operator tests (FNO, DeepONet, GNO, etc.)
  physics/        # Physics solver tests
  data/           # Data loader tests
  training/       # Training loop tests
  optimization/   # Optimization tests
  deployment/     # Deployment infrastructure tests
  integration/    # End-to-end integration tests
  benchmarking/   # Benchmark tests
  visualization/  # Visualization tests
  ...
```

### Running Tests

```bash
source activate.sh

# Run all tests
uv run pytest tests/ -v

# Run a specific test module
uv run pytest tests/core/ -v
uv run pytest tests/neural/ -v
uv run pytest tests/deployment/ -v

# Run with coverage
uv run pytest -vv \
    --json-report --json-report-file=temp/test-results.json \
    --json-report-indent=2 --json-report-verbosity=2 \
    --cov=src/ --cov-report=json:temp/coverage.json \
    --cov-report=term-missing

# Run a single test by name
uv run pytest tests/neural/operators/fno/test_tensorized.py -v -k "test_spectral"
```

## Troubleshooting

### JAX/GPU Not Detected

```bash
# Check JAX sees your GPU
python -c "import jax; print(jax.devices())"

# Check NVIDIA driver
nvidia-smi

# Force CPU mode
JAX_PLATFORMS=cpu python -c "import jax; print(jax.devices())"

# Recreate environment with GPU support
./setup.sh --recreate --backend cuda12
source activate.sh
```

### Import Errors

```bash
# Ensure environment is activated
source activate.sh

# Reinstall
./setup.sh --force-clean
source activate.sh
```

### Docker GPU Issues

```bash
# Verify NVIDIA container toolkit is installed
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Rebuild without cache
docker compose build --no-cache

# Check container logs
docker compose logs opifex-gpu
```

### Port Conflicts

```bash
# Check if port 8080 is in use
lsof -i :8080

# Use a different port
OPIFEX_PORT=9090 python -m opifex.deployment.server
```
