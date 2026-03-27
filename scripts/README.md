# Opifex Scripts

Utility scripts for environment setup, verification, benchmarking, and notebook generation.

Most users only need the root-level `setup.sh` and `activate.sh` -- these scripts are called by them or used for specific workflows.

## Scripts

### setup_env.py

Generates the `.opifex.env` file with backend-specific JAX environment variables. Called automatically by `setup.sh`.

```bash
# Usually not called directly -- setup.sh handles this
python scripts/setup_env.py write --backend auto --output .opifex.env
```

### verify_opifex_gpu.py

GPU verification and diagnostics. Called by `setup.sh` after installation and recommended after `source ./activate.sh`.

```bash
uv run python scripts/verify_opifex_gpu.py
```

### jupytext_converter.py

Converts between Python scripts (`.py` with `py:percent` format) and Jupyter notebooks (`.ipynb`).

```bash
# Convert a Python script to notebook
python scripts/jupytext_converter.py py-to-nb examples/pinns/poisson.py

# Convert a notebook to Python script
python scripts/jupytext_converter.py nb-to-py examples/pinns/poisson.ipynb

# Batch convert all examples
python scripts/jupytext_converter.py batch-py-to-nb examples/
```

### run_ci_benchmark.py

Benchmark regression guard used in CI (`.github/workflows/benchmark-regression.yml`). Runs a small FNO/Burgers benchmark and checks for regressions against stored baselines.

```bash
uv run python scripts/run_ci_benchmark.py
```

### run_pdebench_comparison.py

Standalone research script that benchmarks Opifex neural operators against PDEBench published baselines. Downloads HDF5 datasets from PDEBench.

```bash
source activate.sh
uv run python scripts/run_pdebench_comparison.py
```

## Common Workflows

### Initial Setup

```bash
./setup.sh              # Calls setup_env.py internally
source ./activate.sh    # Prints backend info
```

### After Setup Verification

```bash
uv run python scripts/verify_opifex_gpu.py   # Detailed GPU diagnostics
uv run pytest tests/ -v                       # Run tests
```

### Regenerating Notebooks

```bash
python scripts/jupytext_converter.py batch-py-to-nb examples/
```
