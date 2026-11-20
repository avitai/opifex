# Opifex Scripts

Utility scripts for environment setup, verification, and benchmarking.

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

### run_pdebench_comparison.py

Standalone research script that benchmarks Opifex neural operators against PDEBench published baselines. Downloads HDF5 datasets from PDEBench.

```bash
source activate.sh
uv run python scripts/run_pdebench_comparison.py
```

### check_every_module_imports.py

Imports every `opifex` module with the interpreter that runs it and fails on any import error,
skipping only a module whose missing dependency is a declared optional root. The clean-environment
build check runs it over the installed wheel; `tests/test_capabilities.py` runs it over the
repository's environment.

```bash
python scripts/check_every_module_imports.py
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

### Regenerating a Notebook

An example's notebook is written from its script by substrax's converter, as substrax's
[Writing examples](https://substrax.readthedocs.io/en/latest/guides/writing-examples/) guide
describes:

```bash
uv run substrax-notebooks convert examples/pinns/poisson.py
```
