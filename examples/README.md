# Opifex Examples

Working examples demonstrating the Opifex framework for scientific machine learning.

## Prerequisites

```bash
source ./activate.sh
python -c "import opifex; print('Opifex imported successfully')"
```

---

## Directory Overview

| Directory | Count | Description |
|-----------|-------|-------------|
| [`getting-started/`](getting-started/) | 2 | Minimal first examples for FNO and PINNs |
| [`neural-operators/`](neural-operators/) | 14 | FNO, DeepONet, SFNO, U-FNO, UNO, TFNO, GNO, PINO, and more |
| [`pinns/`](pinns/) | 11 | Physics-informed neural networks for various PDEs |
| [`domain-decomposition/`](domain-decomposition/) | 3 | FBPINN, XPINN, CPINN for parallel PDE solving |
| [`advanced-training/`](advanced-training/) | 3 | NTK analysis, GradNorm, adaptive sampling |
| [`optimization/`](optimization/) | 2 | Learn-to-optimize and meta-optimization (MAML/Reptile) |
| [`uncertainty/`](uncertainty/) | 3 | Calibration, UQNO, Bayesian FNO |
| [`quantum-chemistry/`](quantum-chemistry/) | 2 | Neural DFT and neural XC functionals |
| [`layers/`](layers/) | 4 | DISCO convolutions, grid embeddings, Fourier continuation, spectral norm |
| [`data/`](data/) | 3 | Darcy flow analysis, spectral analysis, PDEBench loading |
| [`benchmarking/`](benchmarking/) | 2 | Operator benchmarks and GPU profiling |
| [`discovery/`](discovery/) | 1 | SINDy equation discovery on Lorenz system |
| [`fields/`](fields/) | 1 | Differential operators, advection, pressure projection |
| [`distributed/`](distributed/) | 1 | Data-parallel multi-GPU training |

Each `.py` file has a corresponding `.ipynb` notebook.

---

## Quick Start

```bash
# Activate environment first
source ./activate.sh

# Getting started
python getting-started/first_neural_operator.py
python getting-started/first_pinn.py

# Neural operators
python neural-operators/fno_darcy.py
python neural-operators/deeponet_darcy.py
python neural-operators/operator_tour.py

# PINNs
python pinns/poisson.py
python pinns/burgers.py

# Discovery
python discovery/sindy_lorenz.py

# Fields
python fields/field_operations.py
```

---

## Code Examples

### Fourier Neural Operator

```python
import jax
from flax import nnx
from opifex.neural.operators.fno import FourierNeuralOperator

rngs = nnx.Rngs(jax.random.PRNGKey(42))
fno = FourierNeuralOperator(
    in_channels=1, out_channels=1, hidden_channels=32,
    modes=8, num_layers=4, rngs=rngs,
)

x = jax.random.normal(jax.random.PRNGKey(0), (4, 1, 64, 64))
y = fno(x)
print(f"FNO: {x.shape} -> {y.shape}")
```

### DISCO Convolutions

```python
import jax
from flax import nnx
from opifex.neural.operators.specialized import DiscreteContinuousConv2d

rngs = nnx.Rngs(jax.random.PRNGKey(42))
disco_conv = DiscreteContinuousConv2d(
    in_channels=3, out_channels=16, kernel_size=5,
    activation=nnx.gelu, rngs=rngs,
)

x = jax.random.normal(jax.random.PRNGKey(0), (8, 64, 64, 3))
y = disco_conv(x)
print(f"DISCO: {x.shape} -> {y.shape}")
```

---

## Troubleshooting

**Import errors:** Run `source ./activate.sh` before running examples.

**GPU availability:** `python -c "import jax; print(jax.default_backend(), jax.devices())"`

**Memory issues:** Reduce `BATCH_SIZE` or `N_TRAIN` in the example configuration section.

**CPU-only mode:** `JAX_PLATFORM_NAME=cpu python your_example.py`

---

## Additional Resources

- [Examples Documentation](../docs/examples/index.md) -- detailed write-ups for each example
- [API Reference](../docs/api/) -- module documentation
- [Development Guide](../docs/development/) -- contributing guidelines
