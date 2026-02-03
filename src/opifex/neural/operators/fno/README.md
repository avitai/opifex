# FNO: Fourier Neural Operators

This module provides comprehensive FNO implementations for learning solution operators of PDEs using spectral methods.

## Available Implementations

### Core Components

- **`base.py`**: Core FNO architecture with `SpectralConvolution`, `FourierLayer`, and `FourierNeuralOperator` classes
- **`__init__.py`**: Module exports for easy imports

### Advanced FNO Variants ✅ **NEW**

- **`tensorized.py`**: `TensorizedFourierNeuralOperator` (TFNO) with tensor decomposition for 10-20x parameter reduction
- **`ufno.py`**: `UFourierNeuralOperator` (U-FNO) with multi-scale encoder-decoder architecture
- **`spherical.py`**: `SphericalFourierNeuralOperator` (SFNO) for global climate and planetary science
- **`local.py`**: `LocalFourierNeuralOperator` combining global Fourier with local convolutions
- **`amortized.py`**: `AmortizedFourierNeuralOperator` (AM-FNO) with neural kernel networks for high-frequency problems

### Classical Variants

- **`factorized.py`**: `FactorizedFourierLayer` with tensor decomposition for memory efficiency
- **`multiscale.py`**: `MultiScaleFourierNeuralOperator` for hierarchical multi-scale physics problems

## Quick Usage

```python
from flax import nnx
import jax

# Basic FNO
from opifex.neural.operators.fno import FourierNeuralOperator

rngs = nnx.Rngs(jax.random.PRNGKey(42))
fno = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    modes=16,
    num_layers=4,
    rngs=rngs
)

# Tensorized FNO for parameter efficiency (10-20x compression)
from opifex.neural.operators.fno import TensorizedFourierNeuralOperator

tfno = TensorizedFourierNeuralOperator(
    in_channels=3,
    out_channels=1,
    hidden_channels=64,
    modes=(16, 16),
    factorization="tucker",  # or "cp"
    rank=0.1,  # Compression ratio
    rngs=rngs
)

# U-FNO for multi-scale problems
from opifex.neural.operators.fno import UFourierNeuralOperator

ufno = UFourierNeuralOperator(
    in_channels=2,
    out_channels=1,
    hidden_channels=64,
    modes=(16, 16),
    num_levels=3,  # Multi-scale levels
    rngs=rngs
)

# Spherical FNO for global climate modeling
from opifex.neural.operators.fno import SphericalFourierNeuralOperator

sfno = SphericalFourierNeuralOperator(
    in_channels=4,  # Temperature, pressure, humidity, wind
    out_channels=1,
    hidden_channels=128,
    num_spectral_layers=6,
    rngs=rngs
)
```

For detailed documentation and examples, see the main [Neural Operators README](../README.md).
