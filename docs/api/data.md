# Data API Reference

The `opifex.data` package provides **datarax-based data loading** infrastructure for scientific machine learning applications with JAX-native performance and single-pass streaming pipelines.

## 🎯 Overview

Opifex uses [datarax](https://github.com/avitai/datarax) `Pipeline` streaming for high-performance data loading with:

- **Eager PDE solution generation**: jit + vmap generators produce whole datasets in one batched call
- **Single-pass pipelines**: memory-efficient streaming that consumers drain once
- **JAX-native batches**: each batch is a plain dict of arrays, ready for JAX training loops
- **Channels-first contract**: every field is `(n, C, *spatial)` for direct use by neural operators
- **Reproducible sampling**: deterministic `seed` controls both generation and the train/val split

### Architecture

```
Data Pipeline Flow:
[generate_* (jit+vmap)] → [train/val split] → [PDELoaders(train, val)] → [Pipeline batches] → Training
```

**Components:**

1. **Generators** (`opifex.data.sources`): eager `generate_*` functions that return a dataset dict `{"input", "output"}`
2. **Loader factories** (`opifex.data.loaders`): `create_*_loader` builders that wrap generated data in datarax `Pipeline`s
3. **PDELoaders**: a frozen dataclass bundling the `train` / `val` pipelines plus split sizes and resolution
4. **Normalization** (`opifex.core.normalization`): `GaussianNormalizer` for optional z-score scaling

## 🏭 Factory Functions

Factory functions are the simplest way to create configured data loaders for common PDE problems. Each returns a frozen `PDELoaders` dataclass:

```python
@dataclass(frozen=True)
class PDELoaders:
    train: Pipeline   # single-pass datarax pipeline of training batches
    val: Pipeline     # single-pass datarax pipeline of validation batches
    n_train: int      # number of training samples
    n_val: int        # number of validation samples
    resolution: int   # spatial resolution of the generated fields
```

A datarax `Pipeline` is **single-pass**: drain it exactly once. Each batch is a dict
`{"input": (b, C, *spatial), "output": (b, C, *spatial)}` in channels-first layout.

### create_burgers_loader

Create loaders for the Burgers equation: `∂u/∂t + u∂u/∂x = ν∂²u/∂x²` (1D, `C=1`).

```python
from opifex.data.loaders import create_burgers_loader

loaders = create_burgers_loader(
    n_samples=1000,               # total dataset size (before split)
    resolution=128,               # spatial grid resolution (1D)
    batch_size=32,                # batch size for both train and val
    viscosity_range=(0.1, 0.1),   # range for the viscosity parameter
    time_range=(0.0, 1.0),        # time integration range
    val_fraction=0.2,             # fraction held out for validation
    seed=42,                      # random seed for generation + split
)

# Drain the single-pass training pipeline
for batch in loaders.train:
    x = batch["input"]    # (b, 1, resolution) initial condition
    y = batch["output"]   # (b, 1, resolution) final-time solution
    # Train model...

# Validation pipeline (drain once)
for batch in loaders.val:
    ...
```

**Parameters:**

- `n_samples` (int): total dataset size before the train/val split
- `resolution` (int): spatial discretization (1D)
- `batch_size` (int): batch size for the train and val pipelines
- `viscosity_range` (tuple): min/max viscosity for generation
- `time_range` (tuple): start/end time for integration
- `val_fraction` (float): fraction of samples held out for validation
- `seed` (int): random seed for reproducible generation and splitting

**Returns:** `PDELoaders` with `train` / `val` pipelines, `n_train`, `n_val`, and `resolution`.

### create_darcy_loader

Create loaders for Darcy flow: `-∇·(a(x)∇u) = f` (2D, `C=1`). Maps the permeability
coefficient field `a(x)` to the steady-state pressure `u(x)`.

```python
from opifex.data.loaders import create_darcy_loader

loaders = create_darcy_loader(
    n_samples=1000,
    resolution=64,                # 64×64 grid
    batch_size=32,
    coeff_range=(0.1, 1.0),       # permeability coefficient range
    field_type="smooth",          # "smooth" (default) or "binary"
    val_fraction=0.2,
    seed=42,
)

for batch in loaders.train:
    permeability = batch["input"]   # (b, 1, 64, 64) a(x)
    pressure = batch["output"]      # (b, 1, 64, 64) u(x)
```

**Key Parameters:**

- `resolution` (int): grid size (default: 64 for a 64×64 grid)
- `coeff_range` (tuple): range for the permeability coefficient
- `field_type` (str): `"smooth"` (default) or `"binary"` coefficient fields
- `val_fraction`, `seed`, `batch_size`, `n_samples`: as in `create_burgers_loader`

### create_diffusion_loader

Create loaders for diffusion-advection: `∂u/∂t + v·∇u = κ∇²u` (2D, `C=1`).

```python
from opifex.data.loaders import create_diffusion_loader

loaders = create_diffusion_loader(
    n_samples=1000,
    resolution=64,
    batch_size=32,
    diffusion_range=(0.01, 0.1),  # κ range
    advection_range=(-1.0, 1.0),  # v range
    val_fraction=0.2,
    seed=42,
)

for batch in loaders.train:
    x = batch["input"]    # (b, 1, 64, 64) initial field
    y = batch["output"]   # (b, 1, 64, 64) final-time field
```

### create_navier_stokes_loader

Create loaders for the incompressible Navier–Stokes equations (2D, `C=2` for the
`(u, v)` velocity components).

```python
from opifex.data.loaders import create_navier_stokes_loader

loaders = create_navier_stokes_loader(
    n_samples=1000,
    resolution=64,
    batch_size=32,
    viscosity_range=(0.001, 0.01),
    time_range=(0.0, 1.0),
    val_fraction=0.2,
    seed=42,
)

for batch in loaders.train:
    x = batch["input"]    # (b, 2, 64, 64) initial (u, v)
    y = batch["output"]   # (b, 2, 64, 64) final-time (u, v)
```

### create_shallow_water_loader

Create loaders for the shallow water equations (2D, `C=3` for `(h, u, v)` — height
and the two velocity components). No extra physics ranges are exposed.

```python
from opifex.data.loaders import create_shallow_water_loader

loaders = create_shallow_water_loader(
    n_samples=1000,
    resolution=64,
    batch_size=32,
    val_fraction=0.2,
    seed=42,
)

for batch in loaders.train:
    x = batch["input"]    # (b, 3, 64, 64) initial (h, u, v)
    y = batch["output"]   # (b, 3, 64, 64) final-time (h, u, v)
```

## 📦 Data Sources

Data sources are eager `generate_*` functions in `opifex.data.sources`. Each is built on
a `jit` + `vmap` kernel and returns the **whole dataset in one call** as a dict:

```python
{"input": ndarray, "output": ndarray}   # both channels-first (n, C, *spatial)
```

The operator maps a conditioning field (`"input"`) to the **final-time** solution
(`"output"`). All signatures are keyword-only.

### generate_burgers

Burgers equation (1D, `C=1`).

```python
from opifex.data.sources import generate_burgers

data = generate_burgers(
    n_samples=1000,
    resolution=128,
    viscosity_range=(0.1, 0.1),
    time_range=(0.0, 1.0),
    seed=42,
)
print(data["input"].shape)   # (1000, 1, 128)
print(data["output"].shape)  # (1000, 1, 128)
```

### generate_darcy

Darcy flow (2D, `C=1`): permeability `a(x)` → pressure `u(x)`.

```python
from opifex.data.sources import generate_darcy

data = generate_darcy(
    n_samples=1000,
    resolution=64,
    coeff_range=(0.1, 1.0),
    field_type="smooth",   # or "binary"
    seed=42,
)
print(data["input"].shape)   # (1000, 1, 64, 64)
```

### generate_diffusion

Diffusion-advection (2D, `C=1`).

```python
from opifex.data.sources import generate_diffusion

data = generate_diffusion(
    n_samples=1000,
    resolution=64,
    diffusion_range=(0.01, 0.1),
    advection_range=(-1.0, 1.0),
    seed=42,
)
```

### generate_navier_stokes

Incompressible Navier–Stokes (2D, `C=2` for `(u, v)`).

```python
from opifex.data.sources import generate_navier_stokes

data = generate_navier_stokes(
    n_samples=1000,
    resolution=64,
    viscosity_range=(0.001, 0.01),
    time_range=(0.0, 1.0),
    seed=42,
)
print(data["output"].shape)  # (1000, 2, 64, 64)
```

### generate_shallow_water

Shallow water equations (2D, `C=3` for `(h, u, v)`).

```python
from opifex.data.sources import generate_shallow_water

data = generate_shallow_water(
    n_samples=500,
    resolution=64,
    seed=42,
)
print(data["output"].shape)  # (500, 3, 64, 64)
```

## 🔄 Normalization

The old Grain `opifex.data.transforms` subpackage (z-score / spectral / noise transforms)
has been removed; there is no in-pipeline transform replacement. Apply normalization
explicitly with `GaussianNormalizer` from `opifex.core.normalization`.

```python
import jax.numpy as jnp
from opifex.core.normalization import GaussianNormalizer
from opifex.data.sources import generate_darcy

data = generate_darcy(n_samples=1000, resolution=64, seed=42)
inputs = jnp.asarray(data["input"])

# Fit on training data, then normalize / denormalize
normalizer = GaussianNormalizer.fit(inputs)
normalized = normalizer.normalize(inputs)
recovered = normalizer.denormalize(normalized)
```

`GaussianNormalizer` is a frozen container of `mean` / `std`; `fit` computes them from
data and `normalize` / `denormalize` apply and invert the z-score scaling.

## 🔧 Advanced Usage

### Materializing a Pipeline into Arrays

A `Pipeline` is single-pass. To collect a whole split into arrays, drain it once and stack:

```python
import jax.numpy as jnp
from opifex.data.loaders import create_darcy_loader

loaders = create_darcy_loader(n_samples=1000, resolution=64, batch_size=32, seed=42)

inputs, outputs = [], []
for batch in loaders.train:        # drain exactly once
    inputs.append(batch["input"])
    outputs.append(batch["output"])

train_inputs = jnp.concatenate(inputs)    # (n_train, C, *spatial)
train_outputs = jnp.concatenate(outputs)
print(loaders.n_train, loaders.n_val, loaders.resolution)
```

### Custom Pipeline from a Generator

To build a bespoke pipeline, start from a `generate_*` dataset and apply your own
preprocessing before batching:

```python
import jax.numpy as jnp
from opifex.core.normalization import GaussianNormalizer
from opifex.data.sources import generate_burgers

# 1. Generate the dataset eagerly
data = generate_burgers(n_samples=1000, resolution=128, seed=42)
inputs = jnp.asarray(data["input"])
outputs = jnp.asarray(data["output"])

# 2. Fit and apply normalization
in_norm = GaussianNormalizer.fit(inputs)
out_norm = GaussianNormalizer.fit(outputs)
inputs = in_norm.normalize(inputs)
outputs = out_norm.normalize(outputs)

# 3. Iterate manual batches in your training loop
batch_size = 32
for start in range(0, inputs.shape[0], batch_size):
    x = inputs[start:start + batch_size]
    y = outputs[start:start + batch_size]
    # Train model...
```

### Multi-Resolution Training

Progressive training from coarse to fine resolution — build fresh loaders per resolution:

```python
from opifex.data.loaders import create_burgers_loader

resolutions = [32, 64, 128]

for resolution in resolutions:
    print(f"Training at resolution {resolution}")

    loaders = create_burgers_loader(
        n_samples=10000,
        resolution=resolution,
        batch_size=32,
        seed=42,
    )

    for epoch in range(epochs_per_resolution):
        for batch in loaders.train:   # rebuild loaders each epoch to re-drain
            # Train model...
            pass
```

> A `Pipeline` is single-pass, so build a fresh `create_*_loader(...)` for each epoch
> (or materialize the arrays once as shown above and slice them per epoch).

### Data Inspection

Examine generated data directly from a generator (no pipeline needed):

```python
from opifex.data.sources import generate_darcy

data = generate_darcy(n_samples=100, resolution=64, seed=42)

print(f"Input shape: {data['input'].shape}")    # (100, 1, 64, 64) permeability
print(f"Output shape: {data['output'].shape}")  # (100, 1, 64, 64) pressure
print(f"Input range: [{data['input'].min():.3f}, {data['input'].max():.3f}]")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(data["input"][0, 0])   # first sample, first channel
plt.colorbar()
plt.title("Permeability Field")

plt.subplot(1, 2, 2)
plt.imshow(data["output"][0, 0])
plt.colorbar()
plt.title("Pressure Field")
plt.show()
```

## 🎓 Training Integration

### With the Unified Trainer

```python
from flax import nnx

from opifex.core.training import Trainer, TrainingConfig
from opifex.neural.operators.fno import FourierNeuralOperator
from opifex.data.loaders import create_darcy_loader

# Create loaders (single PDELoaders bundle holds train + val)
loaders = create_darcy_loader(
    n_samples=10000,
    resolution=64,
    batch_size=32,
    seed=42,
)

# Create model
model = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    modes=12,
    num_layers=4,
    rngs=nnx.Rngs(42),
)

# Configure and train
config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    validation_frequency=10,
)

trainer = Trainer(model, config)
trained_model, history = trainer.train(loaders.train, loaders.val)
```

### Manual Training Loop

For complete control over training:

```python
import jax.numpy as jnp
import optax
from flax import nnx

from opifex.data.loaders import create_burgers_loader

loaders = create_burgers_loader(n_samples=1000, resolution=128, batch_size=32, seed=42)

optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)

for epoch in range(num_epochs):
    # Rebuild the loader each epoch — pipelines are single-pass
    loaders = create_burgers_loader(
        n_samples=1000, resolution=128, batch_size=32, seed=42
    )
    for batch in loaders.train:
        x = batch["input"]
        y_true = batch["output"]

        def loss_fn(model):
            y_pred = model(x)
            return jnp.mean((y_pred - y_true) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(model, grads)

    print(f"Epoch {epoch}, Loss: {loss:.6f}")
```

## 📚 See Also

- [Training API](training.md): Training infrastructure and optimization
- [Neural Operators API](neural.md): Neural network architectures
- [Examples](../examples/index.md): Complete training examples
- [datarax Documentation](https://github.com/avitai/datarax): `Pipeline` streaming internals
