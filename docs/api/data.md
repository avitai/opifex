# Data API Reference

The `opifex.data` package provides **Grain-based data loading** infrastructure for scientific machine learning applications with JAX-native performance and efficient multi-process data pipelines.

## 🎯 Overview

Opifex uses [Grain](https://github.com/google/grain) for high-performance data loading with:

- **On-demand PDE solution generation**: Generate data as needed, no pre-computation
- **Lazy evaluation**: Memory-efficient streaming for large datasets
- **Multi-process parallel loading**: Efficient CPU utilization with worker processes
- **JAX-native pipelines**: Seamless integration with JAX training loops
- **Composable transforms**: Modular data preprocessing and augmentation
- **Automatic sharding**: Distributed training support with `grain.ShardByJaxProcess`

### Architecture

```
Data Pipeline Flow:
[DataSource] → [Sampler] → [Transforms] → [Batching] → [DataLoader] → Training
```

**Components:**

1. **DataSource**: Generates or loads individual samples (e.g., `BurgersDataSource`)
2. **Sampler**: Controls iteration order and sharding (e.g., `IndexSampler`)
3. **Transforms**: Process data (normalization, augmentation, spectral features)
4. **Batching**: Combine samples into batches
5. **DataLoader**: Orchestrates the entire pipeline with multi-processing

## 🏭 Factory Functions

Factory functions provide the simplest way to create configured data loaders for common PDE problems.

### create_burgers_loader

Create a data loader for the Burgers equation: `∂u/∂t + u∂u/∂x = ν∂²u/∂x²`

```python
from opifex.data.loaders import create_burgers_loader

loader = create_burgers_loader(
    n_samples=1000,              # Number of PDE solutions
    batch_size=32,               # Batch size for training
    resolution=64,               # Spatial grid resolution
    time_steps=5,                # Number of time steps
    viscosity_range=(0.01, 0.1), # Range for viscosity parameter
    time_range=(0.0, 2.0),       # Time integration range
    dimension="2d",              # "1d" or "2d"
    shuffle=True,                # Shuffle samples
    seed=42,                     # Random seed
    worker_count=4,              # Parallel workers
    enable_normalization=True,   # Apply z-score normalization
    enable_spectral=False,       # Add FFT features
    enable_augmentation=False,   # Add noise augmentation
)

# Use in training loop
for batch in loader:
    x = batch["input"]   # Initial condition
    y = batch["output"]  # Solution trajectory
    # Train model...
```

**Parameters:**

- `n_samples` (int): Total dataset size
- `batch_size` (int): Training batch size
- `resolution` (int): Spatial discretization resolution
- `time_steps` (int): Number of time steps in trajectory
- `viscosity_range` (tuple): Min/max viscosity for generation
- `time_range` (tuple): Start/end time for integration
- `dimension` (str): "1d" or "2d" problem dimension
- `shuffle` (bool): Randomize sample order
- `seed` (int): Random seed for reproducibility
- `worker_count` (int): Number of parallel data loading workers
- `enable_normalization` (bool): Apply z-score normalization
- `normalization_mean` (float): Mean for normalization (default: 0.0)
- `normalization_std` (float): Std for normalization (default: 1.0)
- `enable_spectral` (bool): Add FFT features as additional input
- `enable_augmentation` (bool): Add Gaussian noise for robustness
- `augmentation_noise_level` (float): Noise standard deviation (default: 0.01)

**Returns:** `grain.DataLoader` ready for iteration

### create_darcy_loader

Create a data loader for Darcy flow: `-∇·(a(x)∇u) = f`

```python
from opifex.data.loaders import create_darcy_loader

loader = create_darcy_loader(
    n_samples=1000,
    batch_size=32,
    resolution=85,               # Grid resolution (85×85)
    viscosity_range=(0.5, 2.0),  # Permeability coefficient range
    shuffle=True,
    seed=42,
    worker_count=4,
    enable_normalization=True,
)

for batch in loader:
    permeability = batch["input"]   # a(x) - permeability field
    pressure = batch["output"]      # u(x) - pressure field
```

**Key Parameters:**

- `resolution` (int): Grid size (default: 85 for 85×85 grid)
- `viscosity_range` (tuple): Range for permeability coefficient
- Other parameters same as `create_burgers_loader`

### create_diffusion_loader

Create a data loader for diffusion-advection: `∂u/∂t + v·∇u = κ∇²u`

```python
from opifex.data.loaders import create_diffusion_loader

loader = create_diffusion_loader(
    n_samples=1000,
    batch_size=32,
    resolution=64,
    time_steps=5,
    shuffle=True,
    seed=42,
    worker_count=4,
)
```

### create_shallow_water_loader

Create a data loader for shallow water equations (conservation of mass and momentum).

```python
from opifex.data.loaders import create_shallow_water_loader

loader = create_shallow_water_loader(
    n_samples=500,
    batch_size=16,
    resolution=64,
    shuffle=True,
    seed=42,
    worker_count=4,
)
```

## 📦 Data Sources

Data sources implement the `grain.RandomAccessDataSource` interface for lazy, on-demand data generation.

### BurgersDataSource

Generates Burgers equation solutions on-demand.

```python
from opifex.data.sources import BurgersDataSource

source = BurgersDataSource(
    n_samples=1000,
    resolution=64,
    time_steps=5,
    viscosity_range=(0.01, 0.1),
    time_range=(0.0, 2.0),
    dimension="2d",
    seed=42,
)

# Access individual samples
sample = source[0]  # Returns dict with 'input', 'output', 'coords', 'times'
print(len(source))  # 1000
```

**Features:**

- Deterministic generation: same index → same sample
- Lazy evaluation: solutions computed on access
- Automatic initial condition generation (Gaussian bumps, sine waves, etc.)
- Numerical PDE solver integration

### DarcyDataSource

Generates Darcy flow solutions (permeability → pressure mapping).

```python
from opifex.data.sources import DarcyDataSource

source = DarcyDataSource(
    n_samples=1000,
    resolution=85,
    viscosity_range=(0.5, 2.0),
    seed=42,
)
```

### DiffusionDataSource

Generates diffusion-advection equation solutions.

```python
from opifex.data.sources import DiffusionDataSource

source = DiffusionDataSource(
    n_samples=1000,
    resolution=64,
    time_steps=5,
    seed=42,
)
```

### ShallowWaterDataSource

Generates shallow water equation solutions.

```python
from opifex.data.sources import ShallowWaterDataSource

source = ShallowWaterDataSource(
    n_samples=500,
    resolution=64,
    seed=42,
)
```

## 🔄 Transforms

Grain-compliant transforms for data preprocessing and augmentation.

### NormalizeTransform

Apply z-score normalization: `(x - mean) / std`

```python
from opifex.data.transforms import NormalizeTransform

transform = NormalizeTransform(
    mean=0.0,
    std=1.0,
    epsilon=1e-8,  # Prevent division by zero
)

# Normalizes both 'input' and 'output' in sample dict
normalized_sample = transform.map(sample)
```

### SpectralTransform

Add FFT features for frequency-domain information.

```python
from opifex.data.transforms import SpectralTransform

transform = SpectralTransform()

# Adds 'input_fft' key with rfft of input
sample_with_fft = transform.map(sample)
# Now sample contains: 'input', 'output', 'input_fft'
```

**Use case:** Neural operators benefit from both spatial and spectral features.

### AddNoiseAugmentation

Add Gaussian noise for data augmentation and robustness.

```python
from opifex.data.transforms import AddNoiseAugmentation

augment = AddNoiseAugmentation(
    noise_level=0.01,  # Standard deviation of noise
    seed=42,
)

# Only augments 'input', leaves 'output' unchanged
noisy_sample = augment.map(sample)
```

**Use case:** Training robust models that handle noisy inputs.

## 🔧 Advanced Usage

### Custom Pipeline

Build a custom data pipeline with explicit Grain components:

```python
import grain.python as grain
from opifex.data.sources import BurgersDataSource
from opifex.data.transforms import NormalizeTransform, SpectralTransform

# 1. Create data source
source = BurgersDataSource(n_samples=1000, resolution=64, seed=42)

# 2. Create sampler
sampler = grain.IndexSampler(
    num_records=len(source),
    shuffle=True,
    seed=42,
    shard_options=grain.ShardByJaxProcess(drop_remainder=True),
)

# 3. Build transformation pipeline
operations = [
    NormalizeTransform(mean=0.0, std=1.0),
    SpectralTransform(),
    grain.Batch(batch_size=32, drop_remainder=True),
]

# 4. Create data loader
loader = grain.DataLoader(
    data_source=source,
    sampler=sampler,
    operations=operations,
    worker_count=4,
    worker_buffer_size=20,
)

# 5. Use in training
for batch in loader:
    # batch["input"]: normalized initial conditions
    # batch["input_fft"]: FFT features
    # batch["output"]: normalized solutions
    pass
```

### Multi-Resolution Training

Progressive training from coarse to fine resolution:

```python
resolutions = [32, 64, 128]

for resolution in resolutions:
    print(f"Training at resolution {resolution}")

    loader = create_burgers_loader(
        n_samples=10000,
        batch_size=32,
        resolution=resolution,
        worker_count=4,
    )

    # Train for N epochs at this resolution
    for epoch in range(epochs_per_resolution):
        for batch in loader:
            # Train model...
            pass
```

### Data Inspection

Examine generated data:

```python
loader = create_darcy_loader(n_samples=100, batch_size=1)

# Get first batch
batch = next(iter(loader))

print(f"Input shape: {batch['input'].shape}")    # Permeability field
print(f"Output shape: {batch['output'].shape}")  # Pressure field
print(f"Input range: [{batch['input'].min():.3f}, {batch['input'].max():.3f}]")

# Visualize
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(batch['input'][0, 0])  # First sample, first channel
plt.colorbar()
plt.title("Permeability Field")

plt.subplot(1, 2, 2)
plt.imshow(batch['output'][0, 0])
plt.colorbar()
plt.title("Pressure Field")
plt.show()
```

## 🎓 Training Integration

### With BasicTrainer

```python
from opifex.training.basic_trainer import BasicTrainer, TrainingConfig
from opifex.neural.operators.fno import FourierNeuralOperator
from opifex.data.loaders import create_darcy_loader

# Create data loaders
train_loader = create_darcy_loader(
    n_samples=8000,
    batch_size=32,
    resolution=85,
    shuffle=True,
    worker_count=4,
)

val_loader = create_darcy_loader(
    n_samples=2000,
    batch_size=32,
    resolution=85,
    shuffle=False,
    worker_count=2,
)

# Create model
model = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=64,
    modes=12,
    num_layers=4,
    rngs=rnx.Rngs(42),
)

# Configure training
config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    validation_frequency=10,
)

# Train
trainer = BasicTrainer(model, config)
trained_model, history = trainer.train(train_loader, val_loader)
```

### With Unified Trainer

```python
from opifex.core.training import Trainer, TrainingConfig

config = TrainingConfig(
    num_epochs=100,
    learning_rate=1e-3,
    batch_size=32,  # Optional, can override loader batch size
)

trainer = Trainer(model, config)
trained_model, history = trainer.train(train_loader, val_loader)
```

### Manual Training Loop

For complete control over training:

```python
import optax
from flax import nnx

# Create optimizer
optimizer = nnx.Optimizer(model, optax.adam(1e-3))

# Training loop
for epoch in range(num_epochs):
    for batch in train_loader:
        x = batch["input"]
        y_true = batch["output"]

        # Loss function
        def loss_fn(model):
            y_pred = model(x)
            return jnp.mean((y_pred - y_true) ** 2)

        # Compute gradients and update
        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)

    print(f"Epoch {epoch}, Loss: {loss:.6f}")
```

## 📊 Performance Optimization

### Worker Count Tuning

```python
# CPU-bound tasks: use multiple workers
loader = create_burgers_loader(
    n_samples=10000,
    batch_size=32,
    worker_count=8,  # Utilize multiple CPU cores
)

# I/O-bound or simple transforms: fewer workers
loader = create_darcy_loader(
    n_samples=1000,
    batch_size=32,
    worker_count=2,
)

# Single process for debugging
loader = create_diffusion_loader(
    n_samples=100,
    batch_size=32,
    worker_count=0,  # No multiprocessing
)
```

### Memory Management

```python
# Adjust buffer size for memory/speed tradeoff
import grain.python as grain

loader = grain.DataLoader(
    data_source=source,
    sampler=sampler,
    operations=operations,
    worker_count=4,
    worker_buffer_size=10,  # Default: 20, lower = less memory
)
```

### Prefetching

Grain automatically prefetches batches in background workers for optimal GPU utilization.

## 📚 See Also

- [Training API](training.md): Training infrastructure and optimization
- [Neural Operators API](neural.md): Neural network architectures
- [Examples](../examples/index.md): Complete training examples
- [Grain Documentation](https://github.com/google/grain): Official Grain docs
