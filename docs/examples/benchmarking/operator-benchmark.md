# Neural Operator Comparative Benchmark

| Metadata | Value |
|----------|-------|
| **Level** | Advanced |
| **Runtime** | ~3 min (GPU) / ~20 min (CPU) |
| **Prerequisites** | JAX, Flax NNX, Neural Operators |
| **Format** | Python + Jupyter |
| **Memory** | ~4 GB RAM |

## Overview

This benchmark provides a full comparative analysis of three neural
operator architectures -- UNO, FNO, and SFNO -- on the Darcy flow equation
using Opifex's benchmarking infrastructure. Each operator is **trained** with
the standard operator-learning recipe (grid positional embedding, Gaussian
input/output normalization, and the relative-L2 loss), so the accuracy column
reflects learned behaviour rather than random initialization. The benchmark
reports relative-L2 accuracy, MSE, parameter count, training time, and
inference time across multiple grid resolutions.

## What You'll Learn

1. Compare UNO, FNO, and SFNO on Darcy flow across grid resolutions
2. Train every operator with the proven recipe so accuracy is meaningful
3. Use Opifex's `BenchmarkEvaluator` and `AnalysisEngine` for systematic evaluation
4. Generate publication-ready plots and statistical analysis
5. Understand the accuracy / parameter / speed trade-offs between architectures

## Files

- **Python Script**: [`examples/benchmarking/operator_benchmark.py`](https://github.com/avitai/opifex/blob/main/examples/benchmarking/operator_benchmark.py)
- **Jupyter Notebook**: [`examples/benchmarking/operator_benchmark.ipynb`](https://github.com/avitai/opifex/blob/main/examples/benchmarking/operator_benchmark.ipynb)

## Quick Start

### Run the Python Script

```bash
source activate.sh && python examples/benchmarking/operator_benchmark.py
```

### Run the Jupyter Notebook

```bash
jupyter lab examples/benchmarking/operator_benchmark.ipynb
```

## Operators Compared

| Operator | Architecture | Strengths |
|----------|-------------|-----------|
| **UNO** | U-Net + Fourier layers | Multi-scale features via encoder-decoder |
| **FNO** | Spectral convolutions | Resolution-invariant, fast inference |
| **SFNO** | Spherical harmonics | Natural for global/spherical domains |

## How It Works

The benchmark creates all three operators at each resolution (32x32 and 64x64
by default), loads Darcy flow data with `create_darcy_loader`, trains each
operator with the standard recipe, and then evaluates accuracy and inference
time with `BenchmarkEvaluator.evaluate_model()`.

```mermaid
flowchart TB
    A[Configure resolutions<br>32, 64] --> B[Create Grid-Embedded<br>Operators: UNO, FNO, SFNO]
    B --> C[Load + Normalize<br>Darcy via create_darcy_loader]
    C --> D[Train Each Operator<br>relative-L2 loss]
    D --> E[Evaluate Accuracy<br>+ Inference Time]
    E --> F[Statistical Analysis<br>Pairwise comparison]
    F --> G[Generate Report<br>Plots + Summary]
```

## Key Code Patterns

### Grid-Embedded Operator Creation

Every operator is wrapped with a `GridEmbedding2D` that appends normalized
`(x, y)` coordinate channels -- the standard positional encoding for spectral
operators -- and exposes a uniform channels-first interface.

```python
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.common.embeddings import GridEmbedding2D
from opifex.neural.operators.fno.base import FourierNeuralOperator


class FNOWithGrid(nnx.Module):
    """FNO with a 2D grid positional embedding (channels-first interface)."""

    def __init__(self, in_channels, out_channels, hidden_channels, modes,
                 num_layers, *, rngs):
        super().__init__()
        self.grid_embedding = GridEmbedding2D(
            in_channels=in_channels,
            grid_boundaries=[[0.0, 1.0], [0.0, 1.0]],
        )
        self.fno = FourierNeuralOperator(
            in_channels=self.grid_embedding.out_channels,
            out_channels=out_channels, hidden_channels=hidden_channels,
            modes=modes, num_layers=num_layers, rngs=rngs,
        )

    def __call__(self, x):  # x: (batch, channels, H, W)
        x_hwc = jnp.moveaxis(x, 1, -1)
        x_embedded = self.grid_embedding(x_hwc)
        return self.fno(jnp.moveaxis(x_embedded, -1, 1))


operators = {
    "UNO": UNOWithGrid(1, 1, hidden_channels=32, modes=16, n_layers=3, rngs=rngs),
    "FNO": FNOWithGrid(1, 1, hidden_channels=32, modes=16, num_layers=4, rngs=rngs),
    "SFNO": SFNOWithGrid(1, 1, hidden_channels=32, lmax=16, num_layers=4, rngs=rngs),
}
```

> **SFNO note**: the spherical harmonic basis is built lazily and cached. Run a
> single forward pass on a concrete (non-traced) batch *before* training so the
> basis holds constants rather than `jit` tracers.

### Benchmarking with Opifex Infrastructure

```python
from calibrax.core import BenchmarkResult, Metric
from opifex.benchmarking.evaluation_engine import BenchmarkEvaluator
from opifex.benchmarking.analysis_engine import AnalysisEngine
from opifex.benchmarking.results_manager import ResultsManager

evaluator = BenchmarkEvaluator(output_dir="benchmark_results")
result = evaluator.evaluate_model(
    model=model_fn,           # un-normalizes predictions to physical space
    model_name="FNO_64",
    input_data=dataset["x_test"],
    target_data=dataset["y_test_raw"],
    dataset_name="Darcy_64",
)
# result.metrics["mse"].value, result.metrics["relative_l2"].value,
# result.metrics["parameters"].value, result.metadata["execution_time"]
```

### Training Recipe

```python
from opifex.core.training import Trainer, TrainingConfig
from opifex.core.training.config import LossConfig

config = TrainingConfig(
    num_epochs=100, learning_rate=1e-3, batch_size=32,
    loss_config=LossConfig(loss_type="relative_l2"),
)
trainer = Trainer(model=operator, config=config, rngs=nnx.Rngs(42))
trained_model, _ = trainer.fit(
    train_data=(x_train_norm, y_train_norm),
    val_data=(x_test_norm, y_test_norm),
)
```

### Data Loading and Normalization

```python
from opifex.data.loaders import create_darcy_loader

loader = create_darcy_loader(
    n_samples=1000, batch_size=32, resolution=64,
    shuffle=False, seed=42, worker_count=0, enable_normalization=False,
)
# Collect into arrays, then fit Gaussian stats on TRAIN and normalize all splits.
x_mean, x_std = x_train.mean(), x_train.std()
y_mean, y_std = y_train.mean(), y_train.std()
x_train_norm = (x_train - x_mean) / x_std
y_train_norm = (y_train - y_mean) / y_std
# Predictions are un-normalized with (y_mean, y_std) before computing errors.
```

## Running the Benchmark

```bash
# Default: resolutions 32 and 64, 1000 training samples, 100 epochs per operator
source activate.sh && python examples/benchmarking/operator_benchmark.py
```

To customize the study, edit the `NeuralOperatorComparativeStudy(...)` call near
the bottom of the script (`resolution_sizes`, `n_train`, `num_epochs`,
`hidden_channels`, etc.).

## Sample Output

```text
INFO:__main__:RESOLUTION 32x32 STUDY
INFO:__main__:Darcy dataset ready: (992, 1, 32, 32)
INFO:__main__:UNO created for resolution 32
INFO:__main__:FNO created for resolution 32
INFO:__main__:SFNO created for resolution 32
INFO:__main__:UNO on Darcy: relL2=0.7190%, MSE=2.299543e-07, params=34,465,921, train=30.6s, infer=0.0011s
INFO:__main__:FNO on Darcy: relL2=0.5017%, MSE=1.173130e-07, params=4,203,009, train=9.6s, infer=0.0009s
INFO:__main__:SFNO on Darcy: relL2=0.7065%, MSE=2.269025e-07, params=2,101,537, train=7.6s, infer=0.0009s
INFO:__main__:RESOLUTION 64x64 STUDY
INFO:__main__:Darcy dataset ready: (992, 1, 64, 64)
INFO:__main__:UNO on Darcy: relL2=1.4759%, MSE=1.004127e-06, params=34,465,921, train=37.7s, infer=0.0035s
INFO:__main__:FNO on Darcy: relL2=0.5381%, MSE=1.346708e-07, params=4,203,009, train=15.2s, infer=0.0040s
INFO:__main__:SFNO on Darcy: relL2=0.7545%, MSE=2.706596e-07, params=2,101,537, train=12.2s, infer=0.0034s
INFO:__main__:BENCHMARK SUMMARY (Darcy flow, trained operators)
INFO:__main__:Operator    Res      Rel L2           MSE        Params   Train(s)   Infer(s)
INFO:__main__:UNO          32    0.7190%     2.300e-07    34,465,921       30.6     0.0011
INFO:__main__:FNO          32    0.5017%     1.173e-07     4,203,009        9.6     0.0009
INFO:__main__:SFNO         32    0.7065%     2.269e-07     2,101,537        7.6     0.0009
INFO:__main__:UNO          64    1.4759%     1.004e-06    34,465,921       37.7     0.0035
INFO:__main__:FNO          64    0.5381%     1.347e-07     4,203,009       15.2     0.0040
INFO:__main__:SFNO         64    0.7545%     2.707e-07     2,101,537       12.2     0.0034
INFO:__main__:Complete study finished in 173.46 seconds!
INFO:__main__:   Success rate: 100.0%
```

### Per-Resolution Darcy Flow Results (Trained)

| Operator | Resolution | Rel L2 | MSE | Parameters | Train (s) | Infer (s) |
|----------|-----------|--------|-----|------------|-----------|-----------|
| UNO  | 32 | 0.72% | 2.30e-07 | 34,465,921 | 30.6 | 0.0011 |
| FNO  | 32 | 0.50% | 1.17e-07 |  4,203,009 |  9.6 | 0.0009 |
| SFNO | 32 | 0.71% | 2.27e-07 |  2,101,537 |  7.6 | 0.0009 |
| UNO  | 64 | 1.48% | 1.00e-06 | 34,465,921 | 37.7 | 0.0035 |
| FNO  | 64 | 0.54% | 1.35e-07 |  4,203,009 | 15.2 | 0.0040 |
| SFNO | 64 | 0.75% | 2.71e-07 |  2,101,537 | 12.2 | 0.0034 |

### Aggregate Results (Averaged Across Resolutions)

| Operator | Rel L2 | MSE | Parameters | Train (s) | Infer (s) |
|----------|--------|-----|------------|-----------|-----------|
| UNO  | 1.10% | 6.17e-07 | 34,465,921 | 34.1 | 0.0023 |
| FNO  | 0.52% | 1.26e-07 |  4,203,009 | 12.4 | 0.0024 |
| SFNO | 0.73% | 2.49e-07 |  2,101,537 |  9.9 | 0.0021 |

All trained operators reach low-single-digit relative L2 error on Darcy flow.
**FNO** delivers the best accuracy, **SFNO** is the most parameter-efficient and
the fastest to train, and **UNO** is by far the most parameter-heavy because of
its dense U-Net encoder/decoder. (Exact numbers vary slightly run-to-run with
GPU non-determinism.)

## Generated Output

```
benchmark_results/operator_benchmark/
    accuracy_comparison.png            # Relative L2 vs resolution plots
    execution_time_comparison.png      # Inference time distributions
    statistical_analysis.json          # Pairwise statistical comparisons
    comparative_study_report.md        # Full summary report
```

## Troubleshooting

### SFNO Tracer Leak During Training

**Symptom**: `UnexpectedTracerError` from the spherical harmonic transform when
training SFNO.

**Cause**: The SHT basis is built lazily and cached; if the first call happens
inside `jit`, the cached arrays become tracers.

**Solution**: Run one forward pass on a concrete batch before calling
`Trainer.fit()` (the benchmark does this in `_train_operator`).

### Out of Memory at Higher Resolutions

**Symptom**: CUDA OOM at 96x96 or higher resolutions, most often with UNO.

**Solution**: Reduce `batch_size` or `hidden_channels`, or test fewer
resolutions by editing the `resolution_sizes` argument.

## Next Steps

### Experiments to Try

1. **Scale up capacity**: Increase `hidden_channels`, `modes`, and `num_epochs`
2. **Add more operators**: Include TFNO, GINO, MGNO for broader comparison
3. **Add resolutions**: Extend `resolution_sizes` (e.g. `[32, 64, 96]`) to study scaling
4. **Memory profiling**: Use the GPU profiling example to measure memory usage

### Related Examples

| Example | Level | What You'll Learn |
|---------|-------|-------------------|
| [GPU Profiling](./gpu-profiling.md) | Advanced | Memory and compute optimization |
| [FNO Darcy](../neural-operators/fno-darcy.md) | Intermediate | Training FNO on Darcy flow |
| [UNO Darcy](../neural-operators/uno-darcy.md) | Intermediate | Multi-resolution neural operator |

### API Reference

- [`BenchmarkResult`](../../api/benchmarking.md) - Core result container (from calibrax)
- [`BenchmarkEvaluator`](../../api/benchmarking.md) - Model evaluation harness
- [`AnalysisEngine`](../../api/benchmarking.md) - Statistical analysis tools
- [`ResultsManager`](../../api/benchmarking.md) - Results storage and retrieval
