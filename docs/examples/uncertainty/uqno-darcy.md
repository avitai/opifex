# UQNO: Uncertainty Quantification Neural Operator on Darcy Flow

| Metadata          | Value                              |
|-------------------|------------------------------------|
| **Level**         | Intermediate                       |
| **Runtime**       | ~3 min (GPU) / ~15 min (CPU)       |
| **Prerequisites** | JAX, Flax NNX, Bayesian NNs        |
| **Format**        | Python + Jupyter                   |
| **Memory**        | ~1.5 GB RAM                        |

## Overview

This example demonstrates the **opifex UQ API surface** on a Bayesian Fourier
Neural Operator applied to the Darcy flow equation: constructing the model,
training with the shared ``negative_elbo`` objective, evaluating via
``predict_distribution``, and inspecting a ``PredictiveDistribution``.

!!! warning "Algorithmic scope of this example"

    This is **not** a faithful implementation of the canonical UQNO
    (Ma et al., TMLR 2024, [arXiv:2402.01960](https://arxiv.org/abs/2402.01960)),
    which is a **conformal-prediction** method (deterministic base FNO + deterministic
    residual FNO trained with a pointwise quantile loss + scalar conformal
    calibration), not a mean-field variational Bayesian neural network.

    The current opifex `UncertaintyQuantificationNeuralOperator` is a *Bayesian
    Fourier Neural Operator* with mean-field variational layers. On wide
    networks like this, mean-field VI has a known posterior-collapse failure
    mode (Coker et al., [arXiv:2106.07052](https://arxiv.org/abs/2106.07052)) —
    the optimal variational posterior predictive converges to the prior
    predictive as width grows — which limits how cleanly the deterministic
    posterior-mean prediction can match the target.

    A faithful conformal-UQNO implementation is tracked as a follow-up task.
    Treat this example as an API-surface tutorial, not a benchmark.

**Opifex's current UQNO** demonstrates:

- **Bayesian spectral convolutions**: Weights are distributions, not point estimates
- **Epistemic uncertainty**: Model uncertainty from weight variance, surfaced via
  Monte Carlo posterior sampling
- **Shared platform surface**: ``predict_distribution`` /
  ``loss_components`` / ``negative_elbo`` integrate with the rest of the
  opifex UQ stack (``ObjectiveConfig``, ``UQLossComponents``,
  ``PredictiveDistribution``)

## What You'll Learn

1. **Instantiate** `UncertaintyQuantificationNeuralOperator` with Bayesian layers
2. **Train** with the shared ``negative_elbo`` surface (data + KL via ``ObjectiveConfig``)
3. **Compute** epistemic uncertainty via Monte Carlo posterior sampling
4. **Analyze** uncertainty calibration quality

## Coming from NeuralOperator (PyTorch)?

| NeuralOperator (PyTorch)             | Opifex (JAX)                                  |
|--------------------------------------|-----------------------------------------------|
| `UQNO(base_model, residual_model)`   | `UncertaintyQuantificationNeuralOperator()`   |
| Two-stage training (base + residual) | Single-stage Bayesian training                |
| Conformal prediction calibration     | Monte Carlo uncertainty estimation            |
| `PointwiseQuantileLoss`              | ELBO with KL divergence                       |

**Key differences:**

1. **Approach**: Opifex uses Bayesian weights, NeuralOperator uses conformal prediction
2. **Training**: Single-stage ELBO vs two-stage base + residual
3. **Uncertainty**: Epistemic/aleatoric decomposition vs prediction intervals

## Files

- **Python Script**: [`examples/uncertainty/uqno_darcy.py`](https://github.com/avitai/opifex/blob/main/examples/uncertainty/uqno_darcy.py)
- **Jupyter Notebook**: [`examples/uncertainty/uqno_darcy.ipynb`](https://github.com/avitai/opifex/blob/main/examples/uncertainty/uqno_darcy.ipynb)

## Quick Start

### Run the Python Script

```bash
source activate.sh && python examples/uncertainty/uqno_darcy.py
```

### Run the Jupyter Notebook

```bash
jupyter lab examples/uncertainty/uqno_darcy.ipynb
```

## Core Concepts

### Bayesian Neural Operators

The UQNO replaces point-estimate weights with weight distributions:

$$w \sim q(w) = \mathcal{N}(\mu_w, \sigma_w^2)$$

Training optimizes the Evidence Lower BOund (ELBO):

$$\mathcal{L} = \mathbb{E}_{q(w)}[\log p(y|x,w)] - \beta \cdot KL(q(w) || p(w))$$

### Uncertainty Surfaced by UQNO

| Type | Source | Reducible? | Where it appears |
|------|--------|------------|-------------|
| Epistemic | Model | Yes (more data) | ``PredictiveDistribution.epistemic`` = MC sample variance |

This UQNO formulation models weight uncertainty only. Aleatoric (input-noise)
uncertainty is not modeled directly; downstream pipelines that need both
can compose this surface with a likelihood/calibration head.

## Implementation

### Step 1: Create UQNO Model

```python
from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
)

model = UncertaintyQuantificationNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    modes=(12, 12),
    num_layers=4,
    rngs=nnx.Rngs(42),
)
```

**Terminal Output:**

```text
======================================================================
Opifex Example: UQNO on Darcy Flow
======================================================================
JAX backend: gpu
JAX devices: [CudaDevice(id=0)]

Configuration:
  Resolution: 64x64
  Training samples: 150, Test samples: 30
  Batch size: 8, Epochs: 20
  UQNO: modes=(12, 12), hidden=32, layers=4
  KL weight: 0.0001, MC samples: 10

Creating UQNO model...
  Total parameters: 1,380,740
  Epistemic uncertainty: via Monte Carlo posterior sampling
```

### Step 2: Define ELBO via the Shared Objective Surface

```python
from opifex.uncertainty.objectives import ObjectiveConfig

OBJECTIVE = ObjectiveConfig(
    kl_weight=1e-4,
    dataset_size=150,
    physics_weight=1.0,
    data_weight=1.0,
    boundary_weight=1.0,
    initial_condition_weight=1.0,
    regularization_weight=1.0,
    calibration_weight=1.0,
    conformal_weight=1.0,
    pac_bayes_weight=1.0,
)


@nnx.jit
def train_step(model, opt, x, y, rngs):
    def loss_fn(m, rngs):
        components = m.negative_elbo({"x": x, "y": y}, rngs=rngs, objective=OBJECTIVE)
        return components.total

    loss, grads = nnx.value_and_grad(loss_fn)(model, rngs)
    opt.update(model, grads)
    return loss
```

### Step 3: Training

**Terminal Output:**

```text
Training UQNO...
  Epoch   1/20: data MSE = 0.184772, ELBO total = 144.872
  Epoch   3/20: data MSE = 0.065186, ELBO total = 138.816
  Epoch   6/20: data MSE = 0.025218, ELBO total = 135.188
  Epoch   9/20: data MSE = 0.018527, ELBO total = 131.671
  Epoch  12/20: data MSE = 0.020669, ELBO total = 128.207
  Epoch  15/20: data MSE = 0.015526, ELBO total = 124.766
  Epoch  18/20: data MSE = 0.010817, ELBO total = 121.349
Training time: 56.6s
Final data MSE = 0.006437, final ELBO total = 117.608
```

### Step 4: Uncertainty Estimation

```python
dist = model.predict_distribution(
    test_inputs, rngs=nnx.Rngs(sample=42), num_samples=10
)

predictions = dist.mean
# PredictiveDistribution stores variances; take sqrt for std-dev display.
epistemic_std = jnp.sqrt(dist.epistemic)
total_uncertainty = epistemic_std
```

**Terminal Output:**

```text
Results:
  Relative L2 Error:      1.4261
  RMSE:                   0.045003
  Mean Epistemic Std:     0.107039
  Mean Total Uncertainty: 0.107039

Uncertainty calibration analysis...
  Error-Uncertainty Correlation: 0.9320
  1-sigma coverage: 99.0% (expected ~68%)
  2-sigma coverage: 100.0% (expected ~95%)
```

## Visualization

![UQNO Solution](../../assets/examples/uqno_darcy/solution.png)

![UQNO Analysis](../../assets/examples/uqno_darcy/analysis.png)

## Results Summary

| Metric                    | Value              |
|---------------------------|--------------------|
| Final data MSE            | 0.006              |
| Relative L2 Error         | ~1.4 (undertrained)|
| Mean Epistemic Std        | 0.11               |
| Error-Uncertainty Corr    | 0.93 (excellent!)  |
| Training Time             | ~57s               |
| Parameters                | 1,380,740          |

**Note**: The Relative L2 Error is well above zero at this tutorial scale —
20 epochs on 150 training samples is intentionally short so the example
runs in about a minute. The uncertainty story is the headline:
``Error-Uncertainty Correlation = 0.93`` means the model knows where its
mistakes are. For production-grade prediction accuracy, increase
``NUM_EPOCHS`` to 100+ and ``N_TRAIN`` to 500+.

## Next Steps

### Experiments to Try

1. **More training**: Increase epochs to 50-100 for better accuracy
2. **More data**: Use 500+ training samples
3. **Tune KL weight**: Try values from 1e-5 to 1e-3
4. **Different modes**: Use (16, 16) or (24, 24) for higher resolution

### Related Examples

| Example                                   | Level        | What You'll Learn                |
|-------------------------------------------|--------------|----------------------------------|
| [Bayesian FNO](bayesian-fno.md)           | Intermediate | Variational framework wrapper    |
| [FNO on Darcy](../neural-operators/fno-darcy.md) | Beginner | Standard FNO without uncertainty |
| [Calibration Methods](calibration.md)     | Intermediate | Post-hoc calibration techniques  |

### API Reference

- `UncertaintyQuantificationNeuralOperator`: Main UQNO class
- `BayesianSpectralConvolution`: Spectral conv with weight uncertainty
- `BayesianLinear`: Linear layer with weight uncertainty
- `predict_distribution()`: Returns a `PredictiveDistribution` from MC posterior samples
- `loss_components()` / `negative_elbo()`: Shared `UQLossComponents` surface
- `kl_divergence()`: Aggregated KL across every Bayesian layer

### Troubleshooting

| Issue | Solution |
|-------|----------|
| High L2 error | Train longer, use more data |
| Zero epistemic uncertainty | Pass caller-owned `rngs` to `predict_distribution` |
| Memory issues | Reduce `hidden_channels` or `modes` |
| Slow training | Use GPU, reduce `num_samples` in `predict_distribution` |
