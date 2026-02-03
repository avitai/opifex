# Neural Tangent Kernel Analysis

The Neural Tangent Kernel (NTK) provides theoretical insights into neural network training dynamics. Opifex's NTK module enables spectral analysis for understanding convergence behavior and diagnosing training issues.

## Overview

NTK analysis offers powerful tools for understanding PINN training:

- **Eigenvalue decomposition** reveals convergence rates for different solution modes
- **Condition number** indicates optimization difficulty
- **Spectral bias detection** identifies slow-converging components
- **Convergence prediction** estimates training time to target accuracy

!!! tip "Survey Reference"
    This implementation follows the theoretical framework from Section 3 of the PINN survey (arXiv:2601.10222v1).

## Theoretical Foundation

### NTK Definition

For a neural network $f(x; \theta)$, the empirical Neural Tangent Kernel is:

$$\Theta(x_1, x_2) = J(x_1) J(x_2)^T$$

where $J(x) = \nabla_\theta f(x; \theta)$ is the Jacobian of the network output with respect to parameters.

### Mode-wise Error Decay

During gradient descent training, the error decomposes into eigenmodes:

$$e_k = \sum_i c_i (1 - \alpha \lambda_i)^k q_i$$

where:

- $e_k$: Error at iteration $k$
- $\lambda_i$: NTK eigenvalues
- $q_i$: Eigenvectors
- $c_i$: Initial mode coefficients
- $\alpha$: Learning rate

### Spectral Bias

Networks exhibit **spectral bias**: modes with larger eigenvalues converge faster. This means:

- **High-frequency components** (small eigenvalues) converge slowly
- **Low-frequency components** (large eigenvalues) converge quickly
- The **condition number** $\kappa = \lambda_{max}/\lambda_{min}$ determines the spread in convergence rates

## Components

Opifex provides a functional API for NTK computation and analysis, designed to work seamlessly with JAX and Flax NNX.

### NTK Computation

The core module `opifex.diagnostics.ntk_computation` handles the efficient computation of the NTK matrix.

```python
from flax import nnx
import jax.numpy as jnp
from opifex.diagnostics.ntk_computation import compute_ntk

# 1. Define your model
class MyModel(nnx.Module):
    def __init__(self, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(2, 64, rngs=rngs)
        self.linear2 = nnx.Linear(64, 1, rngs=rngs)

    def __call__(self, x):
        x = nnx.tanh(self.linear1(x))
        return self.linear2(x)

model = MyModel(rngs=nnx.Rngs(0))

# 2. Prepare input data
x = jnp.linspace(-1, 1, 50).reshape(-1, 1)
x = jnp.hstack([x, x**2])  # 2D input

# 3. Compute NTK
# Returns (batch, batch) matrix
ntk_matrix = compute_ntk(model, x)
```

**Computational Note:** The implementation uses `jax.jacrev` and `jax.vmap` for efficient Jacobian computation. For large datasets, consider computing the NTK on a representative subset of data points to avoid $O(N^2)$ memory scaling.

### Spectrum Analysis

The `opifex.diagnostics.spectrum_analysis` module provides tools to analyze the spectral properties of the NTK.

```python
from opifex.diagnostics.spectrum_analysis import (
    compute_ntk_spectrum,
    compute_condition_number,
    effective_dimension
)

# Compute eigenvalues and eigenvectors
# Eigenvalues are sorted in descending order
eigenvalues, eigenvectors = compute_ntk_spectrum(ntk_matrix)

# 1. Condition Number
# Ratio of largest to smallest eigenvalue (κ = λ_max / λ_min)
kappa = compute_condition_number(ntk_matrix)
print(f"Condition Number: {kappa:.2e}")

# 2. Effective Dimension
# Measures the number of effectively determined parameters/directions
# N_eff(γ) = Σ λ_i / (λ_i + γ)
eff_dim = effective_dimension(eigenvalues, gamma=1e-4)
print(f"Effective Dimension: {eff_dim:.2f}")
```

### Spectral Filtering

You can project signals (like residuals or target functions) onto the principal components of the NTK to analyze which modes are being learned.

```python
from opifex.diagnostics.spectrum_analysis import ntk_spectral_filtering

# Assume 'residuals' is a vector of size (N,) matching the training points
residuals = jnp.ones(50)

# Filter to keep only the top-5 spectral components
# This shows the part of the signal corresponding to the "fastest" learning modes
filtered_residuals = ntk_spectral_filtering(
    gradient_vector=residuals,
    eigenvectors=eigenvectors,
    k=5
)
```

## Interpreting Results

| Metric | Interpretation | Action |
|--------|----------------|--------|
| **Condition Number ($\kappa$)** | High $\kappa$ (> $10^6$) implies severe ill-conditioning and slow convergence for some modes. | Use better initialization, normalization, or multilevel training. |
| **Eigenvalue Decay** | Rapid decay indicates "spectral bias" — the network prefers low-frequency functions. | If target is high-frequency, use Fourier features or sine activations. |
| **Effective Dimension** | Low effective dimension compared to parameter count suggests parameter redundancy. | Network pruning or smaller architecture might suffice. |

## Integration with Training

You can monitor the condition number during training to detect optimization difficulties dynamically.

```python
# In your training loop:
if step % 100 == 0:
    ntk = compute_ntk(model, x_batch)
    kappa = compute_condition_number(ntk)
    print(f"Step {step}, Condition Number: {kappa:.2e}")
```


## See Also

- [Training Guide](../user-guide/training.md) - General training procedures
- [Second-Order Optimization](second-order-optimization.md) - Curvature-based methods
- [GradNorm](gradnorm.md) - Gradient-based loss balancing
- [API Reference](../api/physics.md#ntk) - Complete API documentation
