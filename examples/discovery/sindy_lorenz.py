# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Equation Discovery with SINDy on the Lorenz System
#
# This example demonstrates how to use Opifex's JAX-native SINDy
# (Sparse Identification of Nonlinear Dynamics) to automatically
# discover governing equations from time-series data.
#
# We recover the Lorenz attractor equations from a simulated trajectory.

# %% Imports
import jax
import jax.numpy as jnp
import matplotlib as mpl


mpl.use("Agg")
import matplotlib.pyplot as plt

from opifex.discovery.sindy import EnsembleSINDy, SINDy, SINDyConfig
from opifex.discovery.sindy.config import EnsembleSINDyConfig
from opifex.discovery.sindy.utils import finite_difference


# %% [markdown]
# ## 1. Generate Lorenz Attractor Data
#
# The Lorenz system is defined by:
#
# $$\frac{dx}{dt} = \sigma(y - x)$$
# $$\frac{dy}{dt} = x(\rho - z) - y$$
# $$\frac{dz}{dt} = xy - \beta z$$
#
# Standard parameters: $\sigma = 10$, $\rho = 28$, $\beta = 8/3$.

# %% Generate trajectory
sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
dt = 0.001
n_steps = 5000

x0 = jnp.array([-8.0, 7.0, 27.0])
trajectory = [x0]
for _ in range(n_steps - 1):
    xn = trajectory[-1]
    dx = jnp.array(
        [
            sigma * (xn[1] - xn[0]),
            xn[0] * (rho - xn[2]) - xn[1],
            xn[0] * xn[1] - beta * xn[2],
        ]
    )
    trajectory.append(xn + dt * dx)

x_data = jnp.stack(trajectory)
print(f"Trajectory shape: {x_data.shape}")
print(f"Time span: {n_steps * dt:.1f}s with dt={dt}")

# %% Compute derivatives (clean, from known equations)
x_dot = jnp.stack(
    [
        jnp.array(
            [
                sigma * (xi[1] - xi[0]),
                xi[0] * (rho - xi[2]) - xi[1],
                xi[0] * xi[1] - beta * xi[2],
            ]
        )
        for xi in x_data
    ]
)

# %% [markdown]
# ## 2. Discover Equations with SINDy

# %% Fit SINDy model
config = SINDyConfig(polynomial_degree=2, threshold=0.3)
model = SINDy(config)
model.fit(x_data, x_dot)

# Print discovered equations
print("Discovered equations:")
for eq in model.equations(["x", "y", "z"]):
    print(f"  {eq}")

# %% Evaluate accuracy
r2 = model.score(x_data, x_dot)
print(f"\nR² score: {r2:.6f}")

# %% [markdown]
# ## 3. Coefficient Sparsity Analysis

# %% Show coefficient matrix
coef = model.coefficients
names = model.feature_names(["x", "y", "z"])

print(f"\nCoefficient matrix ({coef.shape[0]} equations x {coef.shape[1]} library terms):")
print(f"Feature names: {names}")
print(f"Nonzero terms: {int(jnp.sum(jnp.abs(coef) > 0.01))}")

# %% Visualize sparsity pattern
fig, ax = plt.subplots(figsize=(10, 3))
im = ax.imshow(jnp.abs(coef), aspect="auto", cmap="viridis")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["dx/dt", "dy/dt", "dz/dt"])
ax.set_title("SINDy Coefficient Magnitude (Sparsity Pattern)")
plt.colorbar(im, ax=ax, label="|coefficient|")
plt.tight_layout()
plt.savefig("docs/assets/examples/sindy_lorenz/sparsity_pattern.png", dpi=150)
print("Saved: docs/assets/examples/sindy_lorenz/sparsity_pattern.png")
plt.close()

# %% Visualize Lorenz trajectory (3D)
fig = plt.figure(figsize=(10, 7))
ax3d = fig.add_subplot(111, projection="3d")
ax3d.plot(x_data[:, 0], x_data[:, 1], x_data[:, 2], lw=0.5, alpha=0.8)
ax3d.set_xlabel("x")
ax3d.set_ylabel("y")
ax3d.set_zlabel("z")
ax3d.set_title("Lorenz Attractor Trajectory (Training Data)")
plt.tight_layout()
plt.savefig("docs/assets/examples/sindy_lorenz/lorenz_trajectory.png", dpi=150)
print("Saved: docs/assets/examples/sindy_lorenz/lorenz_trajectory.png")
plt.close()

# %% Compare true vs predicted derivatives
x_dot_pred = model.predict(x_data)
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
labels = ["dx/dt", "dy/dt", "dz/dt"]
t_plot = jnp.arange(x_data.shape[0]) * dt
for i, (ax_i, label) in enumerate(zip(axes, labels, strict=False)):
    ax_i.plot(t_plot[:500], x_dot[:500, i], "b-", alpha=0.7, label="True")
    ax_i.plot(t_plot[:500], x_dot_pred[:500, i], "r--", alpha=0.7, label="SINDy")
    ax_i.set_xlabel("Time (s)")
    ax_i.set_ylabel(label)
    ax_i.legend(fontsize=8)
    ax_i.set_title(label)
plt.suptitle("True vs SINDy-Predicted Derivatives")
plt.tight_layout()
plt.savefig("docs/assets/examples/sindy_lorenz/derivative_comparison.png", dpi=150)
print("Saved: docs/assets/examples/sindy_lorenz/derivative_comparison.png")
plt.close()

# %% [markdown]
# ## 4. Ensemble SINDy for Uncertainty Quantification

# %% Fit ensemble model
ensemble_config = EnsembleSINDyConfig(
    polynomial_degree=2,
    threshold=0.3,
    n_models=20,
    bagging_fraction=0.8,
)
ensemble = EnsembleSINDy(ensemble_config)
ensemble.fit(x_data, x_dot, key=jax.random.PRNGKey(42))

print("Ensemble equations (mean ± std):")
for eq in ensemble.equations(["x", "y", "z"]):
    print(f"  {eq}")

# %% Show uncertainty
print(f"\nMean coefficient uncertainty: {float(jnp.mean(ensemble.coef_std)):.4f}")
print(f"Max coefficient uncertainty:  {float(jnp.max(ensemble.coef_std)):.4f}")

# %% [markdown]
# ## 5. Comparison with PySINDy Reference Implementation

# %% Compare with PySINDy (reference)
import time

import numpy as np
import pysindy as ps


# Convert to numpy for PySINDy
x_np = np.array(x_data)
x_dot_np = np.array(x_dot)

# PySINDy fit
t_start = time.perf_counter()
pysindy_model = ps.SINDy(
    optimizer=ps.STLSQ(threshold=0.3),
    feature_library=ps.PolynomialLibrary(degree=2),
)
pysindy_model.fit(x_np, t=dt, x_dot=x_dot_np)
pysindy_time = time.perf_counter() - t_start

# Opifex fit (timed)
t_start = time.perf_counter()
opifex_model = SINDy(SINDyConfig(polynomial_degree=2, threshold=0.3))
opifex_model.fit(x_data, x_dot)
opifex_time = time.perf_counter() - t_start

# Compare
print("=== PySINDy (Reference) ===")
pysindy_model.print()
pysindy_r2 = pysindy_model.score(x_np, t=dt, x_dot=x_dot_np)
print(f"R² score: {pysindy_r2:.6f}")
print(f"Time: {pysindy_time:.4f}s")

print()
print("=== Opifex SINDy (JAX-native) ===")
for eq in opifex_model.equations(["x", "y", "z"]):
    print(f"  {eq}")
opifex_r2 = opifex_model.score(x_data, x_dot)
print(f"R² score: {opifex_r2:.6f}")
print(f"Time: {opifex_time:.4f}s")

print()
print(f"Opifex R² matches PySINDy: {abs(opifex_r2 - pysindy_r2) < 0.001}")

# %% [markdown]
# ## 6. Numerical Differentiation (When Derivatives Unknown)

# %% Demonstrate finite difference differentiation
x_dot_numerical = finite_difference(x_data, dt)

# Compare with true derivatives
error = jnp.mean(jnp.abs(x_dot_numerical[10:-10] - x_dot[10:-10]))
print(f"\nNumerical differentiation error (interior points): {float(error):.6f}")

# Fit SINDy with numerical derivatives
model_numerical = SINDy(SINDyConfig(polynomial_degree=2, threshold=0.3))
model_numerical.fit(x_data, x_dot_numerical)
r2_numerical = model_numerical.score(x_data, x_dot)
print(f"R² with numerical derivatives: {r2_numerical:.6f}")

# %% [markdown]
# ## Summary
#
# | Method | R² Score | Nonzero Terms |
# |--------|----------|---------------|
# | SINDy (clean derivatives) | >0.999 | 7 |
# | SINDy (numerical derivatives) | >0.99 | ~7 |
# | EnsembleSINDy | >0.999 | 7 (with uncertainty) |
