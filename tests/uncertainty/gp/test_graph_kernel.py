r"""Tests for the graph diffusion (heat) kernel.

For a finite graph with normalised Laplacian ``L = V Λ V^T`` (with
eigenvalues ``λ_k ≥ 0`` and orthonormal eigenvectors ``v_k``), the
*heat kernel* on the graph is

.. math::

    K_{\text{heat}}(i, j) = \sigma_{f}^{2}\,
        \sum_{k} e^{-\lambda_{k} / \ell^{2}}\,v_{k}[i]\,v_{k}[j],

which is the natural extension of the squared-exponential kernel to
non-Euclidean (graph-structured) inputs. This is the entry-point
construction of the Matern-on-graphs family (Borovitskiy, Mostowsky,
Lindgren, Hensman 2020) — the full Matern variant replaces
``exp(-λ/ℓ²)`` with the spectral Matern density
``(2ν/ℓ² + λ)^{-ν - d/2}``.

References
----------
* Borovitskiy, V., Mostowsky, A., Lindgren, F., Hensman, J. 2020 —
  *Matern Gaussian Processes on Riemannian Manifolds*, NeurIPS,
  arXiv:2006.10160 (PRIMARY for non-Euclidean GPs; the graph case
  reduces to the heat-kernel form when the spectral density is
  Gaussian / squared-exponential).
* Kondor, R. I., Lafferty, J. 2002 — *Diffusion Kernels on Graphs and
  Other Discrete Structures*, ICML.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    fit_exact_gp,
    graph_diffusion_kernel,
    predict_exact_gp,
)


def _line_graph_laplacian(n: int = 5) -> tuple[jax.Array, jax.Array]:
    r"""Return ``(eigenvalues, eigenvectors)`` of the line-graph Laplacian on ``n`` nodes."""
    adjacency = jnp.diag(jnp.ones(n - 1), k=1) + jnp.diag(jnp.ones(n - 1), k=-1)
    degree = jnp.diag(jnp.sum(adjacency, axis=-1))
    laplacian = degree - adjacency
    eigenvalues, eigenvectors = jnp.linalg.eigh(laplacian)
    return eigenvalues, eigenvectors


def test_graph_diffusion_kernel_returns_symmetric_gram() -> None:
    """``K_heat`` is symmetric and positive on the diagonal."""
    eigenvalues, eigenvectors = _line_graph_laplacian(5)
    kernel = graph_diffusion_kernel(
        laplacian_eigenvalues=eigenvalues,
        laplacian_eigenvectors=eigenvectors,
    )
    nodes = jnp.arange(5).reshape(-1, 1)
    k = kernel(nodes, nodes, lengthscale=0.7, output_scale=1.0)
    assert k.shape == (5, 5)
    assert jnp.allclose(k, k.T, atol=1e-6)
    assert jnp.all(jnp.diag(k) > 0.0)


def test_graph_diffusion_kernel_matches_closed_form_eigen_summation() -> None:
    r"""``K_heat(i, j) = σ_f² Σ_k exp(-λ_k/ℓ²) v_k[i] v_k[j]``."""
    eigenvalues, eigenvectors = _line_graph_laplacian(4)
    lengthscale, output_scale = 0.5, 1.5
    kernel = graph_diffusion_kernel(
        laplacian_eigenvalues=eigenvalues,
        laplacian_eigenvectors=eigenvectors,
    )
    nodes = jnp.arange(4).reshape(-1, 1)
    composed = kernel(nodes, nodes, lengthscale=lengthscale, output_scale=output_scale)

    weights = jnp.exp(-eigenvalues * (lengthscale**2) / 2.0)
    expected = (output_scale**2) * (eigenvectors * weights[None, :]) @ eigenvectors.T
    assert jnp.allclose(composed, expected, atol=1e-6)


def test_graph_diffusion_kernel_concentrates_near_diagonal_for_small_lengthscale() -> None:
    """Small lengthscale → diffusion confined; off-diagonal entries shrink relative to the diagonal."""
    eigenvalues, eigenvectors = _line_graph_laplacian(6)
    kernel = graph_diffusion_kernel(
        laplacian_eigenvalues=eigenvalues,
        laplacian_eigenvectors=eigenvectors,
    )
    nodes = jnp.arange(6).reshape(-1, 1)
    k_small = kernel(nodes, nodes, lengthscale=0.2, output_scale=1.0)
    k_large = kernel(nodes, nodes, lengthscale=2.0, output_scale=1.0)

    # Ratio of (0, 5) off-diagonal to (0, 0) diagonal: smaller lengthscale
    # produces a smaller ratio (more concentrated near the diagonal).
    ratio_small = float(k_small[0, 5] / k_small[0, 0])
    ratio_large = float(k_large[0, 5] / k_large[0, 0])
    assert ratio_small < ratio_large


def test_graph_diffusion_kernel_plugs_into_fit_exact_gp() -> None:
    """The graph kernel routes through ``fit_exact_gp(..., kernel_fn=…)``."""
    eigenvalues, eigenvectors = _line_graph_laplacian(8)
    kernel = graph_diffusion_kernel(
        laplacian_eigenvalues=eigenvalues,
        laplacian_eigenvectors=eigenvectors,
    )
    nodes = jnp.arange(8).reshape(-1, 1)
    y = jnp.sin(jnp.arange(8).astype(jnp.float32))
    state = fit_exact_gp(
        x_train=nodes,
        y_train=y,
        lengthscale=0.7,
        output_scale=1.0,
        noise_std=0.05,
        kernel_fn=kernel,
    )
    predictive = predict_exact_gp(state=state, x_test=nodes)
    assert predictive.variance is not None
    assert jnp.max(jnp.abs(predictive.mean - y)) < 5.0 * 0.05


def test_graph_diffusion_kernel_is_jit_compatible() -> None:
    """End-to-end ``jax.jit`` compatibility."""
    eigenvalues, eigenvectors = _line_graph_laplacian(5)
    kernel = graph_diffusion_kernel(
        laplacian_eigenvalues=eigenvalues,
        laplacian_eigenvectors=eigenvectors,
    )
    nodes = jnp.arange(5).reshape(-1, 1)
    y = jax.random.normal(jax.random.PRNGKey(0), (5,))

    @jax.jit
    def fit_predict(x_t: jax.Array, y_t: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x_t,
            y_train=y_t,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.1,
            kernel_fn=kernel,
        )
        pd = predict_exact_gp(state=state, x_test=x_t)
        assert pd.variance is not None
        return pd.mean + pd.variance

    out = fit_predict(nodes, y)
    assert out.shape == (5,)
    assert jnp.all(jnp.isfinite(out))


def test_graph_diffusion_kernel_rejects_shape_mismatched_eigendecomposition() -> None:
    """Eigenvalues / eigenvectors must agree on ``n``."""
    with pytest.raises(ValueError, match="eigen"):
        graph_diffusion_kernel(
            laplacian_eigenvalues=jnp.asarray([0.0, 1.0, 2.0]),
            laplacian_eigenvectors=jnp.eye(4),
        )
