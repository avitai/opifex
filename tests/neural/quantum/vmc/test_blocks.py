"""Tests for the log-domain determinant building blocks.

The :func:`logdet_matmul` primitive must reproduce a sum of weighted
determinants exactly while staying numerically stable in the log domain
(FermiNet ``network_blocks.logdet_matmul``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from opifex.neural.quantum.vmc.wavefunctions._blocks import (
    construct_input_features,
    logdet_matmul,
    slogdet,
)


def test_slogdet_matches_dense_for_1x1() -> None:
    """The fast 1x1 path returns the same sign and log-magnitude as the dense op."""
    x = jnp.array([[[2.5]]])
    sign, logdet = slogdet(x)
    np.testing.assert_allclose(sign, jnp.array([1.0]))
    np.testing.assert_allclose(logdet, jnp.log(jnp.array([2.5])), atol=1e-12)


def test_logdet_matmul_equals_weighted_determinant_sum() -> None:
    """``logdet_matmul`` reproduces ``sum_i w_i det(X_i)`` (exp of log output)."""
    key = jax.random.PRNGKey(0)
    ndet, n = 4, 3
    mats = jax.random.normal(key, (ndet, n, n), dtype=jnp.float64)
    weights = jax.random.normal(jax.random.PRNGKey(1), (ndet, 1), dtype=jnp.float64)

    sign, log_mag = logdet_matmul([mats], weights)
    got = sign * jnp.exp(log_mag)

    dets = jnp.array([jnp.linalg.det(mats[i]) for i in range(ndet)])
    expected = jnp.sum(dets * weights[:, 0])
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_logdet_matmul_uniform_weights() -> None:
    """With no weights the result is the unweighted sum of determinants."""
    mats = jax.random.normal(jax.random.PRNGKey(2), (2, 2, 2), dtype=jnp.float64)
    sign, log_mag = logdet_matmul([mats], None)
    got = sign * jnp.exp(log_mag)
    expected = jnp.linalg.det(mats[0]) + jnp.linalg.det(mats[1])
    np.testing.assert_allclose(got, expected, rtol=1e-10)


def test_logdet_matmul_is_log_stable_for_large_determinants() -> None:
    """Scaling each matrix by 10 leaves the log output finite (log-sum-exp trick)."""
    mats = 10.0 * jnp.eye(5)[None] * jnp.ones((3, 1, 1))
    _, log_mag = logdet_matmul([mats], None)
    assert jnp.isfinite(log_mag)


def test_construct_input_features_shapes_and_masked_diagonal() -> None:
    """Electron-electron distances zero on the diagonal; shapes follow FermiNet."""
    positions = jax.random.normal(jax.random.PRNGKey(3), (4, 3), dtype=jnp.float64)
    atoms = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    ae, ee, r_ae, r_ee = construct_input_features(positions, atoms)

    assert ae.shape == (4, 2, 3)
    assert ee.shape == (4, 4, 3)
    assert r_ae.shape == (4, 2, 1)
    assert r_ee.shape == (4, 4, 1)
    np.testing.assert_allclose(jnp.diagonal(r_ee[..., 0]), jnp.zeros(4), atol=1e-12)
