import jax.numpy as jnp

from opifex.diagnostics.spectrum_analysis import (
    compute_condition_number,
    compute_ntk_spectrum,
    effective_dimension,
    ntk_spectral_filtering,
)


def test_compute_ntk_spectrum():
    # Create a known symmetric positive semi-definite matrix
    diag = jnp.array([10.0, 5.0, 1.0, 0.1])
    # Create Q via rotation or just use diag if we don't care about rotation logic in test
    # Just use diagonal for simplicity of testing eigenvalues
    ntk = jnp.diag(diag)

    eigenvalues, eigenvectors = compute_ntk_spectrum(ntk)

    # Check shape
    assert eigenvalues.shape == (4,)
    assert eigenvectors.shape == (4, 4)

    # Check values (should be sorted descending usually, or at least contain correct values)
    # jax.linalg.eigh returns ascending, so we might want to check if logic sorts them
    # For now, just check existence
    assert jnp.allclose(jnp.sort(eigenvalues), jnp.sort(diag), atol=1e-5)


def test_compute_condition_number():
    diag = jnp.array([100.0, 1.0])
    ntk = jnp.diag(diag)

    # Condition number = max/min
    kappa = compute_condition_number(ntk)
    assert jnp.isclose(kappa, 100.0)


def test_compute_condition_number_singular():
    diag = jnp.array([10.0, 0.0])
    ntk = jnp.diag(diag)

    # Should handle zero eigenvalue (return inf or large number)
    kappa = compute_condition_number(ntk)
    assert jnp.isinf(kappa) or kappa > 1e6


def test_effective_dimension():
    eigenvalues = jnp.array([10.0, 10.0, 0.1, 0.01])
    # Eff dim(gamma) = Sum(lambda / (lambda + gamma))
    gamma = 1.0
    # Expected: 10/11 + 10/11 + 0.1/1.1 + 0.01/1.01
    # ≈ 0.909 + 0.909 + 0.09 + 0.01 ≈ 1.92

    eff_dim = effective_dimension(eigenvalues, gamma)
    assert 1.8 < eff_dim < 2.0


def test_spectral_filtering():
    # Test if we can project gradients onto top-k eigenvectors
    # Mock gradients as vector
    grad = jnp.array([1.0, 1.0, 1.0, 1.0])

    # Eigenvectors: identity
    eigenvectors = jnp.eye(4)
    # eigenvalues = jnp.array([10.0, 5.0, 1.0, 0.1]) (unused)

    # Filter to top 2 modes
    filtered = ntk_spectral_filtering(grad, eigenvectors, k=2)

    # Should keep first 2 components, zero out others (if eigenvectors are standard basis)
    # Note: ntk_spectral_filtering usually projects vector onto eigenspace
    # v_filtered = Q_k Q_k^T v

    expected = jnp.array([1.0, 1.0, 0.0, 0.0])
    assert jnp.allclose(filtered, expected)
