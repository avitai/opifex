"""Tests for the ``BayesianLinear`` / ``BayesianSpectralConvolution`` NNX modules.

Container pattern: NNX state-owning surface (``nnx.Module``) — orthogonal to
the value-object container patterns that govern dataclasses elsewhere.

RNG safety:

* Constructor ``rngs`` initializes parameters only.
* Stochastic sampling routes through caller-owned ``nnx.Rngs`` (advancing
  the named ``"posterior"`` stream) or an explicit ``jax.Array`` key.
* Resolution goes through
  ``artifex.generative_models.core.rng.extract_rng_key`` — canonical Avitai
  helper.
* No hidden ``jax.random.PRNGKey(0)`` fallbacks in production paths.

KL helper: ``kl_divergence()`` delegates to
:func:`opifex.uncertainty.kernels.bayesian.diagonal_gaussian_kl` so the
formula is owned in exactly one place across the platform.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from artifex.generative_models.core.rng import extract_rng_key as artifex_extract_rng_key
from flax import nnx

from opifex.uncertainty.kernels.bayesian import diagonal_gaussian_kl
from opifex.uncertainty.layers.bayesian import BayesianLinear, BayesianSpectralConvolution


def _make_layer(*, prior_std: float = 1.0, seed: int = 0) -> BayesianLinear:
    return BayesianLinear(
        in_features=4,
        out_features=3,
        prior_std=prior_std,
        rngs=nnx.Rngs(seed),
    )


def test_bayesian_linear_initializes_weight_and_bias_distribution_parameters() -> None:
    layer = _make_layer()
    assert layer.weight_mean[...].shape == (3, 4)
    assert layer.weight_logvar[...].shape == (3, 4)
    assert layer.bias_mean[...].shape == (3,)
    assert layer.bias_logvar[...].shape == (3,)


def test_deterministic_mode_returns_identical_outputs() -> None:
    layer = _make_layer()
    x = jnp.ones((2, 4))
    out_a = layer(x, sample=False)
    out_b = layer(x, sample=False)
    assert jnp.array_equal(out_a, out_b)


def test_sampling_with_nnx_rngs_produces_different_outputs_when_stream_advances() -> None:
    layer = _make_layer()
    x = jnp.ones((2, 4))
    rngs = nnx.Rngs(posterior=42)
    out_a = layer(x, sample=True, rngs=rngs)
    out_b = layer(x, sample=True, rngs=rngs)
    assert not jnp.array_equal(out_a, out_b)


def test_sampling_with_explicit_jax_key_is_deterministic_given_key() -> None:
    layer = _make_layer()
    x = jnp.ones((2, 4))
    key = jax.random.PRNGKey(7)
    out_a = layer(x, sample=True, rngs=key)
    out_b = layer(x, sample=True, rngs=key)
    assert jnp.array_equal(out_a, out_b)


def test_sampling_with_different_explicit_keys_produces_different_outputs() -> None:
    layer = _make_layer()
    x = jnp.ones((2, 4))
    out_a = layer(x, sample=True, rngs=jax.random.PRNGKey(0))
    out_b = layer(x, sample=True, rngs=jax.random.PRNGKey(1))
    assert not jnp.array_equal(out_a, out_b)


def test_sampling_without_rngs_raises_value_error_with_posterior_hint() -> None:
    layer = _make_layer()
    x = jnp.ones((2, 4))
    with pytest.raises(ValueError, match=r"posterior"):
        layer(x, sample=True, rngs=None)


def test_sampling_routes_through_artifex_extract_rng_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both nnx.Rngs and jax.Array paths MUST go through Artifex's helper."""
    calls: list[str] = []

    def spy(
        rng: jax.Array | nnx.Rngs | None,
        *,
        streams: tuple[str, ...] = ("sample", "default"),
        context: str = "sampling",
    ) -> jax.Array:
        calls.append(context)
        return artifex_extract_rng_key(rng, streams=streams, context=context)

    monkeypatch.setattr("opifex.uncertainty.layers.bayesian.extract_rng_key", spy)
    layer = _make_layer()
    x = jnp.ones((2, 4))
    layer(x, sample=True, rngs=nnx.Rngs(posterior=0))
    layer(x, sample=True, rngs=jax.random.PRNGKey(0))
    assert len(calls) == 2
    assert all("BayesianLinear" in c for c in calls)


def test_kl_divergence_matches_diagonal_gaussian_kl_helper() -> None:
    """``BayesianLinear.kl_divergence()`` MUST equal the standalone helper sum."""
    layer = _make_layer(prior_std=1.0)
    expected_weight_kl = diagonal_gaussian_kl(
        layer.weight_mean[...],
        layer.weight_logvar[...],
        prior_mean=0.0,
        prior_std=1.0,
    )
    expected_bias_kl = diagonal_gaussian_kl(
        layer.bias_mean[...],
        layer.bias_logvar[...],
        prior_mean=0.0,
        prior_std=1.0,
    )
    expected_total = float(expected_weight_kl + expected_bias_kl)
    assert float(layer.kl_divergence()) == pytest.approx(expected_total, rel=1e-6, abs=1e-7)


def test_kl_divergence_uses_layer_prior_std() -> None:
    layer = _make_layer(prior_std=2.5)
    expected = float(
        diagonal_gaussian_kl(
            layer.weight_mean[...],
            layer.weight_logvar[...],
            prior_mean=0.0,
            prior_std=2.5,
        )
        + diagonal_gaussian_kl(
            layer.bias_mean[...],
            layer.bias_logvar[...],
            prior_mean=0.0,
            prior_std=2.5,
        )
    )
    assert float(layer.kl_divergence()) == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_constructor_rejects_non_positive_prior_std() -> None:
    with pytest.raises(ValueError, match=r"prior_std"):
        BayesianLinear(in_features=4, out_features=3, prior_std=0.0, rngs=nnx.Rngs(0))
    with pytest.raises(ValueError, match=r"prior_std"):
        BayesianLinear(in_features=4, out_features=3, prior_std=-1.0, rngs=nnx.Rngs(0))


def test_no_fixed_prngkey_in_production_path() -> None:
    """AST scan: the layer module must not contain real ``jax.random.PRNGKey(...)`` call sites.

    A plain-text search would false-positive on docstrings; AST walking
    checks only real call expressions.
    """
    import ast
    from pathlib import Path

    tree = ast.parse(Path("src/opifex/uncertainty/layers/bayesian.py").read_text())
    offending: list[str] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "PRNGKey"
        ):
            offending.append(f"line {node.lineno}: {ast.unparse(node)}")
    assert not offending, (
        "BayesianLinear production path must not call jax.random.PRNGKey(...) "
        "directly. Use call-time rngs via extract_rng_key. "
        f"Offending call sites: {offending}"
    )


# ---------------------------------------------------------------------------
# BayesianSpectralConvolution
# ---------------------------------------------------------------------------


def _make_spectral_1d(
    *, in_channels: int = 2, out_channels: int = 3, modes: int = 4, seed: int = 0
) -> BayesianSpectralConvolution:
    return BayesianSpectralConvolution(
        in_channels=in_channels,
        out_channels=out_channels,
        modes=(modes,),
        rngs=nnx.Rngs(seed),
    )


def _make_spectral_2d(
    *,
    in_channels: int = 2,
    out_channels: int = 3,
    modes: tuple[int, int] = (4, 4),
    seed: int = 0,
) -> BayesianSpectralConvolution:
    return BayesianSpectralConvolution(
        in_channels=in_channels,
        out_channels=out_channels,
        modes=modes,
        rngs=nnx.Rngs(seed),
    )


def test_bayesian_spectral_initializes_real_and_imag_weight_parameters_2d() -> None:
    layer = _make_spectral_2d(in_channels=2, out_channels=3, modes=(4, 4))
    expected_shape = (3, 2, 4, 4 // 2 + 1)
    assert layer.weight_mean[...].shape == expected_shape
    assert layer.weight_logvar[...].shape == expected_shape
    assert layer.weight_imag_mean[...].shape == expected_shape
    assert layer.weight_imag_logvar[...].shape == expected_shape


def test_bayesian_spectral_initializes_real_and_imag_weight_parameters_1d() -> None:
    layer = _make_spectral_1d(in_channels=2, out_channels=3, modes=4)
    expected_shape = (3, 2, 4)
    assert layer.weight_mean[...].shape == expected_shape
    assert layer.weight_imag_mean[...].shape == expected_shape


def test_bayesian_spectral_rejects_unsupported_mode_rank() -> None:
    with pytest.raises(ValueError, match=r"modes"):
        BayesianSpectralConvolution(
            in_channels=1,
            out_channels=1,
            modes=(2, 2, 2),
            rngs=nnx.Rngs(0),
        )


def test_bayesian_spectral_rejects_non_positive_prior_std() -> None:
    with pytest.raises(ValueError, match=r"prior_std"):
        BayesianSpectralConvolution(
            in_channels=1, out_channels=1, modes=(4,), prior_std=0.0, rngs=nnx.Rngs(0)
        )


def test_bayesian_spectral_2d_output_shape_matches_uqno_contract() -> None:
    layer = _make_spectral_2d(in_channels=2, out_channels=3, modes=(2, 2))
    x = jnp.ones((1, 2, 8, 8))
    out = layer(x, sample=False)
    assert out.shape == (1, 3, 8, 8)
    assert out.dtype == x.dtype


def test_bayesian_spectral_1d_output_shape_matches_uqno_contract() -> None:
    layer = _make_spectral_1d(in_channels=2, out_channels=3, modes=4)
    x = jnp.ones((1, 2, 16))
    out = layer(x, sample=False)
    assert out.shape == (1, 3, 16)


def test_bayesian_spectral_deterministic_mode_returns_identical_outputs() -> None:
    layer = _make_spectral_2d()
    x = jnp.ones((1, 2, 8, 8))
    a = layer(x, sample=False)
    b = layer(x, sample=False)
    assert jnp.array_equal(a, b)


def test_bayesian_spectral_sampling_with_nnx_rngs_varies_across_calls() -> None:
    layer = _make_spectral_2d()
    x = jnp.ones((1, 2, 8, 8))
    rngs = nnx.Rngs(posterior=0)
    a = layer(x, sample=True, rngs=rngs)
    b = layer(x, sample=True, rngs=rngs)
    assert not jnp.array_equal(a, b)


def test_bayesian_spectral_sampling_with_explicit_key_is_deterministic_given_key() -> None:
    layer = _make_spectral_2d()
    x = jnp.ones((1, 2, 8, 8))
    key = jax.random.PRNGKey(7)
    a = layer(x, sample=True, rngs=key)
    b = layer(x, sample=True, rngs=key)
    assert jnp.array_equal(a, b)


def test_bayesian_spectral_sampling_without_rngs_raises() -> None:
    layer = _make_spectral_2d()
    x = jnp.ones((1, 2, 8, 8))
    with pytest.raises(ValueError, match=r"posterior"):
        layer(x, sample=True, rngs=None)


def test_bayesian_spectral_rejects_mismatched_input_channels() -> None:
    layer = _make_spectral_2d(in_channels=2, out_channels=3)
    x = jnp.ones((1, 5, 8, 8))  # wrong in_channels
    with pytest.raises(ValueError, match=r"in_channels"):
        layer(x, sample=False)


def test_bayesian_spectral_kl_divergence_includes_real_and_imag_weights() -> None:
    """Canonical Li 2D FNO has TWO weight tensors (pos-H, neg-H), each complex.

    KL sums real + imag parts for both positive- and negative-H weight bands.
    """
    layer = _make_spectral_2d(in_channels=2, out_channels=3)
    expected = float(
        diagonal_gaussian_kl(
            layer.weight_mean[...],
            layer.weight_logvar[...],
            prior_mean=0.0,
            prior_std=1.0,
        )
        + diagonal_gaussian_kl(
            layer.weight_imag_mean[...],
            layer.weight_imag_logvar[...],
            prior_mean=0.0,
            prior_std=1.0,
        )
        + diagonal_gaussian_kl(
            layer.weight_neg_h_mean[...],
            layer.weight_neg_h_logvar[...],
            prior_mean=0.0,
            prior_std=1.0,
        )
        + diagonal_gaussian_kl(
            layer.weight_neg_h_imag_mean[...],
            layer.weight_neg_h_imag_logvar[...],
            prior_mean=0.0,
            prior_std=1.0,
        )
    )
    assert float(layer.kl_divergence()) == pytest.approx(expected, rel=1e-6, abs=1e-7)


def test_bayesian_spectral_2d_initializes_positive_and_negative_h_weight_tensors() -> None:
    """Canonical Li 2D FNO has TWO weight tensors per spectral conv."""
    layer = _make_spectral_2d(in_channels=2, out_channels=3, modes=(4, 4))
    expected_shape = (3, 2, 4, 4 // 2 + 1)
    assert layer.weight_neg_h_mean[...].shape == expected_shape
    assert layer.weight_neg_h_imag_mean[...].shape == expected_shape


def test_bayesian_spectral_1d_does_not_create_negative_h_weights() -> None:
    """1D rfft has no negative-frequency band — only positive weights exist."""
    layer = _make_spectral_1d()
    assert not hasattr(layer, "weight_neg_h_mean")


def test_bayesian_spectral_routes_through_extract_rng_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def spy(
        rng: jax.Array | nnx.Rngs | None,
        *,
        streams: tuple[str, ...] = ("sample", "default"),
        context: str = "sampling",
    ) -> jax.Array:
        calls.append(context)
        return artifex_extract_rng_key(rng, streams=streams, context=context)

    monkeypatch.setattr("opifex.uncertainty.layers.bayesian.extract_rng_key", spy)
    layer = _make_spectral_2d()
    x = jnp.ones((1, 2, 8, 8))
    layer(x, sample=True, rngs=nnx.Rngs(posterior=0))
    layer(x, sample=True, rngs=jax.random.PRNGKey(0))
    assert len(calls) == 2
    assert all("BayesianSpectralConvolution" in c for c in calls)
