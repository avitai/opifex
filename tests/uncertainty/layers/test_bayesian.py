"""Phase 2 Task 2.1 — shared ``BayesianLinear`` NNX module tests.

Container pattern: NNX state-owning surface (``nnx.Module``) — orthogonal to
the dual container pattern that governs value-object dataclasses
(GUIDE_ALIGNMENT §5a applies to value objects, not state-owning modules).

RNG safety (GUIDE_ALIGNMENT items 4, 4a, 5, 7, 9):

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
from opifex.uncertainty.layers.bayesian import BayesianLinear


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
        "directly (GUIDE_ALIGNMENT item 5). Use call-time rngs via "
        f"extract_rng_key. Offending call sites: {offending}"
    )
