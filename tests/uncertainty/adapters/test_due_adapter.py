"""DUE (Deterministic Uncertainty Estimation) adapter contracts.

DUE (van Amersfoort, van der Wilk, Hensman 2021, arXiv:2102.11409) pairs a
spectral-normalized / bi-Lipschitz deep feature extractor ``f(x)`` with an
inducing-point sparse variational GP (SVGP) over the features; a single
forward pass yields the GP predictive mean + variance.

The opifex :class:`DUEAdapter` wraps an ALREADY-FITTED deep-kernel feature
extractor + :class:`SVGPState` (the GP posterior over the FEATURES) and reuses
opifex's :func:`opifex.uncertainty.gp.predict_svgp` (Titsias-collapsed SVGP,
GPJax-grounded) for the predictive. The spectral-normalization / bi-Lipschitz
feature training is upstream
(``opifex.neural.operators.specialized.spectral_normalization``); this adapter
does not train features — it evaluates the fitted deep-kernel + SVGP at
predict time.

These tests construct a real :class:`SVGPState` via :func:`fit_svgp` on the
feature space (a fixed linear ``feature_fn``) and assert:

* the adapter satisfies :class:`ModelUncertaintyAdapterProtocol`;
* ``predict_distribution`` cross-checks EXACTLY against ``predict_svgp`` on the
  features, carries non-negative epistemic, and re-tags ``method == "due"``;
* ``wrap`` rejects a non-DUE capability;
* the predictive is ``jit`` / ``vmap`` / ``grad`` safe;
* :class:`DUEState` is a pytree whose ``svgp_state`` arrays travel through
  ``tree_map`` while ``feature_fn`` stays static.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.adapters import (
    DUEAdapter,
    DUEState,
    ModelUncertaintyAdapterProtocol,
)
from opifex.uncertainty.gp import fit_svgp, predict_svgp, rbf_kernel, SVGPState
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import PredictiveDistribution


# A fixed deterministic linear feature map standing in for a frozen,
# spectral-normalized deep feature extractor f(x). The GP is fit over these
# features, so the adapter exercises the genuine deep-kernel + SVGP path.
_FEATURE_SCALE: float = 1.5


def _feature_fn(x: jax.Array) -> jax.Array:
    """Frozen deterministic feature map ``x -> _FEATURE_SCALE * x``."""
    return _FEATURE_SCALE * x


def _due_capability() -> UQCapability:
    """Return the DUE capability advertised by the adapter."""
    return UQCapability(
        supports_calibration=True,
        supports_ood_detection=True,
        native_nnx_module=True,
        default_strategy=DefaultStrategy.DUE,
        source_package="opifex",
    )


def _fit_feature_svgp() -> tuple[SVGPState, jax.Array]:
    """Fit a small SVGP on the FEATURE space; return state + a test batch.

    The training/inducing inputs are pushed through ``_feature_fn`` before
    :func:`fit_svgp` so the GP posterior lives over ``f(x)`` exactly as the
    DUE adapter evaluates it at predict time.
    """
    key = jax.random.key(0)
    x_train = jnp.linspace(-1.0, 1.0, 24).reshape(-1, 1)
    y_train = jnp.sin(3.0 * x_train[:, 0]) + 0.05 * jax.random.normal(key, (24,))
    x_inducing = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    state = fit_svgp(
        x_train=_feature_fn(x_train),
        y_train=y_train,
        x_inducing=_feature_fn(x_inducing),
        lengthscale=0.4,
        output_scale=1.0,
        noise_std=0.1,
        kernel_fn=rbf_kernel,
    )
    x_test = jnp.linspace(-0.8, 0.8, 5).reshape(-1, 1)
    return state, x_test


def test_due_adapter_satisfies_protocol() -> None:
    adapter: object = DUEAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


def test_due_predict_distribution_matches_predict_svgp_on_features() -> None:
    """DUE predictive EQUALS ``predict_svgp`` over ``feature_fn(x)``; method re-tagged."""
    svgp_state, x_test = _fit_feature_svgp()
    state = DUEState(feature_fn=_feature_fn, svgp_state=svgp_state)
    adapter = DUEAdapter()
    wrapped = adapter.wrap(model=state, capability=_due_capability())

    dist = wrapped.predict_distribution(x_test)
    reference = predict_svgp(state=svgp_state, x_test=_feature_fn(x_test))

    assert isinstance(dist, PredictiveDistribution)
    assert dist.variance is not None
    assert reference.variance is not None
    assert dist.mean.shape == (x_test.shape[0],)
    assert dist.variance.shape == (x_test.shape[0],)
    # Cross-check: the moments are byte-for-byte the SVGP-on-features moments.
    assert bool(jnp.array_equal(dist.mean, reference.mean))
    assert bool(jnp.array_equal(dist.variance, reference.variance))
    assert dist.epistemic is not None
    assert bool(jnp.all(dist.epistemic >= 0.0))
    # Provenance is re-tagged from the SVGP's gaussian_process method to DUE.
    meta = dist.metadata_dict()
    assert meta["method"] == "due"
    assert meta["source_package"] == "opifex"
    assert int(meta["num_inducing"]) == int(svgp_state.x_inducing.shape[0])
    assert reference.metadata_dict()["method"] == "gaussian_process"


def test_due_adapter_rejects_non_due_capability() -> None:
    """Only the ``DUE`` strategy is permitted."""
    svgp_state, _ = _fit_feature_svgp()
    state = DUEState(feature_fn=_feature_fn, svgp_state=svgp_state)
    adapter = DUEAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.DETERMINISTIC)
    with pytest.raises(ValueError, match="DUE"):
        adapter.wrap(model=state, capability=capability)


def test_due_predict_is_jit_compatible() -> None:
    """``jax.jit`` of the predictive mean is finite and correctly shaped."""
    svgp_state, x_test = _fit_feature_svgp()
    state = DUEState(feature_fn=_feature_fn, svgp_state=svgp_state)
    wrapped = DUEAdapter().wrap(model=state, capability=_due_capability())

    @jax.jit
    def _mean(x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x).mean

    out = _mean(x_test)
    assert out.shape == (x_test.shape[0],)
    assert bool(jnp.all(jnp.isfinite(out)))
    assert bool(jnp.allclose(out, wrapped.predict_distribution(x_test).mean))


def test_due_predict_is_vmap_compatible() -> None:
    """``jax.vmap`` over a batch of single-point inputs yields finite means."""
    svgp_state, _ = _fit_feature_svgp()
    state = DUEState(feature_fn=_feature_fn, svgp_state=svgp_state)
    wrapped = DUEAdapter().wrap(model=state, capability=_due_capability())

    batch = jnp.linspace(-0.7, 0.7, 4).reshape(4, 1, 1)  # (batch, 1 point, 1 dim)

    def _mean(x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x).mean

    means = jax.vmap(_mean)(batch)
    assert means.shape == (4, 1)
    assert bool(jnp.all(jnp.isfinite(means)))


def test_due_predict_mean_is_differentiable_through_input_scale() -> None:
    """``jax.grad`` through a feature-space scale on the input is finite."""
    svgp_state, x_test = _fit_feature_svgp()
    state = DUEState(feature_fn=_feature_fn, svgp_state=svgp_state)
    wrapped = DUEAdapter().wrap(model=state, capability=_due_capability())

    def _objective(scale: jax.Array) -> jax.Array:
        return jnp.sum(wrapped.predict_distribution(scale * x_test).mean)

    grad = jax.grad(_objective)(jnp.asarray(1.0))
    assert grad.shape == ()
    assert bool(jnp.isfinite(grad))


def test_due_state_pytree_carries_svgp_state_and_keeps_feature_fn_static() -> None:
    """``svgp_state`` rides ``tree_map``/flatten round-trip; ``feature_fn`` stays static.

    ``DUEState`` is a ``flax.struct.dataclass`` pytree: ``svgp_state`` is a
    pytree-node field (NOT static) and ``feature_fn`` is ``pytree_node=False``.
    The nested :class:`opifex.uncertainty.gp.SVGPState` is itself a plain
    ``@dataclass`` (the deliberate ``opifex.uncertainty.gp`` convention shared
    by ``ExactGPState`` / ``RFFGPState`` / ``QuasisepGPState`` /
    ``StochasticSVGPState``), so it travels as a single opaque pytree leaf —
    its own arrays are not independently registered leaves. The contract
    asserted here is therefore the meaningful one: the fitted ``svgp_state``
    survives the ``DUEState`` flatten/unflatten round-trip with every GP array
    intact, and ``feature_fn`` is excluded from the leaves (it is static).
    """
    svgp_state, _ = _fit_feature_svgp()
    state = DUEState(feature_fn=_feature_fn, svgp_state=svgp_state)

    leaves = jax.tree_util.tree_leaves(state)
    assert leaves, "DUEState must expose pytree leaves (its svgp_state node)."
    # feature_fn is static — it must NOT appear among the pytree leaves.
    assert all(leaf is not _feature_fn for leaf in leaves)
    assert all(not callable(leaf) for leaf in leaves)
    # svgp_state IS a (non-static) pytree node: it appears among the leaves.
    assert any(isinstance(leaf, SVGPState) for leaf in leaves)

    # Identity tree_map is a flatten -> unflatten round-trip: the fitted GP
    # state must carry through unchanged and feature_fn stays the same object.
    round_tripped = jax.tree_util.tree_map(lambda leaf: leaf, state)
    assert round_tripped.feature_fn is state.feature_fn
    assert bool(jnp.array_equal(round_tripped.svgp_state.x_inducing, svgp_state.x_inducing))
    assert bool(jnp.array_equal(round_tripped.svgp_state.scaled_alpha, svgp_state.scaled_alpha))
    assert bool(jnp.array_equal(round_tripped.svgp_state.cholesky_b, svgp_state.cholesky_b))
