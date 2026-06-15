"""Real Snapshot-Ensemble / SWAG / BatchEnsemble model→UQ adapters.

These adapters mirror :class:`DeepEnsembleAdapter`: each ``wrap`` validates
the fitted-state container and the capability's ``default_strategy``, then
returns a wrapped model whose ``predict_distribution`` produces an
:class:`opifex.uncertainty.types.PredictiveDistribution` in the same field
shape as :class:`_WrappedDeepEnsembleModel`.

References:

* Snapshot ensemble — Huang, Li, Pleiss, Liu, Hopcroft, Weinberger,
  "Snapshot Ensembles: Train 1, Get M for Free", ICLR 2017
  (arXiv:1704.00109): predict by averaging the cyclic-LR snapshots, exactly
  like a deep ensemble.
* SWAG — Maddox, Garipov, Izmailov, Vetrov, Wilson, "A Simple Baseline for
  Bayesian Uncertainty in Deep Learning", NeurIPS 2019 (arXiv:1902.02476):
  sample weights ``θ ~ N(θ_SWA, ½(Σ_diag + Σ_lowrank))``, forward each draw,
  aggregate the predictive mean/variance. Sampling formula cross-checked
  against ``../torch-uncertainty/src/torch_uncertainty/methods/swag.py``
  (``_fullrank_sample``).
* BatchEnsemble — Wen, Tran, Ba, "BatchEnsemble: An Alternative Approach to
  Efficient Ensemble and Lifelong Learning", ICLR 2020 (arXiv:2002.06715):
  rank-1 per-member fast weights ``y_m = ((x ∘ r_m) W) ∘ s_m``. Layer
  formula cross-checked against
  ``../torch-uncertainty/src/torch_uncertainty/layers/batch_ensemble.py``
  (``BatchLinear``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.adapters import (
    BatchEnsembleAdapter,
    BatchEnsembleState,
    ModelUncertaintyAdapterProtocol,
    SnapshotEnsembleAdapter,
    SnapshotEnsembleState,
    SWAGAdapter,
    SWAGState,
)
from opifex.uncertainty.registry import DefaultStrategy, UQCapability
from opifex.uncertainty.types import PredictiveDistribution


# ---------------------------------------------------------------------------
# Snapshot ensemble — average over cyclic-LR snapshots (deep-ensemble shape)
# ---------------------------------------------------------------------------


def _snapshot_a(x: jax.Array) -> jax.Array:
    return jnp.sum(x, axis=-1, keepdims=True)


def _snapshot_b(x: jax.Array) -> jax.Array:
    return 3.0 * jnp.sum(x, axis=-1, keepdims=True)


def test_snapshot_ensemble_adapter_satisfies_protocol() -> None:
    adapter: object = SnapshotEnsembleAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


def test_snapshot_ensemble_adapter_averages_over_snapshots() -> None:
    """Snapshot mean equals the mean over per-snapshot forward passes."""
    state = SnapshotEnsembleState(members=(_snapshot_a, _snapshot_b))
    adapter = SnapshotEnsembleAdapter()
    capability = UQCapability(
        supports_ensemble=True, default_strategy=DefaultStrategy.SNAPSHOT_ENSEMBLE
    )
    wrapped = adapter.wrap(model=state, capability=capability)
    x = jnp.ones((2, 3))
    dist = wrapped.predict_distribution(x)
    assert isinstance(dist, PredictiveDistribution)
    expected_mean = (_snapshot_a(x) + _snapshot_b(x)) / 2.0
    assert dist.mean.shape == (2, 1)
    assert bool(jnp.allclose(dist.mean, expected_mean))
    assert dist.variance is not None
    assert bool(jnp.all(dist.variance >= 0.0))
    assert dist.epistemic is not None
    assert bool(jnp.all(dist.epistemic > 0.0))
    meta = dist.metadata_dict()
    assert meta["method"] == "snapshot_ensemble"
    assert int(meta["num_snapshots"]) == 2


def test_snapshot_ensemble_adapter_rejects_wrong_strategy() -> None:
    state = SnapshotEnsembleState(members=(_snapshot_a, _snapshot_b))
    adapter = SnapshotEnsembleAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.ENSEMBLE)
    with pytest.raises(ValueError, match="SNAPSHOT_ENSEMBLE"):
        adapter.wrap(model=state, capability=capability)


def test_snapshot_ensemble_adapter_validates_state() -> None:
    """A single snapshot is not an ensemble; ``wrap`` must reject it."""
    state = SnapshotEnsembleState(members=(_snapshot_a,))
    adapter = SnapshotEnsembleAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.SNAPSHOT_ENSEMBLE)
    with pytest.raises(ValueError, match="at least 2"):
        adapter.wrap(model=state, capability=capability)


def test_snapshot_ensemble_adapter_is_vmap_compatible() -> None:
    state = SnapshotEnsembleState(members=(_snapshot_a, _snapshot_b))
    adapter = SnapshotEnsembleAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.SNAPSHOT_ENSEMBLE)
    wrapped = adapter.wrap(model=state, capability=capability)
    batched_x = jnp.ones((5, 2, 3))
    means = jax.vmap(lambda xi: wrapped.predict_distribution(xi).mean)(batched_x)
    assert means.shape == (5, 2, 1)
    assert bool(jnp.all(jnp.isfinite(means)))


def test_snapshot_ensemble_adapter_is_jit_compatible() -> None:
    state = SnapshotEnsembleState(members=(_snapshot_a, _snapshot_b))
    adapter = SnapshotEnsembleAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.SNAPSHOT_ENSEMBLE)
    wrapped = adapter.wrap(model=state, capability=capability)

    @jax.jit
    def predict_std(x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x).std()

    out = predict_std(jnp.ones((2, 3)))
    assert out.shape == (2, 1)
    assert bool(jnp.all(out >= 0.0))


# ---------------------------------------------------------------------------
# SWAG — sample weights from N(θ_SWA, ½(Σ_diag + Σ_lowrank)), forward each
# ---------------------------------------------------------------------------


def _linear_forward(flat_params: jax.Array, x: jax.Array) -> jax.Array:
    """Map a flat weight vector to a prediction: ``y = x @ W`` (W = params)."""
    weight = flat_params.reshape(x.shape[-1], 1)
    return x @ weight


def _make_swag_state(*, num_features: int = 3, rank: int = 2) -> SWAGState:
    first = jnp.arange(1.0, num_features + 1.0)
    second = first**2  # variance == 0 at construction
    deviation = jnp.zeros((num_features, rank))
    return SWAGState(
        first_moment=first,
        second_moment=second,
        deviation_matrix=deviation,
        forward_fn=_linear_forward,
        num_samples=8,
    )


def test_swag_adapter_satisfies_protocol() -> None:
    adapter: object = SWAGAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


def test_swag_adapter_mean_matches_swa_weight_for_zero_deviation() -> None:
    """With zero variance and zero deviation, every draw equals the SWA mean.

    Therefore the predictive mean equals the deterministic forward through
    the SWA weight ``θ_SWA = first_moment`` and the epistemic variance is 0.
    """
    state = _make_swag_state()
    adapter = SWAGAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.SWAG)
    wrapped = adapter.wrap(model=state, capability=capability)
    x = jnp.ones((4, 3))
    dist = wrapped.predict_distribution(x, rngs=nnx.Rngs(sample=0))
    assert isinstance(dist, PredictiveDistribution)
    expected_mean = _linear_forward(state.first_moment, x)
    assert dist.mean.shape == (4, 1)
    assert bool(jnp.allclose(dist.mean, expected_mean, atol=1e-5))
    assert dist.epistemic is not None
    assert bool(jnp.allclose(dist.epistemic, 0.0, atol=1e-6))
    meta = dist.metadata_dict()
    assert meta["method"] == "swag"
    assert int(meta["num_samples"]) == 8


def test_swag_adapter_produces_positive_variance_with_nonzero_covariance() -> None:
    """Non-zero diagonal + low-rank covariance yields positive epistemic var."""
    num_features, rank = 3, 2
    first = jnp.zeros(num_features)
    second = jnp.ones(num_features)  # var = 1 - 0 = 1 > 0
    deviation = 0.5 * jnp.ones((num_features, rank))
    state = SWAGState(
        first_moment=first,
        second_moment=second,
        deviation_matrix=deviation,
        forward_fn=_linear_forward,
        num_samples=256,
    )
    adapter = SWAGAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.SWAG)
    wrapped = adapter.wrap(model=state, capability=capability)
    x = jnp.ones((4, 3))
    dist = wrapped.predict_distribution(x, rngs=nnx.Rngs(sample=1))
    assert dist.variance is not None
    assert bool(jnp.all(dist.variance >= 0.0))
    assert dist.epistemic is not None
    assert bool(jnp.all(dist.epistemic > 0.0))
    assert dist.samples is not None
    assert dist.samples.shape == (256, 4, 1)


def test_swag_adapter_requires_caller_owned_rngs() -> None:
    state = _make_swag_state()
    adapter = SWAGAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.SWAG)
    wrapped = adapter.wrap(model=state, capability=capability)
    x = jnp.ones((4, 3))
    with pytest.raises(TypeError):
        wrapped.predict_distribution(x)  # type: ignore[call-arg]


def test_swag_adapter_rejects_wrong_strategy() -> None:
    state = _make_swag_state()
    adapter = SWAGAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.ENSEMBLE)
    with pytest.raises(ValueError, match="SWAG"):
        adapter.wrap(model=state, capability=capability)


def test_swag_adapter_validates_state_shapes() -> None:
    """``first_moment``/``second_moment`` shape mismatch must be rejected."""
    state = SWAGState(
        first_moment=jnp.zeros(3),
        second_moment=jnp.zeros(4),
        deviation_matrix=jnp.zeros((3, 2)),
        forward_fn=_linear_forward,
    )
    adapter = SWAGAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.SWAG)
    with pytest.raises(ValueError, match="share shape"):
        adapter.wrap(model=state, capability=capability)


def test_swag_adapter_is_nnx_jit_compatible() -> None:
    """SWAG predict traces under ``nnx.jit`` with a traced ``rngs`` argument."""
    state = _make_swag_state()
    adapter = SWAGAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.SWAG)
    wrapped = adapter.wrap(model=state, capability=capability)

    @nnx.jit
    def step(rngs: nnx.Rngs, x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x, rngs=rngs).mean

    out = step(nnx.Rngs(sample=3), jnp.ones((4, 3)))
    assert out.shape == (4, 1)
    assert bool(jnp.all(jnp.isfinite(out)))


# ---------------------------------------------------------------------------
# BatchEnsemble — rank-1 per-member fast weights over M members
# ---------------------------------------------------------------------------


def _batch_ensemble_member(
    x: jax.Array, kernel: jax.Array, alpha_m: jax.Array, gamma_m: jax.Array
) -> jax.Array:
    """Reference forward: ``y_m = ((x ∘ r_m) W) ∘ s_m`` (Wen et al. 2020)."""
    return ((x * alpha_m) @ kernel) * gamma_m


def _make_batch_ensemble_state(*, num_members: int = 3) -> BatchEnsembleState:
    kernel = jnp.arange(1.0, 13.0).reshape(3, 4)
    alpha = 1.0 + 0.1 * jnp.arange(num_members * 3).reshape(num_members, 3)
    gamma = 1.0 + 0.2 * jnp.arange(num_members * 4).reshape(num_members, 4)
    return BatchEnsembleState(shared_kernel=kernel, alpha=alpha, gamma=gamma)


def test_batch_ensemble_adapter_satisfies_protocol() -> None:
    adapter: object = BatchEnsembleAdapter()
    assert isinstance(adapter, ModelUncertaintyAdapterProtocol)


def test_batch_ensemble_adapter_mean_matches_member_average() -> None:
    """BatchEnsemble mean equals the average over the M rank-1 members."""
    state = _make_batch_ensemble_state(num_members=3)
    adapter = BatchEnsembleAdapter()
    capability = UQCapability(
        supports_ensemble=True, default_strategy=DefaultStrategy.BATCH_ENSEMBLE
    )
    wrapped = adapter.wrap(model=state, capability=capability)
    x = jnp.ones((2, 3))
    dist = wrapped.predict_distribution(x)
    assert isinstance(dist, PredictiveDistribution)
    members = jnp.stack(
        [
            _batch_ensemble_member(x, state.shared_kernel, state.alpha[m], state.gamma[m])
            for m in range(state.alpha.shape[0])
        ],
        axis=0,
    )
    expected_mean = jnp.mean(members, axis=0)
    expected_var = jnp.var(members, axis=0)
    assert dist.mean.shape == (2, 4)
    assert bool(jnp.allclose(dist.mean, expected_mean))
    assert dist.variance is not None
    assert bool(jnp.allclose(dist.variance, expected_var))
    assert dist.epistemic is not None
    assert bool(jnp.all(dist.epistemic >= 0.0))
    meta = dist.metadata_dict()
    assert meta["method"] == "batch_ensemble"
    assert int(meta["num_members"]) == 3


def test_batch_ensemble_adapter_rejects_wrong_strategy() -> None:
    state = _make_batch_ensemble_state()
    adapter = BatchEnsembleAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.ENSEMBLE)
    with pytest.raises(ValueError, match="BATCH_ENSEMBLE"):
        adapter.wrap(model=state, capability=capability)


def test_batch_ensemble_adapter_validates_member_axis() -> None:
    """``alpha``/``gamma`` must agree on the member axis."""
    state = BatchEnsembleState(
        shared_kernel=jnp.zeros((3, 4)),
        alpha=jnp.ones((3, 3)),
        gamma=jnp.ones((2, 4)),
    )
    adapter = BatchEnsembleAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.BATCH_ENSEMBLE)
    with pytest.raises(ValueError, match="member axis"):
        adapter.wrap(model=state, capability=capability)


def test_batch_ensemble_adapter_is_vmap_compatible() -> None:
    state = _make_batch_ensemble_state()
    adapter = BatchEnsembleAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.BATCH_ENSEMBLE)
    wrapped = adapter.wrap(model=state, capability=capability)
    batched_x = jnp.ones((5, 2, 3))
    means = jax.vmap(lambda xi: wrapped.predict_distribution(xi).mean)(batched_x)
    assert means.shape == (5, 2, 4)
    assert bool(jnp.all(jnp.isfinite(means)))


def test_batch_ensemble_adapter_is_jit_compatible() -> None:
    state = _make_batch_ensemble_state()
    adapter = BatchEnsembleAdapter()
    capability = UQCapability(default_strategy=DefaultStrategy.BATCH_ENSEMBLE)
    wrapped = adapter.wrap(model=state, capability=capability)

    @jax.jit
    def predict_mean(x: jax.Array) -> jax.Array:
        return wrapped.predict_distribution(x).mean

    out = predict_mean(jnp.ones((2, 3)))
    assert out.shape == (2, 4)
    assert bool(jnp.all(jnp.isfinite(out)))
