"""TDD tests for ``opifex.uncertainty.types``.

These tests pin the Phase 1 Task 1.1 contracts:

* ``PredictionInterval`` (lower/upper/coverage/method/metadata)
* ``PredictionSet`` (values/scores/threshold/method/metadata)
* ``PredictiveDistribution`` (full 11-field set per audit lines 122-138)
* ``PredictiveMode`` (StrEnum: predictive / posterior_predictive /
  prior_predictive / mean_only)

Variance fields (``variance``/``epistemic``/``aleatoric``/``total_uncertainty``)
are *variances*, not standard deviations. Additivity is enforced via
``jnp.allclose(total_uncertainty, epistemic + aleatoric, rtol=1e-5, atol=1e-6)``.

PyTree registration is mandatory: each container must round-trip through
``jax.tree_util.tree_flatten`` / ``tree_unflatten`` with array fields exposed as
leaves and metadata kept as static ``aux_data``.

``validate()`` is a public method, never called from ``__post_init__`` or
``tree_unflatten`` (placeholder values during tracing must not raise).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest


def test_predictive_mode_strenum_members() -> None:
    from opifex.uncertainty.types import PredictiveMode

    assert PredictiveMode.PREDICTIVE.value == "predictive"
    assert PredictiveMode.POSTERIOR_PREDICTIVE.value == "posterior_predictive"
    assert PredictiveMode.PRIOR_PREDICTIVE.value == "prior_predictive"
    assert PredictiveMode.MEAN_ONLY.value == "mean_only"

    # StrEnum: members compare equal to their string values
    assert PredictiveMode.PREDICTIVE == "predictive"


def test_predictive_mode_unknown_value_rejected() -> None:
    from opifex.uncertainty.types import PredictiveMode

    with pytest.raises(ValueError, match="not-a-mode"):
        PredictiveMode("not-a-mode")


def test_prediction_interval_construction_and_shape_contract() -> None:
    from opifex.uncertainty.types import PredictionInterval

    lower = jnp.asarray([0.1, 0.2, 0.3])
    upper = jnp.asarray([0.4, 0.5, 0.6])
    interval = PredictionInterval(lower=lower, upper=upper, coverage=0.9, method="conformal")

    assert interval.lower.shape == interval.upper.shape
    assert interval.coverage == 0.9
    assert interval.method == "conformal"
    # Canonical metadata form: tuple-of-pairs (immutable AND hashable for
    # static aux_data in pytree registration).
    assert isinstance(interval.metadata, tuple)
    # And it must be hashable — that's the whole point of choosing tuple.
    hash(interval.metadata)


def test_prediction_interval_coverage_must_be_in_open_unit_interval() -> None:
    from opifex.uncertainty.types import PredictionInterval

    lower = jnp.zeros(3)
    upper = jnp.ones(3)
    with pytest.raises(ValueError, match="coverage"):
        PredictionInterval(lower=lower, upper=upper, coverage=0.0, method="x").validate()
    with pytest.raises(ValueError, match="coverage"):
        PredictionInterval(lower=lower, upper=upper, coverage=1.0, method="x").validate()


def test_prediction_set_shape_contract_classification() -> None:
    from opifex.uncertainty.types import PredictionSet

    # values: (batch, num_classes) boolean inclusion mask
    values = jnp.asarray([[True, False, True], [False, True, True]])
    # scores: (batch, num_classes) per-class nonconformity scores
    scores = jnp.asarray([[0.1, 0.7, 0.2], [0.6, 0.1, 0.3]])
    pset = PredictionSet(values=values, scores=scores, threshold=0.5, method="aps")

    assert pset.values.shape == pset.scores.shape == (2, 3)
    assert pset.values.dtype == jnp.bool_
    # threshold is a Python scalar (static aux_data), not a 0-d array
    assert isinstance(pset.threshold, float)


def test_predictive_distribution_full_field_set() -> None:
    """All 11 audit-mandated fields are present and accept None for optionals."""
    from opifex.uncertainty.types import PredictiveDistribution

    mean = jnp.zeros((4, 2))
    dist = PredictiveDistribution(mean=mean)
    for field in (
        "mean",
        "samples",
        "variance",
        "covariance",
        "epistemic",
        "aleatoric",
        "total_uncertainty",
        "quantiles",
        "interval",
        "prediction_set",
        "metadata",
    ):
        assert hasattr(dist, field), f"PredictiveDistribution missing field: {field}"


def test_predictive_distribution_quantile_accessor() -> None:
    from opifex.uncertainty.types import PredictiveDistribution

    mean = jnp.zeros((3,))
    q_low = jnp.full((3,), -1.0)
    q_med = jnp.zeros((3,))
    q_high = jnp.full((3,), 1.0)
    dist = PredictiveDistribution(
        mean=mean,
        quantiles={0.025: q_low, 0.5: q_med, 0.975: q_high},
    )

    assert jnp.allclose(dist.quantile(0.5), q_med)
    with pytest.raises(KeyError):
        dist.quantile(0.1)


def test_predictive_distribution_std_from_variance() -> None:
    from opifex.uncertainty.types import PredictiveDistribution

    mean = jnp.zeros((3,))
    variance = jnp.asarray([4.0, 9.0, 16.0])
    dist = PredictiveDistribution(mean=mean, variance=variance)

    assert jnp.allclose(dist.std(), jnp.asarray([2.0, 3.0, 4.0]))


def test_predictive_distribution_std_raises_without_variance() -> None:
    from opifex.uncertainty.types import PredictiveDistribution

    dist = PredictiveDistribution(mean=jnp.zeros((3,)))
    with pytest.raises(ValueError, match="variance"):
        dist.std()


def test_predictive_distribution_variance_additivity() -> None:
    """epistemic + aleatoric == total_uncertainty (variances, not std-devs).

    Tolerance: ``rtol=1e-5, atol=1e-6`` — the canonical Phase 1 constants
    re-used by Phase 6 ``SolutionDistribution``.
    """
    from opifex.uncertainty.types import PredictiveDistribution

    mean = jnp.zeros((3,))
    epistemic = jnp.asarray([0.1, 0.2, 0.3])
    aleatoric = jnp.asarray([0.4, 0.5, 0.6])
    total = epistemic + aleatoric

    PredictiveDistribution(
        mean=mean,
        epistemic=epistemic,
        aleatoric=aleatoric,
        total_uncertainty=total,
    ).validate()  # must pass

    # Off by more than tolerance -> validate raises.
    bad_total = total + 0.1
    with pytest.raises(ValueError, match="total_uncertainty"):
        PredictiveDistribution(
            mean=mean,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=bad_total,
        ).validate()


def test_predictive_distribution_is_a_pytree() -> None:
    """Containers cross transforms; array leaves vs. static aux_data."""
    from opifex.uncertainty.types import PredictiveDistribution

    mean = jnp.asarray([1.0, 2.0, 3.0])
    variance = jnp.asarray([0.1, 0.2, 0.3])
    dist = PredictiveDistribution(mean=mean, variance=variance)

    leaves, treedef = jax.tree_util.tree_flatten(dist)
    # Only the array fields appear as leaves; method/metadata stays static.
    assert any(jnp.allclose(leaf, mean) for leaf in leaves if hasattr(leaf, "shape"))
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert jnp.allclose(rebuilt.mean, mean)
    assert jnp.allclose(rebuilt.variance, variance)


def test_predictive_distribution_unflatten_does_not_validate() -> None:
    """Tree unflatten must not raise even with placeholder leaves.

    Per ``/mnt/ssd2/Works/jax/docs/custom_pytrees.md``: transformations may
    reconstruct containers with placeholder/abstract values during tracing.
    """
    from opifex.uncertainty.types import PredictiveDistribution

    epistemic = jnp.asarray([0.1, 0.2, 0.3])
    aleatoric = jnp.asarray([0.4, 0.5, 0.6])
    # Deliberately mis-sized total — validate() would reject this, but
    # tree_unflatten must not call validate().
    bad_total = jnp.asarray([99.0, 99.0, 99.0])
    dist = PredictiveDistribution(
        mean=jnp.zeros((3,)),
        epistemic=epistemic,
        aleatoric=aleatoric,
        total_uncertainty=bad_total,
    )
    leaves, treedef = jax.tree_util.tree_flatten(dist)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    # Construction + unflatten succeeded despite the inconsistency.
    assert isinstance(rebuilt, PredictiveDistribution)


def test_predictive_distribution_round_trips_through_jit() -> None:
    """Jittability: the container is a transform-safe PyTree."""
    from opifex.uncertainty.types import PredictiveDistribution

    @jax.jit
    def add_noise(dist: PredictiveDistribution, noise: jax.Array) -> PredictiveDistribution:
        return PredictiveDistribution(
            mean=dist.mean + noise,
            variance=dist.variance,
        )

    dist = PredictiveDistribution(
        mean=jnp.asarray([1.0, 2.0, 3.0]),
        variance=jnp.asarray([0.1, 0.2, 0.3]),
    )
    out = add_noise(dist, jnp.full((3,), 0.5))
    assert jnp.allclose(out.mean, jnp.asarray([1.5, 2.5, 3.5]))
    assert out.variance is not None and dist.variance is not None
    assert jnp.allclose(out.variance, dist.variance)


def test_predictive_distribution_vmaps_over_batch_dimension() -> None:
    """Differentiability + vmap: containers compose with jax.vmap."""
    from opifex.uncertainty.types import PredictiveDistribution

    def build(mean: jax.Array) -> PredictiveDistribution:
        return PredictiveDistribution(mean=mean, variance=jnp.ones_like(mean))

    batched_mean = jnp.arange(6.0).reshape(3, 2)
    batched_dist = jax.vmap(build)(batched_mean)
    assert batched_dist.mean.shape == (3, 2)


def test_metadata_defaults_are_immutable_and_hashable() -> None:
    """Per GUIDE_ALIGNMENT §16-17: aux_data must be hashable + immutable.

    Canonical metadata representation is ``tuple[tuple[str, Any], ...]`` —
    ``MappingProxyType`` is immutable but **not** hashable so it cannot serve
    as JAX pytree aux_data (which forms part of the JIT cache key).
    """
    from opifex.uncertainty.types import (
        PredictionInterval,
        PredictionSet,
        PredictiveDistribution,
    )

    lower, upper = jnp.zeros(3), jnp.ones(3)
    pi = PredictionInterval(lower=lower, upper=upper, coverage=0.9, method="m")
    ps = PredictionSet(
        values=jnp.asarray([[True, False]]),
        scores=jnp.asarray([[0.1, 0.9]]),
        threshold=0.5,
        method="m",
    )
    pd = PredictiveDistribution(mean=jnp.zeros(3))

    for obj in (pi, ps, pd):
        assert isinstance(obj.metadata, tuple), (
            f"{type(obj).__name__}.metadata must be a tuple-of-pairs; "
            f"got {type(obj.metadata).__name__}"
        )
        # Hashable — required for static aux_data in pytree registration.
        hash(obj.metadata)
        # Ergonomic dict accessor for read access.
        assert obj.metadata_dict() == {}


def test_metadata_dict_accessor_round_trips_user_metadata() -> None:
    from opifex.uncertainty.types import PredictiveDistribution

    pd = PredictiveDistribution(
        mean=jnp.zeros(3),
        metadata=(("method", "monte_carlo"), ("num_samples", 100)),
    )
    assert pd.metadata_dict() == {"method": "monte_carlo", "num_samples": 100}
    # Mutating the dict view does not affect the underlying tuple.
    d = pd.metadata_dict()
    d["method"] = "tampered"
    assert pd.metadata_dict() == {"method": "monte_carlo", "num_samples": 100}


def test_containers_have_slots_and_no_dict() -> None:
    """Avitai-ecosystem-canonical: frozen + slotted dataclass for value objects."""
    from opifex.uncertainty.types import (
        PredictionInterval,
        PredictionSet,
        PredictiveDistribution,
    )

    pi = PredictionInterval(lower=jnp.zeros(3), upper=jnp.ones(3), coverage=0.9, method="x")
    ps = PredictionSet(
        values=jnp.asarray([[True, False]]),
        scores=jnp.asarray([[0.1, 0.9]]),
        threshold=0.5,
        method="x",
    )
    pd = PredictiveDistribution(mean=jnp.zeros(3))

    for obj in (pi, ps, pd):
        assert hasattr(type(obj), "__slots__"), (
            f"{type(obj).__name__} must use __slots__ (Avitai-canonical frozen+slotted pattern)"
        )
        assert not hasattr(obj, "__dict__"), (
            f"{type(obj).__name__} must not have __dict__ when slots=True"
        )


def test_containers_provide_replace_method_for_immutable_updates() -> None:
    """flax.struct.dataclass adds .replace() for immutable updates — the canonical
    pattern for evolving a frozen container without mutation.
    """
    from opifex.uncertainty.types import (
        PredictionInterval,
        PredictiveDistribution,
    )

    pi = PredictionInterval(lower=jnp.zeros(3), upper=jnp.ones(3), coverage=0.9, method="x")
    pi2 = pi.replace(coverage=0.95)  # pyright: ignore[reportAttributeAccessIssue]
    assert pi2.coverage == 0.95
    assert pi.coverage == 0.9  # original unchanged

    pd = PredictiveDistribution(mean=jnp.zeros(3), variance=jnp.ones(3))
    pd2 = pd.replace(mean=jnp.full(3, 5.0))  # pyright: ignore[reportAttributeAccessIssue]
    assert jnp.allclose(pd2.mean, 5.0)
    assert pd2.variance is not None
    assert jnp.allclose(pd2.variance, 1.0)
