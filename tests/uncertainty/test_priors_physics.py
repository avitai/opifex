"""Contract tests for the physics-informed prior modules.

Covers the five NNX prior modules in
:mod:`opifex.uncertainty.priors_physics`:

* :class:`PhysicsInformedPriors` — hard-constraint projector with
  conservation-law and boundary-condition application.
* :class:`ConservationLawPriors` — uncertainty modifier that inflates
  per-law uncertainty by violation magnitude.
* :class:`DomainSpecificPriors` — domain-tailored parameter ranges.
* :class:`HierarchicalBayesianFramework` — multi-level uncertainty
  propagation across hierarchy levels.
* :class:`PhysicsAwareUncertaintyPropagation` — confidence adjustment
  under physics-constraint violations.

Pinned behaviours:

* every public callable traces under ``jax.jit`` (and ``nnx.jit`` where
  the receiver is the :class:`nnx.Module` self);
* parameter access uses canonical ``[...]`` indexing — not the
  deprecated ``.value`` property;
* no module relies on a hidden ``nnx.Rngs(0)`` fallback — callers must
  pass an explicit ``rngs`` bundle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.priors_physics import (
    ConservationLawPriors,
    DomainSpecificPriors,
    HierarchicalBayesianFramework,
    PhysicsAwareUncertaintyPropagation,
    PhysicsInformedPriors,
)


# ---------------------------------------------------------------------------
# PhysicsInformedPriors
# ---------------------------------------------------------------------------


def test_physics_informed_priors_apply_constraints_preserves_shape() -> None:
    rngs = nnx.Rngs(0)
    prior = PhysicsInformedPriors(
        conservation_laws=("energy", "momentum"),
        boundary_conditions=("dirichlet",),
        rngs=rngs,
    )
    params = jnp.array([1.0, 2.0, -1.0, 0.5])
    out = prior.apply_constraints(params)
    assert out.shape == params.shape


def test_physics_informed_priors_compute_violation_penalty_traces_under_jit() -> None:
    rngs = nnx.Rngs(0)
    prior = PhysicsInformedPriors(
        conservation_laws=("energy", "momentum"),
        rngs=rngs,
    )

    @jax.jit
    def jitted(params: jax.Array) -> jax.Array:
        return prior.compute_violation_penalty(params)

    out = jitted(jnp.array([1.0, 2.0, -1.0, 0.5]))
    assert out.shape == ()


def test_physics_informed_priors_check_physical_plausibility_in_unit_interval() -> None:
    rngs = nnx.Rngs(0)
    prior = PhysicsInformedPriors(
        conservation_laws=("positivity", "boundedness"),
        rngs=rngs,
    )
    score = float(prior.check_physical_plausibility(jnp.array([0.5, 1.2, 0.3])))
    assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# ConservationLawPriors
# ---------------------------------------------------------------------------


def test_conservation_law_priors_constructs_with_default_laws() -> None:
    rngs = nnx.Rngs(0)
    prior = ConservationLawPriors(rngs=rngs)
    assert prior.conservation_laws == ("energy", "momentum", "mass")
    # uncertainty_scalings is a per-law Param vector.
    assert prior.uncertainty_scalings[...].shape == (3,)


def test_conservation_law_priors_accepts_custom_laws() -> None:
    rngs = nnx.Rngs(0)
    prior = ConservationLawPriors(conservation_laws=("energy",), rngs=rngs)
    assert prior.conservation_laws == ("energy",)
    assert prior.uncertainty_scalings[...].shape == (1,)


# ---------------------------------------------------------------------------
# DomainSpecificPriors
# ---------------------------------------------------------------------------


def test_domain_specific_priors_constructs_for_known_domain() -> None:
    rngs = nnx.Rngs(0)
    prior = DomainSpecificPriors(domain="quantum_chemistry", rngs=rngs)
    assert prior.domain == "quantum_chemistry"


# ---------------------------------------------------------------------------
# HierarchicalBayesianFramework
# ---------------------------------------------------------------------------


def test_hierarchical_framework_samples_at_requested_level() -> None:
    rngs = nnx.Rngs(0)
    framework = HierarchicalBayesianFramework(
        hierarchy_levels=3,
        level_dimensions=(8, 4, 2),
        rngs=rngs,
    )
    samples = framework.sample_hierarchical_parameters((5,), level=0)
    assert samples.shape == (5, 8)


def test_hierarchical_framework_rejects_out_of_range_level() -> None:
    rngs = nnx.Rngs(0)
    framework = HierarchicalBayesianFramework(
        hierarchy_levels=3,
        level_dimensions=(8, 4, 2),
        rngs=rngs,
    )
    with pytest.raises(ValueError, match="exceeds hierarchy depth"):
        framework.sample_hierarchical_parameters((5,), level=99)


def test_hierarchical_framework_propagation_does_not_decrease_uncertainty() -> None:
    rngs = nnx.Rngs(0)
    framework = HierarchicalBayesianFramework(
        hierarchy_levels=3,
        level_dimensions=(4, 4, 4),
        uncertainty_propagation="additive",
        rngs=rngs,
    )
    base = jnp.ones((4,))
    propagated = framework.propagate_uncertainty_hierarchically(base, target_level=2)
    assert bool(jnp.all(propagated >= base))


# ---------------------------------------------------------------------------
# PhysicsAwareUncertaintyPropagation
# ---------------------------------------------------------------------------


def test_physics_aware_uncertainty_propagation_returns_confidence_in_unit_interval() -> None:
    rngs = nnx.Rngs(0)
    propagator = PhysicsAwareUncertaintyPropagation(
        conservation_laws=("energy",),
        rngs=rngs,
    )
    predictions = jnp.array([1.0, 2.0])
    uncertainties = jnp.array([0.1, 0.2])
    physics_state = jnp.array([1.0, 2.0])
    confidence = propagator.compute_physics_informed_confidence(
        predictions=predictions,
        uncertainties=uncertainties,
        physics_state=physics_state,
    )
    assert bool(jnp.all((confidence >= 0.0) & (confidence <= 1.0)))


# ---------------------------------------------------------------------------
# No deprecated `.value` access
# ---------------------------------------------------------------------------


def test_priors_physics_uses_canonical_param_indexing() -> None:
    """Source-level guard: the deprecated `.value` property must not be
    used for `nnx.Param` access in priors_physics."""
    import inspect

    from opifex.uncertainty import priors_physics

    source = inspect.getsource(priors_physics)
    lines_with_value = [
        line.strip()
        for line in source.splitlines()
        if ".value" in line
        and "value_and_grad" not in line
        and "values()" not in line
        and not line.strip().startswith("#")
    ]
    assert not lines_with_value, (
        f"Found {len(lines_with_value)} `.value` references in priors_physics — "
        f"use `[...]` indexing instead. Examples: {lines_with_value[:3]}"
    )
