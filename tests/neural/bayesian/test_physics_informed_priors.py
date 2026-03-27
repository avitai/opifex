"""Tests for physics-informed Bayesian priors."""

import jax.numpy as jnp
from flax import nnx

from opifex.neural.bayesian.physics_informed_priors import (
    ConservationLawPriors,
    DomainSpecificPriors,
    PhysicsInformedPriors,
)


class TestPhysicsInformedPriors:
    """Tests for PhysicsInformedPriors."""

    def test_init_no_constraints(self):
        """Initialize with no constraints."""
        priors = PhysicsInformedPriors(rngs=nnx.Rngs(0))
        assert priors.conservation_laws == ()
        assert priors.boundary_conditions == ()
        assert priors.penalty_weight == 1.0

    def test_init_with_conservation_laws(self):
        """Initialize with conservation laws."""
        priors = PhysicsInformedPriors(
            conservation_laws=("energy", "momentum"),
            rngs=nnx.Rngs(0),
        )
        assert len(priors.conservation_laws) == 2
        assert priors.constraint_weights.value.shape == (2,)

    def test_init_with_boundary_conditions(self):
        """Initialize with boundary conditions."""
        priors = PhysicsInformedPriors(
            boundary_conditions=("dirichlet", "neumann"),
            rngs=nnx.Rngs(0),
        )
        assert len(priors.boundary_conditions) == 2

    def test_constraint_weights_default_to_ones(self):
        """Default constraint weights are all ones."""
        priors = PhysicsInformedPriors(
            conservation_laws=("energy",),
            rngs=nnx.Rngs(0),
        )
        assert jnp.allclose(priors.constraint_weights.value, jnp.ones(1))

    def test_apply_constraints_preserves_shape(self):
        """Constraint application preserves parameter shape."""
        priors = PhysicsInformedPriors(rngs=nnx.Rngs(0))
        params = jnp.ones((4, 8))
        constrained = priors.apply_constraints(params)
        assert constrained.shape == (4, 8)

    def test_is_nnx_module(self):
        """PhysicsInformedPriors is an nnx.Module."""
        priors = PhysicsInformedPriors(rngs=nnx.Rngs(0))
        assert isinstance(priors, nnx.Module)


class TestConservationLawPriors:
    """Tests for ConservationLawPriors."""

    def test_init_defaults(self):
        """Initialize with default conservation laws."""
        priors = ConservationLawPriors(rngs=nnx.Rngs(0))
        assert "energy" in priors.conservation_laws
        assert "momentum" in priors.conservation_laws

    def test_init_custom_laws(self):
        """Initialize with custom conservation laws."""
        priors = ConservationLawPriors(
            conservation_laws=("energy",),
            rngs=nnx.Rngs(0),
        )
        assert len(priors.conservation_laws) == 1

    def test_is_nnx_module(self):
        """ConservationLawPriors is an nnx.Module."""
        priors = ConservationLawPriors(rngs=nnx.Rngs(0))
        assert isinstance(priors, nnx.Module)


class TestDomainSpecificPriors:
    """Tests for DomainSpecificPriors."""

    def test_init_fluid_domain(self):
        """Initialize with fluid dynamics domain."""
        priors = DomainSpecificPriors(
            domain="fluid_dynamics",
            rngs=nnx.Rngs(0),
        )
        assert priors.domain == "fluid_dynamics"

    def test_init_quantum_domain(self):
        """Initialize with quantum chemistry domain."""
        priors = DomainSpecificPriors(
            domain="quantum_chemistry",
            rngs=nnx.Rngs(0),
        )
        assert priors.domain == "quantum_chemistry"

    def test_is_nnx_module(self):
        """DomainSpecificPriors is an nnx.Module."""
        priors = DomainSpecificPriors(domain="general", rngs=nnx.Rngs(0))
        assert isinstance(priors, nnx.Module)
