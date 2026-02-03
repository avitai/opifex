"""Tests for InverseProblem implementation."""

import jax.numpy as jnp

from opifex.core.problems import (
    create_inverse_problem,
    create_pde_problem,
    InverseProblem,
)
from opifex.geometry.csg import Rectangle


def dummy_pde(x, u, u_derivs):
    return u


def test_inverse_problem_creation():
    """Test standard creation of inverse problem."""
    rect = Rectangle(center=jnp.array([0.0, 0.0]), width=1.0, height=1.0)
    pde = create_pde_problem(geometry=rect, equation=dummy_pde, boundary_conditions={})

    obs_coords = jnp.array([[0.0, 0.0]])
    obs_values = jnp.array([1.0])

    inverse = create_inverse_problem(pde, (obs_coords, obs_values))

    assert isinstance(inverse, InverseProblem)
    assert inverse.validate()
    assert inverse.get_geometry() == rect


def test_inverse_problem_validation():
    """Test validation fails on mismatch."""
    rect = Rectangle(center=jnp.array([0.0, 0.0]), width=1.0, height=1.0)
    pde = create_pde_problem(geometry=rect, equation=dummy_pde, boundary_conditions={})

    # Mismatch dimensions
    obs_coords = jnp.array([[0.0, 0.0]])
    obs_values = jnp.array([1.0, 2.0])  # Length 2 vs 1

    inverse = create_inverse_problem(pde, (obs_coords, obs_values))
    assert not inverse.validate()


def test_inverse_problem_loss():
    """Test default loss function."""
    rect = Rectangle(center=jnp.array([0.0, 0.0]), width=1.0, height=1.0)
    pde = create_pde_problem(geometry=rect, equation=dummy_pde, boundary_conditions={})

    inverse = create_inverse_problem(pde, (jnp.zeros(1), jnp.zeros(1)))

    pred = jnp.array([1.0])
    target = jnp.array([0.0])
    loss = inverse.parameter_loss(pred, target)
    assert jnp.isclose(loss, 1.0)
