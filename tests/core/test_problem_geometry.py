import jax.numpy as jnp

from opifex.core.problems import create_pde_problem, PDEProblem
from opifex.geometry.base import Geometry
from opifex.geometry.csg import Rectangle


def test_pde_problem_requires_geometry():
    # Geometry instance
    rect = Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0)

    # Simple equation
    def pde_fn(x, u, du):
        return jnp.sum(du["grad_u"])

    # Helper subclass for testing
    class ConcreteTestProblem(PDEProblem):
        def residual(self, x, u, u_derivatives):
            return pde_fn(x, u, u_derivatives)

    # Instantiate with Geometry
    prob = ConcreteTestProblem(geometry=rect, equation=pde_fn, boundary_conditions=[])

    # Check property
    assert isinstance(prob.geometry, Geometry)

    # Old constructor should fail/not match signature if we change it strictly
    # or if we support both, we should test valid conversion
    # But plan says STRICT requirement.

    # assert hasattr(prob, "get_geometry")
    # assert not hasattr(prob, "get_domain") # Plan says DELETE get_domain


def test_create_pde_problem_helper():
    # Helper should probably accept Geometry now, or convert dict to Geometry
    # Let's assume for now it should accept Geometry

    rect = Rectangle(center=jnp.array([0.0, 0.0]), width=2.0, height=2.0)
    prob = create_pde_problem(
        geometry=rect, equation=lambda x, u, du: 0.0, boundary_conditions=[]
    )
    assert prob.geometry is rect
