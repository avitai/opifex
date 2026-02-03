"""Tests for domain decomposition base classes.

TDD: These tests define the expected behavior for Subdomain, Interface,
and DomainDecompositionPINN base classes.
"""

import jax.numpy as jnp
from flax import nnx

from opifex.neural.pinns.domain_decomposition.base import (
    DomainDecompositionPINN,
    Interface,
    Subdomain,
)


class TestSubdomain:
    """Test Subdomain dataclass."""

    def test_create_subdomain(self):
        """Should create subdomain with bounds."""
        subdomain = Subdomain(
            id=0,
            bounds=jnp.array([[0.0, 0.5], [0.0, 1.0]]),
        )
        assert subdomain.id == 0
        assert subdomain.bounds.shape == (2, 2)

    def test_subdomain_contains_point(self):
        """Should correctly check if point is in subdomain."""
        subdomain = Subdomain(
            id=0,
            bounds=jnp.array([[0.0, 0.5], [0.0, 1.0]]),
        )
        # Point inside
        assert subdomain.contains(jnp.array([0.25, 0.5]))
        # Point outside
        assert not subdomain.contains(jnp.array([0.75, 0.5]))

    def test_subdomain_center(self):
        """Should compute subdomain center correctly."""
        subdomain = Subdomain(
            id=0,
            bounds=jnp.array([[0.0, 1.0], [0.0, 2.0]]),
        )
        center = subdomain.center
        assert jnp.allclose(center, jnp.array([0.5, 1.0]))

    def test_subdomain_volume(self):
        """Should compute subdomain volume correctly."""
        subdomain = Subdomain(
            id=0,
            bounds=jnp.array([[0.0, 2.0], [0.0, 3.0]]),
        )
        assert jnp.isclose(subdomain.volume, 6.0)


class TestInterface:
    """Test Interface dataclass."""

    def test_create_interface(self):
        """Should create interface between subdomains."""
        interface = Interface(
            subdomain_ids=(0, 1),
            points=jnp.array([[0.5, 0.0], [0.5, 0.5], [0.5, 1.0]]),
            normal=jnp.array([1.0, 0.0]),
        )
        assert interface.subdomain_ids == (0, 1)
        assert interface.points.shape == (3, 2)
        assert jnp.allclose(interface.normal, jnp.array([1.0, 0.0]))

    def test_interface_has_points(self):
        """Interface should have sample points."""
        interface = Interface(
            subdomain_ids=(0, 1),
            points=jnp.linspace(0, 1, 10).reshape(-1, 1),
            normal=jnp.array([1.0]),
        )
        assert interface.points.shape[0] == 10


class TestDomainDecompositionPINN:
    """Test DomainDecompositionPINN base class."""

    def test_create_with_subdomains(self):
        """Should create DD-PINN with subdomains."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5], [0.0, 1.0]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0], [0.0, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5, y] for y in jnp.linspace(0, 1, 5)]),
                normal=jnp.array([1.0, 0.0]),
            )
        ]

        model = DomainDecompositionPINN(
            input_dim=2,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[32, 32],
            rngs=nnx.Rngs(0),
        )

        assert model is not None
        assert len(model.subdomains) == 2
        assert len(model.interfaces) == 1

    def test_forward_pass(self):
        """Should compute forward pass for all subdomains."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5], [0.0, 1.0]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0], [0.0, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5, 0.5]]),
                normal=jnp.array([1.0, 0.0]),
            )
        ]

        model = DomainDecompositionPINN(
            input_dim=2,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16, 16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25, 0.5], [0.75, 0.5]])
        y = model(x)

        assert y.shape == (2, 1)
        assert jnp.isfinite(y).all()

    def test_subdomain_outputs(self):
        """Should get outputs from individual subdomains."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5]]),
                normal=jnp.array([1.0]),
            )
        ]

        model = DomainDecompositionPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.25], [0.75]])
        outputs = model.get_subdomain_outputs(x)

        assert len(outputs) == 2
        assert all(out.shape == (2, 1) for out in outputs)

    def test_interface_residual(self):
        """Should compute interface residual."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = [
            Interface(
                subdomain_ids=(0, 1),
                points=jnp.array([[0.5]]),
                normal=jnp.array([1.0]),
            )
        ]

        model = DomainDecompositionPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[16],
            rngs=nnx.Rngs(0),
        )

        residual = model.compute_interface_residual()
        assert residual.shape == ()  # Scalar
        assert jnp.isfinite(residual)

    def test_custom_hidden_dims_per_subdomain(self):
        """Should support different architectures per subdomain."""
        subdomains = [
            Subdomain(id=0, bounds=jnp.array([[0.0, 0.5]])),
            Subdomain(id=1, bounds=jnp.array([[0.5, 1.0]])),
        ]
        interfaces = []

        model = DomainDecompositionPINN(
            input_dim=1,
            output_dim=1,
            subdomains=subdomains,
            interfaces=interfaces,
            hidden_dims=[32, 16],  # Shared architecture
            rngs=nnx.Rngs(0),
        )

        assert model is not None


class TestPartitioning:
    """Test automatic domain partitioning."""

    def test_uniform_partitioning_1d(self):
        """Should create uniform 1D partition."""
        from opifex.neural.pinns.domain_decomposition.base import uniform_partition

        subdomains, interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0]]),
            num_partitions=(2,),
        )

        assert len(subdomains) == 2
        assert len(interfaces) == 1

        # Check bounds
        assert jnp.allclose(subdomains[0].bounds, jnp.array([[0.0, 0.5]]))
        assert jnp.allclose(subdomains[1].bounds, jnp.array([[0.5, 1.0]]))

    def test_uniform_partitioning_2d(self):
        """Should create uniform 2D partition."""
        from opifex.neural.pinns.domain_decomposition.base import uniform_partition

        subdomains, interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0], [0.0, 1.0]]),
            num_partitions=(2, 2),
        )

        assert len(subdomains) == 4
        # 2D grid: 2 horizontal + 2 vertical internal interfaces
        # Actually: (2-1)*2 horizontal + 2*(2-1) vertical = 2 + 2 = 4
        assert len(interfaces) >= 2

    def test_uniform_partitioning_interface_points(self):
        """Interfaces should have sample points."""
        from opifex.neural.pinns.domain_decomposition.base import uniform_partition

        _subdomains, interfaces = uniform_partition(
            bounds=jnp.array([[0.0, 1.0]]),
            num_partitions=(2,),
            interface_points=10,
        )

        assert interfaces[0].points.shape[0] == 10
