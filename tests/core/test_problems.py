"""
Tests for Unified Problem Definition Framework

This module provides comprehensive tests for boundary conditions, initial conditions,
and symbolic constraints in the Opifex framework.
"""

import jax.numpy as jnp
import pytest

from opifex.geometry.csg import Rectangle


class TestBoundaryConditions:
    """Test boundary condition specifications and enforcement."""

    def test_dirichlet_boundary_condition(self):
        """Test Dirichlet boundary condition creation and validation."""
        from opifex.core.conditions import DirichletBC

        # Simple constant Dirichlet BC
        bc = DirichletBC(boundary="left", value=1.0)
        assert bc.boundary == "left"
        assert bc.value == 1.0
        assert bc.condition_type == "dirichlet"
        assert bc.validate()

        # Function-based Dirichlet BC
        def boundary_func(x, t=0.0):
            return jnp.sin(x) * jnp.exp(-t)

        bc_func = DirichletBC(
            boundary="right", value=boundary_func, time_dependent=True
        )
        assert callable(bc_func.value)
        assert bc_func.validate()

        # Test evaluation
        test_point = jnp.array([1.0])
        result = bc_func.evaluate(test_point, t=0.5)
        expected = jnp.sin(1.0) * jnp.exp(-0.5)
        assert jnp.allclose(result, expected)

    def test_neumann_boundary_condition(self):
        """Test Neumann boundary condition creation and validation."""
        from opifex.core.conditions import NeumannBC

        # Constant flux BC
        bc = NeumannBC(boundary="top", value=2.0)
        assert bc.boundary == "top"
        assert bc.value == 2.0
        assert bc.condition_type == "neumann"
        assert bc.validate()

        # Variable flux BC
        def flux_func(x, t=0.0):
            return x[0] * t

        bc_func = NeumannBC(boundary="bottom", value=flux_func)
        assert callable(bc_func.value)
        assert bc_func.validate()

    def test_robin_boundary_condition(self):
        """Test Robin (mixed) boundary condition creation and validation."""
        from opifex.core.conditions import RobinBC

        # Robin BC: alpha * u + beta * du/dn = gamma
        bc = RobinBC(boundary="all", alpha=1.0, beta=2.0, gamma=0.0)
        assert bc.boundary == "all"
        assert bc.alpha == 1.0
        assert bc.beta == 2.0
        assert bc.gamma == 0.0
        assert bc.condition_type == "robin"
        assert bc.validate()

        # Time-dependent coefficients
        def time_alpha(t):
            return 1.0 + t

        bc_time = RobinBC(boundary="inlet", alpha=time_alpha, beta=1.0, gamma=0.0)
        assert callable(bc_time.alpha)
        assert bc_time.validate()

    def test_time_dependent_boundary_conditions(self):
        """Test time-dependent boundary conditions."""
        from opifex.core.conditions import DirichletBC, NeumannBC

        # Time-varying Dirichlet BC
        def time_bc(x, t):
            return jnp.sin(2 * jnp.pi * t) * x[0]

        bc = DirichletBC(boundary="left", value=time_bc, time_dependent=True)
        assert bc.time_dependent
        assert bc.validate()

        # Test evaluation at different times
        x = jnp.array([0.5])
        result_t0 = bc.evaluate(x, t=0.0)
        result_t1 = bc.evaluate(x, t=0.25)
        assert jnp.allclose(result_t0, 0.0)
        assert jnp.allclose(result_t1, 0.5)

        # Time-varying Neumann BC
        def time_flux(x, t):
            return jnp.cos(t) * jnp.ones_like(x[..., 0])

        neumann_bc = NeumannBC(boundary="right", value=time_flux, time_dependent=True)
        assert neumann_bc.time_dependent
        assert neumann_bc.validate()

    def test_quantum_boundary_conditions(self):
        """Test quantum mechanical boundary conditions."""
        from opifex.core.conditions import DensityConstraint, WavefunctionBC

        # Wavefunction normalization constraint
        wf_bc = WavefunctionBC(condition_type="normalization", norm_value=1.0)
        assert wf_bc.condition_type == "normalization"
        assert wf_bc.norm_value == 1.0
        assert wf_bc.validate()

        # Wavefunction boundary condition (vanishing at infinity)
        wf_boundary = WavefunctionBC(
            condition_type="boundary", boundary="infinity", value=complex(0.0)
        )
        assert wf_boundary.boundary == "infinity"
        assert wf_boundary.value == complex(0.0)
        assert wf_boundary.validate()

        # Electronic density constraint
        density_constraint = DensityConstraint(
            constraint_type="particle_number", n_electrons=10, tolerance=1e-8
        )
        assert density_constraint.constraint_type == "particle_number"
        assert density_constraint.n_electrons == 10
        assert density_constraint.tolerance == 1e-8
        assert density_constraint.validate()

    def test_molecular_symmetry_constraints(self):
        """Test molecular symmetry constraint handling."""
        from opifex.core.conditions import SymmetryConstraint

        # Point group symmetry
        symmetry = SymmetryConstraint(
            point_group="C2v",
            operations=["E", "C2", "sigma_v", "sigma_v'"],
            enforce_in_loss=True,
        )
        assert symmetry.point_group == "C2v"
        assert symmetry.operations is not None
        assert len(symmetry.operations) == 4
        assert symmetry.enforce_in_loss
        assert symmetry.validate()

        # Translation symmetry for periodic systems
        periodic_symmetry = SymmetryConstraint(
            symmetry_type="translational",
            lattice_vectors=jnp.array(
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
            enforce_in_loss=True,
        )
        assert periodic_symmetry.symmetry_type == "translational"
        assert periodic_symmetry.lattice_vectors is not None
        assert periodic_symmetry.lattice_vectors.shape == (3, 3)
        assert periodic_symmetry.validate()

    def test_boundary_condition_collection(self):
        """Test boundary condition collection and management."""
        from opifex.core.conditions import (
            BoundaryConditionCollection,
            DirichletBC,
            NeumannBC,
        )

        # Create multiple boundary conditions
        bc1 = DirichletBC(boundary="left", value=0.0)
        bc2 = DirichletBC(boundary="right", value=1.0)
        bc3 = NeumannBC(boundary="top", value=0.0)

        # Collect them
        bc_collection = BoundaryConditionCollection([bc1, bc2, bc3])
        assert len(bc_collection) == 3
        assert bc_collection.validate()

        # Check retrieval by boundary
        left_bc = bc_collection.get_boundary_condition("left")
        assert left_bc is not None
        assert getattr(left_bc, "value", None) == 0.0
        assert getattr(left_bc, "condition_type", None) == "dirichlet"

        # Check all Dirichlet conditions
        dirichlet_bcs = bc_collection.get_by_type("dirichlet")
        assert len(dirichlet_bcs) == 2


class TestInitialConditions:
    """Test initial condition specifications for time-dependent problems."""

    def test_scalar_initial_condition(self):
        """Test scalar initial condition specification."""
        from opifex.core.conditions import InitialCondition

        # Constant initial condition
        ic = InitialCondition(value=1.0)
        assert ic.value == 1.0
        assert ic.validate()

        # Function-based initial condition
        def initial_func(x):
            return jnp.sin(jnp.pi * x[0]) * jnp.cos(jnp.pi * x[1])

        ic_func = InitialCondition(value=initial_func)
        assert callable(ic_func.value)
        assert ic_func.validate()

        # Test evaluation
        test_point = jnp.array([0.5, 0.0])
        result = ic_func.evaluate(test_point)
        expected = jnp.sin(jnp.pi * 0.5) * jnp.cos(0.0)
        assert jnp.allclose(result, expected)

    def test_vector_initial_condition(self):
        """Test vector initial condition specification."""
        from opifex.core.conditions import InitialCondition

        # Vector initial condition
        def vector_ic(x):
            return jnp.array([x[0], x[1], x[0] * x[1]])

        ic = InitialCondition(value=vector_ic, dimension=3)
        assert ic.dimension == 3
        assert callable(ic.value)
        assert ic.validate()

        # Test evaluation
        test_point = jnp.array([1.0, 2.0])
        result = ic.evaluate(test_point)
        expected = jnp.array([1.0, 2.0, 2.0])
        assert jnp.allclose(result, expected)

    def test_initial_derivative_conditions(self):
        """Test initial conditions for derivatives (second-order problems)."""
        from opifex.core.conditions import InitialCondition

        # Initial position and velocity for wave equation
        ic_pos = InitialCondition(
            value=lambda x: jnp.sin(jnp.pi * x[0]), derivative_order=0, name="position"
        )
        ic_vel = InitialCondition(
            value=lambda x: jnp.zeros_like(x[0]), derivative_order=1, name="velocity"
        )

        assert ic_pos.derivative_order == 0
        assert ic_vel.derivative_order == 1
        assert ic_pos.name == "position"
        assert ic_vel.name == "velocity"
        assert ic_pos.validate()
        assert ic_vel.validate()

    def test_quantum_initial_conditions(self):
        """Test quantum mechanical initial conditions."""
        from opifex.core.conditions import QuantumInitialCondition

        # Initial density matrix
        def initial_density(positions):
            # Gaussian wavepacket density
            sigma = 0.1
            return jnp.exp(-((positions - 0.5) ** 2) / (2 * sigma**2))

        ic = QuantumInitialCondition(
            condition_type="density",
            value=initial_density,
            normalization=1.0,
            n_electrons=2,
        )
        assert ic.condition_type == "density"
        assert ic.normalization == 1.0
        assert ic.n_electrons == 2
        assert callable(ic.value)
        assert ic.validate()

        # Initial wavefunction
        def initial_wavefunction(x):
            return jnp.exp(-(x**2)) * (1 + 0j)  # Complex-valued

        wf_ic = QuantumInitialCondition(
            condition_type="wavefunction", value=initial_wavefunction, normalization=1.0
        )
        assert wf_ic.condition_type == "wavefunction"
        assert wf_ic.validate()


class TestSymbolicConstraints:
    """Test symbolic constraint expression capabilities."""

    def test_symbolic_constraint_creation(self):
        """Test creation of symbolic constraints."""
        from opifex.core.conditions import SymbolicConstraint

        # Conservation constraint: integral of u over domain = constant
        constraint = SymbolicConstraint(
            expression="integral(u, domain) - C",
            variables=["u"],
            parameters={"C": 1.0},
            constraint_type="conservation",
        )
        assert constraint.expression == "integral(u, domain) - C"
        assert "u" in constraint.variables
        assert constraint.parameters["C"] == 1.0
        assert constraint.constraint_type == "conservation"
        assert constraint.validate()

    def test_physics_constraints(self):
        """Test physics-based symbolic constraints."""
        from opifex.core.conditions import PhysicsConstraint

        # Mass conservation constraint
        mass_constraint = PhysicsConstraint(
            constraint_type="mass_conservation",
            expression="div(rho * v) + d_rho_dt",
            variables=["rho", "v"],
            physics_law="continuity_equation",
        )
        assert mass_constraint.constraint_type == "mass_conservation"
        assert mass_constraint.physics_law == "continuity_equation"
        assert "rho" in mass_constraint.variables
        assert "v" in mass_constraint.variables
        assert mass_constraint.validate()

        # Energy conservation constraint
        energy_constraint = PhysicsConstraint(
            constraint_type="energy_conservation",
            expression="kinetic_energy + potential_energy - total_energy",
            variables=["kinetic_energy", "potential_energy"],
            parameters={"total_energy": 1.0},
            tolerance=1e-6,
        )
        assert energy_constraint.tolerance == 1e-6
        assert energy_constraint.validate()

    def test_quantum_physics_constraints(self):
        """Test quantum mechanical physics constraints."""
        from opifex.core.conditions import QuantumConstraint

        # Particle number conservation
        particle_constraint = QuantumConstraint(
            constraint_type="particle_number",
            expression="integral(rho, all_space) - N",
            variables=["rho"],
            parameters={"N": 10},
            tolerance=1e-8,
        )
        assert particle_constraint.constraint_type == "particle_number"
        assert particle_constraint.parameters["N"] == 10
        assert particle_constraint.tolerance == 1e-8
        assert particle_constraint.validate()

        # Density positivity constraint
        positivity_constraint = QuantumConstraint(
            constraint_type="density_positivity",
            expression="min(rho) >= 0",
            variables=["rho"],
            enforcement_method="penalty",
        )
        assert positivity_constraint.constraint_type == "density_positivity"
        assert positivity_constraint.enforcement_method == "penalty"
        assert positivity_constraint.validate()


class TestConditionIntegration:
    """Test integration of conditions with problem definitions."""

    def test_pde_problem_with_conditions(self):
        """Test PDE problem integration with boundary and initial conditions."""
        from opifex.core.conditions import DirichletBC, InitialCondition, NeumannBC
        from opifex.core.problems import create_pde_problem
        from opifex.geometry.csg import Rectangle

        # Define boundary conditions
        left_bc = DirichletBC(boundary="left", value=0.0)
        right_bc = DirichletBC(boundary="right", value=1.0)
        top_bc = NeumannBC(boundary="top", value=0.0)

        # Define initial condition
        def initial_temp(x):
            return 0.5 * jnp.ones_like(x[0])

        ic = InitialCondition(value=initial_temp)

        # Create PDE problem with conditions
        problem = create_pde_problem(
            geometry=Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0),
            equation=lambda x, u, u_d: u_d["dxx"] + u_d["dyy"],  # Laplace equation
            boundary_conditions=[left_bc, right_bc, top_bc],
            initial_conditions=[ic],
            time_dependent=True,
        )

        assert problem.validate()
        assert len(problem.boundary_conditions) == 3
        assert len(problem.initial_conditions) == 1
        assert problem.time_dependent

    def test_quantum_problem_with_conditions(self):
        """Test quantum problem integration with quantum-specific conditions."""
        from opifex.core.conditions import (
            DensityConstraint,
            SymmetryConstraint,
            WavefunctionBC,
        )
        from opifex.core.problems import (
            create_molecular_system,
            create_neural_dft_problem,
            create_pde_problem,
        )

        # Create molecular system
        water = create_molecular_system(
            [
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.757, 0.586, 0.0)),
                ("H", (-0.757, 0.586, 0.0)),
            ]
        )

        # Define quantum conditions
        wf_bc = WavefunctionBC(condition_type="normalization", norm_value=1.0)
        density_constraint = DensityConstraint(
            constraint_type="particle_number", n_electrons=10, tolerance=1e-8
        )
        symmetry = SymmetryConstraint(
            point_group="C2v", operations=["E", "C2", "sigma_v", "sigma_v'"]
        )

        # Create instances of each problem type
        pde_problem = create_pde_problem(
            geometry=Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0),
            equation=lambda x, u, u_d: u_d["dx"],
            boundary_conditions={"x0": 0.0},
        )
        # Create neural DFT problem with conditions
        problem = create_neural_dft_problem(
            molecular_system=water,
            boundary_conditions=[wf_bc],
            constraints=[density_constraint, symmetry],
        )

        assert problem.validate()
        assert len(problem.boundary_conditions) == 1
        assert len(problem.constraints) == 2
        assert problem.molecular_system.n_electrons == 10
        assert pde_problem.validate()  # Added validation for pde_problem

    def test_condition_validation_errors(self):
        """Test validation errors for invalid conditions."""
        from opifex.core.conditions import DirichletBC, InitialCondition, RobinBC

        # Invalid boundary specification
        with pytest.raises(ValueError, match="Invalid boundary"):
            DirichletBC(boundary="invalid_boundary", value=1.0)

        # Invalid Robin BC coefficients
        with pytest.raises(ValueError, match="Both alpha and beta cannot be zero"):
            RobinBC(boundary="left", alpha=0.0, beta=0.0, gamma=1.0)

        # Invalid initial condition dimension
        with pytest.raises(ValueError, match="Dimension must be positive"):
            InitialCondition(value=1.0, dimension=0)
