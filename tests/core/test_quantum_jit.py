"""
Comprehensive JIT compatibility tests for quantum modules.

Tests JAX JIT compilation for molecular systems, quantum operators,
and related quantum mechanical calculations.
"""

import jax
import jax.numpy as jnp

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.operators import (
    DensityMatrix,
    HamiltonianOperator,
    KineticEnergyOperator,
    MomentumOperator,
    Observable,
)


class TestMolecularSystemJIT:
    """Test JAX JIT compatibility for MolecularSystem operations."""

    def test_distance_matrix_jit_compilation(self):
        """Test JIT compilation of distance matrix computation."""
        # Create simple H2 molecule
        positions = jnp.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        atomic_numbers = jnp.array([1, 1])

        system = MolecularSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )

        # JIT compile distance matrix computation
        @jax.jit
        def jit_distance_matrix():
            return system.distance_matrix()

        distances = jit_distance_matrix()
        assert distances.shape == (2, 2)
        assert jnp.allclose(distances, distances.T)  # Should be symmetric
        assert jnp.allclose(jnp.diag(distances), 0.0)  # Diagonal should be zero

    def test_center_of_mass_jit_compilation(self):
        """Test JIT compilation of center of mass calculation."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # O
                [0.76, 0.59, 0.0],  # H
                [-0.76, 0.59, 0.0],  # H
            ]
        )
        atomic_numbers = jnp.array([8, 1, 1])  # Water molecule

        system = MolecularSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )

        @jax.jit
        def jit_center_of_mass():
            return system.center_of_mass

        com = jit_center_of_mass()
        assert com.shape == (3,)
        assert jnp.all(jnp.isfinite(com))

    def test_center_of_charge_jit_compilation(self):
        """Test JIT compilation of center of charge calculation."""
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # C
                [1.0, 0.0, 0.0],  # H
                [-1.0, 0.0, 0.0],  # H
            ]
        )
        atomic_numbers = jnp.array([6, 1, 1])  # CH2 fragment

        system = MolecularSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )

        @jax.jit
        def jit_center_of_charge():
            return system.center_of_charge

        coc = jit_center_of_charge()
        assert coc.shape == (3,)
        assert jnp.all(jnp.isfinite(coc))

    def test_translation_jit_compilation(self):
        """Test JIT compilation of molecular system translation."""
        positions = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        atomic_numbers = jnp.array([1, 1])

        system = MolecularSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )

        @jax.jit
        def jit_translate(translation):
            return system.translate(translation).positions

        translation = jnp.array([1.0, 2.0, 3.0])
        new_positions = jit_translate(translation)

        expected = positions + translation
        assert jnp.allclose(new_positions, expected)

    def test_batch_molecular_operations_jit(self):
        """Test batch operations on molecular systems with JIT."""
        # Create batch of molecular systems
        positions_batch = jnp.array(
            [
                [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]],  # H2
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],  # H2 stretched
                [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],  # H2 compressed
            ]
        )

        @jax.jit
        def batch_distance_computation(positions_batch):
            """Compute distances for batch of molecular systems."""

            def single_distance_matrix(positions):
                diff = positions[:, None, :] - positions[None, :, :]
                return jnp.sqrt(jnp.sum(diff**2, axis=-1))

            return jax.vmap(single_distance_matrix)(positions_batch)

        distance_matrices = batch_distance_computation(positions_batch)
        assert distance_matrices.shape == (3, 2, 2)

        # Check that all matrices are symmetric
        for i in range(3):
            assert jnp.allclose(distance_matrices[i], distance_matrices[i].T)


class TestQuantumOperatorsJIT:
    """Test JAX JIT compatibility for quantum operators."""

    def test_hamiltonian_operator_jit_compilation(self):
        """Test JIT compilation of Hamiltonian operator."""
        # Create simple molecular system
        positions = jnp.array([[0.0, 0.0, 0.0]])
        atomic_numbers = jnp.array([1])

        system = MolecularSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )

        hamiltonian = HamiltonianOperator(system, kinetic_method="finite_difference")

        # Create test wavefunction
        wavefunction = jnp.array([1.0, 0.8, 0.6, 0.4, 0.2])

        @jax.jit
        def jit_apply_hamiltonian(wf):
            return hamiltonian(wf)

        result = jit_apply_hamiltonian(wavefunction)
        assert result.shape == wavefunction.shape
        assert jnp.all(jnp.isfinite(result))

    def test_kinetic_energy_operator_jit_compilation(self):
        """Test JIT compilation of kinetic energy operator."""
        kinetic_op = KineticEnergyOperator(method="finite_difference")

        # Test wavefunction
        x = jnp.linspace(-5, 5, 100)
        wavefunction = jnp.exp(-(x**2) / 2)  # Gaussian wavefunction

        @jax.jit
        def jit_apply_kinetic(wf, dx):
            return kinetic_op(wf, dx)

        dx = x[1] - x[0]
        result = jit_apply_kinetic(wavefunction, dx)
        assert result.shape == wavefunction.shape
        assert jnp.all(jnp.isfinite(result))

    def test_kinetic_energy_spectral_jit_compilation(self):
        """Test JIT compilation of spectral kinetic energy operator."""
        kinetic_op = KineticEnergyOperator(method="spectral")

        # Test wavefunction
        x = jnp.linspace(-5, 5, 64)  # Power of 2 for FFT efficiency
        wavefunction = jnp.exp(-(x**2) / 2)  # Gaussian wavefunction

        @jax.jit
        def jit_apply_kinetic_spectral(wf, dx):
            return kinetic_op(wf, dx)

        dx = x[1] - x[0]
        result = jit_apply_kinetic_spectral(wavefunction, dx)
        assert result.shape == wavefunction.shape
        assert jnp.all(jnp.isfinite(result))

    def test_momentum_operator_jit_compilation(self):
        """Test JIT compilation of momentum operator."""
        momentum_op = MomentumOperator(method="finite_difference")

        # Test wavefunction
        x = jnp.linspace(-5, 5, 100)
        wavefunction = jnp.exp(-(x**2) / 2) * jnp.exp(1j * x)  # Gaussian with momentum

        @jax.jit
        def jit_apply_momentum(wf, dx):
            return momentum_op(wf, dx)

        dx = x[1] - x[0]
        result = jit_apply_momentum(wavefunction, dx)
        assert result.shape == wavefunction.shape
        assert jnp.all(jnp.isfinite(result))

    def test_momentum_operator_spectral_jit_compilation(self):
        """Test JIT compilation of spectral momentum operator."""
        momentum_op = MomentumOperator(method="spectral")

        # Test wavefunction
        x = jnp.linspace(-5, 5, 64)  # Power of 2 for FFT efficiency
        wavefunction = jnp.exp(-(x**2) / 2) * jnp.exp(1j * x)  # Gaussian with momentum

        @jax.jit
        def jit_apply_momentum_spectral(wf, dx):
            return momentum_op(wf, dx)

        dx = x[1] - x[0]
        result = jit_apply_momentum_spectral(wavefunction, dx)
        assert result.shape == wavefunction.shape
        assert jnp.all(jnp.isfinite(result))

    def test_observable_jit_compilation(self):
        """Test JIT compilation of observable operators."""

        # Create position observable
        def position_operator(state, x):
            return x * state

        observable = Observable(position_operator, name="position")

        # Test state and position grid
        x = jnp.linspace(-5, 5, 100)
        state = jnp.exp(-(x**2) / 2)  # Gaussian state

        @jax.jit
        def jit_apply_observable(state, x):
            return observable(state, x)

        result = jit_apply_observable(state, x)
        assert result.shape == state.shape
        assert jnp.all(jnp.isfinite(result))

    def test_expectation_value_jit_compilation(self):
        """Test JIT compilation of expectation value calculations."""

        # Create simple observable
        def identity_operator(state):
            return state

        observable = Observable(identity_operator, name="identity")

        # Normalized test state
        x = jnp.linspace(-5, 5, 100)
        state = jnp.exp(-(x**2) / 2)
        state = state / jnp.sqrt(jnp.sum(jnp.abs(state) ** 2))  # Normalize

        @jax.jit
        def jit_expectation_value(state):
            return observable.expectation_value(state)

        expectation = jit_expectation_value(state)
        assert jnp.isfinite(expectation)
        assert jnp.isclose(
            jnp.real(expectation), 1.0, atol=1e-6
        )  # Identity should give 1


class TestDensityMatrixJIT:
    """Test JAX JIT compatibility for density matrix operations."""

    def test_density_matrix_trace_jit_compilation(self):
        """Test JIT compilation of density matrix trace."""
        # Create valid density matrix (pure state)
        state = jnp.array([1.0, 0.0, 0.0]) / jnp.sqrt(1.0)
        density_matrix = jnp.outer(state, jnp.conj(state))

        dm = DensityMatrix(density_matrix)

        @jax.jit
        def jit_trace():
            return dm.trace()

        trace_val = jit_trace()
        assert jnp.isclose(jnp.real(trace_val), 1.0, atol=1e-6)

    def test_density_matrix_expectation_jit_compilation(self):
        """Test JIT compilation of density matrix expectation values."""
        # Create density matrix
        state = jnp.array([1.0, 0.0]) / jnp.sqrt(1.0)
        density_matrix = jnp.outer(state, jnp.conj(state))

        dm = DensityMatrix(density_matrix)

        # Pauli-Z observable
        observable = jnp.array([[1.0, 0.0], [0.0, -1.0]])

        @jax.jit
        def jit_expectation(obs):
            return dm.expectation_value(obs)

        expectation = jit_expectation(observable)
        assert jnp.isfinite(expectation)
        assert jnp.isclose(jnp.real(expectation), 1.0, atol=1e-6)

    def test_density_matrix_eigenvalue_jit_compilation(self):
        """Test JIT compilation of density matrix eigenvalue computation."""
        # Create mixed state density matrix
        p1, p2 = 0.7, 0.3
        state1 = jnp.array([1.0, 0.0])
        state2 = jnp.array([0.0, 1.0])

        density_matrix = p1 * jnp.outer(state1, jnp.conj(state1)) + p2 * jnp.outer(
            state2, jnp.conj(state2)
        )

        @jax.jit
        def jit_eigenvalues(matrix):
            return jnp.linalg.eigvals(matrix)

        eigenvals = jit_eigenvalues(density_matrix)
        assert jnp.all(eigenvals >= -1e-8)  # Should be positive semidefinite
        assert jnp.isclose(jnp.sum(eigenvals), 1.0, atol=1e-6)  # Trace should be 1


class TestBatchQuantumOperations:
    """Test batch quantum operations with JIT compilation."""

    def test_batch_hamiltonian_application_jit(self):
        """Test batch application of Hamiltonian with JIT."""
        # Create molecular system
        positions = jnp.array([[0.0, 0.0, 0.0]])
        atomic_numbers = jnp.array([1])

        system = MolecularSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )

        hamiltonian = HamiltonianOperator(system, kinetic_method="finite_difference")

        # Batch of wavefunctions
        batch_size = 5
        wf_size = 50
        wavefunctions = jnp.array(
            [
                jnp.exp(-(jnp.linspace(-2, 2, wf_size) ** 2) / (2 * (i + 1)))
                for i in range(batch_size)
            ]
        )

        @jax.jit
        def batch_hamiltonian_application(wfs):
            return jax.vmap(hamiltonian)(wfs)

        results = batch_hamiltonian_application(wavefunctions)
        assert results.shape == (batch_size, wf_size)
        assert jnp.all(jnp.isfinite(results))

    def test_batch_energy_computation_jit(self):
        """Test batch energy computation with JIT."""
        # Create molecular system
        positions = jnp.array([[0.0, 0.0, 0.0]])
        atomic_numbers = jnp.array([1])

        system = MolecularSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )

        hamiltonian = HamiltonianOperator(system, kinetic_method="finite_difference")

        # Batch of normalized wavefunctions
        batch_size = 3
        wf_size = 50
        x = jnp.linspace(-3, 3, wf_size)

        wavefunctions = jnp.array(
            [jnp.exp(-(x**2) / (2 * (i + 1) ** 2)) for i in range(batch_size)]
        )

        # Normalize wavefunctions
        norms = jnp.sqrt(jnp.sum(jnp.abs(wavefunctions) ** 2, axis=1, keepdims=True))
        wavefunctions = wavefunctions / norms

        @jax.jit
        def batch_energy_computation(wfs):
            return jax.vmap(hamiltonian.compute_energy)(wfs)

        energies = batch_energy_computation(wavefunctions)
        assert energies.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(energies))

    def test_end_to_end_quantum_jit_workflow(self):
        """Test end-to-end quantum calculation workflow with JIT."""
        # Create H2 molecule
        positions = jnp.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        atomic_numbers = jnp.array([1, 1])

        system = MolecularSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )

        @jax.jit
        def quantum_workflow():
            """Complete quantum workflow: system properties + simple energy estimate."""
            # Compute molecular properties
            distances = system.distance_matrix()
            com = system.center_of_mass
            coc = system.center_of_charge

            # Simple energy estimate based on nuclear repulsion and bonding
            bond_length = distances[0, 1]
            nuclear_repulsion = 1.0 / jnp.maximum(bond_length, 0.1)
            bonding_energy = -1.0 / jnp.maximum(bond_length, 0.5)

            total_energy = nuclear_repulsion + bonding_energy

            return {
                "bond_length": bond_length,
                "center_of_mass": com,
                "center_of_charge": coc,
                "energy_estimate": total_energy,
            }

        results = quantum_workflow()

        assert jnp.isfinite(results["bond_length"])
        assert jnp.all(jnp.isfinite(results["center_of_mass"]))
        assert jnp.all(jnp.isfinite(results["center_of_charge"]))
        assert jnp.isfinite(results["energy_estimate"])
        assert jnp.isclose(results["bond_length"], 0.74, atol=1e-6)


class TestJITCompatibility:
    """Test JAX JIT compatibility for problems module."""

    def test_nuclear_repulsion_calculation_jit(self):
        """Test JIT compilation of the optimized nuclear repulsion calculation."""
        import jax

        # Test the optimized nuclear repulsion calculation directly
        positions = jnp.array(
            [
                [0.0, 0.0, 0.0],  # O
                [0.76, 0.59, 0.0],  # H
                [-0.76, 0.59, 0.0],  # H
            ]
        )
        atomic_numbers = jnp.array([8, 1, 1])  # O, H, H

        @jax.jit
        def compute_nuclear_repulsion(positions, atomic_numbers):
            """JIT-compatible nuclear repulsion calculation."""
            n_atoms = positions.shape[0]

            # Vectorized nuclear repulsion calculation using JAX
            if n_atoms > 1:
                # Create pairwise distance matrix using JAX vectorized operations
                def compute_pairwise_distances(pos1, pos2):
                    return jnp.linalg.norm(pos1 - pos2)

                # Vectorize over all pairs using vmap
                distances = jax.vmap(
                    jax.vmap(compute_pairwise_distances, (None, 0)), (0, None)
                )(positions, positions)

                # Create upper triangular mask to avoid double counting
                i_indices, j_indices = jnp.triu_indices(n_atoms, k=1)

                # Extract upper triangular distances and atomic numbers
                pair_distances = distances[i_indices, j_indices]
                atomic_i = atomic_numbers[i_indices]
                atomic_j = atomic_numbers[j_indices]

                # Vectorized nuclear repulsion calculation
                nuclear_repulsion = jnp.sum(
                    atomic_i * atomic_j / jnp.maximum(pair_distances, 0.1)
                )
            else:
                nuclear_repulsion = 0.0

            return nuclear_repulsion

        # Test JIT compilation works
        repulsion = compute_nuclear_repulsion(positions, atomic_numbers)
        assert jnp.isfinite(repulsion)
        assert repulsion > 0.0  # Should be positive (repulsive)

    def test_molecular_system_jit_compatibility(self):
        """Test JIT compilation of molecular system operations."""
        import jax

        from opifex.core.quantum.molecular_system import MolecularSystem

        # Test basic molecular system operations that are JIT-compatible
        positions = jnp.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        atomic_numbers = jnp.array([1, 1])

        system = MolecularSystem(
            atomic_numbers=atomic_numbers,
            positions=positions,
        )

        @jax.jit
        def jit_molecular_properties():
            """Compute molecular properties with JIT."""
            distances = system.distance_matrix()
            com = system.center_of_mass
            coc = system.center_of_charge
            return distances, com, coc

        distances, com, coc = jit_molecular_properties()
        assert distances.shape == (2, 2)
        assert jnp.allclose(distances, distances.T)  # Should be symmetric
        assert com.shape == (3,)
        assert coc.shape == (3,)

    def test_simple_energy_approximation_jit(self):
        """Test JIT compilation of a simple energy approximation."""
        import jax

        @jax.jit
        def simple_energy_approximation(positions, atomic_numbers):
            """Simple energy approximation for testing JIT."""
            # Simple harmonic approximation around equilibrium
            n_atoms = positions.shape[0]

            if n_atoms == 1:
                # Single atom - just return a constant
                return (
                    -atomic_numbers[0] * 0.5
                )  # Simple ionization energy approximation

            # Multi-atom - include nuclear repulsion and simple bonding term
            # Nuclear repulsion (using our optimized calculation)
            def compute_pairwise_distances(pos1, pos2):
                return jnp.linalg.norm(pos1 - pos2)

            distances = jax.vmap(
                jax.vmap(compute_pairwise_distances, (None, 0)), (0, None)
            )(positions, positions)

            i_indices, j_indices = jnp.triu_indices(n_atoms, k=1)
            pair_distances = distances[i_indices, j_indices]
            atomic_i = atomic_numbers[i_indices]
            atomic_j = atomic_numbers[j_indices]

            nuclear_repulsion = jnp.sum(
                atomic_i * atomic_j / jnp.maximum(pair_distances, 0.1)
            )

            # Simple bonding term (attractive)
            bonding_energy = -jnp.sum(
                jnp.sqrt(atomic_i * atomic_j) / jnp.maximum(pair_distances, 0.5)
            )

            return nuclear_repulsion + bonding_energy

        # Test with hydrogen atom
        h_pos = jnp.array([[0.0, 0.0, 0.0]])
        h_atomic = jnp.array([1])
        h_energy = simple_energy_approximation(h_pos, h_atomic)
        assert jnp.isfinite(h_energy)
        assert h_energy < 0.0

        # Test with H2 molecule
        h2_pos = jnp.array([[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])
        h2_atomic = jnp.array([1, 1])
        h2_energy = simple_energy_approximation(h2_pos, h2_atomic)
        assert jnp.isfinite(h2_energy)

    def test_optimization_problem_jit_compilation(self):
        """Test JIT compilation of optimization problem gradient and hessian."""
        import jax

        from opifex.core.problems import create_optimization_problem

        def quadratic_objective(x):
            return jnp.sum(x**2)

        problem = create_optimization_problem(
            dimension=3,
            objective=quadratic_objective,
        )

        # JIT compile gradient computation
        @jax.jit
        def jit_gradient(x):
            return problem.gradient(x)

        # JIT compile hessian computation
        @jax.jit
        def jit_hessian(x):
            return problem.hessian(x)

        x = jnp.array([1.0, 2.0, 3.0])

        # Test JIT compilation works
        grad = jit_gradient(x)
        hessian = jit_hessian(x)

        assert grad.shape == (3,)
        assert hessian.shape == (3, 3)
        assert jnp.allclose(grad, 2.0 * x)
        assert jnp.allclose(hessian, 2.0 * jnp.eye(3))

    def test_batch_energy_computation_jit(self):
        """Test batch energy computation with JIT and vmap."""
        import jax

        from opifex.core.problems import (
            create_molecular_system,
            create_neural_dft_problem,
        )

        # Create multiple molecular systems
        systems = [
            create_molecular_system([("H", (0.0, 0.0, 0.0))]),
            create_molecular_system([("He", (0.0, 0.0, 0.0))]),
            create_molecular_system([("Li", (0.0, 0.0, 0.0))]),
        ]

        problems = [
            create_neural_dft_problem(
                molecular_system=system,
                functional_type="neural_xc",
                scf_method="neural_scf",
                grid_level=1,
            )
            for system in systems
        ]

        # JIT compile batch energy computation
        @jax.jit
        def compute_batch_energies():
            # Use vmap to compute energies for all systems
            return jnp.array([problem.compute_energy() for problem in problems])

        energies = compute_batch_energies()
        assert energies.shape == (3,)
        assert jnp.all(jnp.isfinite(energies))
        assert jnp.all(energies < 0.0)  # All should be negative (bound states)

    def test_molecular_system_distance_matrix_jit(self):
        """Test JIT compilation of molecular system distance calculations."""
        import jax

        from opifex.core.problems import create_molecular_system

        water = create_molecular_system(
            [
                ("O", (0.0, 0.0, 0.0)),
                ("H", (0.76, 0.59, 0.0)),
                ("H", (-0.76, 0.59, 0.0)),
            ]
        )

        # JIT compile distance matrix computation
        @jax.jit
        def jit_distance_matrix():
            return water.distance_matrix()

        distances = jit_distance_matrix()
        assert distances.shape == (3, 3)
        assert jnp.allclose(distances, distances.T)  # Should be symmetric
        assert jnp.allclose(jnp.diag(distances), 0.0)  # Diagonal should be zero

    def test_pde_problem_residual_jit_compilation(self):
        """Test JIT compilation of PDE problem residual computation."""
        import jax

        from opifex.core.problems import create_pde_problem
        from opifex.geometry.csg import Rectangle

        def heat_equation(x, u, u_derivs):
            return u_derivs["dt"] - 0.1 * u_derivs["d2x"]

        problem = create_pde_problem(
            geometry=Rectangle(center=jnp.array([0.5, 0.5]), width=1.0, height=1.0),
            equation=heat_equation,
            boundary_conditions={"x0": 0.0, "x1": 0.0},
        )

        # JIT compile residual computation
        @jax.jit
        def jit_residual(x, u, u_derivs):
            return problem.residual(x, u, u_derivs)

        # Test with sample data
        x = jnp.array([0.5])
        u = jnp.array([1.0])
        u_derivs = {"dt": jnp.array([0.1]), "d2x": jnp.array([0.2])}

        residual = jit_residual(x, u, u_derivs)
        expected = 0.1 - 0.1 * 0.2  # dt - 0.1 * d2x
        assert jnp.allclose(residual, expected)

    def test_ode_problem_rhs_jit_compilation(self):
        """Test JIT compilation of ODE problem RHS computation."""
        import jax

        from opifex.core.problems import create_ode_problem

        def harmonic_oscillator(t, y):
            return jnp.array([y[1], -y[0]])

        problem = create_ode_problem(
            time_span=(0.0, 10.0),
            equation=harmonic_oscillator,
        )

        # JIT compile RHS computation
        @jax.jit
        def jit_rhs(t, y):
            return problem.rhs(t, y)

        t = 1.0
        y = jnp.array([1.0, 0.5])

        rhs = jit_rhs(t, y)
        expected = jnp.array([0.5, -1.0])
        assert jnp.allclose(rhs, expected)

    def test_end_to_end_quantum_jit_workflow(self):
        """Test end-to-end quantum calculation workflow with JIT."""
        import jax

        from opifex.core.problems import (
            create_molecular_system,
            create_neural_dft_problem,
        )

        # Create benzene molecule for complex test
        benzene = create_molecular_system(
            [
                ("C", (1.40, 0.00, 0.00)),
                ("C", (0.70, 1.21, 0.00)),
                ("C", (-0.70, 1.21, 0.00)),
                ("C", (-1.40, 0.00, 0.00)),
                ("C", (-0.70, -1.21, 0.00)),
                ("C", (0.70, -1.21, 0.00)),
            ]
        )

        problem = create_neural_dft_problem(
            molecular_system=benzene,
            functional_type="neural_xc",
            scf_method="neural_scf",
            grid_level=1,
        )

        # JIT compile complete workflow including force computation
        @jax.jit
        def quantum_workflow():
            energy = problem.compute_energy()
            forces = problem.compute_forces()
            return energy, jnp.linalg.norm(forces)

        energy, force_norm = quantum_workflow()
        assert jnp.isfinite(energy)
        assert jnp.isfinite(force_norm)
        assert energy < 0.0  # Should be negative for bound state
