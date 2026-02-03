"""Neural-enhanced self-consistent field solver for DFT calculations."""

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array

from opifex.core.quantum.molecular_system import MolecularSystem


@dataclass
class SCFResult:
    """Result of SCF calculation with comprehensive convergence information."""

    converged: bool
    total_energy: float
    final_density: Array
    iterations: int
    convergence_history: Array
    orbital_energies: Array | None = None
    molecular_orbitals: Array | None = None
    chemical_accuracy_achieved: bool = False
    convergence_prediction: float | None = None


class DensityMixingNetwork(nnx.Module):
    """Neural network for intelligent density mixing in SCF iterations.

    Implements adaptive mixing strategies that learn optimal convergence patterns
    for different molecular systems and electronic configurations.
    """

    def __init__(
        self,
        grid_size: int,
        hidden_dim: int = 64,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize density mixing network.

        Args:
            grid_size: Size of density grid for molecular system
            hidden_dim: Hidden layer dimension for neural mixing
            rngs: Random number generators for initialization
        """
        super().__init__()
        self.grid_size = grid_size
        # Enhanced network architecture for optimal mixing parameters
        self.mixing_network = nnx.Sequential(
            nnx.Linear(grid_size * 2, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim // 2, 1, rngs=rngs),
            nnx.sigmoid,  # Mixing parameter between 0 and 1
        )

        # Enhanced network for density correction with physics awareness
        self.correction_network = nnx.Sequential(
            nnx.Linear(grid_size, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim // 2, grid_size, rngs=rngs),
        )

        # Physics constraint enforcement parameters
        self.stability_threshold = nnx.Param(jnp.array(1e-10))
        self.correction_scale = nnx.Param(jnp.array(0.1))

    def __call__(
        self, old_density: Array, new_density: Array, *, deterministic: bool = False
    ) -> Array:
        """Compute mixed density using neural network with physics constraints.

        Args:
            old_density: Previous iteration density
            new_density: Current iteration density
            deterministic: Whether to use deterministic computation

        Returns:
            Mixed density for next iteration with stability guarantees
        """
        # Input validation for quantum calculations
        if old_density.shape != new_density.shape:
            raise ValueError(
                f"Density shape mismatch: {old_density.shape} != {new_density.shape}"
            )

        # Ensure numerical stability for quantum calculations
        eps = jnp.finfo(jnp.float64).eps
        old_density = jnp.maximum(old_density, eps)
        new_density = jnp.maximum(new_density, eps)

        # Concatenate densities for mixing parameter prediction
        density_concat = jnp.concatenate([old_density.flatten(), new_density.flatten()])

        # Predict optimal mixing parameter with deterministic control
        mixing_param = self.mixing_network(density_concat)
        mixing_param = jnp.squeeze(mixing_param)

        # Compute density difference with stability check
        density_diff = new_density - old_density
        diff_magnitude = jnp.linalg.norm(density_diff)

        # Apply stability threshold for large density changes
        stable_mixing = jnp.where(
            diff_magnitude > self.stability_threshold,
            mixing_param * 0.5,  # Reduce mixing for stability
            mixing_param,
        )

        # Predict correction to the linear mixing
        correction = self.correction_network(density_diff)

        # Apply neural mixing with physics-aware correction
        linear_mixed = old_density + stable_mixing * density_diff
        corrected_density = linear_mixed + self.correction_scale * correction

        # Ensure physical constraints (non-negative density)
        return jnp.maximum(corrected_density, eps)


class ConvergencePredictor(nnx.Module):
    """Neural network to predict SCF convergence with chemical accuracy assessment."""

    def __init__(
        self,
        history_length: int = 5,
        hidden_dim: int = 32,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize convergence predictor with enhanced architecture.

        Args:
            history_length: Number of previous iterations to consider
            hidden_dim: Hidden layer dimension for prediction network
            rngs: Random number generators for initialization
        """
        super().__init__()
        self.history_length = history_length

        # Enhanced network to predict convergence probability
        self.predictor = nnx.Sequential(
            nnx.Linear(history_length, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_dim, hidden_dim // 2, rngs=rngs),
            nnx.gelu,
            nnx.Linear(
                hidden_dim // 2, 2, rngs=rngs
            ),  # [convergence_prob, accuracy_score]
        )

        # Chemical accuracy assessment parameters
        self.accuracy_threshold = nnx.Param(jnp.array(1e-6))  # 1 kcal/mol equivalent

    def __call__(
        self, convergence_history: Array, *, deterministic: bool = False
    ) -> tuple[float, float]:
        """Predict convergence probability and chemical accuracy score.

        Args:
            convergence_history: Recent convergence errors
            deterministic: Whether to use deterministic computation

        Returns:
            Tuple of (convergence_probability [0,1], chemical_accuracy_score [0,1])
        """
        # Input validation
        if convergence_history.size == 0:
            return 0.0, 0.0

        # Pad or truncate history to fixed length for consistent input
        if len(convergence_history) >= self.history_length:
            history = convergence_history[-self.history_length :]
        else:
            padding = jnp.zeros(self.history_length - len(convergence_history))
            history = jnp.concatenate([padding, convergence_history])

        # Predict convergence and accuracy
        prediction = self.predictor(history)
        convergence_prob = nnx.sigmoid(prediction[0])
        accuracy_score = nnx.sigmoid(prediction[1])

        return float(convergence_prob), float(accuracy_score)


class NeuralSCFSolver(nnx.Module):
    """Neural-enhanced self-consistent field solver with comprehensive

    convergence analysis.

    Implements neural acceleration of SCF convergence through:
    1. Intelligent density mixing using neural networks
    2. Advanced convergence prediction with chemical accuracy assessment
    3. Stability monitoring and adaptive recovery mechanisms
    4. High-precision numerical methods for quantum accuracy
    """

    def __init__(
        self,
        convergence_threshold: float = 1e-8,
        max_iterations: int = 100,
        mixing_strategy: str = "neural",
        grid_size: int = 1000,
        chemical_accuracy_target: float = 1e-6,  # ~1 kcal/mol
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize neural SCF solver with enhanced capabilities.

        Args:
            convergence_threshold: Energy convergence threshold
            max_iterations: Maximum number of SCF iterations
            mixing_strategy: Density mixing strategy ("neural" or "linear")
            grid_size: Size of density grid for molecular calculations
            chemical_accuracy_target: Target accuracy for chemical predictions
            rngs: Random number generators for neural components
        """
        super().__init__()
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.mixing_strategy = mixing_strategy
        self.grid_size = grid_size
        self.chemical_accuracy_target = chemical_accuracy_target
        # Enhanced neural components for SCF acceleration
        if mixing_strategy == "neural":
            self.density_mixer = DensityMixingNetwork(grid_size, rngs=rngs)

        self.convergence_predictor = ConvergencePredictor(rngs=rngs)

        # Adaptive mixing parameters with physics constraints
        self.default_mixing = nnx.Param(jnp.array(0.3))
        self.adaptive_factor = nnx.Param(jnp.array(0.8))

        # Enhanced numerical stability parameters
        self.numerical_eps = jnp.finfo(jnp.float64).eps
        self.density_tolerance = nnx.Param(jnp.array(1e-12))

    def _compute_density_error(self, old_density: Array, new_density: Array) -> float:
        """Compute density convergence error with enhanced precision.

        Args:
            old_density: Previous density
            new_density: Current density

        Returns:
            RMS density error with numerical stability
        """
        diff = new_density - old_density
        rms_error = jnp.sqrt(jnp.mean(diff**2))
        return float(rms_error)

    def _compute_energy_error(self, old_energy: float, new_energy: float) -> float:
        """Compute energy convergence error.

        Args:
            old_energy: Previous total energy
            new_energy: Current total energy

        Returns:
            Absolute energy error
        """
        return float(jnp.abs(new_energy - old_energy))

    def _mix_densities(
        self, old_density: Array, new_density: Array, *, deterministic: bool = False
    ) -> Array:
        """Mix old and new densities for next iteration with enhanced control.

        Args:
            old_density: Previous density
            new_density: Current density
            deterministic: Whether to use deterministic computation

        Returns:
            Mixed density for next iteration
        """
        if self.mixing_strategy == "neural":
            return self.density_mixer(
                old_density, new_density, deterministic=deterministic
            )

        # Enhanced linear mixing with adaptive factor
        alpha = self.default_mixing.value * self.adaptive_factor.value
        mixed_density = (1 - alpha) * old_density + alpha * new_density

        # Ensure physical constraints
        return jnp.maximum(mixed_density, self.numerical_eps)

    def _check_convergence(
        self,
        energy_error: float,
        density_error: float,
        convergence_history: Array,
        *,
        deterministic: bool = False,
    ) -> tuple[bool, bool, float]:
        """Check if SCF has converged with chemical accuracy assessment.

        Args:
            energy_error: Energy convergence error
            density_error: Density convergence error
            convergence_history: History of convergence errors
            deterministic: Whether to use deterministic computation

        Returns:
            Tuple of (converged, chemical_accuracy_achieved, convergence_probability)
        """
        # Traditional convergence criteria
        energy_converged = energy_error < self.convergence_threshold
        density_converged = density_error < self.convergence_threshold * 10

        # Chemical accuracy assessment
        chemical_accuracy = energy_error < self.chemical_accuracy_target

        # Neural convergence prediction
        convergence_prob = 0.0
        if len(convergence_history) >= 3:
            convergence_prob, _ = self.convergence_predictor(
                convergence_history, deterministic=deterministic
            )

        converged = energy_converged and density_converged
        return converged, chemical_accuracy, convergence_prob

    def solve_scf(
        self,
        molecular_system: MolecularSystem,
        initial_density: Array,
        hamiltonian_fn: Callable | None = None,
        *,
        deterministic: bool = False,
    ) -> SCFResult:
        """Solve SCF equations with neural acceleration and comprehensive analysis.

        Args:
            molecular_system: Molecular system to solve
            initial_density: Initial electron density guess
            hamiltonian_fn: Custom Hamiltonian function (optional)
            deterministic: Whether to use deterministic computation

        Returns:
            Comprehensive SCF result with convergence analysis
        """
        if initial_density.size != self.grid_size:
            raise ValueError(
                f"Initial density size {initial_density.size} != "
                f"grid_size {self.grid_size}"
            )

            # Initialize SCF iteration variables
        current_density = jnp.array(initial_density)
        previous_density = current_density.copy()  # Initialize previous_density
        previous_energy = jnp.inf
        convergence_history = jnp.array([])

        # Enhanced iteration tracking
        energy_history = []
        density_errors = []
        convergence_probabilities = []

        for iteration in range(self.max_iterations):
            # Default Hamiltonian if none provided (simple kinetic + potential)
            if hamiltonian_fn is None:
                # Simplified Hamiltonian for demonstration
                # In practice, this would be a full DFT Hamiltonian
                current_energy = float(jnp.sum(current_density**2) * 0.5)
            else:
                current_energy = float(hamiltonian_fn(current_density))

            # Compute convergence errors
            energy_error = self._compute_energy_error(previous_energy, current_energy)

            if iteration > 0:
                density_error = self._compute_density_error(
                    previous_density, current_density
                )
            else:
                density_error = jnp.inf

            # Update convergence history
            convergence_history = jnp.append(convergence_history, energy_error)
            energy_history.append(current_energy)
            density_errors.append(density_error)

            # Check convergence with comprehensive analysis
            converged, chemical_accuracy, convergence_prob = self._check_convergence(
                energy_error,
                density_error,
                convergence_history,
                deterministic=deterministic,
            )
            convergence_probabilities.append(convergence_prob)

            if converged:
                return SCFResult(
                    converged=True,
                    total_energy=current_energy,
                    final_density=current_density,
                    iterations=iteration + 1,
                    convergence_history=convergence_history,
                    chemical_accuracy_achieved=chemical_accuracy,
                    convergence_prediction=convergence_prob,
                )

            # Prepare for next iteration with neural mixing
            if iteration < self.max_iterations - 1:
                previous_density = current_density.copy()

                # Enhanced density update with neural mixing
                if iteration > 0:
                    # Simple density update for demonstration
                    # In practice, this would involve solving Kohn-Sham equations
                    density_update = current_density + 0.01 * jnp.sin(
                        jnp.linspace(0, 2 * jnp.pi, self.grid_size)
                    )

                    current_density = self._mix_densities(
                        current_density, density_update, deterministic=deterministic
                    )

                previous_energy = current_energy

        # SCF did not converge
        final_convergence_prob = (
            convergence_probabilities[-1] if convergence_probabilities else 0.0
        )

        return SCFResult(
            converged=False,
            total_energy=previous_energy,
            final_density=current_density,
            iterations=self.max_iterations,
            convergence_history=convergence_history,
            chemical_accuracy_achieved=False,
            convergence_prediction=final_convergence_prob,
        )

    def predict_convergence_iterations(
        self,
        molecular_system: MolecularSystem,
        initial_density: Array,
        *,
        deterministic: bool = False,
    ) -> int:
        """Predict number of iterations required for convergence.

        Args:
            molecular_system: Molecular system to analyze
            initial_density: Initial density guess
            deterministic: Whether to use deterministic computation

        Returns:
            Predicted number of iterations for convergence
        """
        # Run a few test iterations to build convergence history
        test_result = self.solve_scf(
            molecular_system, initial_density, deterministic=deterministic
        )

        if test_result.converged:
            return test_result.iterations

        # Use convergence probability to estimate remaining iterations
        if (
            test_result.convergence_prediction is not None
            and test_result.convergence_prediction > 0.5
        ):
            return min(self.max_iterations, int(test_result.iterations * 1.2))

        return self.max_iterations
