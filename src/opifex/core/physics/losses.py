"""Physics loss composition and PDE residual computation.

This module provides the core functionality for composing physics-informed
losses, including PDE residuals, boundary conditions, and conservation laws.

Single source of truth for ALL physics loss functionality, extracted from
opifex/training/physics_losses.py as part of comprehensive refactoring.

Key Features:
    - Hierarchical loss composition with adaptive weighting
    - Conservation law enforcement for physical constraints
    - Residual computation for various physics equations
    - Quantum mechanical extensions for DFT applications
    - Full JAX compatibility (JIT, vmap, grad)

Classes:
    PhysicsLossConfig: Configuration for physics-informed losses
    PhysicsLossComposer: Hierarchical loss composition system
    AdaptiveWeightScheduler: Dynamic weight scheduling algorithms
    ConservationLawEnforcer: Conservation law constraint enforcement
    ResidualComputer: Physics equation residual computation
    PhysicsInformedLoss: Integrated physics-informed loss system
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp


if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class PhysicsLossConfig:
    """Configuration for physics-informed loss functions.

    This configuration class defines all parameters for physics-informed
    loss computation including weights, adaptive scheduling, and quantum
    mechanical extensions.

    Attributes:
        data_loss_weight: Weight for data fidelity loss term
        physics_loss_weight: Weight for physics residual loss term
        boundary_loss_weight: Weight for boundary condition loss term
        conservation_weights: Dictionary of conservation law weights
        adaptive_weighting: Enable adaptive weight scheduling
        weight_schedule: Type of weight schedule ('linear', 'exponential', 'step')
        quantum_constraints: Enable quantum mechanical constraints
        density_positivity_weight: Weight for density positivity constraint
        wavefunction_normalization_weight: Weight for wavefunction normalization
        step_milestones: Milestones for step scheduling (optional)
    """

    data_loss_weight: float = 1.0
    physics_loss_weight: float = 0.1
    boundary_loss_weight: float = 1.0
    conservation_weights: dict[str, float] = field(default_factory=dict)
    adaptive_weighting: bool = False
    weight_schedule: str = "exponential"
    quantum_constraints: bool = False
    density_positivity_weight: float = 0.0
    wavefunction_normalization_weight: float = 0.0
    step_milestones: list[int] | None = None

    def __post_init__(self):
        """Validate configuration parameters."""
        valid_schedules = ["linear", "exponential", "step"]
        if self.weight_schedule not in valid_schedules:
            raise ValueError(
                f"Invalid weight schedule '{self.weight_schedule}'. "
                f"Must be one of: {valid_schedules}"
            )

        if self.weight_schedule == "step" and self.step_milestones is None:
            self.step_milestones = []


class PhysicsLossComposer:
    """Hierarchical physics loss composition system.

    This class implements a hierarchical approach to composing different
    physics-informed loss terms including data fidelity, physics residuals,
    boundary conditions, and conservation laws.

    Attributes:
        config: Physics loss configuration
    """

    def __init__(self, config: PhysicsLossConfig):
        """Initialize the physics loss composer.

        Args:
            config: Physics loss configuration
        """
        self.config = config

    def compose_loss(
        self,
        data_loss: jax.Array,
        physics_residual: jax.Array,
        boundary_residual: jax.Array,
        conservation_residuals: dict[str, jax.Array] | None = None,
        quantum_residuals: dict[str, jax.Array] | None = None,
    ) -> jax.Array:
        """Compose all loss terms into a single scalar loss.

        Args:
            data_loss: Data fidelity loss term
            physics_residual: Physics equation residual
            boundary_residual: Boundary condition residual
            conservation_residuals: Conservation law residuals
            quantum_residuals: Quantum mechanical constraint residuals

        Returns:
            Composed total loss as scalar
        """
        # Base loss components
        total_loss = (
            self.config.data_loss_weight * data_loss
            + self.config.physics_loss_weight * physics_residual
            + self.config.boundary_loss_weight * boundary_residual
        )

        # Add conservation law terms
        if conservation_residuals is not None:
            for law, residual in conservation_residuals.items():
                weight = self.config.conservation_weights.get(law, 1.0)
                total_loss += weight * residual

        # Add quantum constraint terms
        if quantum_residuals is not None and self.config.quantum_constraints:
            for constraint, residual in quantum_residuals.items():
                if constraint == "density_positivity":
                    total_loss += self.config.density_positivity_weight * residual
                elif constraint == "wavefunction_normalization":
                    total_loss += (
                        self.config.wavefunction_normalization_weight * residual
                    )

        return total_loss

    def compute_residuals(
        self,
        predictions: jax.Array,
        targets: jax.Array,
        inputs: jax.Array,
    ) -> dict[str, jax.Array]:
        """Compute basic residuals for loss composition.

        Args:
            predictions: Model predictions
            targets: Target values
            inputs: Input data

        Returns:
            Dictionary of computed residuals
        """
        # Data loss (MSE)
        data_loss = jnp.mean((predictions - targets) ** 2)

        # Physics residual (placeholder - to be computed by ResidualComputer)
        physics_residual = jnp.array(0.0)

        # Boundary residual (placeholder)
        boundary_residual = jnp.array(0.0)

        return {
            "data_loss": data_loss,
            "physics_residual": physics_residual,
            "boundary_residual": boundary_residual,
        }


class AdaptiveWeightScheduler:
    """Dynamic weight scheduling for physics-informed losses.

    This class implements various scheduling algorithms for dynamically
    adjusting physics loss weights during training to improve convergence
    and balance between data fidelity and physics constraints.

    Attributes:
        schedule_type: Type of scheduling algorithm
        initial_physics_weight: Initial physics weight
        final_physics_weight: Final physics weight
        transition_epochs: Number of epochs for transition
        step_milestones: Milestones for step scheduling
    """

    def __init__(
        self,
        schedule_type: str = "exponential",
        initial_physics_weight: float = 0.01,
        final_physics_weight: float = 1.0,
        transition_epochs: int = 1000,
        step_milestones: list[int] | None = None,
    ):
        """Initialize the adaptive weight scheduler.

        Args:
            schedule_type: Scheduling algorithm type
            initial_physics_weight: Initial weight value
            final_physics_weight: Final weight value
            transition_epochs: Epochs for complete transition
            step_milestones: Milestones for step scheduling
        """
        self.schedule_type = schedule_type
        self.initial_physics_weight = initial_physics_weight
        self.final_physics_weight = final_physics_weight
        self.transition_epochs = transition_epochs
        self.step_milestones = step_milestones or []

    def get_weight(self, epoch: int) -> jax.Array:
        """Get physics weight for given epoch.

        Args:
            epoch: Current training epoch

        Returns:
            Physics weight for current epoch
        """
        epoch_arr = jnp.array(epoch)
        transition_epochs = jnp.array(self.transition_epochs)

        # Clamp epoch to transition range
        t = jnp.clip(epoch_arr / transition_epochs, 0.0, 1.0)

        if self.schedule_type == "linear":
            weight = (
                self.initial_physics_weight
                + (self.final_physics_weight - self.initial_physics_weight) * t
            )
        elif self.schedule_type == "exponential":
            # Exponential schedule: w(t) = w_0 * (w_f/w_0)^t
            ratio = self.final_physics_weight / self.initial_physics_weight
            weight = self.initial_physics_weight * (ratio**t)
        elif self.schedule_type == "step":
            weight = self._compute_step_schedule(epoch_arr)
        else:
            # Default to constant weight
            weight = jnp.array(self.initial_physics_weight)

        return weight

    def _compute_step_schedule(self, epoch: jax.Array) -> jax.Array:
        """Compute step-wise weight schedule.

        Args:
            epoch: Current epoch

        Returns:
            Current weight based on step schedule
        """
        if not self.step_milestones:
            return jnp.array(self.initial_physics_weight)

        weight = jnp.array(self.initial_physics_weight)
        total_steps = len(self.step_milestones)
        weight_increment = (
            self.final_physics_weight - self.initial_physics_weight
        ) / total_steps

        for milestone in self.step_milestones:
            weight = jnp.where(epoch >= milestone, weight + weight_increment, weight)

        return weight


class ConservationLawEnforcer:
    """Conservation law enforcement for physics constraints.

    This class implements enforcement of various conservation laws
    including mass, momentum, energy, and quantum mechanical
    conservation principles.

    Attributes:
        conservation_laws: List of conservation laws to enforce
        tolerance: Numerical tolerance for conservation violations
    """

    def __init__(
        self,
        conservation_laws: list[str],
        tolerance: float = 1e-6,
    ):
        """Initialize conservation law enforcer.

        Args:
            conservation_laws: List of laws to enforce
            tolerance: Numerical tolerance for violations
        """
        self.conservation_laws = conservation_laws
        self.tolerance = tolerance

    def _extract_state_component(
        self, state: jax.Array | tuple[jax.Array, ...], default_shape=None
    ) -> jax.Array:
        """Extract first component from state tuple or return state directly."""
        if isinstance(state, tuple) and len(state) > 0:
            return state[0]
        if isinstance(state, tuple):
            # Empty tuple - return appropriate default
            if default_shape == "matrix":
                return jnp.zeros((1, 1))
            return jnp.array(0.0)
        return state

    def compute_residual(
        self,
        law: str,
        state: jax.Array | tuple[jax.Array, ...],
        coordinates: jax.Array | None = None,
    ) -> jax.Array:
        """Compute conservation law residual.

        Args:
            law: Conservation law name
            state: System state (velocity field, density, etc.)
            coordinates: Spatial coordinates (if needed)

        Returns:
            Conservation law violation residual
        """
        # Dispatch table for conservation law handlers
        handlers = {
            "mass": lambda: self._compute_mass_conservation(
                self._extract_state_component(state), coordinates
            ),
            "momentum": lambda: self._compute_momentum_conservation(state, coordinates),
            "energy": lambda: self._compute_energy_conservation(state, coordinates),
            "particle_number": lambda: self._compute_particle_number_conservation(
                self._extract_state_component(state, default_shape="matrix")
            ),
            "charge": lambda: self._compute_charge_conservation(
                self._extract_state_component(state, default_shape="matrix")
            ),
            "symmetry": lambda: self._compute_symmetry_conservation(
                self._extract_state_component(state)
            ),
        }

        # Execute handler or return zero for unknown laws
        handler = handlers.get(law, lambda: jnp.array(0.0))
        return handler()

    def _compute_mass_conservation(
        self, velocity_field: jax.Array, coordinates: jax.Array | None
    ) -> jax.Array:
        """Compute mass conservation residual (continuity equation).

        Args:
            velocity_field: Velocity field
            coordinates: Spatial coordinates

        Returns:
            Mass conservation residual
        """
        # For demonstration: compute simple L2 norm of velocity divergence
        # In practice, this would use proper finite differences
        # or automatic differentiation
        divergence = jnp.mean(jnp.abs(velocity_field))
        return divergence**2

    def _compute_momentum_conservation(
        self,
        state: jax.Array | tuple[jax.Array, ...],
        coordinates: jax.Array | None,
    ) -> jax.Array:
        """Compute momentum conservation residual.

        Args:
            state: System state
            coordinates: Spatial coordinates

        Returns:
            Momentum conservation residual
        """
        # Simplified momentum conservation check
        if isinstance(state, tuple):
            velocity = state[0] if len(state) > 0 else jnp.array(0.0)
        else:
            velocity = state

        # Compute momentum residual (simplified)
        return jnp.mean(velocity**2)

    def _compute_energy_conservation(
        self,
        state: jax.Array | tuple[jax.Array, ...],
        coordinates: jax.Array | None,
    ) -> jax.Array:
        """Compute energy conservation residual.

        Args:
            state: System state (position, momentum)
            coordinates: Time coordinates

        Returns:
            Energy conservation residual
        """
        if isinstance(state, tuple) and len(state) >= 2:
            q, p = state[0], state[1]
            # Simple Hamiltonian: H = p²/2 + q²/2
            kinetic = jnp.mean(p**2) / 2
            potential = jnp.mean(q**2) / 2
            total_energy = kinetic + potential

            # Energy should be conserved - compute variation
            return jnp.var(total_energy)
        return jnp.array(0.0)

    def _compute_particle_number_conservation(
        self, density_matrix: jax.Array
    ) -> jax.Array:
        """Compute particle number conservation for quantum systems.

        Args:
            density_matrix: Electronic density matrix

        Returns:
            Particle number conservation residual
        """
        # Particle number = Tr(density_matrix)
        # Should be conserved during quantum evolution
        if density_matrix.ndim >= 2:
            particle_number = jnp.trace(density_matrix, axis1=-2, axis2=-1)
            # Compute variation in particle number
            return jnp.var(particle_number)
        return jnp.array(0.0)

    def _compute_charge_conservation(self, density_matrix: jax.Array) -> jax.Array:
        """Compute charge conservation for quantum systems.

        Args:
            density_matrix: Electronic density matrix

        Returns:
            Charge conservation residual
        """
        # Simplified charge conservation check
        # In practice, this would involve proper charge density computation
        if density_matrix.ndim >= 2:
            charge_density = jnp.abs(jnp.trace(density_matrix, axis1=-2, axis2=-1))
            return jnp.var(charge_density)
        return jnp.array(0.0)

    def _compute_symmetry_conservation(self, field: jax.Array) -> jax.Array:
        """Compute symmetry preservation for physical fields.

        Tests whether a field preserves reflection symmetry: f(x) = f(-x).
        This is important for many physical systems with spatial symmetries.

        Args:
            field: Physical field to test for symmetry

        Returns:
            Symmetry preservation residual (0 = perfect symmetry)
        """
        # Check reflection symmetry by comparing field with its flip
        # For a symmetric field: f(x) = f(-x), the residual should be small
        if field.ndim >= 1:
            # Flip along the last spatial dimension
            flipped_field = jnp.flip(field, axis=-1)
            # Compute mean squared difference
            return jnp.mean((field - flipped_field) ** 2)
        return jnp.array(0.0)


class ResidualComputer:
    """Physics equation residual computation system using PDEResidualRegistry.

    This class now serves as a wrapper around PDEResidualRegistry, providing
    backward-compatible API while using proper autodiff-based implementations.

    REFACTORED: Previously had mock implementations. Now uses:
    - PDEResidualRegistry for extensible PDE registration
    - AutoDiffEngine for proper derivative computation
    - Zero code duplication (DRY principle)

    Attributes:
        equation_type: Type of physics equation
        domain_type: Type of computational domain
        equation_params: Additional equation parameters
    """

    def __init__(
        self,
        equation_type: str,
        domain_type: str,
        **equation_params,
    ):
        """Initialize residual computer.

        Args:
            equation_type: Physics equation type (must be registered in
                PDEResidualRegistry)
            domain_type: Computational domain type (maintained for backward
                compatibility)
            **equation_params: Additional equation parameters passed to PDE
                residual function
        """
        self.equation_type = equation_type
        self.domain_type = domain_type
        self.equation_params = equation_params

    def compute_residual(
        self,
        model: Callable[[jax.Array], jax.Array],
        inputs: jax.Array,
        source: jax.Array | None = None,
        **kwargs,
    ) -> jax.Array:
        """Compute physics equation residual using PDEResidualRegistry.

        REFACTORED API: Now expects a model function instead of predictions.
        This enables proper autodiff-based derivative computation.

        Args:
            model: Callable model function that maps inputs to predictions
                   (e.g., lambda x: x[..., 0]**2 + x[..., 1]**2)
            inputs: Input coordinates where to evaluate the residual
            source: Source term (if applicable, for Poisson equation)
            **kwargs: Additional parameters passed to PDE residual function

        Returns:
            Physics equation residual computed using proper autodiff (per-point)

        Raises:
            KeyError: If equation_type is not registered in PDEResidualRegistry

        Examples:
            >>> # Define solution function
            >>> def u(x):
            ...     return x[..., 0]**2 + x[..., 1]**2
            >>> computer = ResidualComputer("poisson", "2d")
            >>> x = jnp.array([[1.0, 1.0]])
            >>> residual = computer.compute_residual(u, x, source=jnp.array([4.0]))
        """
        # Import here to avoid circular imports
        from opifex.core.physics.autodiff_engine import AutoDiffEngine
        from opifex.core.physics.pde_registry import PDEResidualRegistry

        # Get PDE residual function from registry
        pde_residual_fn = PDEResidualRegistry.get(self.equation_type)

        # Merge kwargs with equation_params
        params = {**self.equation_params, **kwargs}

        # Special handling for source term (Poisson equation)
        if source is not None:
            params["source_term"] = source

        # Compute residual using registry function with real autodiff
        return pde_residual_fn(model, inputs, AutoDiffEngine, **params)


class PhysicsInformedLoss:
    """Integrated physics-informed loss system.

    This class provides a complete physics-informed loss system that
    integrates all components: loss composition, adaptive weighting,
    conservation law enforcement, and residual computation.

    Attributes:
        config: Physics loss configuration
        composer: Loss composition system
        weight_scheduler: Adaptive weight scheduler
        conservation_enforcer: Conservation law enforcer
        residual_computer: Physics residual computer
        current_epoch: Current training epoch
    """

    def __init__(
        self,
        config: PhysicsLossConfig,
        equation_type: str,
        domain_type: str,
        **equation_params,
    ):
        """Initialize integrated physics-informed loss system.

        Args:
            config: Physics loss configuration
            equation_type: Physics equation type
            domain_type: Computational domain type
            **equation_params: Additional equation parameters
        """
        self.config = config
        self.current_epoch = 0

        # Initialize components
        self.composer = PhysicsLossComposer(config)

        if config.adaptive_weighting:
            self.weight_scheduler: AdaptiveWeightScheduler | None = (
                AdaptiveWeightScheduler(
                    schedule_type=config.weight_schedule,
                    initial_physics_weight=config.physics_loss_weight,
                    final_physics_weight=1.0,
                    transition_epochs=1000,
                    step_milestones=config.step_milestones,
                )
            )
        else:
            self.weight_scheduler: AdaptiveWeightScheduler | None = None

        if config.conservation_weights:
            self.conservation_enforcer: ConservationLawEnforcer | None = (
                ConservationLawEnforcer(
                    conservation_laws=list(config.conservation_weights.keys()),
                    tolerance=1e-6,
                )
            )
        else:
            self.conservation_enforcer: ConservationLawEnforcer | None = None

        self.residual_computer = ResidualComputer(
            equation_type=equation_type,
            domain_type=domain_type,
            **equation_params,
        )

    def compute_loss(
        self,
        predictions: jax.Array,
        targets: jax.Array,
        inputs: jax.Array,
        model: Callable[[jax.Array], jax.Array] | None = None,
        boundary_predictions: jax.Array | None = None,
        boundary_targets: jax.Array | None = None,
        boundary_inputs: jax.Array | None = None,
        density_matrix: jax.Array | None = None,
        n_electrons: int | None = None,
        epoch: int = 0,
        rngs: Any = None,  # Add RNG support for modern NNX patterns
        **kwargs,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute complete physics-informed loss.

        REFACTORED: Now accepts optional model function for physics residuals.

        Args:
            predictions: Model predictions
            targets: Target values
            inputs: Input data
            model: Optional model function for computing physics residuals via autodiff
                   If None, physics residual will be set to zero
            boundary_predictions: Boundary predictions
            boundary_targets: Boundary targets
            boundary_inputs: Boundary input coordinates
            density_matrix: Quantum density matrix
            n_electrons: Number of electrons (quantum systems)
            epoch: Current training epoch
            rngs: RNGs for NNX patterns
            **kwargs: Additional parameters

        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        return self._execute_loss_computation(
            predictions=predictions,
            targets=targets,
            inputs=inputs,
            model=model,
            boundary_predictions=boundary_predictions,
            boundary_targets=boundary_targets,
            boundary_inputs=boundary_inputs,
            density_matrix=density_matrix,
            n_electrons=n_electrons,
            epoch=epoch,
            **kwargs,
        )

    def _execute_loss_computation(
        self,
        predictions: jax.Array,
        targets: jax.Array,
        inputs: jax.Array,
        model: Callable[[jax.Array], jax.Array] | None,
        boundary_predictions: jax.Array | None,
        boundary_targets: jax.Array | None,
        boundary_inputs: jax.Array | None,
        density_matrix: jax.Array | None,
        n_electrons: int | None,
        epoch: int,
        **kwargs,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Execute the complete loss computation workflow."""
        self.current_epoch = epoch

        # Compute basic loss components
        basic_losses = self._compute_basic_losses(
            predictions,
            targets,
            inputs,
            model,
            boundary_predictions,
            boundary_targets,
            **kwargs,
        )

        # Compute conservation law residuals
        conservation_residuals = self._compute_conservation_residuals(
            predictions, inputs, density_matrix
        )

        # Compute quantum constraint residuals
        quantum_residuals = self._compute_quantum_residuals(density_matrix, n_electrons)

        # Get effective composer with updated weights
        composer = self._get_effective_composer(epoch)

        # Compose total loss
        total_loss = composer.compose_loss(
            data_loss=basic_losses["data_loss"],
            physics_residual=basic_losses["physics_residual"],
            boundary_residual=basic_losses["boundary_residual"],
            conservation_residuals=conservation_residuals,
            quantum_residuals=quantum_residuals,
        )

        # Prepare loss components dictionary
        loss_components = self._prepare_loss_components(
            basic_losses, conservation_residuals, quantum_residuals, total_loss
        )

        return total_loss, loss_components

    def _compute_basic_losses(
        self,
        predictions: jax.Array,
        targets: jax.Array,
        inputs: jax.Array,
        model: Callable[[jax.Array], jax.Array] | None,
        boundary_predictions: jax.Array | None,
        boundary_targets: jax.Array | None,
        **kwargs,
    ) -> dict[str, jax.Array]:
        """Compute basic loss components (data, physics, boundary)."""
        data_loss = jnp.mean((predictions - targets) ** 2)

        # Compute physics residual only if model function is provided
        if model is not None:
            residual_per_point = self.residual_computer.compute_residual(
                model, inputs, **kwargs
            )
            # Average over batch to get scalar physics_residual for loss composition
            physics_residual = jnp.mean(residual_per_point**2)
        else:
            # No model provided - skip physics residual computation
            physics_residual = jnp.array(0.0)

        if boundary_predictions is not None and boundary_targets is not None:
            boundary_residual = jnp.mean((boundary_predictions - boundary_targets) ** 2)
        else:
            boundary_residual = jnp.array(0.0)

        return {
            "data_loss": data_loss,
            "physics_residual": physics_residual,
            "boundary_residual": boundary_residual,
        }

    def _compute_conservation_residuals(
        self,
        predictions: jax.Array,
        inputs: jax.Array,
        density_matrix: jax.Array | None,
    ) -> dict[str, jax.Array] | None:
        """Compute conservation law residuals."""
        if self.conservation_enforcer is None:
            return None

        conservation_residuals = {}
        for law in self.conservation_enforcer.conservation_laws:
            if law in ["particle_number", "charge"] and density_matrix is not None:
                residual = self.conservation_enforcer.compute_residual(
                    law, density_matrix
                )
            else:
                residual = self.conservation_enforcer.compute_residual(
                    law, predictions, inputs
                )
            conservation_residuals[law] = residual

        return conservation_residuals

    def _compute_quantum_residuals(
        self,
        density_matrix: jax.Array | None,
        n_electrons: int | None,
    ) -> dict[str, jax.Array] | None:
        """Compute quantum constraint residuals."""
        if not self.config.quantum_constraints or density_matrix is None:
            return None

        quantum_residuals = {}

        # Density positivity constraint
        if self.config.density_positivity_weight > 0:
            density_positivity_violation = jnp.mean(
                jnp.maximum(0.0, -jnp.real(density_matrix))
            )
            quantum_residuals["density_positivity"] = density_positivity_violation

        # Wavefunction normalization constraint
        if self.config.wavefunction_normalization_weight > 0:
            trace_dm = jnp.trace(density_matrix, axis1=-2, axis2=-1)
            if n_electrons is not None:
                normalization_violation = jnp.mean((trace_dm - n_electrons) ** 2)
            else:
                normalization_violation = jnp.mean((trace_dm - 1.0) ** 2)
            quantum_residuals["wavefunction_normalization"] = normalization_violation

        return quantum_residuals

    def _get_effective_composer(self, epoch: int) -> PhysicsLossComposer:
        """Get effective composer with updated weights for current epoch."""
        if self.weight_scheduler is None:
            return self.composer

        current_physics_weight = self.weight_scheduler.get_weight(epoch)
        # Convert JAX array to float for config
        current_physics_weight_float = float(current_physics_weight)

        # Update config for this computation
        effective_config = PhysicsLossConfig(
            data_loss_weight=self.config.data_loss_weight,
            physics_loss_weight=current_physics_weight_float,
            boundary_loss_weight=self.config.boundary_loss_weight,
            conservation_weights=self.config.conservation_weights,
            quantum_constraints=self.config.quantum_constraints,
            density_positivity_weight=self.config.density_positivity_weight,
            wavefunction_normalization_weight=self.config.wavefunction_normalization_weight,
        )
        return PhysicsLossComposer(effective_config)

    def _prepare_loss_components(
        self,
        basic_losses: dict[str, jax.Array],
        conservation_residuals: dict[str, jax.Array] | None,
        quantum_residuals: dict[str, jax.Array] | None,
        total_loss: jax.Array,
    ) -> dict[str, jax.Array]:
        """Prepare comprehensive loss components dictionary."""
        loss_components = {
            "data_loss": basic_losses["data_loss"],
            "physics_loss": basic_losses["physics_residual"],
            "boundary_loss": basic_losses["boundary_residual"],
            "total_loss": total_loss,
        }

        if conservation_residuals is not None:
            loss_components.update(conservation_residuals)

        if quantum_residuals is not None:
            loss_components.update(quantum_residuals)
            # Add combined quantum constraints key for consistency with
            # test expectations
            quantum_total = jnp.array(0.0)
            for residual in quantum_residuals.values():
                quantum_total = quantum_total + residual
            loss_components["quantum_constraints"] = quantum_total

        return loss_components

    def update_weights(self, epoch: int) -> None:
        """Update adaptive weights for given epoch.

        Args:
            epoch: Current training epoch
        """
        self.current_epoch = epoch

    def get_current_physics_weight(self) -> jax.Array:
        """Get current physics loss weight.

        Returns:
            Current physics weight
        """
        if self.weight_scheduler is not None:
            return self.weight_scheduler.get_weight(self.current_epoch)
        return jnp.array(self.config.physics_loss_weight)
