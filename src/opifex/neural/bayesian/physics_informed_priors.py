"""Physics-informed priors for enforcing physical constraints in probabilistic models.

This module provides tools for incorporating physical knowledge into Bayesian
priors, ensuring that sampled parameters respect conservation laws and other
physical constraints.
"""

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.core.physics import apply_dirichlet, apply_neumann, apply_periodic


class PhysicsInformedPriors(nnx.Module):
    """Physics-informed prior constraints for Bayesian models.

    Enforces conservation laws, boundary conditions, and other physical
    constraints through learnable constraint weights and penalty functions.
    """

    def __init__(
        self,
        conservation_laws: Sequence[str] = (),
        boundary_conditions: Sequence[str] = (),
        constraint_weights: jax.Array | None = None,
        penalty_weight: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize physics-informed priors.

        Args:
            conservation_laws: List of conservation laws to enforce
            boundary_conditions: List of boundary conditions to enforce
            constraint_weights: Optional custom weights for constraints
            penalty_weight: Weight for constraint violation penalties
            rngs: Random number generators
        """
        super().__init__()

        self.conservation_laws = conservation_laws
        self.boundary_conditions = boundary_conditions
        self.penalty_weight = penalty_weight

        # Initialize constraint weights
        num_constraints = len(conservation_laws) + len(boundary_conditions)
        if constraint_weights is not None:
            initial_weights = constraint_weights
        else:
            initial_weights = jnp.ones(num_constraints)

        self.constraint_weights = nnx.Param(initial_weights)

    def apply_constraints(self, params: jax.Array) -> jax.Array:
        """Apply physics constraints to sampled parameters.

        Args:
            params: Unconstrained parameter samples

        Returns:
            Constrained parameters that satisfy physics laws
        """
        constrained_params = params

        # Apply conservation law constraints
        for i, law in enumerate(self.conservation_laws):
            weight = float(self.constraint_weights.value[i])
            constrained_params = self._apply_conservation_law(
                constrained_params, law, weight
            )

        # Apply boundary condition constraints
        boundary_start_idx = len(self.conservation_laws)
        for i, condition in enumerate(self.boundary_conditions):
            weight = float(self.constraint_weights.value[boundary_start_idx + i])
            constrained_params = self._apply_boundary_condition(
                constrained_params, condition, weight
            )

        return constrained_params

    def _apply_conservation_law(
        self, params: jax.Array, law: str, weight: float
    ) -> jax.Array:
        """Apply specific conservation law constraint.

        Args:
            params: Parameter values to constrain
            law: Conservation law type
            weight: Constraint weight

        Returns:
            Constrained parameters
        """
        if law == "energy":
            # Energy conservation: ensure total energy is preserved
            # Simple projection: normalize parameter magnitude
            param_norm = jnp.linalg.norm(params, axis=-1, keepdims=True)
            target_norm = jnp.sqrt(params.shape[-1])  # Target norm
            constrained = params * (target_norm / (param_norm + 1e-8))
            return weight * constrained + (1 - weight) * params

        if law == "momentum":
            # Momentum conservation: ensure momentum sum is zero
            # Project out the mean to achieve zero total momentum
            mean_momentum = jnp.mean(params, axis=-1, keepdims=True)
            constrained = params - mean_momentum
            return weight * constrained + (1 - weight) * params

        if law == "mass":
            # Mass conservation: ensure positive masses
            constrained = jnp.abs(params)
            return weight * constrained + (1 - weight) * params

        if law == "positivity":
            # Positivity constraint: ensure all values are positive
            constrained = jnp.maximum(params, 1e-8)
            return weight * constrained + (1 - weight) * params

        if law == "boundedness":
            # Boundedness constraint: keep values within reasonable bounds
            constrained = jnp.tanh(params)  # Bounded to [-1, 1]
            return weight * constrained + (1 - weight) * params

        return params

    def _apply_boundary_condition(
        self, params: jax.Array, condition: str, weight: float
    ) -> jax.Array:
        """Apply specific boundary condition constraint.

        This method now uses the centralized boundary condition functions
        from opifex.core.physics.boundaries.

        Args:
            params: Parameter values to constrain
            condition: Boundary condition type
            weight: Constraint weight

        Returns:
            Constrained parameters
        """
        if condition == "dirichlet":
            return apply_dirichlet(params, boundary_value=0.0, weight=weight)

        if condition == "neumann":
            return apply_neumann(params, weight=weight)

        if condition == "periodic":
            return apply_periodic(params, weight=weight)

        return params

    def compute_violation_penalty(self, params: jax.Array) -> float:
        """Compute penalty for physics constraint violations.

        Args:
            params: Parameter values to evaluate

        Returns:
            Violation penalty (higher = more violation)
        """
        penalty = 0.0

        # Check for numerical issues first and return immediately if found
        has_nan = jnp.any(jnp.isnan(params))
        has_inf = jnp.any(jnp.isinf(params))

        if has_nan or has_inf:
            # Return large finite penalty immediately to avoid NaN propagation
            return 1e6 * self.penalty_weight

        # Check conservation law violations (only if params are finite)
        for law in self.conservation_laws:
            if law == "energy":
                # Penalize large deviations from target energy
                target_energy = params.shape[-1]
                actual_energy = jnp.sum(params**2)
                penalty += jnp.abs(actual_energy - target_energy)

            elif law == "momentum":
                # Penalize non-zero total momentum
                total_momentum = jnp.sum(params)
                penalty += jnp.abs(total_momentum)

            elif law == "positivity":
                # Penalize negative values
                negative_penalty = jnp.sum(jnp.maximum(-params, 0.0))
                penalty += negative_penalty

        return float(penalty * self.penalty_weight)

    def check_physical_plausibility(self, params: jax.Array) -> float:
        """Check physical plausibility of parameters.

        Args:
            params: Parameter values to check

        Returns:
            Plausibility score between 0 (implausible) and 1 (plausible)
        """
        # Start with perfect plausibility
        plausibility = 1.0

        # Check for numerical issues
        if jnp.any(jnp.isnan(params)) or jnp.any(jnp.isinf(params)):
            return 0.0

        # Check magnitude plausibility
        param_magnitude = jnp.linalg.norm(params)
        if param_magnitude > 1e3:  # Very large parameters
            plausibility *= 0.1
        elif param_magnitude > 10:  # Moderately large parameters
            plausibility *= 0.5

        # Check constraint violations
        for law in self.conservation_laws:
            if law == "positivity":
                # Check if all values are positive
                if jnp.any(params < 0):
                    plausibility *= 0.3

            elif law == "boundedness" and jnp.any(jnp.abs(params) > 10):
                # Check if values are within reasonable bounds
                plausibility *= 0.5

        return plausibility


# Phase 3: Physics-Informed Integration - NEW CLASSES


class ConservationLawPriors(nnx.Module):
    """Conservation law priors for uncertainty estimation.

    This class implements physics-aware priors that incorporate conservation
    laws directly into uncertainty quantification, enabling physically
    consistent uncertainty estimates.
    """

    def __init__(
        self,
        conservation_laws: Sequence[str] = ("energy", "momentum", "mass"),
        uncertainty_scale: float = 0.1,
        prior_strength: float = 1.0,
        adaptive_weighting: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize conservation law priors for uncertainty estimation.

        Args:
            conservation_laws: List of conservation laws to enforce
            uncertainty_scale: Scale factor for uncertainty estimates
            prior_strength: Strength of physics constraints in prior
            adaptive_weighting: Whether to use adaptive constraint weighting
            rngs: Random number generators
        """
        super().__init__()

        self.conservation_laws = conservation_laws
        self.uncertainty_scale = uncertainty_scale
        self.prior_strength = prior_strength
        self.adaptive_weighting = adaptive_weighting

        # Initialize learnable conservation strength parameters
        num_laws = len(conservation_laws)
        self.conservation_strengths = nnx.Param(jnp.ones(num_laws))

        # Initialize uncertainty scaling parameters for each conservation law
        self.uncertainty_scalings = nnx.Param(uncertainty_scale * jnp.ones(num_laws))

    def compute_physics_aware_uncertainty(
        self,
        predictions: jax.Array,
        model_uncertainty: jax.Array,
        physics_state: jax.Array,
    ) -> jax.Array:
        """Compute physics-aware uncertainty estimates.

        Args:
            predictions: Model predictions
            model_uncertainty: Basic model uncertainty
            physics_state: Physical state variables for constraint evaluation

        Returns:
            Physics-aware uncertainty estimates
        """
        # Start with model uncertainty
        physics_uncertainty = model_uncertainty

        # Add physics-informed uncertainty contributions
        for i, law in enumerate(self.conservation_laws):
            constraint_violation = self._evaluate_conservation_violation(
                predictions, physics_state, law
            )

            # Scale uncertainty based on constraint violations
            law_scaling = float(self.uncertainty_scalings.value[i])
            law_strength = float(self.conservation_strengths.value[i])

            physics_contribution = (
                law_scaling * law_strength * jnp.abs(constraint_violation)
            )
            physics_uncertainty = physics_uncertainty + physics_contribution

        return physics_uncertainty

    def _evaluate_conservation_violation(
        self, predictions: jax.Array, physics_state: jax.Array, law: str
    ) -> jax.Array:
        """Evaluate violation of specific conservation law.

        Args:
            predictions: Model predictions
            physics_state: Current physics state
            law: Conservation law type

        Returns:
            Violation magnitude for the law
        """
        if law == "energy":
            return self._evaluate_energy_violation(predictions, physics_state)
        if law == "momentum":
            return self._evaluate_momentum_violation(predictions, physics_state)
        if law == "mass":
            return self._evaluate_mass_violation(predictions)

        # Unknown conservation law - return zeros with proper shape
        return self._get_zero_shaped_like(predictions)

    def _evaluate_energy_violation(
        self, predictions: jax.Array, physics_state: jax.Array
    ) -> jax.Array:
        """Evaluate energy conservation violation."""
        initial_energy = physics_state[..., 0]  # First state component

        # Handle predictions properly - if last dim > 1, take the first component
        if predictions.shape[-1] == 1:
            predicted_energy = initial_energy + predictions.squeeze(-1)
        else:
            # Take first component for energy predictions
            predicted_energy = initial_energy + predictions[..., 0]

        return jnp.abs(predicted_energy - initial_energy)

    def _evaluate_momentum_violation(
        self, predictions: jax.Array, physics_state: jax.Array
    ) -> jax.Array:
        """Evaluate momentum conservation violation."""
        # Momentum conservation: check momentum balance
        # Assume physics_state contains momentum components
        if physics_state.shape[-1] >= 3:
            initial_momentum = physics_state[..., 1:4]  # Components 1-3
            pred_momentum = self._get_compatible_momentum(predictions, initial_momentum)
            return jnp.linalg.norm(initial_momentum + pred_momentum, axis=-1)

        # Fallback for insufficient state dimensions
        return self._get_zero_shaped_like(predictions)

    def _evaluate_mass_violation(self, predictions: jax.Array) -> jax.Array:
        """Evaluate mass conservation violation."""
        return jnp.abs(jnp.sum(predictions, axis=-1))

    def _get_compatible_momentum(
        self, predictions: jax.Array, initial_momentum: jax.Array
    ) -> jax.Array:
        """Get momentum predictions compatible with initial momentum shape."""
        pred_dim = predictions.shape[-1]
        momentum_dim = initial_momentum.shape[-1]

        if pred_dim >= momentum_dim:
            # Predictions have enough dimensions
            return predictions[..., :momentum_dim]

        # Pad predictions to match momentum dimensions
        padding_needed = momentum_dim - pred_dim
        return jnp.concatenate(
            [
                predictions,
                jnp.zeros((*predictions.shape[:-1], padding_needed)),
            ],
            axis=-1,
        )

    def _get_zero_shaped_like(self, predictions: jax.Array) -> jax.Array:
        """Get zeros with shape compatible with predictions."""
        if predictions.shape[-1] == 1:
            return jnp.zeros_like(predictions.squeeze(-1))
        return jnp.zeros(predictions.shape[:-1])

    def sample_physics_constrained_params(
        self, base_params: jax.Array, constraint_strength: float = 1.0
    ) -> jax.Array:
        """Sample parameters that satisfy physics constraints.

        Args:
            base_params: Base parameter samples
            constraint_strength: Strength of constraint enforcement

        Returns:
            Physics-constrained parameter samples
        """
        constrained_params = base_params

        for i, law in enumerate(self.conservation_laws):
            law_strength = (
                float(self.conservation_strengths.value[i]) * constraint_strength
            )
            constrained_params = self._apply_conservation_constraint(
                constrained_params, law, law_strength
            )

        return constrained_params

    def _apply_conservation_constraint(
        self, params: jax.Array, law: str, strength: float
    ) -> jax.Array:
        """Apply conservation constraint to parameters.

        Args:
            params: Parameter values to constrain
            law: Conservation law type
            strength: Constraint strength

        Returns:
            Constrained parameters
        """
        if law == "energy":
            # Energy conservation: normalize to preserve energy
            param_energy = jnp.sum(params**2, axis=-1, keepdims=True)
            target_energy = params.shape[-1]  # Target total energy
            energy_correction = jnp.sqrt(target_energy / (param_energy + 1e-8))
            constrained = params * energy_correction
            return strength * constrained + (1 - strength) * params

        if law == "momentum":
            # Momentum conservation: remove net momentum
            net_momentum = jnp.mean(params, axis=-1, keepdims=True)
            constrained = params - net_momentum
            return strength * constrained + (1 - strength) * params

        if law == "mass":
            # Mass conservation: ensure positive definite
            constrained = jnp.abs(params)
            return strength * constrained + (1 - strength) * params

        return params


class DomainSpecificPriors(nnx.Module):
    """Domain-specific prior distributions for scientific computing.

    Provides specialized priors for different scientific domains including
    quantum mechanics, molecular dynamics, fluid dynamics, and materials science.
    """

    def __init__(
        self,
        domain: str = "quantum_chemistry",
        parameter_ranges: dict[str, tuple[float, float]] | None = None,
        distribution_types: dict[str, str] | None = None,
        correlation_structure: str = "independent",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize domain-specific priors.

        Args:
            domain: Scientific domain (quantum_chemistry, molecular_dynamics, etc.)
            parameter_ranges: Custom parameter ranges for specific parameters
            distribution_types: Distribution types for each parameter
            correlation_structure: Correlation structure between parameters
            rngs: Random number generators
        """
        super().__init__()

        self.domain = domain
        self.correlation_structure = correlation_structure

        # Store RNG for proper random number generation
        self.rngs = rngs

        # Set up domain-specific parameter ranges and distributions
        self.parameter_ranges = parameter_ranges or self._get_default_ranges(domain)
        self.distribution_types = distribution_types or self._get_default_distributions(
            domain
        )

        # Initialize prior parameters for each parameter type
        n_params = len(self.parameter_ranges)
        self.prior_means = nnx.Param(jnp.zeros(n_params))
        self.prior_scales = nnx.Param(jnp.ones(n_params))

        # Initialize correlation matrix if needed
        if correlation_structure != "independent":
            self.correlation_matrix = nnx.Param(jnp.eye(n_params))
        else:
            self.correlation_matrix = None

    def _get_default_ranges(self, domain: str) -> dict[str, tuple[float, float]]:
        """Get default parameter ranges for scientific domain.

        Args:
            domain: Scientific domain

        Returns:
            Dictionary of parameter ranges
        """
        if domain == "quantum_chemistry":
            return {
                "bond_length": (0.5, 3.0),  # Angstroms
                "bond_angle": (60.0, 180.0),  # Degrees
                "charge": (-2.0, 2.0),  # Elementary charge
                "energy": (-1000.0, 100.0),  # kcal/mol
                "dipole": (0.0, 10.0),  # Debye
            }
        if domain == "molecular_dynamics":
            return {
                "temperature": (200.0, 400.0),  # Kelvin
                "pressure": (0.8, 1.2),  # Bar
                "density": (0.8, 1.2),  # g/cm³
                "velocity": (-100.0, 100.0),  # m/s
                "force": (-1000.0, 1000.0),  # pN
            }
        if domain == "fluid_dynamics":
            return {
                "velocity": (0.0, 100.0),  # m/s
                "pressure": (0.9, 1.1),  # atm
                "temperature": (273.0, 373.0),  # Kelvin
                "viscosity": (1e-6, 1e-3),  # Pa·s
                "reynolds": (10.0, 1e6),  # Dimensionless
            }
        # Generic scientific computing ranges
        return {
            "parameter_1": (-10.0, 10.0),
            "parameter_2": (0.0, 100.0),
            "parameter_3": (-1.0, 1.0),
        }

    def _get_default_distributions(self, domain: str) -> dict[str, str]:
        """Get default distribution types for scientific domain.

        Args:
            domain: Scientific domain

        Returns:
            Dictionary of distribution types
        """
        if domain == "quantum_chemistry":
            return {
                "bond_length": "lognormal",
                "bond_angle": "beta",
                "charge": "normal",
                "energy": "normal",
                "dipole": "gamma",
            }
        if domain == "molecular_dynamics":
            return {
                "temperature": "gamma",
                "pressure": "lognormal",
                "density": "normal",
                "velocity": "normal",
                "force": "normal",
            }
        # Default to normal distributions
        param_names = list(self._get_default_ranges(domain).keys())
        return dict.fromkeys(param_names, "normal")

    def sample_domain_priors(
        self, sample_shape: tuple[int, ...], parameter_type: str
    ) -> jax.Array:
        """Sample from domain-specific priors.

        Args:
            sample_shape: Shape of samples to generate
            parameter_type: Type of parameter to sample

        Returns:
            Samples from domain-specific prior distribution
        """
        if parameter_type not in self.parameter_ranges:
            raise ValueError(f"Unknown parameter type: {parameter_type}")

        param_idx = list(self.parameter_ranges.keys()).index(parameter_type)
        prior_mean = self.prior_means.value[param_idx]
        prior_scale = self.prior_scales.value[param_idx]

        param_range = self.parameter_ranges[parameter_type]
        distribution_type = self.distribution_types[parameter_type]

        # Generate samples based on distribution type
        key = self.rngs.params()  # Use proper RNG from module

        if distribution_type == "normal":
            samples = jax.random.normal(key, sample_shape) * prior_scale + prior_mean
        elif distribution_type == "lognormal":
            log_samples = (
                jax.random.normal(key, sample_shape) * prior_scale + prior_mean
            )
            samples = jnp.exp(log_samples)
        elif distribution_type == "gamma":
            # Use shape=2, scale=prior_scale for gamma distribution
            samples = jax.random.gamma(key, 2.0, sample_shape) * prior_scale
        elif distribution_type == "beta":
            # Beta distribution between 0 and 1, then scale to range
            beta_samples = jax.random.beta(key, 2.0, 2.0, sample_shape)
            range_min, range_max = param_range
            samples = beta_samples * (range_max - range_min) + range_min
        else:
            # Default to normal
            samples = jax.random.normal(key, sample_shape) * prior_scale + prior_mean

        # Clip to valid range
        range_min, range_max = param_range
        return jnp.clip(samples, range_min, range_max)

    def evaluate_prior_log_prob(
        self, values: jax.Array, parameter_type: str
    ) -> jax.Array:
        """Evaluate log probability under domain-specific prior.

        Args:
            values: Parameter values to evaluate
            parameter_type: Type of parameter

        Returns:
            Log probability under domain prior
        """
        if parameter_type not in self.parameter_ranges:
            # Return very low probability for unknown parameters
            return jnp.full_like(values, -1e6)

        param_idx = list(self.parameter_ranges.keys()).index(parameter_type)
        prior_mean = self.prior_means.value[param_idx]
        prior_scale = self.prior_scales.value[param_idx]

        distribution_type = self.distribution_types[parameter_type]

        if distribution_type == "normal":
            # Normal log probability
            normalized_values = (values - prior_mean) / prior_scale
            log_prob = (
                -0.5 * normalized_values**2
                - jnp.log(prior_scale)
                - 0.5 * jnp.log(2 * jnp.pi)
            )
        elif distribution_type == "lognormal":
            # Log-normal log probability
            log_values = jnp.log(jnp.maximum(values, 1e-8))
            normalized_log_values = (log_values - prior_mean) / prior_scale
            log_prob = (
                -0.5 * normalized_log_values**2
                - jnp.log(prior_scale)
                - 0.5 * jnp.log(2 * jnp.pi)
                - log_values
            )
        else:
            # Approximate with normal for other distributions
            normalized_values = (values - prior_mean) / prior_scale
            log_prob = -0.5 * normalized_values**2 - jnp.log(prior_scale)

        return log_prob


class HierarchicalBayesianFramework(nnx.Module):
    """Hierarchical Bayesian framework for multi-level uncertainty estimation.

    Implements hierarchical models that can capture uncertainty at multiple
    scales and levels, suitable for complex scientific computing applications.
    """

    def __init__(
        self,
        hierarchy_levels: int = 3,
        level_dimensions: Sequence[int] = (64, 32, 16),
        uncertainty_propagation: str = "multiplicative",
        correlation_structure: str = "exchangeable",
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize hierarchical Bayesian framework.

        Args:
            hierarchy_levels: Number of hierarchy levels
            level_dimensions: Dimensions for each hierarchy level
            uncertainty_propagation: How uncertainty propagates between levels
            correlation_structure: Correlation structure between levels
            rngs: Random number generators
        """
        super().__init__()

        self.hierarchy_levels = hierarchy_levels
        self.level_dimensions = level_dimensions
        self.uncertainty_propagation = uncertainty_propagation
        self.correlation_structure = correlation_structure

        # Store RNG for proper random number generation
        self.rngs = rngs

        # Initialize parameters for each hierarchy level
        self.level_means = nnx.List([])
        self.level_scales = nnx.List([])
        self.level_correlations = nnx.List([])

        for _i, dim in enumerate(level_dimensions):
            # Mean parameters for this level
            self.level_means.append(nnx.Param(jnp.zeros(dim)))

            # Scale parameters for this level
            self.level_scales.append(nnx.Param(jnp.ones(dim)))

            # Correlation structure for this level
            if correlation_structure == "exchangeable":
                correlation_matrix = jnp.eye(dim)
            else:
                correlation_matrix = jnp.eye(dim)
            self.level_correlations.append(nnx.Param(correlation_matrix))

        # Propagation weights between levels
        self.propagation_weights = nnx.Param(jnp.ones(hierarchy_levels - 1))

    def sample_hierarchical_parameters(
        self, sample_shape: tuple[int, ...], level: int = 0
    ) -> jax.Array:
        """Sample parameters from hierarchical model at specified level.

        Args:
            sample_shape: Shape of samples to generate
            level: Hierarchy level to sample from

        Returns:
            Hierarchical parameter samples
        """
        if level >= len(self.level_dimensions):
            raise ValueError(f"Level {level} exceeds hierarchy depth")

        # Sample from the specified level
        mean = self.level_means[level].value
        scale = self.level_scales[level].value

        key = self.rngs.params()  # Use proper RNG from module
        base_samples = jax.random.normal(
            key, (*sample_shape, self.level_dimensions[level])
        )

        # Apply mean and scale
        return base_samples * scale + mean

    def propagate_uncertainty_hierarchically(
        self, base_uncertainty: jax.Array, target_level: int
    ) -> jax.Array:
        """Propagate uncertainty through hierarchy levels.

        Args:
            base_uncertainty: Base uncertainty estimates
            target_level: Target hierarchy level

        Returns:
            Hierarchically propagated uncertainty
        """
        propagated_uncertainty = base_uncertainty

        for level in range(target_level):
            weight = self.propagation_weights.value[level]
            level_scale = jnp.mean(self.level_scales[level].value)

            if self.uncertainty_propagation == "multiplicative":
                propagated_uncertainty = propagated_uncertainty * (
                    1 + weight * level_scale
                )
            elif self.uncertainty_propagation == "additive":
                propagated_uncertainty = propagated_uncertainty + weight * level_scale
            else:
                # Default to additive
                propagated_uncertainty = propagated_uncertainty + weight * level_scale

        return propagated_uncertainty

    def compute_hierarchical_log_prob(self, values: jax.Array, level: int) -> jax.Array:
        """Compute log probability under hierarchical model.

        Args:
            values: Parameter values to evaluate
            level: Hierarchy level

        Returns:
            Log probability under hierarchical model
        """
        if level >= len(self.level_dimensions):
            return jnp.full(values.shape[:-1], -1e6)

        mean = self.level_means[level].value
        scale = self.level_scales[level].value

        # Compute log probability for each dimension
        normalized_values = (values - mean) / scale
        log_prob_per_dim = (
            -0.5 * normalized_values**2 - jnp.log(scale) - 0.5 * jnp.log(2 * jnp.pi)
        )

        # Sum over dimensions
        return jnp.sum(log_prob_per_dim, axis=-1)

    def adaptive_hierarchy_weighting(
        self, observed_data: jax.Array, predictions: jax.Array
    ) -> jax.Array:
        """Adaptively weight hierarchy levels based on data fit.

        Args:
            observed_data: Observed data for adaptation
            predictions: Model predictions at different levels

        Returns:
            Adaptive weights for hierarchy levels
        """
        # Compute fit quality for each level
        fit_scores = []

        for level in range(self.hierarchy_levels):
            # Simple MSE-based fit score
            if level < predictions.shape[0]:
                level_predictions = predictions[level]
                mse = jnp.mean((observed_data - level_predictions) ** 2)
                fit_score = 1.0 / (1.0 + mse)  # Higher score for better fit
                fit_scores.append(fit_score)
            else:
                fit_scores.append(0.1)  # Low score for missing levels

        fit_scores = jnp.array(fit_scores)

        # Normalize to get weights
        return fit_scores / (jnp.sum(fit_scores) + 1e-8)


class PhysicsAwareUncertaintyPropagation(nnx.Module):
    """Physics-aware uncertainty propagation for scientific computing.

    Propagates uncertainty through physics-informed models while respecting
    conservation laws and physical constraints.
    """

    def __init__(
        self,
        conservation_laws: Sequence[str] = ("energy", "momentum"),
        constraint_tolerance: float = 1e-6,
        uncertainty_inflation: float = 1.1,
        correlation_aware: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """Initialize physics-aware uncertainty propagation.

        Args:
            conservation_laws: Conservation laws to respect during propagation
            constraint_tolerance: Tolerance for constraint violations
            uncertainty_inflation: Factor to inflate uncertainty for safety
            correlation_aware: Whether to account for parameter correlations
            rngs: Random number generators
        """
        super().__init__()

        self.conservation_laws = conservation_laws
        self.constraint_tolerance = constraint_tolerance
        self.uncertainty_inflation = uncertainty_inflation
        self.correlation_aware = correlation_aware

        # Initialize constraint enforcement weights
        num_laws = len(conservation_laws)
        self.constraint_weights = nnx.Param(jnp.ones(num_laws))

    def propagate_with_physics_constraints(
        self,
        input_uncertainty: jax.Array,
        model_jacobian: jax.Array,
        physics_state: jax.Array,
    ) -> jax.Array:
        """Propagate uncertainty while respecting physics constraints.

        Args:
            input_uncertainty: Input uncertainty estimates
            model_jacobian: Jacobian of the model wrt inputs
            physics_state: Current physics state for constraint evaluation

        Returns:
            Physics-constrained uncertainty propagation
        """
        # Handle broadcasting for uncertainty propagation
        # input_uncertainty: (batch,) or (batch, input_dim)
        # model_jacobian: (batch, output_dim, input_dim)

        # Ensure input_uncertainty has compatible shape for broadcasting
        if input_uncertainty.ndim == 1 and model_jacobian.ndim == 3:
            # Broadcast input_uncertainty to match jacobian dimensions
            # (batch,) -> (batch, 1, input_dim) to match (batch, output_dim, input_dim)
            input_uncertainty_expanded = input_uncertainty[:, None, None]
        elif input_uncertainty.ndim == 2 and model_jacobian.ndim == 3:
            # (batch, input_dim) -> (batch, 1, input_dim)
            input_uncertainty_expanded = input_uncertainty[:, None, :]
        else:
            input_uncertainty_expanded = input_uncertainty

        # Standard uncertainty propagation via Jacobian
        # Sum over input dimensions for each output
        standard_propagation = jnp.sum(
            (model_jacobian**2) * (input_uncertainty_expanded**2), axis=-1
        )

        # Sum over output dimensions to get total uncertainty per batch
        if standard_propagation.ndim > 1:
            standard_propagation = jnp.sum(standard_propagation, axis=-1)

        # Add physics constraint contributions
        constraint_contributions = jnp.zeros_like(standard_propagation)

        for i, law in enumerate(self.conservation_laws):
            constraint_violation = self._evaluate_physics_constraint(physics_state, law)

            weight = float(self.constraint_weights.value[i])
            contribution = weight * jnp.abs(constraint_violation)
            constraint_contributions = constraint_contributions + contribution

        # Combine standard and physics-aware contributions
        total_constraint_contribution = constraint_contributions

        physics_aware_uncertainty = jnp.sqrt(
            standard_propagation + total_constraint_contribution**2
        )

        # Apply uncertainty inflation for safety
        physics_aware_uncertainty *= self.uncertainty_inflation

        return physics_aware_uncertainty

    def _evaluate_physics_constraint(
        self, physics_state: jax.Array, law: str
    ) -> jax.Array:
        """Evaluate physics constraint violation.

        Args:
            physics_state: Physics state variables
            law: Conservation law to evaluate

        Returns:
            Constraint violation measure
        """
        if law == "energy":
            # Simple energy constraint: total energy should be preserved
            # Assume first component is energy
            energy = physics_state[..., 0] if physics_state.shape[-1] > 0 else 0.0
            target_energy = 1.0  # Example target
            return jnp.abs(energy - target_energy)

        if law == "momentum":
            # Momentum conservation: total momentum should be zero
            if physics_state.shape[-1] >= 3:
                momentum = physics_state[..., 1:4]  # Components 1-3
                return jnp.linalg.norm(momentum, axis=-1)
            return jnp.zeros_like(physics_state[..., 0])

        # Unknown constraint
        return jnp.zeros_like(physics_state[..., 0])

    def compute_physics_informed_confidence(
        self,
        predictions: jax.Array,
        uncertainties: jax.Array,
        physics_state: jax.Array,
    ) -> jax.Array:
        """Compute physics-informed confidence intervals.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates
            physics_state: Physics state for constraint evaluation

        Returns:
            Physics-informed confidence measures
        """
        # Base confidence from uncertainties
        base_confidence = 1.0 / (1.0 + uncertainties)

        # Adjust confidence based on physics constraint satisfaction
        physics_penalty = 0.0

        for law in self.conservation_laws:
            constraint_violation = self._evaluate_physics_constraint(physics_state, law)

            # Penalize confidence for constraint violations
            violation_penalty = constraint_violation / (1.0 + constraint_violation)
            physics_penalty += violation_penalty

        # Reduce confidence for physics violations
        return base_confidence * jnp.exp(-physics_penalty)

    def uncertainty_aware_constraint_projection(
        self,
        parameters: jax.Array,
        uncertainties: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Project parameters to satisfy constraints while accounting for uncertainty.

        Args:
            parameters: Parameter values to project
            uncertainties: Parameter uncertainties

        Returns:
            Tuple of (projected_parameters, adjusted_uncertainties)
        """
        projected_params = parameters
        adjusted_uncertainties = uncertainties

        for law in self.conservation_laws:
            if law == "energy":
                # Energy conservation projection
                current_energy = jnp.sum(projected_params**2, axis=-1, keepdims=True)
                target_energy = projected_params.shape[-1]

                # Scale parameters to conserve energy
                scale_factor = jnp.sqrt(target_energy / (current_energy + 1e-8))
                projected_params = projected_params * scale_factor

                # Adjust uncertainties accordingly
                adjusted_uncertainties = adjusted_uncertainties * jnp.abs(scale_factor)

            elif law == "momentum":
                # Momentum conservation projection
                net_momentum = jnp.mean(projected_params, axis=-1, keepdims=True)
                projected_params = projected_params - net_momentum

                # Uncertainties remain the same for momentum projection
                # (it's a linear transformation)

        return projected_params, adjusted_uncertainties
