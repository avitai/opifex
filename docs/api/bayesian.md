# Bayesian & Uncertainty Quantification API Reference

The `opifex.neural.bayesian` package provides comprehensive Bayesian neural networks and uncertainty quantification capabilities for scientific machine learning applications.

## Overview

The Bayesian package implements advanced probabilistic methods:

- **Advanced Uncertainty Quantification**: Multi-source uncertainty aggregation with adaptive weighting
- **Enhanced Epistemic Uncertainty**: Ensemble disagreement methods and predictive diversity computation
- **Advanced Aleatoric Uncertainty**: Distributional uncertainty for multiple distribution types
- **Uncertainty Quality Assessment**: Coverage probability, calibration metrics, and reliability estimation
- **Bayesian Inference**: MCMC sampling with BlackJAX integration
- **Calibration Tools**: Temperature scaling, isotonic regression, conformal prediction
- **Conformal Prediction**: Split conformal method with `ConformalPredictor` for calibrated prediction intervals without distributional assumptions

## Advanced Uncertainty Quantification

### AdvancedUncertaintyAggregator

```python
class AdvancedUncertaintyAggregator:
    """Advanced uncertainty aggregation with multiple sources and weighting strategies."""
```

#### Methods

##### `weighted_uncertainty_aggregation(uncertainty_sources, weights=None, aggregation_method="weighted_variance") -> Array`

Aggregate uncertainties from multiple sources with optional weighting.

```python
@staticmethod
def weighted_uncertainty_aggregation(
    uncertainty_sources: list[Float[Array, "batch output"]],
    weights: Float[Array, "sources"] | None = None,
    aggregation_method: str = "weighted_variance",
) -> Float[Array, "batch output"]:
    """
    Aggregate uncertainties from multiple sources with optional weighting.

    Args:
        uncertainty_sources: List of uncertainty estimates from different sources
        weights: Optional weights for each source (normalized automatically)
        aggregation_method: Method for aggregation
            - "weighted_variance": Weighted sum of variances
            - "weighted_mean": Simple weighted average
            - "max_weighted": Maximum weighted uncertainty
            - "robust_weighted": Robust aggregation using median

    Returns:
        Aggregated uncertainty estimates

    Example:
        >>> aggregator = AdvancedUncertaintyAggregator()
        >>> sources = [ensemble_uncertainty, gaussian_uncertainty]
        >>> aggregated = aggregator.weighted_uncertainty_aggregation(
        ...     sources, aggregation_method="weighted_variance"
        ... )
    """
```

##### `adaptive_weighting(uncertainty_sources, reliability_scores=None, adaptation_method="reliability_based") -> Array`

Compute adaptive weights for uncertainty sources based on reliability.

```python
@staticmethod
def adaptive_weighting(
    uncertainty_sources: list[Float[Array, "batch output"]],
    reliability_scores: list[Float[Array, "batch"]] | None = None,
    adaptation_method: str = "reliability_based",
) -> Float[Array, "sources batch"]:
    """
    Compute adaptive weights for uncertainty sources based on reliability.

    Args:
        uncertainty_sources: List of uncertainty estimates
        reliability_scores: Optional reliability scores for each source
        adaptation_method: Method for computing adaptive weights
            - "reliability_based": Weight by reliability scores
            - "inverse_variance": Weight inversely proportional to variance
            - "entropy_based": Weight based on predictive entropy
            - "uniform": Uniform weighting

    Returns:
        Adaptive weights for each source and batch element

    Example:
        >>> reliability_scores = [
        ...     jnp.ones((100,)) * 0.9,  # High reliability
        ...     jnp.ones((100,)) * 0.7   # Medium reliability
        ... ]
        >>> weights = aggregator.adaptive_weighting(
        ...     sources, reliability_scores, "reliability_based"
        ... )
    """
```

##### `uncertainty_quality_assessment(predictions, uncertainties, true_values=None) -> dict`

Assess the quality of uncertainty estimates.

```python
@staticmethod
def uncertainty_quality_assessment(
    predictions: Float[Array, "batch output"],
    uncertainties: Float[Array, "batch output"],
    true_values: Float[Array, "batch output"] | None = None,
) -> dict[str, float]:
    """
    Assess the quality of uncertainty estimates.

    Args:
        predictions: Model predictions
        uncertainties: Uncertainty estimates
        true_values: Optional ground truth values

    Returns:
        Dictionary containing quality metrics:
        - coverage_probability: Fraction of true values within prediction intervals
        - mean_interval_width: Average width of prediction intervals
        - calibration_error: Normalized prediction errors
        - mean_uncertainty: Average uncertainty magnitude
        - uncertainty_std: Standard deviation of uncertainties
        - uncertainty_range: Range of uncertainty values
        - mean_confidence: Average prediction confidence

    Example:
        >>> quality = aggregator.uncertainty_quality_assessment(
        ...     predictions, uncertainties, true_values
        ... )
        >>> print(f"Coverage: {quality['coverage_probability']:.3f}")
    """
```

### AdvancedEpistemicUncertainty

```python
class AdvancedEpistemicUncertainty:
    """Advanced epistemic uncertainty estimation methods."""
```

#### Methods

##### `compute_ensemble_disagreement(ensemble_predictions, aggregation_method="variance") -> Array`

Compute epistemic uncertainty from ensemble disagreement.

```python
@staticmethod
def compute_ensemble_disagreement(
    ensemble_predictions: Float[Array, "models batch output"],
    aggregation_method: str = "variance",
) -> Float[Array, "batch output"]:
    """
    Compute epistemic uncertainty from ensemble disagreement.

    Args:
        ensemble_predictions: Predictions from multiple models
        aggregation_method: Method for computing disagreement
            - "variance": Variance across ensemble
            - "std": Standard deviation across ensemble
            - "range": Range (max - min) across ensemble
            - "iqr": Interquartile range across ensemble

    Returns:
        Epistemic uncertainty estimates

    Example:
        >>> epistemic = AdvancedEpistemicUncertainty()
        >>> ensemble_preds = jax.random.normal(key, (5, 100, 1))
        >>> uncertainty = epistemic.compute_ensemble_disagreement(
        ...     ensemble_preds, "variance"
        ... )
    """
```

##### `compute_predictive_diversity(ensemble_predictions, diversity_metric="pairwise_distance") -> Array`

Compute predictive diversity as a measure of epistemic uncertainty.

```python
@staticmethod
def compute_predictive_diversity(
    ensemble_predictions: Float[Array, "models batch output"],
    diversity_metric: str = "pairwise_distance",
) -> Float[Array, "batch output"]:
    """
    Compute predictive diversity as a measure of epistemic uncertainty.

    Args:
        ensemble_predictions: Predictions from multiple models
        diversity_metric: Metric for computing diversity
            - "pairwise_distance": Average pairwise L2 distance
            - "cosine_diversity": Average cosine diversity

    Returns:
        Predictive diversity estimates

    Example:
        >>> diversity = epistemic.compute_predictive_diversity(
        ...     ensemble_preds, "pairwise_distance"
        ... )
    """
```

### AdvancedAleatoricUncertainty

```python
class AdvancedAleatoricUncertainty:
    """Advanced aleatoric uncertainty estimation methods."""
```

#### Methods

##### `distributional_uncertainty(distribution_params, distribution_type="gaussian") -> Array`

Compute aleatoric uncertainty from distributional outputs.

```python
@staticmethod
def distributional_uncertainty(
    distribution_params: dict[str, Float[Array, "batch ..."]],
    distribution_type: str = "gaussian",
) -> Float[Array, "batch output"]:
    """
    Compute aleatoric uncertainty from distributional outputs.

    Args:
        distribution_params: Parameters of the output distribution
        distribution_type: Type of distribution
            - "gaussian": Requires 'log_std', 'std', or 'variance'
            - "laplace": Requires 'scale' parameter
            - "mixture": Requires 'weights', 'means', 'variances'

    Returns:
        Aleatoric uncertainty estimates

    Example:
        >>> aleatoric = AdvancedAleatoricUncertainty()
        >>> gaussian_params = {"log_std": log_std_predictions}
        >>> uncertainty = aleatoric.distributional_uncertainty(
        ...     gaussian_params, "gaussian"
        ... )

        >>> mixture_params = {
        ...     "weights": mixture_weights,
        ...     "means": mixture_means,
        ...     "variances": mixture_variances
        ... }
        >>> mixture_uncertainty = aleatoric.distributional_uncertainty(
        ...     mixture_params, "mixture"
        ... )
    """
```

## Enhanced Calibration Framework

The Enhanced Calibration Framework provides physics-aware temperature scaling and constraint-aware calibration methods for improved uncertainty calibration in scientific machine learning applications.

### TemperatureScaling

The enhanced `TemperatureScaling` class now supports physics-aware calibration with constraint enforcement:

```python
class TemperatureScaling:
    """Enhanced temperature scaling with physics-aware constraint capabilities."""

    def __init__(
        self,
        physics_constraints: Sequence[str] = (),
        adaptive: bool = False,
        learning_rate: float = 0.01,
        constraint_strength: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize enhanced temperature scaling with physics constraints.

        Args:
            physics_constraints: List of physics constraints to enforce
            adaptive: Whether to use adaptive temperature learning
            learning_rate: Learning rate for temperature optimization
            constraint_strength: Strength of physics constraint enforcement (default: 1.0)
            rngs: Random number generators for parameter initialization
        """
```

#### Enhanced Methods

##### `apply_physics_aware_calibration(predictions, inputs) -> tuple[Array, float]`

Apply temperature scaling with physics-aware constraint enforcement.

```python
def apply_physics_aware_calibration(
    self, predictions: jax.Array, inputs: jax.Array
) -> tuple[jax.Array, float]:
    """
    Apply temperature scaling with physics-aware constraint enforcement.

    Args:
        predictions: Model predictions to calibrate
        inputs: Input data for constraint evaluation

    Returns:
        Tuple of (calibrated_predictions, constraint_penalty)

    Example:
        >>> import jax
        >>> import flax.nnx as nnx
        >>> from opifex.neural.bayesian import TemperatureScaling
        >>>
        >>> key = jax.random.PRNGKey(42)
        >>> rngs = nnx.Rngs(key)
        >>> temp_scaler = TemperatureScaling(
        ...     physics_constraints=['energy_conservation', 'positivity'],
        ...     constraint_strength=0.2,
        ...     rngs=rngs
        ... )
        >>>
        >>> predictions = jax.random.normal(key, (100, 1))
        >>> inputs = jax.random.normal(key, (100, 5))
        >>> calibrated_preds, penalty = temp_scaler.apply_physics_aware_calibration(
        ...     predictions, inputs
        ... )
        >>> print(f"Constraint penalty: {penalty:.6f}")
    """
```

##### `optimize_temperature_with_physics_constraints(predictions, targets, inputs) -> float`

Optimize temperature parameter with physics constraint awareness.

```python
def optimize_temperature_with_physics_constraints(
    self, predictions: jax.Array, targets: jax.Array, inputs: jax.Array
) -> float:
    """
    Optimize temperature parameter with physics constraint awareness.

    Args:
        predictions: Model predictions
        targets: Target values
        inputs: Input data for constraint evaluation

    Returns:
        Optimized temperature value

    Example:
        >>> temp_scaler = TemperatureScaling(constraint_strength=0.15)
        >>> optimal_temp = temp_scaler.optimize_temperature_with_physics_constraints(
        ...     predictions, targets, inputs
        ... )
        >>> print(f"Optimal temperature: {optimal_temp:.4f}")
    """
```

#### Physics Constraints

The framework supports multiple physics constraint types:

**Energy Conservation**

```python
constraint = {'type': 'energy_conservation', 'params': {}}
# Enforces non-negative energy values (E ≥ 0)
```

**Mass Conservation**

```python
constraint = {'type': 'mass_conservation', 'params': {}}
# Enforces conservation of total mass (∑m = constant)
```

**Positivity**

```python
constraint = {'type': 'positivity', 'params': {}}
# Enforces positive values (x > 0)
```

**Boundedness**

```python
constraint = {'type': 'boundedness', 'params': {}}
# Enforces bounded range (-10 ≤ x ≤ 10)
```

### Comprehensive Usage Example

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.neural.bayesian import TemperatureScaling

# Generate sample data
key = jax.random.PRNGKey(42)
batch_size = 100
n_features = 5

predictions = jax.random.normal(key, (batch_size, 1))
targets = jax.random.normal(key, (batch_size, 1))
inputs = jax.random.normal(key, (batch_size, n_features))

# Initialize enhanced temperature scaling with physics constraints
rngs = nnx.Rngs(key)
temp_scaler = TemperatureScaling(
    physics_constraints=['energy_conservation', 'positivity', 'boundedness'],
    constraint_strength=0.2,  # 20% constraint penalty weight
    adaptive=True,  # Enable adaptive temperature scaling
    rngs=rngs
)

# Apply physics-aware calibration
calibrated_predictions, constraint_penalty = temp_scaler.apply_physics_aware_calibration(
    predictions, inputs
)

print(f"Original predictions range: [{jnp.min(predictions):.3f}, {jnp.max(predictions):.3f}]")
print(f"Calibrated predictions range: [{jnp.min(calibrated_predictions):.3f}, {jnp.max(calibrated_predictions):.3f}]")
print(f"Constraint penalty: {constraint_penalty:.6f}")
print(f"Penalty history length: {len(temp_scaler.constraint_penalty_history)}")

# Optimize temperature with physics constraints
optimal_temperature = temp_scaler.optimize_temperature_with_physics_constraints(
    predictions, targets, inputs
)
print(f"Optimal temperature: {optimal_temperature:.4f}")

# Access the current temperature parameter
print(f"Current temperature: {temp_scaler.temperature.value:.4f}")

# Use the calibrated model for inference with uncertainty
calibrated_preds, aleatoric_uncertainty = temp_scaler(predictions, inputs)
print(f"Aleatoric uncertainty mean: {jnp.mean(aleatoric_uncertainty):.6f}")
```

## Enhanced Uncertainty Quantifier

### EnhancedUncertaintyQuantifier

```python
class EnhancedUncertaintyQuantifier:
    """Enhanced uncertainty quantifier with multiple decomposition methods."""

    def __init__(
        self,
        ensemble_size: int = 5,
        distributional_output: bool = True,
        multi_source_aggregation: bool = True,
        confidence_level: float = 0.95,
    ):
        """
        Initialize enhanced uncertainty quantifier.

        Args:
            ensemble_size: Number of models in ensemble
            distributional_output: Whether to use distributional outputs
            multi_source_aggregation: Whether to aggregate multiple uncertainty sources
            confidence_level: Confidence level for intervals
        """
```

#### Methods

##### `enhanced_decompose_uncertainty(...) -> EnhancedUncertaintyComponents`

Enhanced uncertainty decomposition with multiple sources.

```python
def enhanced_decompose_uncertainty(
    self,
    ensemble_predictions: Float[Array, "models batch output"],
    distributional_std: Float[Array, "batch output"] | None = None,
    inputs: Float[Array, "batch input_dim"] | None = None,
    dropout_predictions: Float[Array, "samples batch output"] | None = None,
) -> EnhancedUncertaintyComponents:
    """
    Enhanced uncertainty decomposition with multiple sources.

    Args:
        ensemble_predictions: Predictions from ensemble models
        distributional_std: Standard deviation from distributional output
        inputs: Input data for context-dependent uncertainty
        dropout_predictions: Predictions with dropout for additional epistemic uncertainty

    Returns:
        Enhanced uncertainty components with detailed breakdown

    Example:
        >>> quantifier = EnhancedUncertaintyQuantifier()
        >>> components = quantifier.enhanced_decompose_uncertainty(
        ...     ensemble_predictions=ensemble_preds,
        ...     distributional_std=distributional_std,
        ...     inputs=input_features
        ... )
        >>> print(f"Total uncertainty: {components.total_uncertainty}")
        >>> print(f"Sources: {list(components.uncertainty_breakdown.keys())}")
    """
```

## Data Structures

### EnhancedUncertaintyComponents

```python
@dataclasses.dataclass
class EnhancedUncertaintyComponents:
    """Enhanced uncertainty components with multiple sources."""

    epistemic_ensemble: Float[Array, "batch output"]  # Ensemble-based epistemic uncertainty
    epistemic_dropout: Float[Array, "batch output"] | None  # Dropout-based epistemic uncertainty
    aleatoric_distributional: Float[Array, "batch output"]  # Distributional aleatoric uncertainty
    total_uncertainty: Float[Array, "batch output"]  # Combined uncertainty
    uncertainty_breakdown: dict[str, Float[Array, "batch output"]]  # Detailed breakdown
```

## Usage Examples

### Basic Uncertainty Analysis

```python
import jax
import jax.numpy as jnp
from opifex.neural.bayesian import (
    AdvancedUncertaintyAggregator,
    AdvancedEpistemicUncertainty,
    AdvancedAleatoricUncertainty
)

# Generate ensemble predictions
key = jax.random.PRNGKey(42)
ensemble_predictions = jax.random.normal(key, (5, 100, 1))

# Epistemic uncertainty analysis
epistemic_analyzer = AdvancedEpistemicUncertainty()
epistemic_uncertainty = epistemic_analyzer.compute_ensemble_disagreement(
    ensemble_predictions, aggregation_method="variance"
)

# Aleatoric uncertainty analysis
aleatoric_analyzer = AdvancedAleatoricUncertainty()
gaussian_params = {"log_std": jax.random.normal(key, (100, 1)) * 0.1}
aleatoric_uncertainty = aleatoric_analyzer.distributional_uncertainty(
    gaussian_params, distribution_type="gaussian"
)

# Multi-source aggregation
aggregator = AdvancedUncertaintyAggregator()
total_uncertainty = aggregator.weighted_uncertainty_aggregation(
    [epistemic_uncertainty, aleatoric_uncertainty],
    aggregation_method="weighted_variance"
)
```

### Model Comparison with Uncertainty

```python
def compare_models_with_uncertainty(models_predictions, true_values):
    """Compare multiple models based on uncertainty quality."""
    aggregator = AdvancedUncertaintyAggregator()
    results = {}

    for model_name, predictions in models_predictions.items():
        # Compute uncertainty
        uncertainty = jnp.std(predictions, axis=0)

        # Assess quality
        quality = aggregator.uncertainty_quality_assessment(
            predictions=jnp.mean(predictions, axis=0),
            uncertainties=uncertainty,
            true_values=true_values
        )

        results[model_name] = quality

    return results

# Example usage
models_predictions = {
    "model_a": ensemble_predictions_a,
    "model_b": ensemble_predictions_b
}
comparison = compare_models_with_uncertainty(models_predictions, true_values)
```

## Integration with Existing Components

The advanced uncertainty quantification components are designed to work seamlessly with existing Opifex components:

- **Neural Networks**: Compatible with all neural network architectures
- **Training Infrastructure**: Integrates with training loops and optimization
- **Physics-Informed Models**: Uncertainty quantification for PINNs and neural operators
- **Benchmarking**: Uncertainty metrics for model evaluation and comparison

## Physics-Informed Bayesian Components (NEW)

### PhysicsInformedPriors

```python
class PhysicsInformedPriors(nnx.Module):
    """Physics-informed prior constraints for Bayesian models."""

    def __init__(
        self,
        conservation_laws: Sequence[str] = (),
        boundary_conditions: Sequence[str] = (),
        constraint_weights: jax.Array | None = None,
        penalty_weight: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize physics-informed priors.

        Args:
            conservation_laws: List of conservation laws to enforce
            boundary_conditions: List of boundary conditions to enforce
            constraint_weights: Optional custom weights for constraints
            penalty_weight: Weight for constraint violation penalties
            rngs: Random number generators
        """
```

#### Methods

##### `apply_constraints(params: jax.Array) -> jax.Array`

Apply physics constraints to sampled parameters.

```python
def apply_constraints(self, params: jax.Array) -> jax.Array:
    """
    Apply physics constraints to sampled parameters.

    Args:
        params: Unconstrained parameter samples

    Returns:
        Constrained parameters that satisfy physics laws

    Supported conservation laws:
        - "energy": Energy conservation with normalization
        - "momentum": Momentum conservation (zero total momentum)
        - "mass": Mass conservation (positive masses)
        - "positivity": Positivity constraint
        - "boundedness": Bounded values using tanh

    Supported boundary conditions:
        - "dirichlet": Fixed boundary values
        - "neumann": Zero derivative at boundaries
        - "periodic": Periodic boundary conditions

    Example:
        >>> priors = PhysicsInformedPriors(
        ...     conservation_laws=['energy', 'momentum'],
        ...     boundary_conditions=['dirichlet'],
        ...     rngs=rngs
        ... )
        >>> constrained = priors.apply_constraints(unconstrained_params)
    """
```

##### `compute_violation_penalty(params: jax.Array) -> float`

Compute penalty for physics constraint violations.

```python
def compute_violation_penalty(self, params: jax.Array) -> float:
    """
    Compute penalty for physics constraint violations.

    Args:
        params: Parameter values to evaluate

    Returns:
        Violation penalty (higher = more violation)

    Example:
        >>> penalty = priors.compute_violation_penalty(params)
        >>> print(f"Constraint violation: {penalty:.6f}")
    """
```

### ConservationLawPriors

```python
class ConservationLawPriors(nnx.Module):
    """Advanced conservation law enforcement with adaptive weighting."""

    def __init__(
        self,
        conservation_laws: Sequence[str] = ("energy", "momentum", "mass"),
        uncertainty_scale: float = 0.1,
        prior_strength: float = 1.0,
        adaptive_weighting: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize conservation law priors.

        Args:
            conservation_laws: Conservation laws to enforce
            uncertainty_scale: Scale for uncertainty inflation
            prior_strength: Strength of prior constraints
            adaptive_weighting: Enable adaptive constraint weighting
            rngs: Random number generators
        """
```

#### Methods

##### `compute_physics_aware_uncertainty(...) -> jax.Array`

Compute uncertainty with physics constraint awareness.

```python
def compute_physics_aware_uncertainty(
    self,
    predictions: jax.Array,
    model_uncertainty: jax.Array,
    physics_state: jax.Array,
) -> jax.Array:
    """
    Compute physics-aware uncertainty estimates.

    Args:
        predictions: Model predictions
        model_uncertainty: Base model uncertainty
        physics_state: Physical state representation

    Returns:
        Enhanced uncertainty estimates incorporating physics constraints

    Example:
        >>> conservation_priors = ConservationLawPriors(rngs=rngs)
        >>> physics_uncertainty = conservation_priors.compute_physics_aware_uncertainty(
        ...     predictions, model_uncertainty, physics_state
        ... )
    """
```

##### `sample_physics_constrained_params(...) -> jax.Array`

Sample parameters subject to physics constraints.

```python
def sample_physics_constrained_params(
    self, base_params: jax.Array, constraint_strength: float = 1.0
) -> jax.Array:
    """
    Sample physics-constrained parameters.

    Args:
        base_params: Base parameter distribution
        constraint_strength: Strength of constraint enforcement

    Returns:
        Constrained parameter samples

    Example:
        >>> constrained_samples = conservation_priors.sample_physics_constrained_params(
        ...     base_params, constraint_strength=0.8
        ... )
    """
```

### DomainSpecificPriors

```python
class DomainSpecificPriors(nnx.Module):
    """Domain-specific priors for scientific applications."""

    def __init__(
        self,
        domain: str = "quantum_chemistry",
        parameter_ranges: dict[str, tuple[float, float]] | None = None,
        distribution_types: dict[str, str] | None = None,
        correlation_structure: str = "independent",
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize domain-specific priors.

        Args:
            domain: Scientific domain ("quantum_chemistry", "fluid_dynamics", "materials")
            parameter_ranges: Custom parameter ranges
            distribution_types: Distribution types for each parameter
            correlation_structure: Parameter correlation structure
            rngs: Random number generators

        Supported domains:
            - "quantum_chemistry": Molecular parameters
            - "fluid_dynamics": Flow parameters
            - "materials": Material properties
        """
```

#### Methods

##### `sample_domain_priors(sample_shape: tuple, parameter_type: str) -> jax.Array`

Sample from domain-specific parameter distributions.

```python
def sample_domain_priors(
    self, sample_shape: tuple[int, ...], parameter_type: str
) -> jax.Array:
    """
    Sample from domain-specific parameter distributions.

    Args:
        sample_shape: Shape of samples to generate
        parameter_type: Type of parameter to sample

    Returns:
        Domain-appropriate parameter samples

    Quantum chemistry parameters:
        - "bond_length": Typical chemical bond lengths
        - "angle": Bond angles in degrees
        - "energy": Molecular energies
        - "charge": Atomic charges

    Example:
        >>> quantum_priors = DomainSpecificPriors(domain="quantum_chemistry", rngs=rngs)
        >>> bond_samples = quantum_priors.sample_domain_priors((100,), "bond_length")
    """
```

### HierarchicalBayesianFramework

```python
class HierarchicalBayesianFramework(nnx.Module):
    """Hierarchical Bayesian modeling with multi-level uncertainty."""

    def __init__(
        self,
        hierarchy_levels: int = 3,
        level_dimensions: Sequence[int] = (64, 32, 16),
        uncertainty_propagation: str = "multiplicative",
        correlation_structure: str = "exchangeable",
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize hierarchical# Bayesian Networks

```python
from jaxtyping import Float, Array
import jax.numpy as jnp
import jax
```

        Args:
            hierarchy_levels: Number of hierarchy levels
            level_dimensions: Dimensions at each level
            uncertainty_propagation: How uncertainty propagates between levels
            correlation_structure: Correlation structure between levels
            rngs: Random number generators
        """
```

#### Methods

##### `sample_hierarchical_parameters(sample_shape: tuple, level: int) -> jax.Array`

Sample parameters from specified hierarchy level.

##### `propagate_uncertainty_hierarchically(base_uncertainty: jax.Array, target_level: int) -> jax.Array`

Propagate uncertainty through hierarchy levels.

### PhysicsAwareUncertaintyPropagation

```python
class PhysicsAwareUncertaintyPropagation(nnx.Module):
    """Physics-aware uncertainty propagation with constraint enforcement."""

    def __init__(
        self,
        conservation_laws: Sequence[str] = ("energy", "momentum"),
        constraint_tolerance: float = 1e-6,
        uncertainty_inflation: float = 1.1,
        correlation_aware: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        """
        Initialize physics-aware uncertainty propagation.

        Args:
            conservation_laws: Conservation laws to enforce
            constraint_tolerance: Tolerance for constraint violations
            uncertainty_inflation: Factor for uncertainty inflation
            correlation_aware: Whether to account for correlations
            rngs: Random number generators
        """
```

#### Methods

##### `propagate_with_physics_constraints(...) -> jax.Array`

Propagate uncertainty while enforcing physics constraints.

##### `compute_physics_informed_confidence(...) -> jax.Array`

Compute confidence measures that respect physics constraints.

## Physics-Informed Usage Examples

### Conservation Law Enforcement

```python
from opifex.neural.bayesian import PhysicsInformedPriors

# Initialize physics priors
physics_priors = PhysicsInformedPriors(
    conservation_laws=['energy', 'momentum', 'mass'],
    boundary_conditions=['dirichlet', 'neumann'],
    penalty_weight=1.0,
    rngs=rngs
)

# Apply constraints
unconstrained_params = jax.random.normal(key, (100,))
constrained_params = physics_priors.apply_constraints(unconstrained_params)
violation_penalty = physics_priors.compute_violation_penalty(constrained_params)

print(f"Constraint violation penalty: {violation_penalty:.6f}")
```

### Domain-Specific Modeling

```python
from opifex.neural.bayesian import DomainSpecificPriors

# Quantum chemistry modeling
quantum_priors = DomainSpecificPriors(
    domain="quantum_chemistry",
    rngs=rngs
)

# Sample molecular parameters (using default ranges)
bond_lengths = quantum_priors.sample_domain_priors((50,), "bond_length")
energies = quantum_priors.sample_domain_priors((50,), "energy")
```

### Hierarchical Uncertainty

```python
from opifex.neural.bayesian import HierarchicalBayesianFramework

# Multi-level uncertainty modeling
hierarchical_framework = HierarchicalBayesianFramework(
    hierarchy_levels=3,
    level_dimensions=[64, 32, 16],
    uncertainty_propagation="multiplicative",
    rngs=rngs
)

# Sample at different levels
global_params = hierarchical_framework.sample_hierarchical_parameters((10,), level=0)
local_params = hierarchical_framework.sample_hierarchical_parameters((10,), level=2)

# Propagate uncertainty
base_uncertainty = jnp.ones((10, 64)) * 0.1
propagated = hierarchical_framework.propagate_uncertainty_hierarchically(
    base_uncertainty, target_level=2
)
```

## Performance Considerations

- **JAX Compilation**: All methods are JIT-compilable for optimal performance
- **Memory Efficiency**: Streaming computation for large ensembles
- **Vectorization**: Batch processing for multiple uncertainty sources
- **Adaptive Computation**: Dynamic weighting reduces computational overhead
- **Physics Constraints**: Efficient constraint enforcement with minimal overhead
