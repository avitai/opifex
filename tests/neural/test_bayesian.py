"""Tests for Bayesian neural network components."""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.base import StandardMLP
from opifex.neural.bayesian import (
    AleatoricUncertainty,
    AmortizedVariationalFramework,
    BlackJAXIntegration,
    CalibrationTools,
    ConformalPrediction,
    ConservationLawPriors,
    DomainSpecificPriors,
    EpistemicUncertainty,
    HierarchicalBayesianFramework,
    IsotonicRegression,
    MeanFieldGaussian,
    PhysicsAwareUncertaintyPropagation,
    PhysicsInformedPriors,
    PlattScaling,
    PriorConfig,
    TemperatureScaling,
    UncertaintyComponents,
    UncertaintyEncoder,
    UncertaintyQuantifier,
    VariationalConfig,
)


class TestMeanFieldGaussian:
    """Test MeanFieldGaussian variational posterior."""

    def test_initialization(self):
        """Test MeanFieldGaussian initialization."""
        rngs = nnx.Rngs(42)
        num_params = 10

        posterior = MeanFieldGaussian(num_params=num_params, rngs=rngs)

        assert posterior.num_params == num_params
        assert posterior.mean.value.shape == (num_params,)
        assert posterior.log_std.value.shape == (num_params,)

        # Check initial values
        assert jnp.allclose(posterior.mean.value, 0.0)
        assert jnp.allclose(posterior.log_std.value, -2.0)

    def test_sampling(self):
        """Test sampling from posterior."""
        rngs = nnx.Rngs(42)
        num_params = 10
        num_samples = 5

        posterior = MeanFieldGaussian(num_params=num_params, rngs=rngs)
        samples = posterior.sample(num_samples, rngs=rngs)

        assert samples.shape == (num_samples, num_params)

    def test_log_prob(self):
        """Test log probability computation."""
        rngs = nnx.Rngs(42)
        num_params = 10
        num_samples = 5

        posterior = MeanFieldGaussian(num_params=num_params, rngs=rngs)
        samples = posterior.sample(num_samples, rngs=rngs)
        log_probs = posterior.log_prob(samples)

        assert log_probs.shape == (num_samples,)
        assert jnp.all(jnp.isfinite(log_probs))

    def test_kl_divergence(self):
        """Test KL divergence computation."""
        rngs = nnx.Rngs(42)
        num_params = 10

        posterior = MeanFieldGaussian(num_params=num_params, rngs=rngs)
        kl_div = posterior.kl_divergence()

        assert kl_div.shape == ()
        assert jnp.isfinite(kl_div)
        assert kl_div >= 0.0  # KL divergence is non-negative


class TestUncertaintyEncoder:
    """Test UncertaintyEncoder network."""

    def test_initialization(self):
        """Test UncertaintyEncoder initialization."""
        rngs = nnx.Rngs(42)
        input_dim = 5
        hidden_dims = [10, 8]
        output_dim = 20

        encoder = UncertaintyEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            rngs=rngs,
        )

        # Check that layers are properly initialized
        # nnx.Sequential stores layers in .layers tuple
        layers = encoder.layers.layers
        assert (
            len(layers) == 2 * len(hidden_dims) + 1
        )  # linear + activation per hidden layer + output
        assert isinstance(layers[0], nnx.Linear)
        assert isinstance(layers[-1], nnx.Linear)

    def test_forward_pass(self):
        """Test forward pass through encoder."""
        rngs = nnx.Rngs(42)
        input_dim = 5
        hidden_dims = [10, 8]
        output_dim = 20
        batch_size = 3

        encoder = UncertaintyEncoder(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            rngs=rngs,
        )

        x = jnp.ones((batch_size, input_dim))
        output = encoder(x)

        assert output.shape == (batch_size, output_dim)

    def test_different_architectures(self):
        """Test encoder with different architectures."""
        rngs = nnx.Rngs(42)
        input_dim = 5
        output_dim = 20
        batch_size = 3

        # Test single hidden layer
        encoder1 = UncertaintyEncoder(
            input_dim=input_dim, hidden_dims=[10], output_dim=output_dim, rngs=rngs
        )

        x = jnp.ones((batch_size, input_dim))
        output1 = encoder1(x)
        assert output1.shape == (batch_size, output_dim)

        # Test no hidden layers (direct mapping)
        encoder2 = UncertaintyEncoder(
            input_dim=input_dim, hidden_dims=[], output_dim=output_dim, rngs=rngs
        )

        output2 = encoder2(x)
        assert output2.shape == (batch_size, output_dim)


class TestVariationalConfig:
    """Test VariationalConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = VariationalConfig(input_dim=10)

        assert config.input_dim == 10
        assert config.hidden_dims == (64, 32)
        assert config.num_samples == 10
        assert config.kl_weight == 1.0
        assert config.temperature == 1.0

    def test_custom_values(self):
        """Test custom config values."""
        config = VariationalConfig(
            input_dim=5,
            hidden_dims=[20, 15],
            num_samples=20,
            kl_weight=0.5,
            temperature=2.0,
        )

        assert config.input_dim == 5
        assert config.hidden_dims == [20, 15]
        assert config.num_samples == 20
        assert config.kl_weight == 0.5
        assert config.temperature == 2.0


class TestPriorConfig:
    """Test PriorConfig dataclass."""

    def test_default_values(self):
        """Test default config values."""
        config = PriorConfig()

        assert config.conservation_laws == ()
        assert config.boundary_conditions == ()
        assert config.physics_constraints == ()
        assert config.prior_scale == 1.0

    def test_custom_values(self):
        """Test custom config values."""
        config = PriorConfig(
            conservation_laws=["energy", "momentum"],
            boundary_conditions=["dirichlet"],
            physics_constraints=["positivity"],
            prior_scale=0.5,
        )

        assert config.conservation_laws == ["energy", "momentum"]
        assert config.boundary_conditions == ["dirichlet"]
        assert config.physics_constraints == ["positivity"]
        assert config.prior_scale == 0.5


class TestAmortizedVariationalFramework:
    """Test AmortizedVariationalFramework."""

    def test_initialization(self):
        """Test framework initialization."""
        rngs = nnx.Rngs(42)

        # Create base model
        base_model = StandardMLP([4, 8, 1], rngs=rngs)

        # Create configs
        prior_config = PriorConfig()
        variational_config = VariationalConfig(input_dim=4)

        # Create framework
        framework = AmortizedVariationalFramework(
            base_model=base_model,
            prior_config=prior_config,
            variational_config=variational_config,
            rngs=rngs,
        )

        assert framework.base_model is base_model
        assert framework.config is variational_config
        assert framework.prior_config is prior_config
        assert framework.num_params > 0
        assert isinstance(framework.variational_posterior, MeanFieldGaussian)
        assert isinstance(framework.amortization_network, UncertaintyEncoder)

    def test_parameter_counting(self):
        """Test parameter counting functionality."""
        rngs = nnx.Rngs(42)

        # Create simple model
        base_model = StandardMLP([2, 4, 1], rngs=rngs)

        # Create framework
        framework = AmortizedVariationalFramework(
            base_model=base_model,
            prior_config=PriorConfig(),
            variational_config=VariationalConfig(input_dim=2),
            rngs=rngs,
        )

        # Calculate expected parameters
        # Layer 1: 2*4 + 4 = 12 (weights + bias)
        # Layer 2: 4*1 + 1 = 5 (weights + bias)
        # Total: 17
        expected_params = 17

        assert framework.num_params == expected_params

    def test_deterministic_forward(self):
        """Test deterministic forward pass."""
        rngs = nnx.Rngs(42)
        batch_size = 3

        # Create model and framework
        base_model = StandardMLP([2, 4, 1], rngs=rngs)
        framework = AmortizedVariationalFramework(
            base_model=base_model,
            prior_config=PriorConfig(),
            variational_config=VariationalConfig(input_dim=2),
            rngs=rngs,
        )

        # Test forward pass
        x = jnp.ones((batch_size, 2))
        mean_output, uncertainty_output = framework(x)

        assert mean_output.shape == (batch_size, 1)
        assert uncertainty_output.shape == (batch_size, 1)

    def test_split_posterior_params(self):
        """Test parameter splitting functionality."""
        rngs = nnx.Rngs(42)

        # Create framework
        base_model = StandardMLP([2, 4, 1], rngs=rngs)
        framework = AmortizedVariationalFramework(
            base_model=base_model,
            prior_config=PriorConfig(),
            variational_config=VariationalConfig(input_dim=2),
            rngs=rngs,
        )

        # Create test parameters
        batch_size = 3
        total_params = 2 * framework.num_params
        posterior_params = jnp.ones((batch_size, total_params))

        # Split parameters
        mean_params, log_std_params = framework._split_posterior_params(
            posterior_params
        )

        assert mean_params.shape == (batch_size, framework.num_params)
        assert log_std_params.shape == (batch_size, framework.num_params)
        assert jnp.allclose(mean_params, 1.0)
        assert jnp.allclose(log_std_params, 1.0)


class TestBayesianIntegration:
    """Test integration between Bayesian components."""

    def test_end_to_end_workflow(self):
        """Test complete Bayesian workflow."""
        rngs = nnx.Rngs(42)
        batch_size = 4
        input_dim = 3

        # Create base model
        base_model = StandardMLP([input_dim, 8, 1], rngs=rngs)

        # Create framework
        framework = AmortizedVariationalFramework(
            base_model=base_model,
            prior_config=PriorConfig(prior_scale=0.5),
            variational_config=VariationalConfig(
                input_dim=input_dim, num_samples=5, kl_weight=0.1
            ),
            rngs=rngs,
        )

        # Test data
        x = jax.random.normal(rngs.sample(), (batch_size, input_dim))
        y = jax.random.normal(rngs.sample(), (batch_size, 1))

        # Test ELBO computation
        elbo = framework.compute_elbo(x, y, rngs=rngs)

        assert elbo.shape == ()
        assert jnp.isfinite(elbo)

        # Test deterministic prediction
        mean_pred, uncertainty_pred = framework(x)
        assert mean_pred.shape == (batch_size, 1)
        assert uncertainty_pred.shape == (batch_size, 1)

        # Test components work together
        posterior = framework.variational_posterior
        kl_div = posterior.kl_divergence(prior_std=0.5)
        assert jnp.isfinite(kl_div)
        assert kl_div >= 0.0

    def test_multiple_architectures(self):
        """Test framework with different base model architectures."""
        rngs = nnx.Rngs(42)
        batch_size = 2
        input_dim = 4

        # Test with different architectures
        architectures = [
            [input_dim, 1],  # Single layer
            [input_dim, 5, 1],  # Two layers
            [input_dim, 10, 5, 1],  # Three layers
        ]

        for arch in architectures:
            base_model = StandardMLP(arch, rngs=rngs)
            framework = AmortizedVariationalFramework(
                base_model=base_model,
                prior_config=PriorConfig(),
                variational_config=VariationalConfig(input_dim=input_dim),
                rngs=rngs,
            )

            # Test forward pass
            x = jnp.ones((batch_size, input_dim))
            mean_output, uncertainty_output = framework(x)

            assert mean_output.shape == (batch_size, 1)
            assert uncertainty_output.shape == (batch_size, 1)
            assert framework.num_params > 0


class TestUncertaintyQuantification:
    """Test UncertaintyQuantifier class."""

    def test_epistemic_uncertainty_variance(self):
        """Test epistemic uncertainty computation using variance."""
        # Create sample predictions from multiple models
        samples = 10
        batch_size = 5
        output_dim = 3

        predictions = jax.random.normal(
            jax.random.PRNGKey(42), (samples, batch_size, output_dim)
        )

        # Add some variation to make uncertainty meaningful
        predictions = predictions * jnp.arange(samples)[:, None, None] * 0.1

        epistemic_uncertainty = EpistemicUncertainty.compute_variance(predictions)

        assert epistemic_uncertainty.shape == (batch_size, output_dim)
        assert jnp.all(epistemic_uncertainty >= 0)

    def test_uncertainty_decomposition(self):
        """Test uncertainty decomposition into epistemic and aleatoric."""
        quantifier = UncertaintyQuantifier(num_samples=10)

        # Create synthetic predictions with known uncertainty structure
        samples = 10
        batch_size = 5
        output_dim = 2

        predictions = jax.random.normal(
            jax.random.PRNGKey(42), (samples, batch_size, output_dim)
        )
        aleatoric_variance = jnp.ones((samples, batch_size, output_dim)) * 0.1

        uncertainty_components = quantifier.decompose_uncertainty(
            predictions, aleatoric_variance
        )

        assert isinstance(uncertainty_components, UncertaintyComponents)
        assert uncertainty_components.epistemic.shape == (batch_size, output_dim)
        assert uncertainty_components.aleatoric.shape == (batch_size, output_dim)
        assert uncertainty_components.total.shape == (batch_size, output_dim)

        # Total should be approximately epistemic + aleatoric
        expected_total = (
            uncertainty_components.epistemic + uncertainty_components.aleatoric
        )
        assert jnp.allclose(uncertainty_components.total, expected_total, rtol=1e-5)

    def test_confidence_intervals(self):
        """Test confidence interval computation."""
        quantifier = UncertaintyQuantifier(confidence_level=0.95)

        samples = 100
        batch_size = 5
        output_dim = 2

        predictions = jax.random.normal(
            jax.random.PRNGKey(42), (samples, batch_size, output_dim)
        )

        lower, upper = quantifier.compute_confidence_intervals(predictions)

        assert lower.shape == (batch_size, output_dim)
        assert upper.shape == (batch_size, output_dim)
        assert jnp.all(lower <= upper)

    def test_calibration_assessment(self):
        """Test calibration assessment with ground truth."""
        quantifier = UncertaintyQuantifier()

        batch_size = 100
        output_dim = 1

        # Create synthetic data with known properties
        predictions = jax.random.normal(
            jax.random.PRNGKey(42), (batch_size, output_dim)
        )
        uncertainties = jnp.ones((batch_size, output_dim)) * 0.5
        true_values = (
            predictions
            + jax.random.normal(jax.random.PRNGKey(43), (batch_size, output_dim)) * 0.3
        )

        metrics = quantifier._assess_uncertainty_calibration(
            predictions, uncertainties, true_values
        )

        assert hasattr(metrics, "expected_calibration_error")
        assert hasattr(metrics, "maximum_calibration_error")
        assert hasattr(metrics, "reliability_diagram")
        assert hasattr(metrics, "confidence_histogram")
        assert hasattr(metrics, "accuracy_histogram")


class TestAdvancedUncertaintyDecomposition:
    """Test advanced uncertainty decomposition methods including ensemble approaches."""

    def test_ensemble_epistemic_uncertainty_initialization(self):
        """Test initialization of ensemble-based epistemic uncertainty estimator."""
        from opifex.neural.bayesian.uncertainty_quantification import (
            EnsembleEpistemicUncertainty,
        )

        num_models = 5
        estimator = EnsembleEpistemicUncertainty(num_models=num_models)

        assert estimator.num_models == num_models
        assert estimator.models == []

    def test_ensemble_epistemic_uncertainty_add_model(self):
        """Test adding models to ensemble uncertainty estimator."""
        from opifex.neural.bayesian.uncertainty_quantification import (
            EnsembleEpistemicUncertainty,
        )

        estimator = EnsembleEpistemicUncertainty(num_models=3)

        # Create mock models
        rngs = nnx.Rngs(42)
        model1 = StandardMLP(layer_sizes=[5, 10, 3], rngs=rngs)
        model2 = StandardMLP(layer_sizes=[5, 8, 3], rngs=rngs)

        estimator.add_model(model1)
        estimator.add_model(model2)

        assert len(estimator.models) == 2

    def test_ensemble_prediction_aggregation(self):
        """Test ensemble prediction aggregation methods."""
        from opifex.neural.bayesian.uncertainty_quantification import (
            EnsembleEpistemicUncertainty,
        )

        estimator = EnsembleEpistemicUncertainty(num_models=3)

        # Create synthetic ensemble predictions
        batch_size, output_dim = 5, 2
        ensemble_predictions = jax.random.normal(
            jax.random.PRNGKey(42), (3, batch_size, output_dim)
        )

        # Test mean aggregation
        mean_pred = estimator.aggregate_predictions(ensemble_predictions, method="mean")
        expected_mean = jnp.mean(ensemble_predictions, axis=0)
        assert jnp.allclose(mean_pred, expected_mean)

        # Test median aggregation
        median_pred = estimator.aggregate_predictions(
            ensemble_predictions, method="median"
        )
        expected_median = jnp.median(ensemble_predictions, axis=0)
        assert jnp.allclose(median_pred, expected_median)

    def test_ensemble_uncertainty_computation(self):
        """Test ensemble-based uncertainty computation."""
        from opifex.neural.bayesian.uncertainty_quantification import (
            EnsembleEpistemicUncertainty,
        )

        estimator = EnsembleEpistemicUncertainty(num_models=5)

        # Create synthetic ensemble predictions with varying uncertainty
        batch_size, output_dim = 10, 3
        ensemble_predictions = jax.random.normal(
            jax.random.PRNGKey(42), (5, batch_size, output_dim)
        )

        # Add deliberate variation to create measurable uncertainty
        variation_scale = jnp.array([1.0, 2.0, 0.5, 1.5, 0.8])[:, None, None]
        ensemble_predictions = ensemble_predictions * variation_scale

        uncertainty = estimator.compute_epistemic_uncertainty(ensemble_predictions)

        assert uncertainty.shape == (batch_size, output_dim)
        assert jnp.all(uncertainty >= 0)

    def test_distributional_aleatoric_uncertainty(self):
        """Test distributional modeling of aleatoric uncertainty."""
        from opifex.neural.bayesian.uncertainty_quantification import (
            DistributionalAleatoricUncertainty,
        )

        estimator = DistributionalAleatoricUncertainty()

        batch_size, output_dim = 8, 2

        # Test Gaussian distributional output
        mean = jax.random.normal(jax.random.PRNGKey(42), (batch_size, output_dim))
        log_std = jax.random.normal(jax.random.PRNGKey(43), (batch_size, output_dim))

        samples = estimator.sample_gaussian(mean, log_std, num_samples=100)
        assert samples.shape == (100, batch_size, output_dim)

        uncertainty = estimator.compute_gaussian_uncertainty(mean, log_std)
        expected_uncertainty = jnp.exp(2 * log_std)  # variance = exp(2 * log_std)
        assert jnp.allclose(uncertainty, expected_uncertainty)

    def test_distributional_mixture_uncertainty(self):
        """Test mixture-based distributional uncertainty."""
        from opifex.neural.bayesian.uncertainty_quantification import (
            DistributionalAleatoricUncertainty,
        )

        estimator = DistributionalAleatoricUncertainty()

        batch_size, output_dim, num_components = 6, 2, 3

        # Mixture parameters
        mixture_weights = jax.random.uniform(
            jax.random.PRNGKey(42), (batch_size, num_components)
        )
        mixture_weights = mixture_weights / jnp.sum(
            mixture_weights, axis=-1, keepdims=True
        )

        means = jax.random.normal(
            jax.random.PRNGKey(43), (batch_size, num_components, output_dim)
        )
        log_stds = jax.random.normal(
            jax.random.PRNGKey(44), (batch_size, num_components, output_dim)
        )

        uncertainty = estimator.compute_mixture_uncertainty(
            mixture_weights, means, log_stds
        )
        assert uncertainty.shape == (batch_size, output_dim)
        assert jnp.all(uncertainty >= 0)

    def test_multi_source_uncertainty_aggregation(self):
        """Test multi-source uncertainty aggregation."""
        from opifex.neural.bayesian.uncertainty_quantification import (
            MultiSourceUncertaintyAggregator,
        )

        aggregator = MultiSourceUncertaintyAggregator()

        batch_size, output_dim = 7, 2

        # Create multiple uncertainty sources
        epistemic_ensemble = (
            jax.random.uniform(jax.random.PRNGKey(42), (batch_size, output_dim)) * 0.5
        )
        epistemic_dropout = (
            jax.random.uniform(jax.random.PRNGKey(43), (batch_size, output_dim)) * 0.3
        )
        aleatoric_distributional = (
            jax.random.uniform(jax.random.PRNGKey(44), (batch_size, output_dim)) * 0.2
        )

        # Test variance-based aggregation
        total_uncertainty = aggregator.aggregate_uncertainties(
            epistemic_sources=[epistemic_ensemble, epistemic_dropout],
            aleatoric_sources=[aleatoric_distributional],
            method="variance_sum",
        )

        assert total_uncertainty.shape == (batch_size, output_dim)
        assert jnp.all(total_uncertainty >= 0)

        # Test weighted aggregation
        weights = jnp.array([0.6, 0.4])  # weights for epistemic sources
        weighted_uncertainty = aggregator.aggregate_uncertainties(
            epistemic_sources=[epistemic_ensemble, epistemic_dropout],
            aleatoric_sources=[aleatoric_distributional],
            method="weighted_sum",
            epistemic_weights=weights,
        )

        assert weighted_uncertainty.shape == (batch_size, output_dim)
        assert jnp.all(weighted_uncertainty >= 0)

    def test_enhanced_uncertainty_decomposition_integration(self):
        """Test integration of advanced uncertainty decomposition methods."""
        from opifex.neural.bayesian.uncertainty_quantification import (
            EnhancedUncertaintyQuantifier,
        )

        quantifier = EnhancedUncertaintyQuantifier(
            ensemble_size=5, distributional_output=True, multi_source_aggregation=True
        )

        batch_size, input_dim, output_dim = 8, 5, 3

        # Mock input data
        inputs = jax.random.normal(jax.random.PRNGKey(42), (batch_size, input_dim))

        # Mock ensemble predictions
        ensemble_predictions = jax.random.normal(
            jax.random.PRNGKey(43), (5, batch_size, output_dim)
        )

        # Mock distributional parameters
        aleatoric_std = (
            jax.random.uniform(jax.random.PRNGKey(44), (batch_size, output_dim)) * 0.5
        )

        # Test enhanced decomposition
        result = quantifier.enhanced_decompose_uncertainty(
            ensemble_predictions=ensemble_predictions,
            distributional_std=aleatoric_std,
            inputs=inputs,
        )

        assert hasattr(result, "epistemic_ensemble")
        assert hasattr(result, "aleatoric_distributional")
        assert hasattr(result, "total_uncertainty")
        assert hasattr(result, "uncertainty_breakdown")

        assert result.epistemic_ensemble.shape == (batch_size, output_dim)
        assert result.aleatoric_distributional.shape == (batch_size, output_dim)
        assert result.total_uncertainty.shape == (batch_size, output_dim)


class TestBayesianDataStructures:
    """Test Bayesian data structures and configurations."""

    def test_uncertainty_components_dataclass(self):
        """Test UncertaintyComponents dataclass."""
        epistemic = jnp.array([0.1, 0.2])
        aleatoric = jnp.array([0.05, 0.15])
        total = epistemic + aleatoric

        components = UncertaintyComponents(
            epistemic=epistemic, aleatoric=aleatoric, total=total
        )

        assert jnp.array_equal(components.epistemic, epistemic)
        assert jnp.array_equal(components.aleatoric, aleatoric)
        assert jnp.array_equal(components.total, total)

    def test_aleatoric_uncertainty_methods(self):
        """Test aleatoric uncertainty computation methods."""
        # Test homoscedastic uncertainty
        predictions = jnp.ones((4, 2))
        log_variance = jnp.log(jnp.ones((4, 2)) * 0.1)

        homo_uncertainty = AleatoricUncertainty.homoscedastic_uncertainty(
            predictions, log_variance
        )
        expected = jnp.exp(log_variance)
        assert jnp.allclose(homo_uncertainty, expected)

        # Test heteroscedastic uncertainty
        input_var = jnp.ones((4, 2)) * 0.2
        hetero_uncertainty = AleatoricUncertainty.heteroscedastic_uncertainty(input_var)
        assert jnp.array_equal(hetero_uncertainty, input_var)


class TestBlackJAXIntegration:
    """Test BlackJAX MCMC sampling integration."""

    def test_initialization(self):
        """Test BlackJAX integration initialization."""
        rngs = nnx.Rngs(42)

        # Create base model for MCMC sampling
        base_model = StandardMLP([4, 8, 1], rngs=rngs)

        # Create BlackJAX integration
        blackjax_sampler = BlackJAXIntegration(
            base_model=base_model,
            sampler_type="nuts",
            num_warmup=100,
            num_samples=500,
            rngs=rngs,
        )

        assert blackjax_sampler.base_model is base_model
        assert blackjax_sampler.sampler_type == "nuts"
        assert blackjax_sampler.num_warmup == 100
        assert blackjax_sampler.num_samples == 500

    def test_mcmc_sampling(self):
        """Test MCMC posterior sampling with BlackJAX."""
        rngs = nnx.Rngs(42)

        # Create simple model
        base_model = StandardMLP([2, 4, 1], rngs=rngs)

        # Create synthetic data with 2 features to match model input dimension
        x1 = jnp.linspace(-1, 1, 20)
        x2 = jnp.linspace(0.5, 1.5, 20)
        x_data = jnp.column_stack([x1, x2])  # Shape: (20, 2)
        y_data = (x1**2 + x2).reshape(-1, 1) + 0.1 * jax.random.normal(
            rngs.sample(), (20, 1)
        )

        # Create BlackJAX sampler
        sampler = BlackJAXIntegration(
            base_model=base_model,
            sampler_type="nuts",
            num_warmup=50,
            num_samples=100,
            rngs=rngs,
        )

        # Sample posterior
        samples = sampler.sample_posterior(x_data, y_data)

        assert samples.shape[0] == 100  # num_samples
        assert samples.shape[1] > 0  # parameter dimension
        assert jnp.all(jnp.isfinite(samples))

    def test_posterior_predictive(self):
        """Test posterior predictive sampling."""
        rngs = nnx.Rngs(42)

        # Create model and data
        base_model = StandardMLP([1, 4, 1], rngs=rngs)
        x_test = jnp.linspace(-1, 1, 10).reshape(-1, 1)

        # Create sampler
        sampler = BlackJAXIntegration(
            base_model=base_model,
            sampler_type="hmc",
            num_warmup=20,
            num_samples=50,
            rngs=rngs,
        )

        # Generate synthetic posterior samples
        num_params = sampler._count_parameters(base_model)
        posterior_samples = jax.random.normal(rngs.sample(), (50, num_params))

        # Posterior predictive sampling
        predictions = sampler.posterior_predictive(x_test, posterior_samples)

        assert predictions.shape == (50, 10, 1)  # (num_samples, num_test, output_dim)
        assert jnp.all(jnp.isfinite(predictions))

    def test_multiple_samplers(self):
        """Test different MCMC sampler types."""
        rngs = nnx.Rngs(42)
        base_model = StandardMLP([2, 1], rngs=rngs)

        # Test different sampler types
        sampler_types = ["nuts", "hmc", "mala"]

        for sampler_type in sampler_types:
            sampler = BlackJAXIntegration(
                base_model=base_model,
                sampler_type=sampler_type,
                num_warmup=10,
                num_samples=20,
                rngs=rngs,
            )

            assert sampler.sampler_type == sampler_type
            # Test that sampler can be initialized without errors


class TestCalibrationTools:
    """Test uncertainty calibration tools."""

    def test_temperature_scaling_initialization(self):
        """Test temperature scaling initialization."""
        rngs = nnx.Rngs(42)

        calibrator = TemperatureScaling(
            physics_constraints=["energy_conservation"],
            rngs=rngs,
        )

        assert calibrator.physics_constraints == ["energy_conservation"]
        assert hasattr(calibrator, "temperature")
        assert calibrator.temperature.value > 0.0

    def test_calibration_assessment(self):
        """Test calibration quality assessment."""
        rngs = nnx.Rngs(42)

        # Create synthetic predictions and uncertainties
        num_samples = 100
        predictions = jax.random.normal(rngs.sample(), (num_samples,))
        uncertainties = jnp.abs(jax.random.normal(rngs.sample(), (num_samples,))) + 0.1
        true_values = predictions + 0.5 * jax.random.normal(
            rngs.sample(), (num_samples,)
        )

        # Create calibration tools
        calibration_tools = CalibrationTools(rngs=rngs)

        # Assess calibration
        calibration_metrics = calibration_tools.assess_calibration(
            predictions=predictions,
            uncertainties=uncertainties,
            true_values=true_values,
        )

        assert "expected_calibration_error" in calibration_metrics
        assert "reliability_diagram_data" in calibration_metrics
        assert "brier_score" in calibration_metrics
        ece_value = calibration_metrics["expected_calibration_error"]
        assert isinstance(ece_value, (int, float)) and ece_value >= 0.0

    def test_reliability_diagram_computation(self):
        """Test reliability diagram data computation."""
        rngs = nnx.Rngs(42)

        # Create synthetic calibrated data
        num_samples = 200
        confidences = jax.random.uniform(rngs.sample(), (num_samples,))
        # Make accuracies correlated with confidences for better calibration
        accuracies = (
            confidences + 0.1 * jax.random.normal(rngs.sample(), (num_samples,))
        ) > 0.5

        calibration_tools = CalibrationTools(rngs=rngs)

        # Compute reliability diagram
        reliability_data = calibration_tools.compute_reliability_diagram(
            confidences=confidences,
            accuracies=accuracies,
            num_bins=10,
        )

        assert "bin_confidences" in reliability_data
        assert "bin_accuracies" in reliability_data
        assert "bin_counts" in reliability_data
        assert "bin_boundaries" in reliability_data
        assert "bin_centers" in reliability_data
        assert len(reliability_data["bin_centers"]) == 10

    def test_adaptive_temperature_scaling(self):
        """Test adaptive temperature scaling."""
        rngs = nnx.Rngs(42)

        # Create temperature scaling with adaptive learning
        calibrator = TemperatureScaling(
            physics_constraints=["energy_conservation"],
            adaptive=True,
            learning_rate=0.01,
            rngs=rngs,
        )

        # Create synthetic validation data
        logits = jax.random.normal(rngs.sample(), (50, 3))
        labels = jax.random.randint(rngs.sample(), (50,), 0, 3)

        # Optimize temperature
        optimized_temp = calibrator.optimize_temperature(logits, labels)

        assert optimized_temp > 0.0
        assert jnp.isfinite(optimized_temp)

    def test_physics_aware_temperature_scaling(self):
        """Test physics-aware temperature scaling with constraint enforcement."""
        rngs = nnx.Rngs(42)

        # Create temperature scaling with physics constraints
        calibrator = TemperatureScaling(
            physics_constraints=["energy_conservation", "mass_conservation"],
            adaptive=True,
            constraint_strength=1.0,
            rngs=rngs,
        )

        # Create synthetic physics-informed data
        batch_size = 32
        input_dim = 3
        inputs = jax.random.uniform(rngs.sample(), (batch_size, input_dim))

        # Predictions that violate physics constraints (negative energy)
        predictions = jax.random.normal(rngs.sample(), (batch_size, 1)) - 2.0

        # Apply physics-aware calibration
        calibrated_predictions, physics_penalty = (
            calibrator.apply_physics_aware_calibration(predictions, inputs)
        )

        # Physics-aware calibration should:
        # 1. Return finite, physically plausible predictions
        assert jnp.all(jnp.isfinite(calibrated_predictions))

        # 2. Compute physics constraint penalty for violations
        assert physics_penalty >= 0.0
        assert jnp.isfinite(physics_penalty)

        # 3. Modify predictions to respect physics constraints
        # (energy should be non-negative for energy_conservation constraint)
        energy_violations = jnp.sum(calibrated_predictions < 0)
        assert energy_violations <= jnp.sum(predictions < 0)  # Should reduce violations

    def test_constraint_aware_temperature_optimization(self):
        """Test temperature optimization with physics constraint awareness."""
        rngs = nnx.Rngs(42)

        calibrator = TemperatureScaling(
            physics_constraints=["positivity", "boundedness"],
            adaptive=True,
            constraint_strength=0.5,
            rngs=rngs,
        )

        # Create physics problem data
        batch_size = 50
        physics_inputs = jax.random.uniform(rngs.sample(), (batch_size, 2))
        physics_outputs = jnp.abs(
            jax.random.normal(rngs.sample(), (batch_size, 1))
        )  # Positive targets

        # Optimize temperature with physics constraints
        optimized_temp = calibrator.optimize_temperature_with_physics_constraints(
            predictions=jax.random.normal(rngs.sample(), (batch_size, 1)),
            targets=physics_outputs,
            inputs=physics_inputs,
        )

        assert optimized_temp > 0.0
        assert jnp.isfinite(optimized_temp)

        # Temperature should be adjusted based on physics constraint violations
        assert hasattr(calibrator, "constraint_penalty_history")
        assert len(calibrator.constraint_penalty_history) > 0


class TestPhysicsInformedPriors:
    """Test physics-informed prior constraints."""

    def test_conservation_law_priors(self):
        """Test conservation law enforcement in priors."""
        rngs = nnx.Rngs(42)

        # Create physics-informed prior
        physics_prior = PhysicsInformedPriors(
            conservation_laws=["energy", "momentum", "mass"],
            boundary_conditions=["dirichlet", "neumann"],
            constraint_weights=jnp.array([1.0, 0.8, 0.9, 0.7, 0.6]),
            rngs=rngs,
        )

        assert len(physics_prior.conservation_laws) == 3
        assert len(physics_prior.boundary_conditions) == 2
        assert physics_prior.constraint_weights.value.shape == (5,)

    def test_parameter_constraint_application(self):
        """Test applying physics constraints to sampled parameters."""
        rngs = nnx.Rngs(42)

        # Create physics prior
        physics_prior = PhysicsInformedPriors(
            conservation_laws=["energy"],
            boundary_conditions=["periodic"],
            rngs=rngs,
        )

        # Sample unconstrained parameters
        unconstrained_params = jax.random.normal(rngs.sample(), (10, 20))

        # Apply constraints
        constrained_params = physics_prior.apply_constraints(unconstrained_params)

        assert constrained_params.shape == unconstrained_params.shape
        assert jnp.all(jnp.isfinite(constrained_params))
        # Constraints should modify parameters
        assert not jnp.allclose(constrained_params, unconstrained_params)

    def test_constraint_violation_detection(self):
        """Test detection of physics constraint violations."""
        rngs = nnx.Rngs(42)

        physics_prior = PhysicsInformedPriors(
            conservation_laws=["energy", "momentum"],
            penalty_weight=1.0,
            rngs=rngs,
        )

        # Create parameters that violate constraints
        violating_params = jnp.array([1e6, -1e6, jnp.nan, jnp.inf])

        # Compute violation penalty
        violation_penalty = physics_prior.compute_violation_penalty(violating_params)

        assert violation_penalty >= 0.0
        assert jnp.isfinite(violation_penalty)

    def test_physical_plausibility_check(self):
        """Test physical plausibility validation."""
        rngs = nnx.Rngs(42)

        physics_prior = PhysicsInformedPriors(
            conservation_laws=["positivity", "boundedness"],
            rngs=rngs,
        )

        # Test physically plausible parameters
        plausible_params = jnp.array([0.5, 1.2, -0.3, 2.1])
        plausibility_score = physics_prior.check_physical_plausibility(plausible_params)

        assert 0.0 <= plausibility_score <= 1.0

        # Test physically implausible parameters
        implausible_params = jnp.array([jnp.inf, -1e10, jnp.nan])
        implausibility_score = physics_prior.check_physical_plausibility(
            implausible_params
        )

        assert implausibility_score < plausibility_score


class TestConservationLawPriors:
    """Test conservation law priors for uncertainty estimation (Phase 3)."""

    def test_conservation_law_priors_initialization(self):
        """Test ConservationLawPriors initialization."""
        rngs = nnx.Rngs(42)

        conservation_priors = ConservationLawPriors(
            conservation_laws=["energy", "momentum", "mass"],
            uncertainty_scale=0.1,
            prior_strength=1.0,
            adaptive_weighting=True,
            rngs=rngs,
        )

        assert len(conservation_priors.conservation_laws) == 3
        assert conservation_priors.uncertainty_scale == 0.1
        assert conservation_priors.prior_strength == 1.0
        assert conservation_priors.adaptive_weighting is True
        assert conservation_priors.conservation_strengths.value.shape == (3,)
        assert conservation_priors.uncertainty_scalings.value.shape == (3,)

    def test_physics_aware_uncertainty_computation(self):
        """Test physics-aware uncertainty computation."""
        rngs = nnx.Rngs(42)

        conservation_priors = ConservationLawPriors(
            conservation_laws=["energy", "momentum"],
            rngs=rngs,
        )

        # Create test data
        batch_size = 10
        predictions = jax.random.normal(rngs.sample(), (batch_size, 3))
        model_uncertainty = jax.random.uniform(rngs.sample(), (batch_size,)) * 0.1
        physics_state = jax.random.normal(rngs.sample(), (batch_size, 4))

        # Compute physics-aware uncertainty
        physics_uncertainty = conservation_priors.compute_physics_aware_uncertainty(
            predictions, model_uncertainty, physics_state
        )

        assert physics_uncertainty.shape == (batch_size,)
        assert jnp.all(physics_uncertainty >= model_uncertainty)
        assert jnp.all(jnp.isfinite(physics_uncertainty))

    def test_physics_constrained_parameter_sampling(self):
        """Test physics-constrained parameter sampling."""
        rngs = nnx.Rngs(42)

        conservation_priors = ConservationLawPriors(
            conservation_laws=["energy", "momentum"],
            rngs=rngs,
        )

        # Sample base parameters
        base_params = jax.random.normal(rngs.sample(), (5, 10))

        # Apply physics constraints
        constrained_params = conservation_priors.sample_physics_constrained_params(
            base_params, constraint_strength=0.8
        )

        assert constrained_params.shape == base_params.shape
        assert jnp.all(jnp.isfinite(constrained_params))
        # Physics constraints should modify parameters
        assert not jnp.allclose(constrained_params, base_params)


class TestDomainSpecificPriors:
    """Test domain-specific prior distributions (Phase 3)."""

    def test_domain_specific_priors_initialization(self):
        """Test DomainSpecificPriors initialization."""
        rngs = nnx.Rngs(42)

        domain_priors = DomainSpecificPriors(
            domain="quantum_chemistry",
            correlation_structure="independent",
            rngs=rngs,
        )

        assert domain_priors.domain == "quantum_chemistry"
        assert domain_priors.correlation_structure == "independent"
        assert len(domain_priors.parameter_ranges) == 5  # QC has 5 default params
        assert len(domain_priors.distribution_types) == 5

    def test_quantum_chemistry_domain_defaults(self):
        """Test quantum chemistry domain defaults."""
        rngs = nnx.Rngs(42)

        domain_priors = DomainSpecificPriors(domain="quantum_chemistry", rngs=rngs)

        # Check expected parameter ranges
        assert "bond_length" in domain_priors.parameter_ranges
        assert "bond_angle" in domain_priors.parameter_ranges
        assert "charge" in domain_priors.parameter_ranges
        assert "energy" in domain_priors.parameter_ranges
        assert "dipole" in domain_priors.parameter_ranges

        # Check distribution types
        assert domain_priors.distribution_types["bond_length"] == "lognormal"
        assert domain_priors.distribution_types["bond_angle"] == "beta"
        assert domain_priors.distribution_types["charge"] == "normal"

    def test_domain_prior_sampling(self):
        """Test sampling from domain-specific priors."""
        rngs = nnx.Rngs(42)

        domain_priors = DomainSpecificPriors(domain="quantum_chemistry", rngs=rngs)

        # Sample bond length parameters
        sample_shape = (20,)
        bond_length_samples = domain_priors.sample_domain_priors(
            sample_shape, "bond_length"
        )

        assert bond_length_samples.shape == sample_shape
        assert jnp.all(jnp.isfinite(bond_length_samples))
        # Bond lengths should be positive (lognormal distribution)
        assert jnp.all(bond_length_samples > 0)

        # Check that values are within expected range
        bond_range = domain_priors.parameter_ranges["bond_length"]
        assert jnp.all(bond_length_samples >= bond_range[0])
        assert jnp.all(bond_length_samples <= bond_range[1])

    def test_prior_log_probability_evaluation(self):
        """Test log probability evaluation under domain priors."""
        rngs = nnx.Rngs(42)

        domain_priors = DomainSpecificPriors(domain="quantum_chemistry", rngs=rngs)

        # Test log probability for bond length
        test_values = jnp.array([1.0, 1.5, 2.0, 2.5])
        log_probs = domain_priors.evaluate_prior_log_prob(test_values, "bond_length")

        assert log_probs.shape == test_values.shape
        assert jnp.all(jnp.isfinite(log_probs))
        # All values should have negative log probability (proper distribution)
        assert jnp.all(log_probs <= 0.0)


class TestHierarchicalBayesianFramework:
    """Test hierarchical Bayesian framework (Phase 3)."""

    def test_hierarchical_framework_initialization(self):
        """Test HierarchicalBayesianFramework initialization."""
        rngs = nnx.Rngs(42)

        hierarchical_framework = HierarchicalBayesianFramework(
            hierarchy_levels=3,
            level_dimensions=(64, 32, 16),
            uncertainty_propagation="multiplicative",
            correlation_structure="exchangeable",
            rngs=rngs,
        )

        assert hierarchical_framework.hierarchy_levels == 3
        assert hierarchical_framework.level_dimensions == (64, 32, 16)
        assert hierarchical_framework.uncertainty_propagation == "multiplicative"
        assert len(hierarchical_framework.level_means) == 3
        assert len(hierarchical_framework.level_scales) == 3

    def test_hierarchical_parameter_sampling(self):
        """Test hierarchical parameter sampling."""
        rngs = nnx.Rngs(42)

        hierarchical_framework = HierarchicalBayesianFramework(
            hierarchy_levels=3,
            level_dimensions=(32, 16, 8),
            rngs=rngs,
        )

        # Sample from level 0
        sample_shape = (10,)
        level_0_samples = hierarchical_framework.sample_hierarchical_parameters(
            sample_shape, level=0
        )

        assert level_0_samples.shape == (*sample_shape, 32)
        assert jnp.all(jnp.isfinite(level_0_samples))

        # Sample from level 1
        level_1_samples = hierarchical_framework.sample_hierarchical_parameters(
            sample_shape, level=1
        )

        assert level_1_samples.shape == (*sample_shape, 16)
        assert jnp.all(jnp.isfinite(level_1_samples))

    def test_hierarchical_uncertainty_propagation(self):
        """Test hierarchical uncertainty propagation."""
        rngs = nnx.Rngs(42)

        hierarchical_framework = HierarchicalBayesianFramework(
            hierarchy_levels=3,
            level_dimensions=(32, 16, 8),
            uncertainty_propagation="additive",
            rngs=rngs,
        )

        # Base uncertainty
        base_uncertainty = jnp.array([0.1, 0.2, 0.15, 0.3])

        # Propagate to level 2
        propagated_uncertainty = (
            hierarchical_framework.propagate_uncertainty_hierarchically(
                base_uncertainty, target_level=2
            )
        )

        assert propagated_uncertainty.shape == base_uncertainty.shape
        assert jnp.all(jnp.isfinite(propagated_uncertainty))
        # Propagated uncertainty should be larger than base
        assert jnp.all(propagated_uncertainty >= base_uncertainty)

    def test_hierarchical_log_probability(self):
        """Test hierarchical log probability computation."""
        rngs = nnx.Rngs(42)

        hierarchical_framework = HierarchicalBayesianFramework(
            hierarchy_levels=2,
            level_dimensions=(16, 8),
            rngs=rngs,
        )

        # Test values for level 0
        test_values = jax.random.normal(rngs.sample(), (5, 16))
        log_probs = hierarchical_framework.compute_hierarchical_log_prob(
            test_values, level=0
        )

        assert log_probs.shape == (5,)
        assert jnp.all(jnp.isfinite(log_probs))


class TestPhysicsAwareUncertaintyPropagation:
    """Test physics-aware uncertainty propagation (Phase 3)."""

    def test_physics_aware_propagation_initialization(self):
        """Test PhysicsAwareUncertaintyPropagation initialization."""
        rngs = nnx.Rngs(42)

        physics_propagation = PhysicsAwareUncertaintyPropagation(
            conservation_laws=["energy", "momentum"],
            constraint_tolerance=1e-6,
            uncertainty_inflation=1.1,
            correlation_aware=True,
            rngs=rngs,
        )

        assert len(physics_propagation.conservation_laws) == 2
        assert physics_propagation.constraint_tolerance == 1e-6
        assert physics_propagation.uncertainty_inflation == 1.1
        assert physics_propagation.correlation_aware is True
        assert physics_propagation.constraint_weights.value.shape == (2,)

    def test_physics_constrained_uncertainty_propagation(self):
        """Test physics-constrained uncertainty propagation."""
        rngs = nnx.Rngs(42)

        physics_propagation = PhysicsAwareUncertaintyPropagation(
            conservation_laws=["energy"],
            rngs=rngs,
        )

        batch_size = 8
        input_dim = 4
        output_dim = 3

        # Create test data
        input_uncertainty = jax.random.uniform(rngs.sample(), (batch_size,)) * 0.1
        model_jacobian = jax.random.normal(
            rngs.sample(), (batch_size, output_dim, input_dim)
        )
        physics_state = jax.random.normal(rngs.sample(), (batch_size, 4))

        # Propagate uncertainty with physics constraints
        physics_uncertainty = physics_propagation.propagate_with_physics_constraints(
            input_uncertainty, model_jacobian, physics_state
        )

        assert physics_uncertainty.shape == (batch_size,)
        assert jnp.all(physics_uncertainty >= 0.0)
        assert jnp.all(jnp.isfinite(physics_uncertainty))

    def test_physics_informed_confidence_computation(self):
        """Test physics-informed confidence computation."""
        rngs = nnx.Rngs(42)

        physics_propagation = PhysicsAwareUncertaintyPropagation(
            conservation_laws=["energy", "momentum"],
            rngs=rngs,
        )

        batch_size = 10
        predictions = jax.random.normal(rngs.sample(), (batch_size, 2))
        uncertainties = jax.random.uniform(rngs.sample(), (batch_size,)) * 0.2
        physics_state = jax.random.normal(rngs.sample(), (batch_size, 4))

        # Compute physics-informed confidence
        confidence = physics_propagation.compute_physics_informed_confidence(
            predictions, uncertainties, physics_state
        )

        assert confidence.shape == (batch_size,)
        assert jnp.all(confidence >= 0.0)
        assert jnp.all(confidence <= 1.0)
        assert jnp.all(jnp.isfinite(confidence))

    def test_uncertainty_aware_constraint_projection(self):
        """Test uncertainty-aware constraint projection."""
        rngs = nnx.Rngs(42)

        physics_propagation = PhysicsAwareUncertaintyPropagation(
            conservation_laws=["energy", "momentum"],
            rngs=rngs,
        )

        # Test parameters and uncertainties
        parameters = jax.random.normal(rngs.sample(), (5, 8))
        uncertainties = jax.random.uniform(rngs.sample(), (5, 8)) * 0.1

        # Project parameters while accounting for uncertainty
        projected_params, adjusted_uncertainties = (
            physics_propagation.uncertainty_aware_constraint_projection(
                parameters, uncertainties
            )
        )

        assert projected_params.shape == parameters.shape
        assert adjusted_uncertainties.shape == uncertainties.shape
        assert jnp.all(jnp.isfinite(projected_params))
        assert jnp.all(jnp.isfinite(adjusted_uncertainties))
        assert jnp.all(adjusted_uncertainties >= 0.0)


class TestProbabilisticIntegration:
    """Test integration between probabilistic components."""

    def test_blackjax_calibration_integration(self):
        """Test integration between BlackJAX sampling and calibration tools."""
        rngs = nnx.Rngs(42)

        # Create model and sampler
        base_model = StandardMLP([2, 4, 1], rngs=rngs)
        sampler = BlackJAXIntegration(
            base_model=base_model,
            sampler_type="nuts",
            num_warmup=20,
            num_samples=50,
            rngs=rngs,
        )

        # Create calibration tools
        calibrator = CalibrationTools(rngs=rngs)

        # This tests that the components can work together
        assert sampler is not None
        assert calibrator is not None

    def test_physics_prior_variational_integration(self):
        """Test integration between physics priors and variational framework."""
        rngs = nnx.Rngs(42)

        # Create base model
        base_model = StandardMLP([3, 6, 1], rngs=rngs)

        # Create physics-informed prior config
        prior_config = PriorConfig(
            conservation_laws=["energy", "momentum"],
            boundary_conditions=["periodic"],
            physics_constraints=["positivity"],
            prior_scale=0.5,
        )

        # Create variational framework with physics priors
        framework = AmortizedVariationalFramework(
            base_model=base_model,
            prior_config=prior_config,
            variational_config=VariationalConfig(input_dim=3),
            rngs=rngs,
        )

        # Test that physics constraints are properly integrated
        assert framework.prior_config.conservation_laws == ["energy", "momentum"]
        assert framework.prior_config.physics_constraints == ["positivity"]

    def test_end_to_end_probabilistic_workflow(self):
        """Test complete probabilistic workflow with all components."""
        rngs = nnx.Rngs(42)

        # Create synthetic physics problem data
        batch_size = 30
        input_dim = 2
        x_data = jax.random.uniform(
            rngs.sample(), (batch_size, input_dim), minval=-1, maxval=1
        )
        # Create model with all probabilistic components
        base_model = StandardMLP([input_dim, 8, 1], rngs=rngs)

        # Physics-informed prior
        PhysicsInformedPriors(
            conservation_laws=["energy"],
            rngs=rngs,
        )

        # Temperature scaling for calibration
        TemperatureScaling(
            physics_constraints=["energy_conservation"],
            rngs=rngs,
        )

        # Variational framework integration
        prior_config = PriorConfig(
            conservation_laws=["energy"],
            physics_constraints=["positivity"],
        )

        framework = AmortizedVariationalFramework(
            base_model=base_model,
            prior_config=prior_config,
            variational_config=VariationalConfig(input_dim=input_dim),
            rngs=rngs,
        )

        # Test that all components can work together
        predictions, uncertainties = framework(x_data[:5], num_samples=10)

        assert predictions.shape == (5, 1)
        assert uncertainties.shape == (5, 1)
        assert jnp.all(jnp.isfinite(predictions))
        assert jnp.all(uncertainties >= 0.0)

    def test_phase_3_integration_workflow(self):
        """Test Phase 3 physics-informed integration workflow."""
        rngs = nnx.Rngs(42)

        # Create Phase 3 components
        conservation_priors = ConservationLawPriors(
            conservation_laws=["energy", "momentum"],
            rngs=rngs,
        )

        domain_priors = DomainSpecificPriors(
            domain="quantum_chemistry",
            rngs=rngs,
        )

        hierarchical_framework = HierarchicalBayesianFramework(
            hierarchy_levels=2,
            level_dimensions=(16, 8),
            rngs=rngs,
        )

        physics_propagation = PhysicsAwareUncertaintyPropagation(
            conservation_laws=["energy"],
            rngs=rngs,
        )

        # Test integration between components
        batch_size = 5

        # Sample from domain priors
        bond_length_samples = domain_priors.sample_domain_priors(
            (batch_size,), "bond_length"
        )

        # Sample hierarchical parameters
        hierarchical_samples = hierarchical_framework.sample_hierarchical_parameters(
            (batch_size,), level=0
        )

        # Create mock physics data
        predictions = jax.random.normal(rngs.sample(), (batch_size, 2))
        model_uncertainty = jax.random.uniform(rngs.sample(), (batch_size,)) * 0.1
        physics_state = jax.random.normal(rngs.sample(), (batch_size, 4))

        # Compute physics-aware uncertainty
        physics_uncertainty = conservation_priors.compute_physics_aware_uncertainty(
            predictions, model_uncertainty, physics_state
        )

        # Compute physics-informed confidence
        confidence = physics_propagation.compute_physics_informed_confidence(
            predictions, physics_uncertainty, physics_state
        )

        # Verify all components work together
        assert bond_length_samples.shape == (batch_size,)
        assert hierarchical_samples.shape == (batch_size, 16)
        assert physics_uncertainty.shape == (batch_size,)
        assert confidence.shape == (batch_size,)
        assert jnp.all(jnp.isfinite(bond_length_samples))
        assert jnp.all(jnp.isfinite(hierarchical_samples))
        assert jnp.all(jnp.isfinite(physics_uncertainty))
        assert jnp.all(jnp.isfinite(confidence))


class TestEnhancedCalibrationMethods:
    """Test enhanced calibration methods from Phase 2."""

    def test_platt_scaling_initialization(self):
        """Test PlattScaling initialization."""
        rngs = nnx.Rngs(42)

        platt_scaler = PlattScaling(rngs=rngs)

        assert hasattr(platt_scaler, "a")
        assert hasattr(platt_scaler, "b")
        assert platt_scaler.a.value.shape == ()
        assert platt_scaler.b.value.shape == ()

    def test_platt_scaling_calibration(self):
        """Test Platt scaling calibration process."""
        rngs = nnx.Rngs(42)

        platt_scaler = PlattScaling(rngs=rngs)

        # Create synthetic binary classification data
        num_samples = 100
        logits = jax.random.normal(rngs.sample(), (num_samples,))
        labels = logits > 0  # Remove explicit dtype casting

        # Fit Platt scaling parameters
        platt_scaler.fit(logits, labels)

        # Apply calibration
        calibrated_probs = platt_scaler(logits)

        assert calibrated_probs.shape == logits.shape
        assert jnp.all(calibrated_probs >= 0.0)
        assert jnp.all(calibrated_probs <= 1.0)
        assert jnp.all(jnp.isfinite(calibrated_probs))

    def test_isotonic_regression_initialization(self):
        """Test IsotonicRegression initialization."""
        rngs = nnx.Rngs(42)

        isotonic_regressor = IsotonicRegression(n_bins=50, rngs=rngs)

        assert isotonic_regressor.n_bins == 50
        assert hasattr(isotonic_regressor, "calibration_map")
        assert hasattr(isotonic_regressor, "bin_edges")

    def test_isotonic_regression_calibration(self):
        """Test isotonic regression calibration."""
        rngs = nnx.Rngs(42)

        isotonic_regressor = IsotonicRegression(n_bins=20, rngs=rngs)

        # Create synthetic calibration data
        num_samples = 200
        confidences = jax.random.uniform(rngs.sample(), (num_samples,))
        # Make labels correlated with confidences for better calibration
        labels = (
            confidences + 0.1 * jax.random.normal(rngs.sample(), (num_samples,))
        ) > 0.5

        # Fit isotonic regression
        isotonic_regressor.fit(confidences, labels)  # Remove explicit dtype casting

        # Apply calibration
        calibrated_confidences = isotonic_regressor(confidences)

        assert calibrated_confidences.shape == confidences.shape
        assert jnp.all(calibrated_confidences >= 0.0)
        assert jnp.all(calibrated_confidences <= 1.0)
        assert jnp.all(jnp.isfinite(calibrated_confidences))

    def test_conformal_prediction_initialization(self):
        """Test ConformalPrediction initialization."""
        rngs = nnx.Rngs(42)

        conformal_predictor = ConformalPrediction(alpha=0.05, rngs=rngs)

        assert conformal_predictor.alpha == 0.05
        assert hasattr(conformal_predictor, "quantile")

    def test_conformal_prediction_intervals(self):
        """Test conformal prediction interval computation."""
        rngs = nnx.Rngs(42)

        conformal_predictor = ConformalPrediction(alpha=0.1, rngs=rngs)

        # Create calibration data
        num_cal_samples = 100
        cal_predictions = jax.random.normal(rngs.sample(), (num_cal_samples,))
        cal_true_values = cal_predictions + 0.5 * jax.random.normal(
            rngs.sample(), (num_cal_samples,)
        )

        # Calibrate conformal predictor
        conformal_predictor.calibrate(cal_predictions, cal_true_values)

        # Create test predictions
        num_test_samples = 50
        test_predictions = jax.random.normal(rngs.sample(), (num_test_samples,))

        # Compute prediction intervals
        lower_bounds, upper_bounds = conformal_predictor.predict_intervals(
            test_predictions
        )

        assert lower_bounds.shape == test_predictions.shape
        assert upper_bounds.shape == test_predictions.shape
        assert jnp.all(lower_bounds <= upper_bounds)
        assert jnp.all(jnp.isfinite(lower_bounds))
        assert jnp.all(jnp.isfinite(upper_bounds))

    def test_conformal_prediction_coverage(self):
        """Test conformal prediction coverage computation."""
        rngs = nnx.Rngs(42)

        conformal_predictor = ConformalPrediction(alpha=0.1, rngs=rngs)

        # Create synthetic test data
        num_samples = 100
        predictions = jax.random.normal(rngs.sample(), (num_samples,))
        true_values = predictions + 0.3 * jax.random.normal(
            rngs.sample(), (num_samples,)
        )

        # Calibrate with part of the data
        conformal_predictor.calibrate(predictions[:50], true_values[:50])

        # Test coverage on remaining data
        test_predictions = predictions[50:]
        test_true_values = true_values[50:]

        lower_bounds, upper_bounds = conformal_predictor.predict_intervals(
            test_predictions
        )
        coverage = conformal_predictor.compute_coverage(
            lower_bounds, upper_bounds, test_true_values
        )

        assert isinstance(coverage, (int, float))
        assert 0.0 <= coverage <= 1.0
        # With alpha=0.1, we expect coverage to be around 0.9
        assert coverage > 0.5  # Reasonable lower bound for this test

    def test_enhanced_calibration_integration(self):
        """Test integration between enhanced calibration methods."""
        rngs = nnx.Rngs(42)

        # Create all enhanced calibration methods
        platt_scaler = PlattScaling(rngs=rngs)
        isotonic_regressor = IsotonicRegression(rngs=rngs)
        conformal_predictor = ConformalPrediction(rngs=rngs)

        # Create synthetic data
        num_samples = 150
        logits = jax.random.normal(rngs.sample(), (num_samples,))
        labels = logits > 0  # Remove explicit dtype casting
        predictions = jax.nn.sigmoid(logits)

        # Test that all methods can work with the same data
        platt_scaler.fit(logits[:100], labels[:100])
        isotonic_regressor.fit(predictions[:100], labels[:100])
        conformal_predictor.calibrate(predictions[:100], labels[:100])

        # Apply all calibration methods
        platt_calibrated = platt_scaler(logits[100:])
        isotonic_calibrated = isotonic_regressor(predictions[100:])
        conformal_lower, conformal_upper = conformal_predictor.predict_intervals(
            predictions[100:]
        )

        # All methods should produce valid outputs
        assert jnp.all(jnp.isfinite(platt_calibrated))
        assert jnp.all(jnp.isfinite(isotonic_calibrated))
        assert jnp.all(jnp.isfinite(conformal_lower))
        assert jnp.all(jnp.isfinite(conformal_upper))

        # Outputs should be in valid ranges
        assert jnp.all((platt_calibrated >= 0.0) & (platt_calibrated <= 1.0))
        assert jnp.all((isotonic_calibrated >= 0.0) & (isotonic_calibrated <= 1.0))
        assert jnp.all(conformal_lower <= conformal_upper)
