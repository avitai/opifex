"""Tests for warm-starting optimization strategies."""

import jax.numpy as jnp

from opifex.optimization.meta_optimization.warm_starting import WarmStartingStrategy


class TestWarmStartingInit:
    """Tests for WarmStartingStrategy initialization."""

    def test_default_config(self):
        """Default config uses parameter_transfer strategy."""
        ws = WarmStartingStrategy()
        assert ws.strategy_type == "parameter_transfer"
        assert ws.similarity_threshold == 0.8
        assert ws.memory_size == 10

    def test_custom_config(self):
        """Custom config overrides defaults."""
        ws = WarmStartingStrategy(
            strategy_type="molecular_similarity",
            adaptation_ratio=0.5,
        )
        assert ws.strategy_type == "molecular_similarity"
        assert ws.adaptation_ratio == 0.5


class TestParameterTransfer:
    """Tests for parameter transfer warm-starting."""

    def test_returns_adapted_params(self):
        """Parameter transfer scales and perturbs previous params."""
        ws = WarmStartingStrategy(strategy_type="parameter_transfer", adaptation_ratio=0.9)
        prev_params = jnp.ones((4,))
        features = jnp.ones((2,))

        result = ws.get_warm_start_params(prev_params, features)
        assert result.shape == prev_params.shape
        # Should be close to 0.9 * 1.0 = 0.9 plus noise (0.1 * Normal(0,1))
        # Use 3-sigma tolerance: 3 * 0.1 = 0.3
        assert jnp.allclose(result, prev_params * 0.9, atol=0.3)

    def test_non_transfer_returns_input(self):
        """Non-transfer strategy returns params unchanged."""
        ws = WarmStartingStrategy(strategy_type="other")
        prev_params = jnp.array([1.0, 2.0, 3.0])
        result = ws.get_warm_start_params(prev_params, jnp.ones((2,)))
        assert jnp.allclose(result, prev_params)


class TestOptimizerStateAdaptation:
    """Tests for optimizer state adaptation."""

    def test_resets_step_count(self):
        """Step count is scaled down."""
        ws = WarmStartingStrategy()
        state = {"step": jnp.array(100)}
        adapted = ws.adapt_optimizer_state(state)
        assert int(adapted["step"]) == 10

    def test_scales_momentum(self):
        """Array state values are scaled by adaptation_ratio."""
        ws = WarmStartingStrategy(adaptation_ratio=0.5)
        momentum = jnp.ones((4,)) * 2.0
        state = {"momentum": momentum}
        adapted = ws.adapt_optimizer_state(state)
        assert jnp.allclose(adapted["momentum"], jnp.ones((4,)))

    def test_preserves_non_array_state(self):
        """Non-array state values pass through unchanged."""
        ws = WarmStartingStrategy()
        state = {"config_name": "adam", "learning_rate": 0.001}
        adapted = ws.adapt_optimizer_state(state)
        assert adapted["config_name"] == "adam"
        assert adapted["learning_rate"] == 0.001


class TestMolecularWarmStart:
    """Tests for molecular similarity warm-starting."""

    def test_cosine_similarity_selects_most_similar(self):
        """Cosine similarity selects the closest molecule."""
        ws = WarmStartingStrategy(similarity_metric="cosine", min_similarity=0.0)

        fingerprints = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        params = jnp.array([[10.0], [20.0], [30.0]])
        query = jnp.array([0.9, 0.1])

        result = ws.get_molecular_warm_start(fingerprints, params, query)
        # query is most similar to first fingerprint [1, 0]
        assert jnp.allclose(result, jnp.array([10.0]))

    def test_euclidean_similarity(self):
        """Euclidean distance-based similarity works."""
        ws = WarmStartingStrategy(similarity_metric="euclidean", min_similarity=0.0)

        fingerprints = jnp.array([[0.0, 0.0], [10.0, 10.0]])
        params = jnp.array([[1.0], [2.0]])
        query = jnp.array([0.1, 0.1])

        result = ws.get_molecular_warm_start(fingerprints, params, query)
        # query is closest to first fingerprint [0, 0]
        assert jnp.allclose(result, jnp.array([1.0]))

    def test_low_similarity_returns_average(self):
        """When no molecule is similar enough, returns average params."""
        ws = WarmStartingStrategy(similarity_metric="cosine", min_similarity=0.99)

        fingerprints = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        params = jnp.array([[10.0], [20.0]])
        query = jnp.array([0.5, 0.5])  # equidistant

        result = ws.get_molecular_warm_start(fingerprints, params, query)
        # Neither is > 0.99 similar, should return mean
        assert jnp.allclose(result, jnp.array([15.0]))
