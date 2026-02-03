"""Tests for Coarse-to-Fine multilevel training.

TDD: These tests define the expected behavior for multilevel network hierarchies.
"""

import jax
import jax.numpy as jnp
from flax import nnx


class TestMultilevelConfig:
    """Test multilevel training configuration."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        from opifex.training.multilevel.coarse_to_fine import MultilevelConfig

        config = MultilevelConfig()
        assert config.num_levels >= 2
        assert config.coarsening_factor > 0
        assert config.coarsening_factor < 1

    def test_custom_levels(self):
        """Should accept custom number of levels."""
        from opifex.training.multilevel.coarse_to_fine import MultilevelConfig

        config = MultilevelConfig(num_levels=4)
        assert config.num_levels == 4


class TestNetworkHierarchy:
    """Test network hierarchy creation."""

    def test_create_hierarchy(self):
        """Should create hierarchy of networks with decreasing width."""
        from opifex.training.multilevel.coarse_to_fine import create_network_hierarchy

        hierarchy = create_network_hierarchy(
            input_dim=2,
            output_dim=1,
            base_hidden_dims=[64, 64],
            num_levels=3,
            coarsening_factor=0.5,
            rngs=nnx.Rngs(0),
        )

        assert len(hierarchy) == 3

    def test_coarser_levels_smaller(self):
        """Coarser levels should have smaller networks."""
        from opifex.training.multilevel.coarse_to_fine import create_network_hierarchy

        hierarchy = create_network_hierarchy(
            input_dim=2,
            output_dim=1,
            base_hidden_dims=[64, 64],
            num_levels=3,
            coarsening_factor=0.5,
            rngs=nnx.Rngs(0),
        )

        # Count parameters at each level
        def count_params(model):
            leaves = [p.size for p in jax.tree.leaves(nnx.state(model))]
            return sum(leaves)

        params_counts = [count_params(m) for m in hierarchy]

        # Coarser levels (lower index) should have fewer parameters
        for i in range(len(params_counts) - 1):
            assert params_counts[i] < params_counts[i + 1]


class TestTransferOperators:
    """Test transfer operators between levels."""

    def test_prolongation(self):
        """Should transfer parameters from coarse to fine."""
        from opifex.training.multilevel.coarse_to_fine import (
            create_network_hierarchy,
            prolongate,
        )

        hierarchy = create_network_hierarchy(
            input_dim=1,
            output_dim=1,
            base_hidden_dims=[32],
            num_levels=2,
            coarsening_factor=0.5,
            rngs=nnx.Rngs(0),
        )

        coarse_model = hierarchy[0]
        fine_model = hierarchy[1]

        # Prolongate from coarse to fine
        prolongated = prolongate(coarse_model, fine_model)

        assert prolongated is not None

    def test_restriction(self):
        """Should transfer parameters from fine to coarse."""
        from opifex.training.multilevel.coarse_to_fine import (
            create_network_hierarchy,
            restrict,
        )

        hierarchy = create_network_hierarchy(
            input_dim=1,
            output_dim=1,
            base_hidden_dims=[32],
            num_levels=2,
            coarsening_factor=0.5,
            rngs=nnx.Rngs(0),
        )

        coarse_model = hierarchy[0]
        fine_model = hierarchy[1]

        # Restrict from fine to coarse
        restricted = restrict(fine_model, coarse_model)

        assert restricted is not None


class TestCascadeTrainer:
    """Test cascade (sequential) training."""

    def test_create_trainer(self):
        """Should create cascade trainer."""
        from opifex.training.multilevel.coarse_to_fine import (
            CascadeTrainer,
            MultilevelConfig,
        )

        config = MultilevelConfig(num_levels=3)
        trainer = CascadeTrainer(
            input_dim=1,
            output_dim=1,
            base_hidden_dims=[32],
            config=config,
            rngs=nnx.Rngs(0),
        )

        assert trainer is not None
        assert len(trainer.hierarchy) == 3

    def test_get_current_model(self):
        """Should return current level model."""
        from opifex.training.multilevel.coarse_to_fine import (
            CascadeTrainer,
            MultilevelConfig,
        )

        config = MultilevelConfig(num_levels=2)
        trainer = CascadeTrainer(
            input_dim=1,
            output_dim=1,
            base_hidden_dims=[32],
            config=config,
            rngs=nnx.Rngs(0),
        )

        model = trainer.get_current_model()
        assert model is not None

    def test_advance_level(self):
        """Should advance to next level."""
        from opifex.training.multilevel.coarse_to_fine import (
            CascadeTrainer,
            MultilevelConfig,
        )

        config = MultilevelConfig(num_levels=3)
        trainer = CascadeTrainer(
            input_dim=1,
            output_dim=1,
            base_hidden_dims=[32],
            config=config,
            rngs=nnx.Rngs(0),
        )

        initial_level = trainer.current_level
        trainer.advance_level()

        assert trainer.current_level == initial_level + 1


class TestMultilevelForward:
    """Test forward pass through multilevel models."""

    def test_forward_pass(self):
        """Should compute forward pass at each level."""
        from opifex.training.multilevel.coarse_to_fine import create_network_hierarchy

        hierarchy = create_network_hierarchy(
            input_dim=1,
            output_dim=1,
            base_hidden_dims=[32],
            num_levels=2,
            coarsening_factor=0.5,
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.5]])

        for model in hierarchy:
            y = model(x)
            assert y.shape == (1, 1)
            assert jnp.isfinite(y).all()

    def test_multilevel_consistency(self):
        """All levels should produce valid outputs for same input."""
        from opifex.training.multilevel.coarse_to_fine import create_network_hierarchy

        hierarchy = create_network_hierarchy(
            input_dim=2,
            output_dim=1,
            base_hidden_dims=[64, 32],
            num_levels=3,
            coarsening_factor=0.5,
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.1, 0.2], [0.3, 0.4]])

        outputs = [model(x) for model in hierarchy]

        # All outputs should have same shape
        for out in outputs:
            assert out.shape == (2, 1)


class TestTrainingIntegration:
    """Test multilevel training integration."""

    def test_train_coarse_then_fine(self):
        """Should train coarse level first, then fine."""
        import optax

        from opifex.training.multilevel.coarse_to_fine import (
            CascadeTrainer,
            MultilevelConfig,
        )

        config = MultilevelConfig(num_levels=2)
        trainer = CascadeTrainer(
            input_dim=1,
            output_dim=1,
            base_hidden_dims=[16],
            config=config,
            rngs=nnx.Rngs(0),
        )

        x = jnp.array([[0.1], [0.5], [0.9]])
        y_target = jnp.array([[0.1], [0.5], [0.9]])

        # Train at coarse level
        model = trainer.get_current_model()
        optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

        def loss_fn(model):
            return jnp.mean((model(x) - y_target) ** 2)

        for _ in range(5):
            _loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        # Advance and train at fine level
        trainer.advance_level()
        fine_model = trainer.get_current_model()

        assert fine_model is not None
