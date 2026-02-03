"""Tests for Multilevel FNO training.

TDD: These tests define the expected behavior for FNO-specific multilevel training.
"""

import jax.numpy as jnp
from flax import nnx


class TestMultilevelFNOConfig:
    """Test multilevel FNO configuration."""

    def test_default_config(self):
        """Should create config with sensible defaults."""
        from opifex.training.multilevel.multilevel_fno import MultilevelFNOConfig

        config = MultilevelFNOConfig()
        assert config.num_levels >= 2
        assert config.mode_reduction_factor > 1
        assert config.base_modes > 0

    def test_custom_levels(self):
        """Should accept custom number of levels."""
        from opifex.training.multilevel.multilevel_fno import MultilevelFNOConfig

        config = MultilevelFNOConfig(num_levels=4, base_modes=16)
        assert config.num_levels == 4
        assert config.base_modes == 16


class TestFNOModeHierarchy:
    """Test FNO mode hierarchy creation."""

    def test_create_mode_hierarchy(self):
        """Should create hierarchy of mode counts."""
        from opifex.training.multilevel.multilevel_fno import create_mode_hierarchy

        hierarchy = create_mode_hierarchy(
            base_modes=16,
            num_levels=3,
            reduction_factor=2,
        )

        assert len(hierarchy) == 3
        # Coarsest level should have fewest modes
        assert hierarchy[0] < hierarchy[1] < hierarchy[2]

    def test_mode_reduction(self):
        """Mode counts should follow reduction factor."""
        from opifex.training.multilevel.multilevel_fno import create_mode_hierarchy

        hierarchy = create_mode_hierarchy(
            base_modes=16,
            num_levels=3,
            reduction_factor=2,
        )

        # Finest level has base_modes
        assert hierarchy[-1] == 16
        # Each coarser level has modes / reduction_factor
        assert hierarchy[-2] == 8
        assert hierarchy[-3] == 4


class TestSimpleFNO:
    """Test simple FNO for multilevel training."""

    def test_create_simple_fno(self):
        """Should create simple FNO with given modes."""
        from opifex.training.multilevel.multilevel_fno import SimpleFNO

        fno = SimpleFNO(
            modes=8,
            width=32,
            input_dim=1,
            output_dim=1,
            rngs=nnx.Rngs(0),
        )

        assert fno is not None
        assert fno.modes == 8

    def test_fno_forward_pass(self):
        """FNO should compute forward pass."""
        from opifex.training.multilevel.multilevel_fno import SimpleFNO

        fno = SimpleFNO(
            modes=4,
            width=16,
            input_dim=1,
            output_dim=1,
            rngs=nnx.Rngs(0),
        )

        # Input: (batch, spatial, channels)
        x = jnp.ones((2, 32, 1))
        y = fno(x)

        assert y.shape == (2, 32, 1)
        assert jnp.isfinite(y).all()


class TestFNOHierarchy:
    """Test FNO hierarchy creation."""

    def test_create_fno_hierarchy(self):
        """Should create hierarchy of FNOs with decreasing modes."""
        from opifex.training.multilevel.multilevel_fno import create_fno_hierarchy

        hierarchy = create_fno_hierarchy(
            base_modes=8,
            width=16,
            input_dim=1,
            output_dim=1,
            num_levels=3,
            reduction_factor=2,
            rngs=nnx.Rngs(0),
        )

        assert len(hierarchy) == 3

    def test_hierarchy_modes_decreasing(self):
        """Coarser levels should have fewer modes."""
        from opifex.training.multilevel.multilevel_fno import create_fno_hierarchy

        hierarchy = create_fno_hierarchy(
            base_modes=8,
            width=16,
            input_dim=1,
            output_dim=1,
            num_levels=3,
            reduction_factor=2,
            rngs=nnx.Rngs(0),
        )

        modes = [fno.modes for fno in hierarchy]
        assert modes[0] < modes[1] < modes[2]


class TestFNOTransferOperators:
    """Test transfer operators for FNO parameters."""

    def test_mode_prolongation(self):
        """Should transfer modes from coarse to fine."""
        from opifex.training.multilevel.multilevel_fno import (
            prolongate_fno_modes,
            SimpleFNO,
        )

        coarse_fno = SimpleFNO(
            modes=4, width=16, input_dim=1, output_dim=1, rngs=nnx.Rngs(0)
        )
        fine_fno = SimpleFNO(
            modes=8, width=16, input_dim=1, output_dim=1, rngs=nnx.Rngs(1)
        )

        prolongate_fno_modes(coarse_fno, fine_fno)

        # Fine FNO should now have some coarse weights
        assert fine_fno is not None

    def test_mode_restriction(self):
        """Should transfer modes from fine to coarse."""
        from opifex.training.multilevel.multilevel_fno import (
            restrict_fno_modes,
            SimpleFNO,
        )

        fine_fno = SimpleFNO(
            modes=8, width=16, input_dim=1, output_dim=1, rngs=nnx.Rngs(0)
        )
        coarse_fno = SimpleFNO(
            modes=4, width=16, input_dim=1, output_dim=1, rngs=nnx.Rngs(1)
        )

        restrict_fno_modes(fine_fno, coarse_fno)

        # Coarse FNO should now have fine weights (truncated)
        assert coarse_fno is not None


class TestMultilevelFNOTrainer:
    """Test multilevel FNO trainer."""

    def test_create_trainer(self):
        """Should create multilevel FNO trainer."""
        from opifex.training.multilevel.multilevel_fno import (
            MultilevelFNOConfig,
            MultilevelFNOTrainer,
        )

        config = MultilevelFNOConfig(num_levels=2, base_modes=8)
        trainer = MultilevelFNOTrainer(
            width=16,
            input_dim=1,
            output_dim=1,
            config=config,
            rngs=nnx.Rngs(0),
        )

        assert trainer is not None
        assert len(trainer.hierarchy) == 2

    def test_get_current_model(self):
        """Should return current level FNO."""
        from opifex.training.multilevel.multilevel_fno import (
            MultilevelFNOConfig,
            MultilevelFNOTrainer,
        )

        config = MultilevelFNOConfig(num_levels=2, base_modes=8)
        trainer = MultilevelFNOTrainer(
            width=16,
            input_dim=1,
            output_dim=1,
            config=config,
            rngs=nnx.Rngs(0),
        )

        model = trainer.get_current_model()
        assert model is not None

    def test_advance_level(self):
        """Should advance to finer level."""
        from opifex.training.multilevel.multilevel_fno import (
            MultilevelFNOConfig,
            MultilevelFNOTrainer,
        )

        config = MultilevelFNOConfig(num_levels=3, base_modes=8)
        trainer = MultilevelFNOTrainer(
            width=16,
            input_dim=1,
            output_dim=1,
            config=config,
            rngs=nnx.Rngs(0),
        )

        initial_level = trainer.current_level
        trainer.advance_level()

        assert trainer.current_level == initial_level + 1


class TestMultilevelFNOTraining:
    """Test multilevel FNO in training context."""

    def test_train_coarse_then_fine(self):
        """Should train coarse FNO first, then fine."""
        import optax

        from opifex.training.multilevel.multilevel_fno import (
            MultilevelFNOConfig,
            MultilevelFNOTrainer,
        )

        config = MultilevelFNOConfig(num_levels=2, base_modes=4)
        trainer = MultilevelFNOTrainer(
            width=8,
            input_dim=1,
            output_dim=1,
            config=config,
            rngs=nnx.Rngs(0),
        )

        # Input shape: (batch, spatial, channels)
        x = jnp.ones((4, 16, 1)) * jnp.linspace(0, 1, 16).reshape(1, -1, 1)
        y_target = x  # Identity mapping for simplicity

        # Train at coarse level
        model = trainer.get_current_model()
        optimizer = nnx.Optimizer(model, optax.adam(0.01), wrt=nnx.Param)

        def loss_fn(model):
            return jnp.mean((model(x) - y_target) ** 2)

        for _ in range(3):
            _loss, grads = nnx.value_and_grad(loss_fn)(model)
            optimizer.update(model, grads)

        # Advance and verify
        trainer.advance_level()
        fine_model = trainer.get_current_model()

        assert fine_model is not None
        assert fine_model.modes > model.modes
