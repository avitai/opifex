"""Tests for second-order optimization configuration.

TDD: These tests define the expected behavior for SecondOrderConfig,
LBFGSConfig, GaussNewtonConfig, and HybridOptimizerConfig.
"""

import pytest

from opifex.optimization.second_order.config import (
    GaussNewtonConfig,
    HybridOptimizerConfig,
    LBFGSConfig,
    LinesearchType,
    SecondOrderConfig,
    SecondOrderMethod,
    SwitchCriterion,
)


class TestSecondOrderConfig:
    """Test unified SecondOrderConfig dataclass."""

    def test_default_config_valid(self):
        """Default configuration should have sensible values."""
        config = SecondOrderConfig()
        assert config.method == SecondOrderMethod.LBFGS
        assert config.max_iterations > 0
        assert config.tolerance > 0

    def test_method_enum_values(self):
        """SecondOrderMethod enum should have all expected values."""
        assert SecondOrderMethod.LBFGS.value == "lbfgs"
        assert SecondOrderMethod.GAUSS_NEWTON.value == "gauss_newton"
        assert SecondOrderMethod.LEVENBERG_MARQUARDT.value == "levenberg_marquardt"
        assert SecondOrderMethod.BFGS.value == "bfgs"

    def test_config_immutable(self):
        """Config should be frozen (immutable)."""
        config = SecondOrderConfig()
        with pytest.raises(AttributeError):
            config.method = SecondOrderMethod.BFGS  # type: ignore # noqa: PGH003

    def test_custom_config_values(self):
        """Custom values should be accepted."""
        config = SecondOrderConfig(
            method=SecondOrderMethod.BFGS,
            max_iterations=50,
            tolerance=1e-8,
        )
        assert config.method == SecondOrderMethod.BFGS
        assert config.max_iterations == 50
        assert config.tolerance == 1e-8


class TestLBFGSConfig:
    """Test L-BFGS specific configuration."""

    def test_default_config_valid(self):
        """Default L-BFGS config should have sensible values."""
        config = LBFGSConfig()
        assert config.memory_size == 10
        assert config.scale_init_precond is True
        assert config.linesearch == LinesearchType.ZOOM

    def test_memory_size_positive(self):
        """Memory size must be positive."""
        with pytest.raises(ValueError, match="memory_size must be positive"):
            LBFGSConfig(memory_size=0)

        with pytest.raises(ValueError, match="memory_size must be positive"):
            LBFGSConfig(memory_size=-5)

    def test_linesearch_enum_values(self):
        """LinesearchType enum should have expected values."""
        assert LinesearchType.ZOOM.value == "zoom"
        assert LinesearchType.BACKTRACKING.value == "backtracking"

    def test_custom_config(self):
        """Custom L-BFGS config should work."""
        config = LBFGSConfig(
            memory_size=20,
            scale_init_precond=False,
            linesearch=LinesearchType.BACKTRACKING,
            max_linesearch_steps=30,
        )
        assert config.memory_size == 20
        assert config.scale_init_precond is False
        assert config.linesearch == LinesearchType.BACKTRACKING
        assert config.max_linesearch_steps == 30


class TestGaussNewtonConfig:
    """Test Gauss-Newton/Levenberg-Marquardt configuration."""

    def test_default_config_valid(self):
        """Default GN config should have sensible values."""
        config = GaussNewtonConfig()
        assert config.damping_factor >= 0
        assert config.damping_increase_factor > 1
        assert config.damping_decrease_factor < 1
        assert config.min_damping > 0
        assert config.max_damping > config.min_damping

    def test_invalid_damping_factors(self):
        """Damping factors must satisfy constraints."""
        # Increase factor must be > 1
        with pytest.raises(ValueError, match="increase_factor must be > 1"):
            GaussNewtonConfig(damping_increase_factor=0.5)

        # Decrease factor must be < 1
        with pytest.raises(ValueError, match="decrease_factor must be < 1"):
            GaussNewtonConfig(damping_decrease_factor=2.0)

    def test_damping_bounds_valid(self):
        """Min damping must be less than max damping."""
        with pytest.raises(ValueError, match="min_damping must be < max_damping"):
            GaussNewtonConfig(min_damping=1.0, max_damping=0.1)


class TestHybridOptimizerConfig:
    """Test hybrid Adam→L-BFGS optimizer configuration."""

    def test_default_config_valid(self):
        """Default hybrid config should have sensible values."""
        config = HybridOptimizerConfig()
        assert config.first_order_steps > 0
        assert config.switch_criterion == SwitchCriterion.LOSS_VARIANCE
        assert config.adam_learning_rate > 0
        assert config.lbfgs_config is not None

    def test_switch_criterion_enum(self):
        """SwitchCriterion enum should have expected values."""
        assert SwitchCriterion.EPOCH.value == "epoch"
        assert SwitchCriterion.LOSS_VARIANCE.value == "loss_variance"
        assert SwitchCriterion.GRADIENT_NORM.value == "gradient_norm"
        assert SwitchCriterion.RELATIVE_IMPROVEMENT.value == "relative_improvement"

    def test_loss_variance_requires_threshold(self):
        """Loss variance criterion requires threshold parameter."""
        config = HybridOptimizerConfig(
            switch_criterion=SwitchCriterion.LOSS_VARIANCE,
            loss_variance_threshold=1e-4,
        )
        assert config.loss_variance_threshold == 1e-4

    def test_loss_history_window_positive(self):
        """Loss history window must be positive."""
        with pytest.raises(ValueError, match="loss_history_window must be positive"):
            HybridOptimizerConfig(loss_history_window=0)

    def test_nested_lbfgs_config(self):
        """Hybrid config should accept custom L-BFGS config."""
        lbfgs_config = LBFGSConfig(memory_size=20)
        config = HybridOptimizerConfig(lbfgs_config=lbfgs_config)
        assert config.lbfgs_config.memory_size == 20

    def test_gradient_norm_threshold(self):
        """Gradient norm criterion should work with threshold."""
        config = HybridOptimizerConfig(
            switch_criterion=SwitchCriterion.GRADIENT_NORM,
            gradient_norm_threshold=1e-3,
        )
        assert config.switch_criterion == SwitchCriterion.GRADIENT_NORM
        assert config.gradient_norm_threshold == 1e-3
