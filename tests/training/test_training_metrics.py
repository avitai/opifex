"""Tests for training metrics module."""

from __future__ import annotations

import jax.numpy as jnp
import optax
from flax import nnx

from opifex.neural.base import StandardMLP
from opifex.training.metrics import (
    AdvancedMetricsCollector,
    HARTREE_TO_KCAL_MOL,
    TrainingMetrics,
    TrainingState,
)


class TestTrainingMetrics:
    """Test training metrics tracking."""

    def test_metrics_creation(self):
        """Test metrics creation and updating."""
        metrics = TrainingMetrics()

        # Test initial state
        assert len(metrics.train_losses) == 0
        assert len(metrics.val_losses) == 0
        assert metrics.best_val_loss is None

    def test_metrics_update(self):
        """Test metrics updating functionality."""
        metrics = TrainingMetrics()

        # Update training loss
        metrics.update_train_loss(0.5)
        assert len(metrics.train_losses) == 1
        assert metrics.train_losses[0] == 0.5

        # Update validation loss
        metrics.update_val_loss(0.3)
        assert len(metrics.val_losses) == 1
        assert metrics.val_losses[0] == 0.3
        assert metrics.best_val_loss == 0.3

        # Update with worse validation loss
        metrics.update_val_loss(0.4)
        assert len(metrics.val_losses) == 2
        assert metrics.best_val_loss == 0.3  # Should remain the best

    def test_quantum_metrics_tracking(self):
        """Test quantum-specific metrics tracking."""
        metrics = TrainingMetrics()

        # Test chemical accuracy tracking
        metrics.update_chemical_accuracy(0.5e-3)  # kcal/mol
        assert len(metrics.chemical_accuracies) == 1
        assert metrics.chemical_accuracies[0] == 0.5e-3

        # Test SCF convergence tracking
        metrics.update_scf_convergence(True, 15)
        assert len(metrics.scf_converged) == 1
        assert metrics.scf_converged[0] is True
        assert len(metrics.scf_iterations) == 1
        assert metrics.scf_iterations[0] == 15

    def test_physics_losses_tracking(self):
        """Test physics-informed loss components tracking."""
        metrics = TrainingMetrics()

        # Test physics loss tracking
        metrics.physics_losses.append(0.1)
        metrics.physics_losses.append(0.05)
        assert len(metrics.physics_losses) == 2
        assert metrics.physics_losses[0] == 0.1

        # Test boundary loss tracking
        metrics.boundary_losses.append(0.02)
        assert len(metrics.boundary_losses) == 1
        assert metrics.boundary_losses[0] == 0.02

    def test_learning_rate_tracking(self):
        """Test learning rate tracking."""
        metrics = TrainingMetrics()

        metrics.update_learning_rate(1e-3)
        metrics.update_learning_rate(5e-4)
        assert len(metrics.learning_rates) == 2
        assert metrics.learning_rates[0] == 1e-3
        assert metrics.learning_rates[1] == 5e-4

    def test_constraint_violations_tracking(self):
        """Test constraint violations tracking."""
        metrics = TrainingMetrics()

        metrics.update_constraint_violation(1e-6)
        metrics.update_constraint_violation(5e-7)
        assert len(metrics.constraint_violations) == 2
        assert metrics.constraint_violations[0] == 1e-6


class TestTrainingState:
    """Test training state management."""

    def test_training_state_creation(self):
        """Test training state initialization."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            step=0,
            epoch=0,
        )

        assert state.step == 0
        assert state.epoch == 0
        assert state.optimizer is optimizer
        assert isinstance(state.model, StandardMLP)

    def test_training_state_step_increment(self):
        """Test training state step incrementation."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
            step=0,
            epoch=0,
        )

        # Test increment
        state.step += 1
        state.epoch += 1

        assert state.step == 1
        assert state.epoch == 1

    def test_enhanced_training_state_physics_metrics(self):
        """Test enhanced training state physics metrics tracking."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Test physics metrics
        state.update_physics_metric("energy_error", 0.1)
        state.update_physics_metric("energy_error", 0.05)
        assert len(state.physics_metrics["energy_error"]) == 2
        assert state.physics_metrics["energy_error"][0] == 0.1
        assert state.physics_metrics["energy_error"][1] == 0.05

        # Test conservation violations
        state.update_conservation_violation("mass", 1e-6)
        state.update_conservation_violation("energy", 2e-6)
        assert len(state.conservation_violations["mass"]) == 1
        assert len(state.conservation_violations["energy"]) == 1
        assert state.conservation_violations["mass"][0] == 1e-6

    def test_enhanced_training_state_chemical_accuracy(self):
        """Test chemical accuracy tracking in enhanced training state."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Test chemical accuracy tracking
        state.update_chemical_accuracy(2e-3)  # Above target
        state.update_chemical_accuracy(0.5e-3)  # Below target

        assert len(state.chemical_accuracy_history) == 2
        assert state.chemical_accuracy_history[0] == 2e-3
        assert state.chemical_accuracy_history[1] == 0.5e-3

        # Test convergence check based on chemical accuracy
        assert not state.is_converged(1.0)  # High loss, above target accuracy
        # Test with loss below threshold (1e-6) and chemical accuracy below target (1e-3)
        assert state.is_converged(1e-7)  # Very low loss, below target accuracy

    def test_enhanced_training_state_scf_convergence(self):
        """Test SCF convergence tracking in enhanced training state."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Test SCF convergence tracking
        state.update_scf_convergence(True, 15)
        state.update_scf_convergence(False, 100)
        state.update_scf_convergence(True, 8)

        assert len(state.scf_convergence_history) == 3
        assert state.scf_convergence_history[0] == (True, 15)
        assert state.scf_convergence_history[1] == (False, 100)
        assert state.scf_convergence_history[2] == (True, 8)

    def test_enhanced_training_state_diagnostics(self):
        """Test training diagnostics in enhanced training state."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Test gradient norm tracking
        state.update_gradient_norm(1.5)
        state.update_gradient_norm(0.8)
        assert len(state.gradient_norms) == 2
        assert state.gradient_norms[0] == 1.5

        # Test learning rate tracking
        state.update_learning_rate(1e-3)
        state.update_learning_rate(5e-4)
        assert len(state.learning_rates) == 2
        assert state.learning_rates[1] == 5e-4

        # Test wall time tracking
        state.update_wall_time(10.5)
        state.update_wall_time(25.3)
        assert len(state.wall_time_history) == 2
        assert state.wall_time_history[0] == 10.5

    def test_enhanced_training_state_physics_summary(self):
        """Test physics summary generation in enhanced training state."""
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Add various metrics
        state.update_physics_metric("energy_error", 0.1)
        state.update_physics_metric("energy_error", 0.05)
        state.update_conservation_violation("mass", 1e-6)
        state.update_conservation_violation("mass", 2e-6)
        state.update_chemical_accuracy(1e-3)
        state.update_chemical_accuracy(0.5e-3)
        state.update_scf_convergence(True, 15)
        state.update_scf_convergence(False, 100)
        state.update_scf_convergence(True, 8)

        summary = state.get_physics_summary()

        # Check physics metrics
        assert "latest_energy_error" in summary
        assert summary["latest_energy_error"] == 0.05
        assert "avg_energy_error" in summary
        assert abs(summary["avg_energy_error"] - 0.075) < 1e-10

        # Check conservation violations
        assert "max_mass_violation" in summary
        assert summary["max_mass_violation"] == 2e-6
        assert "avg_mass_violation" in summary
        assert summary["avg_mass_violation"] == 1.5e-6

        # Check chemical accuracy
        assert "latest_chemical_accuracy" in summary
        assert summary["latest_chemical_accuracy"] == 0.5e-3
        assert "best_chemical_accuracy" in summary
        assert summary["best_chemical_accuracy"] == 0.5e-3

        # Check SCF convergence
        assert "scf_convergence_rate" in summary
        assert abs(summary["scf_convergence_rate"] - 2 / 3) < 1e-10
        assert "avg_scf_iterations" in summary
        assert abs(summary["avg_scf_iterations"] - (15 + 100 + 8) / 3) < 1e-10


class TestAdvancedMetricsCollector:
    """Test advanced metrics collection."""

    def test_metrics_collector_creation(self):
        """Test advanced metrics collector initialization."""
        collector = AdvancedMetricsCollector()

        assert collector.training_start_time is None
        assert collector.epoch_start_time is None
        assert len(collector.metrics_history) == 0

    def test_timing_tracking(self):
        """Test timing tracking functionality."""
        collector = AdvancedMetricsCollector()

        collector.start_training()
        assert collector.training_start_time is not None

        collector.start_epoch()
        assert collector.epoch_start_time is not None

    def test_physics_metrics_collection(self):
        """Test physics metrics collection."""
        collector = AdvancedMetricsCollector()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        # Create dummy data
        x = jnp.ones((2, 4))
        y_true = jnp.ones((2, 1))

        metrics = collector.collect_physics_metrics(model, x, y_true)

        assert "mse_loss" in metrics
        assert "mae_loss" in metrics
        assert "max_error" in metrics

    def test_physics_metrics_with_energy_prediction(self):
        """Test chemical accuracy metric for energy predictions."""
        collector = AdvancedMetricsCollector()
        model = StandardMLP([4, 1], rngs=nnx.Rngs(42))

        # Create dummy energy data
        x = jnp.ones((2, 4))
        y_true = jnp.array([[0.1], [0.2]])  # Energy in Hartree

        metrics = collector.collect_physics_metrics(model, x, y_true)

        assert "chemical_accuracy" in metrics
        # Chemical accuracy should be in kcal/mol
        assert metrics["chemical_accuracy"] > 0

    def test_training_diagnostics_collection(self):
        """Test training diagnostics collection."""
        collector = AdvancedMetricsCollector()
        collector.start_epoch()

        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        # Create dummy gradients
        @nnx.value_and_grad
        def loss_fn(model_inner):
            x = jnp.ones((2, 4))
            y = model_inner(x)
            return jnp.mean(y**2)

        _, grads = loss_fn(model)

        metrics = collector.collect_training_diagnostics(
            model, grads, learning_rate=1e-3
        )

        assert "gradient_norm" in metrics
        assert "learning_rate" in metrics
        assert "parameter_norm" in metrics
        assert "epoch_time" in metrics
        assert metrics["learning_rate"] == 1e-3

    def test_convergence_metrics_collection(self):
        """Test convergence metrics collection."""
        collector = AdvancedMetricsCollector()
        model = StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))
        optimizer = optax.adam(1e-3)
        params = nnx.to_tree(nnx.state(model, nnx.Param))
        opt_state = optimizer.init(params)

        state = TrainingState(
            model=model,
            optimizer=optimizer,
            opt_state=opt_state,
        )

        # Add some history
        state.update_chemical_accuracy(2e-3)
        state.update_chemical_accuracy(0.5e-3)
        state.update_scf_convergence(True, 15)

        metrics = collector.collect_convergence_metrics(state)

        assert "is_converged" in metrics
        assert "best_loss" in metrics
        assert "best_val_loss" in metrics
        assert "best_chemical_accuracy" in metrics
        assert "current_chemical_accuracy" in metrics
        assert "scf_convergence_rate" in metrics

    def test_metrics_history_update(self):
        """Test metrics history updating."""
        collector = AdvancedMetricsCollector()

        new_metrics = {"loss": 0.5, "accuracy": 0.8}
        collector.update_metrics_history(new_metrics)

        assert "loss" in collector.metrics_history
        assert "accuracy" in collector.metrics_history
        assert len(collector.metrics_history["loss"]) == 1
        assert collector.metrics_history["loss"][0] == 0.5

        # Add more metrics
        new_metrics = {"loss": 0.3, "accuracy": 0.85}
        collector.update_metrics_history(new_metrics)

        assert len(collector.metrics_history["loss"]) == 2
        assert collector.metrics_history["loss"][1] == 0.3

    def test_metrics_summary(self):
        """Test metrics summary generation."""
        collector = AdvancedMetricsCollector()

        # Add some history
        for i in range(15):
            collector.update_metrics_history({"loss": 1.0 - i * 0.05})

        summary = collector.get_metrics_summary(window_size=10)

        assert "loss" in summary
        assert "current" in summary["loss"]
        assert "mean" in summary["loss"]
        assert "min" in summary["loss"]
        assert "max" in summary["loss"]
        assert "trend" in summary["loss"]

        # Check trend calculation (loss should be decreasing)
        assert summary["loss"]["trend"] < 0

    def test_hartree_to_kcal_mol_constant(self):
        """Test HARTREE_TO_KCAL_MOL constant value."""
        # Verify the conversion constant is correct
        assert abs(HARTREE_TO_KCAL_MOL - 627.50960803) < 1e-8

    def test_collect_quantum_metrics(self):
        """Test quantum metrics collection."""
        collector = AdvancedMetricsCollector()
        StandardMLP([4, 8, 1], rngs=nnx.Rngs(42))

        # Create dummy quantum data
        x = jnp.ones((2, 4))
        y_pred = jnp.array([[0.1], [0.2]])

        # Mock quantum config - simulate DFT functionality
        quantum_config = {
            "quantum_training": True,
            "dft_functional": "pbe",
            "track_quantum_states": True,
        }

        metrics = collector.collect_quantum_metrics(
            x=x, y_pred=y_pred, quantum_config=quantum_config
        )

        # Should collect DFT energy metrics
        assert "dft_energy" in metrics
        assert isinstance(metrics["dft_energy"], float)
        assert jnp.isfinite(metrics["dft_energy"])

        # Should collect quantum state metrics
        assert "quantum_state" in metrics
        assert isinstance(metrics["quantum_state"], float)
        assert jnp.isfinite(metrics["quantum_state"])

    def test_collect_quantum_metrics_disabled(self):
        """Test quantum metrics when quantum training is disabled."""
        collector = AdvancedMetricsCollector()

        # Create dummy data
        x = jnp.ones((2, 4))
        y_pred = jnp.array([[0.1], [0.2]])

        # No quantum config
        quantum_config = {"quantum_training": False}

        metrics = collector.collect_quantum_metrics(
            x=x, y_pred=y_pred, quantum_config=quantum_config
        )

        # Should return empty metrics when disabled
        assert len(metrics) == 0

    def test_collect_conservation_metrics(self):
        """Test conservation law metrics collection."""
        collector = AdvancedMetricsCollector()

        # Create dummy data
        x = jnp.ones((2, 4))
        y_pred = jnp.array([[0.1, 0.2], [0.3, 0.4]])
        y_true = jnp.array([[0.11, 0.19], [0.31, 0.39]])

        # Configure conservation laws
        conservation_config = {
            "conservation_laws": ["energy", "momentum", "symmetry"],
        }

        metrics = collector.collect_conservation_metrics(
            x=x, y_pred=y_pred, y_true=y_true, conservation_config=conservation_config
        )

        # Should collect metrics for each conservation law
        assert "energy_conservation" in metrics
        assert "momentum_conservation" in metrics
        assert "symmetry_conservation" in metrics

        # All metrics should be finite floats
        for key in [
            "energy_conservation",
            "momentum_conservation",
            "symmetry_conservation",
        ]:
            assert isinstance(metrics[key], float)
            assert jnp.isfinite(metrics[key])

    def test_collect_conservation_metrics_empty(self):
        """Test conservation metrics with no conservation laws configured."""
        collector = AdvancedMetricsCollector()

        # Create dummy data
        x = jnp.ones((2, 4))
        y_pred = jnp.array([[0.1], [0.2]])
        y_true = jnp.array([[0.11], [0.19]])

        # No conservation laws
        conservation_config = {"conservation_laws": []}

        metrics = collector.collect_conservation_metrics(
            x=x, y_pred=y_pred, y_true=y_true, conservation_config=conservation_config
        )

        # Should return empty metrics when no conservation laws
        assert len(metrics) == 0
