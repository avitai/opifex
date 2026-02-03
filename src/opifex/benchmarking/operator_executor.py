"""Operator Executor - Runs actual Opifex operators for benchmarking.

This module replaces the mock execution in BenchmarkRunner with real
operator training and evaluation.
"""

import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import optax
from calibrax.core import BenchmarkResult
from calibrax.core.models import Metric
from calibrax.metrics import (
    mae as calc_mae,
    mse as calc_mse,
    relative_error as calc_relative_error,
)
from flax import nnx


def _prepare_input(x: jnp.ndarray) -> jnp.ndarray:
    """Prepare input for FNO: ensure (batch, C, H, W) format."""
    if x.ndim == 3:
        # (batch, H, W) -> (batch, 1, H, W)
        return x[:, jnp.newaxis, :, :]
    if x.ndim == 4:
        # Check if channel-last: (batch, H, W, C) -> (batch, C, H, W)
        if x.shape[-1] <= 4 and x.shape[1] > 4:
            return jnp.transpose(x, (0, 3, 1, 2))
        return x
    return x


def _prepare_target(y: jnp.ndarray, pred: jnp.ndarray) -> jnp.ndarray:
    """Prepare target to match prediction shape."""
    if y.ndim == 3 and pred.ndim == 4:
        # Add channel dim to match
        return y[:, jnp.newaxis, :, :]
    return y


@dataclass(frozen=True, slots=True, kw_only=True)
class ExecutionConfig:
    """Configuration for benchmark execution."""

    n_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    warmup_steps: int = 5
    eval_frequency: int = 10
    use_mixed_precision: bool = False
    seed: int = 42


class OperatorExecutor:
    """Executes actual Opifex operators for benchmarking.

    This class provides the core execution logic that was missing from
    the original BenchmarkRunner implementation. It uses:
    - Real Opifex operators (TFNO, DeepONet, etc.)
    - Real Opifex data loaders (create_darcy_loader, etc.)
    - Flax NNX 0.11.0+ optimizer pattern
    - calibrax.metrics for evaluation (DRY)
    """

    def __init__(self, config: ExecutionConfig | None = None) -> None:
        """Initialize executor with configuration.

        Args:
            config: Execution configuration. Uses defaults if None.
        """
        self.config = config or ExecutionConfig()

    def execute_training_benchmark(
        self,
        operator_class: type,
        operator_config: dict[str, Any],
        train_loader: Any,
        test_loader: Any,
        benchmark_name: str,
    ) -> BenchmarkResult:
        """Execute a training benchmark with actual operator.

        Args:
            operator_class: Opifex operator class to instantiate
            operator_config: Configuration dict for operator
            train_loader: Training data loader (from opifex.data.loaders)
            test_loader: Test data loader
            benchmark_name: Name of benchmark for results

        Returns:
            BenchmarkResult with real metrics from training
        """
        rngs = nnx.Rngs(self.config.seed)

        # Instantiate operator
        model = self._create_operator(operator_class, operator_config, rngs)

        # Create optimizer (Flax NNX 0.11.0+ pattern)
        opt = nnx.Optimizer(
            model,
            optax.adam(self.config.learning_rate),
            wrt=nnx.Param,
        )

        # Training loop
        start_time = time.perf_counter()
        train_metrics = self._train_loop(model, opt, train_loader)
        training_time = time.perf_counter() - start_time

        # Evaluation
        eval_metrics = self._evaluate(model, test_loader)

        all_metrics = {**train_metrics, **eval_metrics}
        return BenchmarkResult(
            name=operator_class.__name__,
            domain="scientific_ml",
            tags={"dataset": benchmark_name},
            metrics={k: Metric(value=v) for k, v in all_metrics.items()},
            metadata={
                "execution_time": training_time,
                "framework_version": "flax_nnx",
            },
        )

    def _create_operator(
        self,
        operator_class: type,
        config: dict[str, Any],
        rngs: nnx.Rngs,
    ) -> nnx.Module:
        """Create operator instance with proper configuration.

        Args:
            operator_class: Operator class to instantiate
            config: Configuration dictionary for operator
            rngs: Flax NNX random number generators

        Returns:
            Instantiated operator module

        Raises:
            TypeError: If operator cannot be instantiated
        """
        try:
            return operator_class(**config, rngs=rngs)
        except TypeError as e:
            raise TypeError(
                f"Failed to create operator {operator_class.__name__}: {e}. "
                f"Config provided: {config}"
            ) from e

    def _train_loop(
        self,
        model: nnx.Module,
        opt: nnx.Optimizer,
        train_loader: Any,
    ) -> dict[str, float]:
        """Execute training loop and return metrics.

        Uses the correct Flax NNX 0.11.0+ optimizer pattern:
        - opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)
        - opt.update(model, grads)

        Args:
            model: Neural operator model
            opt: Flax NNX optimizer
            train_loader: Training data loader

        Returns:
            Dictionary with training metrics (initial_train_loss, final_train_loss)
        """

        @jax.jit
        def train_step(model, x, y):
            """Single training step - returns loss and grads."""

            def loss_fn(m):
                x_input = _prepare_input(x)
                pred = m(x_input)
                y_target = _prepare_target(y, pred)
                return jnp.mean((pred - y_target) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(model)
            return loss, grads

        losses = []
        for _ in range(self.config.n_epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                x, y = batch["input"], batch["output"]

                loss, grads = train_step(model, x, y)
                opt.update(model, grads)

                epoch_loss += float(loss)
                n_batches += 1

            if n_batches > 0:
                losses.append(epoch_loss / n_batches)

        return {
            "final_train_loss": losses[-1] if losses else 0.0,
            "initial_train_loss": losses[0] if losses else 0.0,
        }

    def _evaluate(
        self,
        model: nnx.Module,
        test_loader: Any,
    ) -> dict[str, float]:
        """Evaluate model and return metrics.

        DRY: Uses calibrax.metrics functions.

        Args:
            model: Trained neural operator
            test_loader: Test data loader

        Returns:
            Dictionary with evaluation metrics (mse, mae, relative_error)
        """

        all_preds = []
        all_targets = []

        for batch in test_loader:
            x, y = batch["input"], batch["output"]

            x_input = _prepare_input(x)
            pred = model(x_input)  # type: ignore[operator]  # nnx.Module is callable
            y_target = _prepare_target(y, pred)

            all_preds.append(pred)
            all_targets.append(y_target)

        if not all_preds:
            return {"mse": 0.0, "mae": 0.0, "relative_error": 0.0}

        preds = jnp.concatenate(all_preds, axis=0)
        targets = jnp.concatenate(all_targets, axis=0)

        return {
            "mse": calc_mse(preds, targets),
            "mae": calc_mae(preds, targets),
            "relative_error": calc_relative_error(preds, targets),
        }


__all__ = ["ExecutionConfig", "OperatorExecutor"]
