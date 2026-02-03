"""Performance monitoring and analytics for meta-optimization.

This module provides comprehensive performance monitoring capabilities
including metric tracking, convergence detection, and performance
analytics for optimization algorithms.

Author: Opifex Framework Team
Date: December 2024
License: MIT
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


class PerformanceMonitor:
    """Performance monitoring and analytics for meta-optimization.

    This class provides comprehensive performance monitoring capabilities
    including metric tracking, convergence detection, and performance
    analytics for optimization algorithms.

    Attributes:
        metrics: List of metrics to track
        window_size: Size of rolling window for metrics
        tracking_frequency: Frequency of metric updates
        convergence_tolerance: Tolerance for convergence detection
        convergence_patience: Patience for convergence detection
        analytics_enabled: Enable detailed analytics
        quantum_aware: Enable quantum-specific metrics
    """

    def __init__(
        self,
        metrics: list[str] | None = None,
        window_size: int = 100,
        tracking_frequency: int = 1,
        convergence_tolerance: float = 1e-6,
        convergence_patience: int = 10,
        analytics_enabled: bool = False,
        quantum_aware: bool = False,
    ):
        """Initialize performance monitor.

        Args:
            metrics: List of metrics to track
            window_size: Rolling window size
            tracking_frequency: How often to update metrics
            convergence_tolerance: Tolerance for convergence
            convergence_patience: Patience for convergence detection
            analytics_enabled: Enable detailed analytics
            quantum_aware: Enable quantum metrics
        """
        if metrics is None:
            metrics = ["loss", "gradient_norm"]

        self.metrics = metrics
        self.window_size = window_size
        self.tracking_frequency = tracking_frequency
        self.convergence_tolerance = convergence_tolerance
        self.convergence_patience = convergence_patience
        self.analytics_enabled = analytics_enabled
        self.quantum_aware = quantum_aware

        # Metric storage
        self._metric_history = {metric: [] for metric in metrics}
        self._step_history = []

        # Convergence tracking
        self._convergence_state = dict.fromkeys(metrics, False)
        self._steps_since_improvement = dict.fromkeys(metrics, 0)
        self._best_values = {metric: float("inf") for metric in metrics}

    def update_metrics(self, step: int, **metric_values: float) -> None:
        """Update tracked metrics.

        Args:
            step: Current optimization step
            **metric_values: Metric values to update
        """
        if step % self.tracking_frequency != 0:
            return

        self._step_history.append(step)

        for metric, value in metric_values.items():
            if metric in self.metrics:
                self._metric_history[metric].append(float(value))

                # Keep only recent history
                if len(self._metric_history[metric]) > self.window_size:
                    self._metric_history[metric] = self._metric_history[metric][
                        -self.window_size :
                    ]

                # Update convergence tracking
                self._update_convergence_tracking(metric, value)

    def _update_convergence_tracking(self, metric: str, value: float) -> None:
        """Update convergence tracking for a metric."""
        if value < self._best_values[metric] - self.convergence_tolerance:
            self._best_values[metric] = value
            self._steps_since_improvement[metric] = 0
        else:
            self._steps_since_improvement[metric] += 1

    def get_metric_history(self, metric: str) -> list[float]:
        """Get history of a specific metric.

        Args:
            metric: Metric name

        Returns:
            List of metric values
        """
        return self._metric_history.get(metric, [])

    def check_convergence(self, metric: str) -> bool:
        """Check if a metric has converged.

        Args:
            metric: Metric name

        Returns:
            True if metric has converged
        """
        if metric not in self._metric_history:
            return False

        history = self._metric_history[metric]
        if len(history) < self.convergence_patience:
            return False

        # Check if recent values are stable
        recent_values = history[-self.convergence_patience :]
        value_range = max(recent_values) - min(recent_values)

        return value_range < self.convergence_tolerance

    def get_performance_analytics(self) -> dict[str, Any]:
        """Get comprehensive performance analytics.

        Returns:
            Dictionary containing performance analytics
        """
        if not self.analytics_enabled:
            return {}

        analytics = {}

        for metric in self.metrics:
            history = self._metric_history[metric]
            if not history:
                continue

            # Basic statistics
            analytics[f"{metric}_mean"] = jnp.mean(jnp.array(history))
            analytics[f"{metric}_std"] = jnp.std(jnp.array(history))
            analytics[f"{metric}_min"] = min(history)
            analytics[f"{metric}_max"] = max(history)

            # Convergence analysis
            if len(history) > 10:
                # Simple convergence rate estimation
                recent_slope = (history[-1] - history[-10]) / 10
                analytics[f"{metric}_convergence_rate"] = recent_slope

                # Stability analysis
                recent_history = history[-min(20, len(history)) :]
                stability = 1.0 / (1.0 + jnp.var(jnp.array(recent_history)))
                analytics[f"{metric}_stability"] = float(stability)

        # Overall optimization efficiency
        if self._metric_history.get("loss"):
            loss_history = self._metric_history["loss"]
            if len(loss_history) > 1:
                total_improvement = loss_history[0] - loss_history[-1]
                steps_taken = len(loss_history)
                efficiency = total_improvement / steps_taken if steps_taken > 0 else 0.0
                analytics["optimization_efficiency"] = efficiency

        analytics["convergence_rate"] = sum(self._convergence_state.values()) / len(
            self.metrics
        )
        analytics["stability_metrics"] = {
            metric: self._steps_since_improvement[metric] < self.convergence_patience
            for metric in self.metrics
        }

        return analytics

    def get_quantum_analytics(self) -> dict[str, Any]:
        """Get quantum-specific performance analytics.

        Returns:
            Dictionary containing quantum analytics
        """
        if not self.quantum_aware:
            return {}

        quantum_analytics = {}

        # SCF efficiency analysis
        if "scf_iterations" in self._metric_history:
            scf_history = self._metric_history["scf_iterations"]
            if scf_history:
                avg_scf_iters = jnp.mean(jnp.array(scf_history))
                scf_trend = (
                    jnp.diff(jnp.array(scf_history[-10:]))
                    if len(scf_history) > 1
                    else [0]
                )
                quantum_analytics["scf_efficiency"] = {
                    "average_iterations": float(avg_scf_iters),
                    "recent_trend": float(jnp.mean(jnp.array(scf_trend))),
                    "acceleration": float(
                        jnp.mean(jnp.array(scf_trend)) < 0
                    ),  # Decreasing is good
                }

        # Energy convergence analysis
        if "energy_error" in self._metric_history:
            energy_history = self._metric_history["energy_error"]
            if energy_history:
                convergence_rate = (
                    (energy_history[0] - energy_history[-1]) / len(energy_history)
                    if len(energy_history) > 1
                    else 0
                )
                quantum_analytics["energy_convergence_rate"] = float(convergence_rate)

        # Chemical accuracy tracking
        if "chemical_accuracy" in self._metric_history:
            accuracy_history = self._metric_history["chemical_accuracy"]
            if accuracy_history:
                target_achieved = [
                    acc < 1e-3 for acc in accuracy_history
                ]  # 1 kcal/mol target
                quantum_analytics["chemical_accuracy_progress"] = {
                    "target_achieved_ratio": sum(target_achieved)
                    / len(target_achieved),
                    "best_accuracy": min(accuracy_history),
                    "current_accuracy": accuracy_history[-1],
                }

        return quantum_analytics


__all__ = ["PerformanceMonitor"]
