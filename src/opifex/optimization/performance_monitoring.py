"""Performance monitoring and prediction for Opifex production optimization.

This module implements AI-powered performance monitoring, anomaly detection,
and predictive scaling for the Phase 7.4 Production Optimization system.

Part of: Hybrid Performance Platform + Intelligent Edge + Adaptive Optimization
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import jax
import jax.numpy as jnp
from flax import nnx


class AnomalySeverity(Enum):
    """Severity levels for performance anomalies."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    """Performance anomaly detection result."""

    anomaly_id: str
    timestamp: float
    severity: AnomalySeverity
    anomaly_type: str
    description: str
    metrics: dict[str, float]
    confidence: float
    recommended_action: str


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for monitoring."""

    timestamp: float
    latency_ms: float
    throughput_rps: float
    memory_usage_gb: float
    gpu_utilization: float
    cpu_utilization: float
    energy_efficiency: float
    error_rate: float

    # Scientific computing specific metrics
    numerical_stability: float = 0.0
    conservation_score: float = 0.0
    physics_consistency: float = 0.0

    # Time series data for trend analysis
    time_series_data: dict[str, list[float]] = field(default_factory=dict)
    feature_matrix: jnp.ndarray | None = None

    # GPU-specific metrics
    gpu_metrics: dict[str, float] = field(default_factory=dict)
    conservation_metrics: dict[str, float] = field(default_factory=dict)
    numerical_metrics: dict[str, float] = field(default_factory=dict)
    physics_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of performance prediction."""

    predicted_latency: float
    predicted_throughput: float
    predicted_memory_usage: float
    confidence_interval: tuple[float, float]
    prediction_horizon_minutes: int
    recommended_scaling_action: str


class PerformanceMonitorProtocol(Protocol):
    """Protocol for performance monitoring implementations."""

    async def collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        ...

    async def analyze_trends(
        self, metrics_history: list[PerformanceMetrics]
    ) -> dict[str, Any]:
        """Analyze performance trends."""
        ...


class PerformancePredictor(nnx.Module):
    """Neural network-based performance predictor."""

    def __init__(
        self,
        input_features: int = 32,
        hidden_features: int = 128,
        prediction_horizon: int = 60,  # minutes
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.prediction_horizon = prediction_horizon

        # Time series encoder
        self.time_encoder = nnx.Sequential(
            nnx.Linear(input_features, hidden_features, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_features, hidden_features // 2, rngs=rngs),
            nnx.gelu,
        )

        # Prediction head
        self.predictor = nnx.Linear(
            hidden_features // 2, 3, rngs=rngs
        )  # latency, throughput, memory

    def __call__(self, performance_history: jnp.ndarray) -> jnp.ndarray:
        """Predict future performance metrics."""
        encoded = self.time_encoder(performance_history)
        return self.predictor(encoded)


class AIAnomalyDetector(nnx.Module):
    """AI-powered anomaly detection for scientific computing performance."""

    def __init__(
        self,
        input_features: int = 16,
        hidden_features: int = 64,
        anomaly_threshold: float = 0.8,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.anomaly_threshold = anomaly_threshold

        # Autoencoder for anomaly detection
        self.encoder = nnx.Sequential(
            nnx.Linear(input_features, hidden_features, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_features, hidden_features // 2, rngs=rngs),
        )

        self.decoder = nnx.Sequential(
            nnx.Linear(hidden_features // 2, hidden_features, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_features, input_features, rngs=rngs),
        )

    def __call__(self, metrics: jnp.ndarray) -> jnp.ndarray:
        """Detect anomalies in performance metrics."""
        encoded = self.encoder(metrics)
        return self.decoder(encoded)

    def detect_anomalies(self, metrics: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Detect anomalies and return reconstruction error."""
        reconstructed = self(metrics)
        reconstruction_error = jnp.mean((metrics - reconstructed) ** 2, axis=-1)
        is_anomaly = reconstruction_error > self.anomaly_threshold
        return is_anomaly, reconstruction_error


class PerformanceMonitor:
    """Real-time performance monitoring system."""

    def __init__(
        self,
        anomaly_detector: AIAnomalyDetector,
        performance_predictor: PerformancePredictor,
        collection_interval: float = 1.0,  # seconds
    ):
        self.anomaly_detector = anomaly_detector
        self.performance_predictor = performance_predictor
        self.collection_interval = collection_interval
        self.metrics_history: list[PerformanceMetrics] = []
        self.is_monitoring = False

    async def start_monitoring(self) -> None:
        """Start continuous performance monitoring."""
        self.is_monitoring = True

        while self.is_monitoring:
            try:
                # Collect current metrics
                current_metrics = await self.collect_current_metrics()
                self.metrics_history.append(current_metrics)

                # Keep only last 1000 metrics for memory efficiency
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]

                # Detect anomalies
                anomalies = await self.detect_performance_anomalies(current_metrics)

                # Generate predictions if we have enough history
                if len(self.metrics_history) >= 10:
                    predictions = await self.predict_future_performance()
                    await self.handle_predictions(predictions)

                # Handle any detected anomalies
                if anomalies:
                    await self.handle_anomalies(anomalies)

                await asyncio.sleep(self.collection_interval)

            except Exception:
                await asyncio.sleep(self.collection_interval)

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self.is_monitoring = False

    async def collect_current_metrics(self) -> PerformanceMetrics:
        """Collect current system performance metrics."""
        current_time = time.time()

        # Simulate metric collection (in real implementation, this would
        # collect from actual system monitoring)
        return PerformanceMetrics(
            timestamp=current_time,
            latency_ms=float(
                jax.random.normal(jax.random.PRNGKey(int(current_time)), ()) * 2 + 10
            ),
            throughput_rps=float(
                jax.random.normal(jax.random.PRNGKey(int(current_time + 1)), ()) * 20
                + 100
            ),
            memory_usage_gb=float(
                jax.random.normal(jax.random.PRNGKey(int(current_time + 2)), ()) * 0.5
                + 2.0
            ),
            gpu_utilization=float(
                jax.random.uniform(jax.random.PRNGKey(int(current_time + 3)), ()) * 0.3
                + 0.7
            ),
            cpu_utilization=float(
                jax.random.uniform(jax.random.PRNGKey(int(current_time + 4)), ()) * 0.4
                + 0.5
            ),
            energy_efficiency=float(
                jax.random.uniform(jax.random.PRNGKey(int(current_time + 5)), ()) * 0.2
                + 0.8
            ),
            error_rate=float(
                jax.random.uniform(jax.random.PRNGKey(int(current_time + 6)), ()) * 0.01
            ),
            numerical_stability=float(
                jax.random.uniform(jax.random.PRNGKey(int(current_time + 7)), ()) * 0.1
                + 0.9
            ),
            conservation_score=float(
                jax.random.uniform(jax.random.PRNGKey(int(current_time + 8)), ()) * 0.1
                + 0.9
            ),
            physics_consistency=float(
                jax.random.uniform(jax.random.PRNGKey(int(current_time + 9)), ()) * 0.1
                + 0.9
            ),
        )

    async def detect_performance_anomalies(
        self, metrics: PerformanceMetrics
    ) -> list[Anomaly]:
        """Detect performance anomalies using AI models."""
        anomalies = []

        # Create feature vector from metrics
        feature_vector = jnp.array(
            [
                metrics.latency_ms,
                metrics.throughput_rps,
                metrics.memory_usage_gb,
                metrics.gpu_utilization,
                metrics.cpu_utilization,
                metrics.energy_efficiency,
                metrics.error_rate,
                metrics.numerical_stability,
                metrics.conservation_score,
                metrics.physics_consistency,
            ]
        )

        # Pad to match expected input size
        if feature_vector.shape[0] < 16:
            padding = jnp.zeros(16 - feature_vector.shape[0])
            feature_vector = jnp.concatenate([feature_vector, padding])

        # Detect anomalies
        is_anomaly, reconstruction_error = self.anomaly_detector.detect_anomalies(
            feature_vector.reshape(1, -1)
        )

        if is_anomaly[0]:
            # Determine severity based on reconstruction error
            error_value = float(reconstruction_error[0])
            if error_value > 2.0:
                severity = AnomalySeverity.CRITICAL
            elif error_value > 1.5:
                severity = AnomalySeverity.HIGH
            elif error_value > 1.0:
                severity = AnomalySeverity.MEDIUM
            else:
                severity = AnomalySeverity.LOW

            anomaly = Anomaly(
                anomaly_id=f"anomaly_{int(metrics.timestamp)}",
                timestamp=metrics.timestamp,
                severity=severity,
                anomaly_type="performance_anomaly",
                description=(
                    f"Performance anomaly detected with error {error_value:.3f}"
                ),
                metrics={
                    "latency_ms": metrics.latency_ms,
                    "throughput_rps": metrics.throughput_rps,
                    "memory_usage_gb": metrics.memory_usage_gb,
                },
                confidence=min(error_value / 2.0, 1.0),
                recommended_action="investigate_performance_degradation",
            )
            anomalies.append(anomaly)

        return anomalies

    async def predict_future_performance(self) -> PredictionResult:
        """Predict future performance metrics."""
        if len(self.metrics_history) < 10:
            raise ValueError("Need at least 10 metrics for prediction")

        # Prepare input data
        recent_metrics = self.metrics_history[-10:]
        feature_matrix = jnp.array(
            [
                [
                    m.latency_ms,
                    m.throughput_rps,
                    m.memory_usage_gb,
                    m.gpu_utilization,
                    m.cpu_utilization,
                    m.energy_efficiency,
                    m.error_rate,
                    m.numerical_stability,
                    m.conservation_score,
                    m.physics_consistency,
                ]
                for m in recent_metrics
            ]
        )

        # Pad to match expected input size
        if feature_matrix.shape[1] < 32:
            padding = jnp.zeros((feature_matrix.shape[0], 32 - feature_matrix.shape[1]))
            feature_matrix = jnp.concatenate([feature_matrix, padding], axis=1)

        # Make prediction
        predictions = self.performance_predictor(feature_matrix[-1:])
        pred_latency, pred_throughput, pred_memory = predictions[0]

        # Ensure positive predictions for throughput and memory
        pred_latency = float(jnp.maximum(pred_latency, 0.1))  # Minimum 0.1ms latency
        pred_throughput = float(jnp.maximum(pred_throughput, 0.1))  # Minimum 0.1 RPS
        pred_memory = float(jnp.maximum(pred_memory, 0.1))  # Minimum 0.1GB memory

        # Generate confidence interval (simplified)
        confidence_interval = (float(pred_latency * 0.9), float(pred_latency * 1.1))

        # Determine recommended scaling action
        current_latency = recent_metrics[-1].latency_ms
        if pred_latency > current_latency * 1.2:
            recommended_action = "scale_up"
        elif pred_latency < current_latency * 0.8:
            recommended_action = "scale_down"
        else:
            recommended_action = "maintain"

        return PredictionResult(
            predicted_latency=pred_latency,
            predicted_throughput=pred_throughput,
            predicted_memory_usage=pred_memory,
            confidence_interval=confidence_interval,
            prediction_horizon_minutes=self.performance_predictor.prediction_horizon,
            recommended_scaling_action=recommended_action,
        )

    async def handle_anomalies(self, anomalies: list[Anomaly]) -> None:
        """Handle detected anomalies."""
        for _anomaly in anomalies:
            pass
            # In real implementation, this would trigger alerts, notifications, etc.

    async def handle_predictions(self, predictions: PredictionResult) -> None:
        """Handle performance predictions."""
        if predictions.recommended_scaling_action != "maintain":
            pass
            # In real implementation, this would trigger auto-scaling actions


class PredictiveScaler:
    """Predictive scaling engine for scientific workloads."""

    def __init__(
        self,
        performance_monitor: PerformanceMonitor,
        scale_up_threshold: float = 1.2,
        scale_down_threshold: float = 0.8,
        min_replicas: int = 1,
        max_replicas: int = 10,
    ):
        self.performance_monitor = performance_monitor
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.current_replicas = 1

    async def evaluate_scaling_decision(self) -> dict[str, Any]:
        """Evaluate whether scaling is needed based on predictions."""
        try:
            predictions = await self.performance_monitor.predict_future_performance()

            # Get current performance
            current_metrics = self.performance_monitor.metrics_history[-1]

            # Calculate scaling decision
            latency_ratio = predictions.predicted_latency / current_metrics.latency_ms

            scaling_decision = {
                "action": "maintain",
                "target_replicas": self.current_replicas,
                "reason": "Performance within acceptable bounds",
                "confidence": 0.0,
            }

            if latency_ratio > self.scale_up_threshold:
                new_replicas = min(self.current_replicas + 1, self.max_replicas)
                scaling_decision.update(
                    {
                        "action": "scale_up",
                        "target_replicas": new_replicas,
                        "reason": f"Predicted latency increase of {latency_ratio:.2f}x",
                        "confidence": min(latency_ratio - 1.0, 1.0),
                    }
                )
            elif latency_ratio < self.scale_down_threshold:
                new_replicas = max(self.current_replicas - 1, self.min_replicas)
                scaling_decision.update(
                    {
                        "action": "scale_down",
                        "target_replicas": new_replicas,
                        "reason": f"Predicted latency decrease of {latency_ratio:.2f}x",
                        "confidence": min(1.0 - latency_ratio, 1.0),
                    }
                )

            return scaling_decision

        except Exception as e:
            return {
                "action": "maintain",
                "target_replicas": self.current_replicas,
                "reason": f"Prediction failed: {e}",
                "confidence": 0.0,
            }

    async def execute_scaling_action(self, scaling_decision: dict[str, Any]) -> bool:
        """Execute the scaling action."""
        if scaling_decision["action"] == "maintain":
            return True

        try:
            target_replicas = scaling_decision["target_replicas"]

            # In real implementation, this would call Kubernetes API or other
            # orchestration
            self.current_replicas = target_replicas

            return True

        except Exception:
            return False
