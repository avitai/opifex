"""Tests for performance monitoring and prediction components.

This module tests the Phase 7.4 Performance Monitoring & Prediction implementation
including AI-powered anomaly detection and predictive scaling.
"""

import asyncio
import contextlib
import time

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.optimization.performance_monitoring import (
    AIAnomalyDetector,
    Anomaly,
    AnomalySeverity,
    PerformanceMetrics,
    PerformanceMonitor,
    PerformancePredictor,
    PredictionResult,
    PredictiveScaler,
)


@pytest.fixture
def sample_performance_metrics():
    """Create sample performance metrics for testing."""
    return PerformanceMetrics(
        timestamp=time.time(),
        latency_ms=10.5,
        throughput_rps=150.0,
        memory_usage_gb=2.5,
        gpu_utilization=0.85,
        cpu_utilization=0.60,
        energy_efficiency=0.92,
        error_rate=0.001,
        numerical_stability=0.95,
        conservation_score=0.98,
        physics_consistency=0.97,
    )


@pytest.fixture
def anomaly_detector():
    """Create an AI anomaly detector for testing."""
    rngs = nnx.Rngs(0)
    return AIAnomalyDetector(
        input_features=16,
        hidden_features=32,
        anomaly_threshold=0.5,
        rngs=rngs,
    )


@pytest.fixture
def performance_predictor():
    """Create a performance predictor for testing."""
    rngs = nnx.Rngs(0)
    return PerformancePredictor(
        input_features=32,
        hidden_features=64,
        prediction_horizon=30,
        rngs=rngs,
    )


@pytest.fixture
def performance_monitor(anomaly_detector, performance_predictor):
    """Create a performance monitor for testing."""
    return PerformanceMonitor(
        anomaly_detector=anomaly_detector,
        performance_predictor=performance_predictor,
        collection_interval=0.1,  # Faster for testing
    )


class TestPerformanceMetrics:
    """Test PerformanceMetrics data structure."""

    def test_performance_metrics_creation(self, sample_performance_metrics):
        """Test basic performance metrics creation."""
        metrics = sample_performance_metrics

        assert metrics.latency_ms == 10.5
        assert metrics.throughput_rps == 150.0
        assert metrics.memory_usage_gb == 2.5
        assert metrics.gpu_utilization == 0.85
        assert metrics.cpu_utilization == 0.60
        assert metrics.energy_efficiency == 0.92
        assert metrics.error_rate == 0.001
        assert metrics.numerical_stability == 0.95
        assert metrics.conservation_score == 0.98
        assert metrics.physics_consistency == 0.97

    def test_performance_metrics_scientific_fields(self, sample_performance_metrics):
        """Test scientific computing specific fields."""
        metrics = sample_performance_metrics

        # Test scientific metrics
        assert metrics.numerical_stability > 0.9
        assert metrics.conservation_score > 0.9
        assert metrics.physics_consistency > 0.9

        # Test default collections
        assert isinstance(metrics.time_series_data, dict)
        assert isinstance(metrics.gpu_metrics, dict)
        assert isinstance(metrics.conservation_metrics, dict)
        assert isinstance(metrics.numerical_metrics, dict)
        assert isinstance(metrics.physics_metrics, dict)


class TestAnomaly:
    """Test Anomaly data structure."""

    def test_anomaly_creation(self):
        """Test basic anomaly creation."""
        anomaly = Anomaly(
            anomaly_id="test_anomaly_001",
            timestamp=time.time(),
            severity=AnomalySeverity.HIGH,
            anomaly_type="performance_anomaly",
            description="Test anomaly for validation",
            metrics={"latency_ms": 25.0, "throughput_rps": 50.0},
            confidence=0.95,
            recommended_action="investigate_performance",
        )

        assert anomaly.anomaly_id == "test_anomaly_001"
        assert anomaly.severity == AnomalySeverity.HIGH
        assert anomaly.anomaly_type == "performance_anomaly"
        assert anomaly.confidence == 0.95
        assert "latency_ms" in anomaly.metrics
        assert anomaly.recommended_action == "investigate_performance"


class TestAnomalySeverity:
    """Test AnomalySeverity enum."""

    def test_anomaly_severity_values(self):
        """Test all anomaly severity levels."""
        assert AnomalySeverity.LOW.value == "low"
        assert AnomalySeverity.MEDIUM.value == "medium"
        assert AnomalySeverity.HIGH.value == "high"
        assert AnomalySeverity.CRITICAL.value == "critical"


class TestPerformancePredictor:
    """Test PerformancePredictor neural network."""

    def test_predictor_initialization(self, performance_predictor):
        """Test performance predictor initialization."""
        assert performance_predictor.prediction_horizon == 30
        assert hasattr(performance_predictor, "time_encoder")
        assert hasattr(performance_predictor, "predictor")

    def test_predictor_forward_pass(self, performance_predictor):
        """Test performance predictor forward pass."""
        # Create sample input
        input_data = jnp.ones((1, 32))

        # Forward pass
        predictions = performance_predictor(input_data)

        # Check output shape and content
        assert predictions.shape == (1, 3)  # latency, throughput, memory
        assert jnp.all(jnp.isfinite(predictions))

    def test_predictor_batch_processing(self, performance_predictor):
        """Test performance predictor with multiple samples."""
        # Create batch input
        batch_input = jnp.ones((5, 32))

        # Forward pass
        predictions = performance_predictor(batch_input)

        # Check batch output
        assert predictions.shape == (5, 3)
        assert jnp.all(jnp.isfinite(predictions))


class TestAIAnomalyDetector:
    """Test AIAnomalyDetector neural network."""

    def test_anomaly_detector_initialization(self, anomaly_detector):
        """Test anomaly detector initialization."""
        assert anomaly_detector.anomaly_threshold == 0.5
        assert hasattr(anomaly_detector, "encoder")
        assert hasattr(anomaly_detector, "decoder")

    def test_anomaly_detector_forward_pass(self, anomaly_detector):
        """Test anomaly detector forward pass."""
        # Create sample input
        input_data = jnp.ones((1, 16))

        # Forward pass
        reconstructed = anomaly_detector(input_data)

        # Check output
        assert reconstructed.shape == (1, 16)
        assert jnp.all(jnp.isfinite(reconstructed))

    def test_anomaly_detection(self, anomaly_detector):
        """Test anomaly detection functionality."""
        # Normal data
        normal_data = jnp.ones((1, 16)) * 0.5

        # Detect anomalies
        is_anomaly, reconstruction_error = anomaly_detector.detect_anomalies(
            normal_data
        )

        # Check results
        assert is_anomaly.shape == (1,)
        assert reconstruction_error.shape == (1,)
        assert jnp.all(jnp.isfinite(reconstruction_error))

    def test_anomaly_detection_threshold(self, anomaly_detector):
        """Test anomaly detection threshold behavior."""
        # Create data that should trigger high reconstruction error
        anomalous_data = jnp.ones((1, 16)) * 100.0  # Very different from training

        _, reconstruction_error = anomaly_detector.detect_anomalies(anomalous_data)

        # High reconstruction error should be detected
        assert reconstruction_error[0] > 0.0


class TestPredictionResult:
    """Test PredictionResult data structure."""

    def test_prediction_result_creation(self):
        """Test basic prediction result creation."""
        result = PredictionResult(
            predicted_latency=12.5,
            predicted_throughput=180.0,
            predicted_memory_usage=3.2,
            confidence_interval=(11.0, 14.0),
            prediction_horizon_minutes=60,
            recommended_scaling_action="scale_up",
        )

        assert result.predicted_latency == 12.5
        assert result.predicted_throughput == 180.0
        assert result.predicted_memory_usage == 3.2
        assert result.confidence_interval == (11.0, 14.0)
        assert result.prediction_horizon_minutes == 60
        assert result.recommended_scaling_action == "scale_up"


class TestPerformanceMonitor:
    """Test PerformanceMonitor system."""

    def test_monitor_initialization(self, performance_monitor):
        """Test performance monitor initialization."""
        assert performance_monitor.collection_interval == 0.1
        assert len(performance_monitor.metrics_history) == 0
        assert not performance_monitor.is_monitoring

    @pytest.mark.asyncio
    async def test_collect_current_metrics(self, performance_monitor):
        """Test metrics collection."""
        metrics = await performance_monitor.collect_current_metrics()

        assert isinstance(metrics, PerformanceMetrics)
        assert metrics.timestamp > 0
        assert metrics.latency_ms > 0
        assert metrics.throughput_rps > 0
        assert 0 <= metrics.gpu_utilization <= 1
        assert 0 <= metrics.cpu_utilization <= 1

    @pytest.mark.asyncio
    async def test_detect_performance_anomalies(
        self, performance_monitor, sample_performance_metrics
    ):
        """Test anomaly detection."""
        anomalies = await performance_monitor.detect_performance_anomalies(
            sample_performance_metrics
        )

        assert isinstance(anomalies, list)
        # Should be able to detect anomalies (or not) without errors
        for anomaly in anomalies:
            assert isinstance(anomaly, Anomaly)
            assert anomaly.severity in AnomalySeverity

    @pytest.mark.asyncio
    async def test_predict_future_performance_insufficient_data(
        self, performance_monitor
    ):
        """Test prediction with insufficient data."""
        # Add only a few metrics (less than 10)
        for _ in range(5):
            metrics = await performance_monitor.collect_current_metrics()
            performance_monitor.metrics_history.append(metrics)

        with pytest.raises(ValueError, match="Need at least 10 metrics"):
            await performance_monitor.predict_future_performance()

    @pytest.mark.asyncio
    async def test_predict_future_performance_sufficient_data(
        self, performance_monitor
    ):
        """Test prediction with sufficient data."""
        # Add enough metrics for prediction
        for _ in range(15):
            metrics = await performance_monitor.collect_current_metrics()
            performance_monitor.metrics_history.append(metrics)
            await asyncio.sleep(0.01)  # Small delay to vary timestamps

        prediction = await performance_monitor.predict_future_performance()

        assert isinstance(prediction, PredictionResult)
        assert prediction.predicted_latency > 0
        assert prediction.predicted_throughput > 0
        assert prediction.predicted_memory_usage > 0
        assert len(prediction.confidence_interval) == 2
        assert prediction.recommended_scaling_action in [
            "scale_up",
            "scale_down",
            "maintain",
        ]

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self, performance_monitor):
        """Test starting and stopping monitoring."""
        # Start monitoring in background
        monitor_task = asyncio.create_task(performance_monitor.start_monitoring())

        # Wait a short time for some metrics collection
        await asyncio.sleep(0.3)

        # Check that monitoring is active and metrics are collected
        assert performance_monitor.is_monitoring
        assert len(performance_monitor.metrics_history) > 0

        # Stop monitoring
        await performance_monitor.stop_monitoring()

        # Wait for the task to complete
        await asyncio.sleep(0.1)

        assert not performance_monitor.is_monitoring

        # Cancel the task to clean up
        monitor_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await monitor_task


class TestPredictiveScaler:
    """Test PredictiveScaler system."""

    def test_scaler_initialization(self, performance_monitor):
        """Test predictive scaler initialization."""
        scaler = PredictiveScaler(
            performance_monitor=performance_monitor,
            scale_up_threshold=1.3,
            scale_down_threshold=0.7,
            min_replicas=2,
            max_replicas=20,
        )

        assert scaler.performance_monitor == performance_monitor
        assert scaler.scale_up_threshold == 1.3
        assert scaler.scale_down_threshold == 0.7
        assert scaler.min_replicas == 2
        assert scaler.max_replicas == 20
        assert scaler.current_replicas == 1

    @pytest.mark.asyncio
    async def test_evaluate_scaling_decision_insufficient_data(
        self, performance_monitor
    ):
        """Test scaling decision with insufficient data."""
        scaler = PredictiveScaler(performance_monitor=performance_monitor)

        decision = await scaler.evaluate_scaling_decision()

        assert decision["action"] == "maintain"
        assert decision["target_replicas"] == 1
        assert "failed" in decision["reason"].lower()

    @pytest.mark.asyncio
    async def test_evaluate_scaling_decision_sufficient_data(self, performance_monitor):
        """Test scaling decision with sufficient data."""
        # Add enough metrics for prediction
        for _ in range(15):
            metrics = await performance_monitor.collect_current_metrics()
            performance_monitor.metrics_history.append(metrics)
            await asyncio.sleep(0.01)

        scaler = PredictiveScaler(performance_monitor=performance_monitor)
        decision = await scaler.evaluate_scaling_decision()

        assert "action" in decision
        assert "target_replicas" in decision
        assert "reason" in decision
        assert "confidence" in decision
        assert decision["action"] in ["scale_up", "scale_down", "maintain"]

    @pytest.mark.asyncio
    async def test_execute_scaling_action_maintain(self, performance_monitor):
        """Test executing maintain scaling action."""
        scaler = PredictiveScaler(performance_monitor=performance_monitor)

        decision = {
            "action": "maintain",
            "target_replicas": 1,
            "reason": "Performance stable",
            "confidence": 0.0,
        }

        result = await scaler.execute_scaling_action(decision)
        assert result is True
        assert scaler.current_replicas == 1

    @pytest.mark.asyncio
    async def test_execute_scaling_action_scale_up(self, performance_monitor):
        """Test executing scale up action."""
        scaler = PredictiveScaler(performance_monitor=performance_monitor)

        decision = {
            "action": "scale_up",
            "target_replicas": 3,
            "reason": "High latency predicted",
            "confidence": 0.8,
        }

        result = await scaler.execute_scaling_action(decision)
        assert result is True
        assert scaler.current_replicas == 3

    @pytest.mark.asyncio
    async def test_execute_scaling_action_scale_down(self, performance_monitor):
        """Test executing scale down action."""
        scaler = PredictiveScaler(performance_monitor=performance_monitor)
        scaler.current_replicas = 5  # Start with more replicas

        decision = {
            "action": "scale_down",
            "target_replicas": 2,
            "reason": "Low latency predicted",
            "confidence": 0.7,
        }

        result = await scaler.execute_scaling_action(decision)
        assert result is True
        assert scaler.current_replicas == 2


class TestPerformanceMonitoringIntegration:
    """Integration tests for performance monitoring components."""

    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_and_scaling(self):
        """Test complete monitoring and scaling workflow."""
        # Create components
        rngs = nnx.Rngs(42)
        anomaly_detector = AIAnomalyDetector(rngs=rngs)
        performance_predictor = PerformancePredictor(rngs=rngs)

        monitor = PerformanceMonitor(
            anomaly_detector=anomaly_detector,
            performance_predictor=performance_predictor,
            collection_interval=0.05,
        )

        scaler = PredictiveScaler(performance_monitor=monitor)

        # Simulate metrics collection
        for _ in range(20):
            metrics = await monitor.collect_current_metrics()
            monitor.metrics_history.append(metrics)
            await asyncio.sleep(0.01)

        # Test prediction
        prediction = await monitor.predict_future_performance()
        assert isinstance(prediction, PredictionResult)

        # Test scaling decision
        decision = await scaler.evaluate_scaling_decision()
        assert decision["action"] in ["scale_up", "scale_down", "maintain"]

        # Test scaling execution
        result = await scaler.execute_scaling_action(decision)
        assert result is True

    @pytest.mark.asyncio
    async def test_anomaly_detection_workflow(self):
        """Test complete anomaly detection workflow."""
        rngs = nnx.Rngs(42)
        anomaly_detector = AIAnomalyDetector(
            anomaly_threshold=0.1,  # Lower threshold for easier detection
            rngs=rngs,
        )

        performance_predictor = PerformancePredictor(rngs=rngs)

        monitor = PerformanceMonitor(
            anomaly_detector=anomaly_detector,
            performance_predictor=performance_predictor,
        )

        # Create potentially anomalous metrics
        anomalous_metrics = PerformanceMetrics(
            timestamp=time.time(),
            latency_ms=1000.0,  # Very high latency
            throughput_rps=1.0,  # Very low throughput
            memory_usage_gb=50.0,  # Very high memory usage
            gpu_utilization=0.1,  # Very low GPU utilization
            cpu_utilization=0.99,  # Very high CPU utilization
            energy_efficiency=0.1,  # Very low efficiency
            error_rate=0.5,  # Very high error rate
            numerical_stability=0.1,  # Very low stability
            conservation_score=0.1,  # Very low conservation
            physics_consistency=0.1,  # Very low consistency
        )

        # Detect anomalies
        anomalies = await monitor.detect_performance_anomalies(anomalous_metrics)

        # Verify anomaly detection
        # Note: Due to the neural network being untrained, results may vary
        # This test mainly verifies the workflow doesn't crash
        assert isinstance(anomalies, list)
        for anomaly in anomalies:
            assert isinstance(anomaly, Anomaly)
            assert hasattr(anomaly, "severity")
            assert hasattr(anomaly, "recommended_action")
