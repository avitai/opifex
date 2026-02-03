"""Comprehensive tests for Opifex metrics collection system."""

from unittest.mock import Mock, patch

from opifex.deployment.monitoring.metrics import (
    CustomMetrics,
    MetricConfig,
    PrometheusMetrics,
)


class TestMetricConfig:
    """Test MetricConfig dataclass."""

    def test_metric_config_default_values(self):
        """Test MetricConfig with default values."""
        config = MetricConfig(name="test_metric", description="Test metric")

        assert config.name == "test_metric"
        assert config.description == "Test metric"
        assert config.labels == []
        assert config.metric_type == "counter"
        assert config.buckets is None
        assert config.namespace == "opifex"
        assert config.subsystem == ""

    def test_metric_config_custom_values(self):
        """Test MetricConfig with custom values."""
        config = MetricConfig(
            name="custom_metric",
            description="Custom metric",
            labels=["model", "version"],
            metric_type="histogram",
            buckets=[0.1, 0.5, 1.0],
            namespace="custom",
            subsystem="test",
        )

        assert config.name == "custom_metric"
        assert config.description == "Custom metric"
        assert config.labels == ["model", "version"]
        assert config.metric_type == "histogram"
        assert config.buckets == [0.1, 0.5, 1.0]
        assert config.namespace == "custom"
        assert config.subsystem == "test"


class TestPrometheusMetrics:
    """Test PrometheusMetrics class."""

    def test_init_without_prometheus(self):
        """Test initialization when prometheus is not available."""
        with patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", False):
            metrics = PrometheusMetrics()
            assert not metrics._metrics_enabled

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    @patch("opifex.deployment.monitoring.metrics.CollectorRegistry")
    @patch("opifex.deployment.monitoring.metrics.Counter")
    @patch("opifex.deployment.monitoring.metrics.Gauge")
    @patch("opifex.deployment.monitoring.metrics.Histogram")
    def test_init_with_prometheus(
        self, mock_histogram, mock_gauge, mock_counter, mock_registry
    ):
        """Test initialization when prometheus is available."""
        mock_registry_instance = Mock()
        mock_registry.return_value = mock_registry_instance

        metrics = PrometheusMetrics(namespace="test")

        assert metrics.namespace == "test"
        assert metrics._metrics_enabled
        assert metrics.registry == mock_registry_instance

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    @patch("opifex.deployment.monitoring.metrics.CollectorRegistry")
    def test_init_with_custom_registry(self, mock_registry):
        """Test initialization with custom registry."""
        custom_registry = Mock()
        metrics = PrometheusMetrics(registry=custom_registry)

        assert metrics.registry == custom_registry

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    @patch("opifex.deployment.monitoring.metrics.HAS_PSUTIL", True)
    @patch("opifex.deployment.monitoring.metrics.psutil")
    def test_update_system_metrics(self, mock_psutil):
        """Test system metrics update."""
        # Mock psutil return values
        mock_memory = Mock()
        mock_memory.used = 8000000000
        mock_memory.total = 16000000000
        mock_psutil.cpu_percent.return_value = 75.5
        mock_psutil.virtual_memory.return_value = mock_memory

        with (
            patch("opifex.deployment.monitoring.metrics.CollectorRegistry"),
            patch("opifex.deployment.monitoring.metrics.Gauge"),
        ):
            metrics = PrometheusMetrics()

            # Simply test that the method runs without error
            metrics.update_system_metrics()

            # Verify psutil methods were called
            assert mock_psutil.cpu_percent.called or mock_psutil.virtual_memory.called

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    @patch("opifex.deployment.monitoring.metrics.HAS_JAX", True)
    def test_update_gpu_metrics_with_jax(self):
        """Test GPU metrics update when JAX is available."""
        with (
            patch("opifex.deployment.monitoring.metrics.CollectorRegistry"),
            patch("opifex.deployment.monitoring.metrics.jax") as mock_jax,
        ):
            # Mock JAX device info
            mock_device = Mock()
            mock_device.id = 0
            mock_jax.devices.return_value = [mock_device]
            mock_jax.device_get.return_value = 0.85

            metrics = PrometheusMetrics()

            # Simply test that the method runs without error
            metrics.update_gpu_metrics()

            # Verify JAX methods were called
            assert mock_jax.devices.called

    def test_update_gpu_metrics_without_jax(self):
        """Test GPU metrics update when JAX is not available."""
        with (
            patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True),
            patch("opifex.deployment.monitoring.metrics.HAS_JAX", False),
        ):
            metrics = PrometheusMetrics()

            # Should not raise exception when JAX is not available
            metrics.update_gpu_metrics()

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_metrics_basic_functionality(self):
        """Test basic metrics functionality without mocking non-existent attributes."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            # Test that basic methods exist and can be called
            assert hasattr(metrics, "record_training_metrics")
            assert hasattr(metrics, "record_inference_metrics")
            assert hasattr(metrics, "update_system_metrics")
            assert hasattr(metrics, "update_gpu_metrics")

            # Test actual method calls
            metrics.record_training_metrics("FNO", "job123", 0.05)
            metrics.record_inference_metrics("FNO", "v1.0", 0.02)

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_record_training_metrics(self):
        """Test training metrics recording."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            # Test the actual interface
            metrics.record_training_metrics(
                model_type="FNO", job_id="job123", loss=0.05
            )

            # Basic verification that the object exists and method worked
            assert metrics is not None
            assert hasattr(metrics, "record_training_metrics")

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_record_inference_accuracy(self):
        """Test inference accuracy recording."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            # Mock accuracy metric
            metrics.model_accuracy = Mock()

            metrics.record_inference_accuracy("test_model", "v1", 0.89)

            # Verify accuracy was recorded
            assert metrics.model_accuracy.labels.called

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    @patch("opifex.deployment.monitoring.metrics.Counter")
    @patch("opifex.deployment.monitoring.metrics.Histogram")
    @patch("opifex.deployment.monitoring.metrics.Gauge")
    def test_create_custom_metric_counter(
        self, mock_gauge, mock_histogram, mock_counter
    ):
        """Test creating custom counter metric."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            config = MetricConfig(
                name="custom_counter",
                description="Custom counter",
                labels=["model"],
                metric_type="counter",
            )

            result = metrics.create_custom_metric(config)

            assert result is not None
            mock_counter.assert_called()

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    @patch("opifex.deployment.monitoring.metrics.Histogram")
    def test_create_custom_metric_histogram(self, mock_histogram):
        """Test creating custom histogram metric."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            config = MetricConfig(
                name="custom_histogram",
                description="Custom histogram",
                labels=["model"],
                metric_type="histogram",
                buckets=[0.1, 0.5, 1.0],
            )

            result = metrics.create_custom_metric(config)

            assert result is not None
            mock_histogram.assert_called()

    def test_create_custom_metric_without_prometheus(self):
        """Test creating custom metric when prometheus is not available."""
        with patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", False):
            metrics = PrometheusMetrics()

            config = MetricConfig(name="test", description="Test")
            result = metrics.create_custom_metric(config)

            assert result is None

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_get_custom_metric(self):
        """Test getting custom metric."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            # Add a mock custom metric
            mock_metric = Mock()
            metrics.custom_metrics["test_metric"] = mock_metric

            result = metrics.get_custom_metric("test_metric")
            assert result == mock_metric

            # Test non-existent metric
            result = metrics.get_custom_metric("non_existent")
            assert result is None

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    @patch("opifex.deployment.monitoring.metrics.start_http_server")
    def test_start_metrics_server(self, mock_start_server):
        """Test starting metrics server."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics(port=9090)

            metrics.start_metrics_server()

            mock_start_server.assert_called_with(9090, registry=metrics.registry)

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    @patch("opifex.deployment.monitoring.metrics.generate_latest")
    def test_get_metrics_data(self, mock_generate_latest):
        """Test getting metrics data."""
        mock_generate_latest.return_value = b"metrics data"

        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            result = metrics.get_metrics_data()

            assert result == "metrics data"
            mock_generate_latest.assert_called_with(metrics.registry)

    def test_get_metrics_data_without_prometheus(self):
        """Test getting metrics data when prometheus is not available."""
        with patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", False):
            metrics = PrometheusMetrics()

            result = metrics.get_metrics_data()

            assert result == "Prometheus metrics not available"

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_health_check(self):
        """Test health check."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            result = metrics.health_check()

            assert isinstance(result, dict)
            assert "status" in result
            assert "metrics_enabled" in result
            assert "timestamp" in result

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_record_gpu_metrics(self):
        """Test GPU metrics recording."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            # Test the system and GPU metrics update methods
            metrics.update_system_metrics()
            metrics.update_gpu_metrics()

            # Basic verification
            assert metrics is not None
            assert hasattr(metrics, "update_gpu_metrics")
            assert hasattr(metrics, "update_system_metrics")

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_record_inference_metrics(self):
        """Test inference metrics recording."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            # Test the actual interface
            metrics.record_inference_metrics(
                model_type="FNO", model_version="v1.0", duration=0.02
            )

            metrics.record_inference_accuracy(
                model_type="FNO", model_version="v1.0", accuracy=0.95
            )

            # Basic verification
            assert metrics is not None
            assert hasattr(metrics, "record_inference_metrics")
            assert hasattr(metrics, "record_inference_accuracy")

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_context_managers_basic(self):
        """Test that context manager methods can be checked for existence."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            # Test that basic methods exist - don't try to use non-existent context managers
            assert hasattr(metrics, "record_training_metrics")
            assert hasattr(metrics, "record_inference_metrics")

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_training_methods_basic(self):
        """Test training methods basic functionality."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()

            # Test basic training metrics functionality
            assert hasattr(metrics, "record_training_metrics")

            # Call with proper parameters
            metrics.record_training_metrics("FNO", "job123", 0.05)


class TestCustomMetrics:
    """Test CustomMetrics class."""

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_custom_metrics_init(self):
        """Test CustomMetrics initialization."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            prometheus_metrics = PrometheusMetrics()
            custom_metrics = CustomMetrics(prometheus_metrics)

            assert custom_metrics.prometheus_metrics == prometheus_metrics

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_record_fno_metrics(self):
        """Test FNO metrics recording."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            metrics = PrometheusMetrics()
            custom_metrics = CustomMetrics(metrics)

            # Mock the histogram metric that is actually used
            custom_metrics.fno_forward_time = Mock()  # type: ignore[attr-defined]

            custom_metrics.record_fno_metrics(
                model_id="test_fno",
                modes={"x": 32, "y": 32},
                forward_time=0.1,
                resolution="256x256",
            )

            # Verify the forward time metric was recorded
            assert custom_metrics.fno_forward_time.labels.called  # type: ignore[attr-defined]

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_record_deeponet_metrics(self):
        """Test recording DeepONet metrics."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            prometheus_metrics = PrometheusMetrics()
            custom_metrics = CustomMetrics(prometheus_metrics)

            # Mock metric objects
            custom_metrics.deeponet_branch_size = Mock()  # type: ignore[attr-defined]
            custom_metrics.deeponet_trunk_size = Mock()  # type: ignore[attr-defined]

            custom_metrics.record_deeponet_metrics(
                model_id="deeponet_001", branch_size=256, trunk_size=128
            )

            # Verify metrics were recorded
            assert custom_metrics.deeponet_branch_size.labels.called  # type: ignore[attr-defined]
            assert custom_metrics.deeponet_trunk_size.labels.called  # type: ignore[attr-defined]

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_record_simulation_metrics(self):
        """Test recording simulation metrics."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            prometheus_metrics = PrometheusMetrics()
            custom_metrics = CustomMetrics(prometheus_metrics)

            # Mock metric objects
            custom_metrics.simulation_step_time = Mock()  # type: ignore[attr-defined]
            custom_metrics.simulation_convergence = Mock()  # type: ignore[attr-defined]

            custom_metrics.record_simulation_metrics(
                simulation_type="CFD",
                grid_size="1024x1024",
                step_time=0.1,
                convergence_iterations=50,
                tolerance="1e-6",
            )

            # Verify metrics were recorded
            assert custom_metrics.simulation_step_time.labels.called  # type: ignore[attr-defined]

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_record_pde_accuracy(self):
        """Test recording PDE accuracy metrics."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            prometheus_metrics = PrometheusMetrics()
            custom_metrics = CustomMetrics(prometheus_metrics)

            # Mock metric objects
            custom_metrics.pde_accuracy = Mock()  # type: ignore[attr-defined]

            custom_metrics.record_pde_accuracy(
                equation_type="heat", method="FNO", error=0.01
            )

            # Verify accuracy was recorded
            assert custom_metrics.pde_accuracy.labels.called  # type: ignore[attr-defined]

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_record_computational_efficiency(self):
        """Test recording computational efficiency metrics."""
        with patch("opifex.deployment.monitoring.metrics.CollectorRegistry"):
            prometheus_metrics = PrometheusMetrics()
            custom_metrics = CustomMetrics(prometheus_metrics)

            # Mock metric objects
            custom_metrics.computational_efficiency = Mock()  # type: ignore[attr-defined]

            custom_metrics.record_computational_efficiency(
                algorithm="FNO", problem_size="large", efficiency=0.85
            )

            # Verify efficiency was recorded
            assert custom_metrics.computational_efficiency.labels.called  # type: ignore[attr-defined]


class TestIntegration:
    """Integration tests for metrics system."""

    @patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", True)
    def test_full_metrics_workflow(self):
        """Test complete metrics workflow."""
        with (
            patch("opifex.deployment.monitoring.metrics.CollectorRegistry"),
            patch("opifex.deployment.monitoring.metrics.start_http_server"),
        ):
            # Initialize metrics system
            prometheus_metrics = PrometheusMetrics(namespace="test")
            custom_metrics = CustomMetrics(prometheus_metrics)

            # Mock all metric objects
            prometheus_metrics.training_loss = Mock()  # type: ignore[attr-defined]
            prometheus_metrics.inference_duration = Mock()  # type: ignore[attr-defined]
            custom_metrics.fno_forward_time = Mock()  # type: ignore[attr-defined]

            # Record various metrics
            prometheus_metrics.record_training_metrics("FNO", "job1", 0.05)
            custom_metrics.record_fno_metrics("fno1", {"x": 16}, 0.05, "256x256")

            # Get health check
            health = prometheus_metrics.health_check()
            assert isinstance(health, dict)

            # Start metrics server
            prometheus_metrics.start_metrics_server(9091)

    def test_error_handling_without_dependencies(self):
        """Test error handling when dependencies are missing."""
        with patch("opifex.deployment.monitoring.metrics.HAS_PROMETHEUS", False):
            # Should not raise exceptions
            metrics = PrometheusMetrics()
            assert not metrics._metrics_enabled

            # All methods should handle missing dependencies gracefully
            metrics.update_system_metrics()
            metrics.update_gpu_metrics()
            metrics.record_training_metrics("test", "job1", 0.1)

            health = metrics.health_check()
            assert health["metrics_enabled"] is False
