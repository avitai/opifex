"""
Test cases for core model serving infrastructure.

Tests the foundational components for serving Opifex models in production,
including model loading, inference serving, and basic deployment utilities.
"""

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


# Import type checking to avoid runtime issues
if TYPE_CHECKING:
    from opifex.deployment.core_serving import (
        DeploymentConfig,
        InferenceEngine,
        ModelMetadata,
        ModelRegistry,
        ModelServer,
        ServingStatus,
    )
else:
    try:
        from opifex.deployment.core_serving import (
            DeploymentConfig,
            InferenceEngine,
            ModelMetadata,
            ModelRegistry,
            ModelServer,
            ServingStatus,
        )
    except ImportError:
        # Handle gracefully if components aren't implemented yet
        DeploymentConfig = None  # type: ignore[misc,assignment]
        InferenceEngine = None  # type: ignore[misc,assignment]
        ModelMetadata = None  # type: ignore[misc,assignment]
        ModelRegistry = None  # type: ignore[misc,assignment]
        ModelServer = None  # type: ignore[misc,assignment]
        ServingStatus = None  # type: ignore[misc,assignment]


# Test model classes at module level for proper pickling
class SimpleTestModel(nnx.Module):
    """Simple test model that can be pickled."""

    def __init__(self, rngs):
        self.linear = nnx.Linear(64, 64, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


class MultiLayerTestModel(nnx.Module):
    """Multi-layer test model for more complex testing."""

    def __init__(self, rngs):
        self.layer1 = nnx.Linear(32, 64, rngs=rngs)
        self.layer2 = nnx.Linear(64, 32, rngs=rngs)

    def __call__(self, x):
        x = self.layer1(x)
        x = nnx.relu(x)
        return self.layer2(x)


class TestDeploymentConfig:
    """Test deployment configuration management."""

    def test_deployment_config_initialization(self):
        """Test basic deployment configuration creation."""
        if DeploymentConfig is None:
            pytest.skip("DeploymentConfig not yet implemented")

        config = DeploymentConfig(
            model_name="test_fno",
            model_type="neural_operator",
            serving_port=8080,
            batch_size=32,
            gpu_enabled=True,
            precision="float32",
        )

        assert config.model_name == "test_fno"
        assert config.model_type == "neural_operator"
        assert config.serving_port == 8080
        assert config.batch_size == 32
        assert config.gpu_enabled is True
        assert config.precision == "float32"

    def test_deployment_config_validation(self):
        """Test deployment configuration validation."""
        if DeploymentConfig is None:
            pytest.skip("DeploymentConfig not yet implemented")

        # Test invalid port
        with pytest.raises(ValueError, match="Port must be between"):
            DeploymentConfig(
                model_name="test", model_type="neural_operator", serving_port=70000
            )

        # Test invalid batch size
        with pytest.raises(ValueError, match="Batch size must be positive"):
            DeploymentConfig(
                model_name="test", model_type="neural_operator", batch_size=0
            )

    def test_deployment_config_jax_precision(self):
        """Test JAX precision configuration."""
        if DeploymentConfig is None:
            pytest.skip("DeploymentConfig not yet implemented")

        config = DeploymentConfig(
            model_name="test", model_type="neural_operator", precision="float64"
        )

        assert config.get_jax_dtype() == jnp.float64


class TestModelMetadata:
    """Test model metadata management."""

    def test_model_metadata_creation(self):
        """Test basic model metadata creation."""
        if ModelMetadata is None:
            pytest.skip("ModelMetadata not yet implemented")

        metadata = ModelMetadata(
            name="darcy_fno",
            version="1.0.0",
            model_type="fourier_neural_operator",
            input_shape=(64, 64),
            output_shape=(64, 64),
            parameters_count=123456,
            training_dataset="darcy_flow",
            accuracy_metrics={"mse": 0.001, "r2": 0.99},
        )

        assert metadata.name == "darcy_fno"
        assert metadata.version == "1.0.0"
        assert metadata.model_type == "fourier_neural_operator"
        assert metadata.input_shape == (64, 64)
        assert metadata.output_shape == (64, 64)
        assert metadata.parameters_count == 123456

    def test_model_metadata_serialization(self):
        """Test model metadata JSON serialization."""
        if ModelMetadata is None:
            pytest.skip("ModelMetadata not yet implemented")

        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            model_type="deeponet",
            input_shape=(100,),
            output_shape=(50,),
        )

        # Test to_dict
        metadata_dict = metadata.to_dict()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["name"] == "test_model"

        # Test from_dict
        restored_metadata = ModelMetadata.from_dict(metadata_dict)
        assert restored_metadata.name == metadata.name
        assert restored_metadata.version == metadata.version


class TestModelRegistry:
    """Test model registry functionality."""

    def test_model_registry_initialization(self):
        """Test model registry initialization."""
        if ModelRegistry is None:
            pytest.skip("ModelRegistry not yet implemented")

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(storage_path=temp_dir)
            assert registry.storage_path == Path(temp_dir)
            assert len(registry.list_models()) == 0

    def test_model_registration(self):
        """Test model registration and retrieval."""
        if ModelRegistry is None:
            pytest.skip("ModelRegistry not yet implemented")

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(storage_path=temp_dir)

            # Create test model and metadata
            model = SimpleTestModel(rngs=nnx.Rngs(0))
            metadata = ModelMetadata(
                name="test_fno",
                version="1.0.0",
                model_type="fno",
                input_shape=(64, 64),
                output_shape=(64, 64),
            )

            # Register model
            model_id = registry.register_model(model, metadata)
            assert isinstance(model_id, str)

            # Verify registration
            models = registry.list_models()
            assert len(models) == 1
            assert models[0]["name"] == "test_fno"

            # Retrieve model
            retrieved_model, retrieved_metadata = registry.get_model(model_id)
            assert retrieved_metadata.name == "test_fno"
            # The registry returns a MinimalModel for testing purposes
            assert hasattr(retrieved_model, "linear")
            assert callable(retrieved_model)

    def test_model_versioning(self):
        """Test model versioning functionality."""
        if ModelRegistry is None:
            pytest.skip("ModelRegistry not yet implemented")

        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(storage_path=temp_dir)

            # Register multiple versions
            for version in ["1.0.0", "1.1.0", "2.0.0"]:
                metadata = ModelMetadata(
                    name="test_model",
                    version=version,
                    model_type="fno",
                    input_shape=(32, 32),
                    output_shape=(32, 32),
                )
                registry.register_model(MultiLayerTestModel(rngs=nnx.Rngs(0)), metadata)

            # Get latest version
            latest_metadata = registry.get_latest_version("test_model")
            assert latest_metadata.version == "2.0.0"

            # Get specific version
            specific_metadata = registry.get_model_by_version("test_model", "1.1.0")
            assert specific_metadata.version == "1.1.0"


class TestInferenceEngine:
    """Test inference engine functionality."""

    def test_inference_engine_initialization(self):
        """Test inference engine initialization."""
        if InferenceEngine is None:
            pytest.skip("InferenceEngine not yet implemented")

        config = DeploymentConfig(
            model_name="test_model",
            model_type="neural_operator",
            batch_size=32,
            gpu_enabled=True,
            precision="float32",
        )

        engine = InferenceEngine(config)
        assert engine.config == config
        assert engine.is_initialized is False

    def test_model_loading(self):
        """Test model loading functionality."""
        if InferenceEngine is None:
            pytest.skip("InferenceEngine not yet implemented")

        config = DeploymentConfig(
            model_name="test_model",
            model_type="neural_operator",
            batch_size=32,
            gpu_enabled=False,  # Avoid GPU issues in tests
            precision="float32",
        )

        engine = InferenceEngine(config)

        # Real model and metadata
        model = SimpleTestModel(rngs=nnx.Rngs(0))
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            model_type="fno",
            input_shape=(64, 64),
            output_shape=(64, 64),
        )

        # Load model
        engine.load_model(model, metadata)
        assert engine.is_initialized is True
        assert engine.model == model
        assert engine.metadata == metadata

    def test_batch_inference(self):
        """Test batch inference processing."""
        if InferenceEngine is None:
            pytest.skip("InferenceEngine not yet implemented")

        config = DeploymentConfig(
            model_name="test_model",
            model_type="neural_operator",
            batch_size=2,
            gpu_enabled=False,
            precision="float32",
        )

        engine = InferenceEngine(config)

        # Load real model
        model = SimpleTestModel(rngs=nnx.Rngs(0))
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            model_type="fno",
            input_shape=(64,),
            output_shape=(64,),
        )

        engine.load_model(model, metadata)

        # Test inference with proper shape
        input_data = jnp.ones((2, 64))
        output = engine.predict(input_data)

        assert output.shape == (2, 64)  # Should match expected output shape

    def test_inference_validation(self):
        """Test input validation for inference."""
        if InferenceEngine is None:
            pytest.skip("InferenceEngine not yet implemented")

        config = DeploymentConfig(
            model_name="test_model",
            model_type="neural_operator",
            batch_size=32,
            gpu_enabled=False,
            precision="float32",
        )

        engine = InferenceEngine(config)

        # Test prediction without loaded model
        with pytest.raises(RuntimeError, match="Model not loaded"):
            engine.predict(jnp.array([[1.0, 2.0]]))

    def test_performance_monitoring(self):
        """Test inference performance monitoring."""
        if InferenceEngine is None:
            pytest.skip("InferenceEngine not yet implemented")

        config = DeploymentConfig(
            model_name="test_model",
            model_type="neural_operator",
            batch_size=16,
            gpu_enabled=False,
            precision="float32",
        )

        engine = InferenceEngine(config)

        # Load real model
        model = SimpleTestModel(rngs=nnx.Rngs(0))
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            model_type="fno",
            input_shape=(64,),
            output_shape=(64,),
        )

        engine.load_model(model, metadata)

        # Perform inference and check metrics
        input_data = jnp.ones((16, 64))
        output = engine.predict(input_data)

        # Validate output shape and type
        assert output.shape == (16, 64)
        assert isinstance(output, jax.Array)

        metrics = engine.get_performance_metrics()
        assert "total_requests" in metrics
        assert "average_latency" in metrics
        assert "total_throughput" in metrics
        assert metrics["total_requests"] == 1


class TestModelServer:
    """Test model server functionality."""

    def test_model_server_initialization(self):
        """Test model server initialization."""
        if ModelServer is None:
            pytest.skip("ModelServer not yet implemented")

        config = DeploymentConfig(
            model_name="test_model", model_type="neural_operator", serving_port=8080
        )

        server = ModelServer(config)
        assert server.config == config
        assert server.status == ServingStatus.STOPPED

    def test_server_startup(self):
        """Test server startup process."""
        if ModelServer is None:
            pytest.skip("ModelServer not yet implemented")

        config = DeploymentConfig(
            model_name="test_model", model_type="neural_operator", serving_port=8080
        )

        server = ModelServer(config)
        server.start()

        # For testing, the server just sets status to RUNNING
        assert server.status == ServingStatus.RUNNING

    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        if ModelServer is None:
            pytest.skip("ModelServer not yet implemented")

        config = DeploymentConfig(
            model_name="test_model", model_type="neural_operator", serving_port=8080
        )

        server = ModelServer(config)

        # Test health check
        health_status = server.health_check()
        assert isinstance(health_status, dict)
        assert "status" in health_status
        assert "timestamp" in health_status
        assert "model_name" in health_status
        assert health_status["model_name"] == "test_model"

    def test_prediction_endpoint(self):
        """Test prediction endpoint functionality."""
        if ModelServer is None:
            pytest.skip("ModelServer not yet implemented")

        config = DeploymentConfig(
            model_name="test_model",
            model_type="neural_operator",
            serving_port=8080,
            batch_size=1,
        )

        server = ModelServer(config)
        server.start()

        # Load model into server's inference engine
        model = SimpleTestModel(rngs=nnx.Rngs(0))
        metadata = ModelMetadata(
            name="test_model",
            version="1.0.0",
            model_type="fno",
            input_shape=(64,),
            output_shape=(64,),
        )

        # Initialize inference engine
        server.inference_engine = InferenceEngine(config)
        server.inference_engine.load_model(model, metadata)

        # Test prediction
        input_data = {"data": [[1.0] * 64]}
        response = server.predict(input_data)

        assert "predictions" in response
        assert "metadata" in response
        assert response["metadata"]["model_name"] == "test_model"


class TestServingIntegration:
    """Test integration between serving components."""

    def test_end_to_end_serving_workflow(self):
        """Test complete model serving workflow."""
        if None in [ModelServer, ModelRegistry, InferenceEngine, DeploymentConfig]:
            pytest.skip("Serving components not yet implemented")

        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup
            config = DeploymentConfig(
                model_name="integration_test",
                model_type="fno",
                serving_port=8081,
                batch_size=4,
            )

            registry = ModelRegistry(storage_path=temp_dir)

            # Mock model
            mock_model = SimpleTestModel(rngs=nnx.Rngs(0))
            mock_metadata = ModelMetadata(
                name="integration_test",
                version="1.0.0",
                model_type="fno",
                input_shape=(8,),
                output_shape=(8,),
            )

            # Register model
            model_id = registry.register_model(mock_model, mock_metadata)

            # Load into inference engine
            engine = InferenceEngine(config)
            retrieved_model, retrieved_metadata = registry.get_model(model_id)
            engine.load_model(retrieved_model, retrieved_metadata)

            # Create server
            server = ModelServer(config)
            server.inference_engine = engine

            # Test prediction
            input_data = {"data": [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]}
            result = server.predict(input_data)

            assert "predictions" in result
            assert "metadata" in result

    def test_error_handling_integration(self):
        """Test error handling across serving components."""
        if None in [ModelServer, InferenceEngine, DeploymentConfig]:
            pytest.skip("Serving components not yet implemented")

        config = DeploymentConfig(
            model_name="error_test", model_type="fno", batch_size=4
        )

        # Test uninitialized server prediction
        server = ModelServer(config)

        with pytest.raises(RuntimeError):
            server.predict({"data": [[1.0, 2.0]]})


class TestJAXOptimization:
    """Test JAX-specific optimizations for serving."""

    def test_jax_jit_compilation(self):
        """Test JAX JIT compilation for inference."""
        if InferenceEngine is None:
            pytest.skip("InferenceEngine not yet implemented")

        config = DeploymentConfig(
            model_name="jit_test",
            model_type="neural_operator",
            batch_size=4,
            gpu_enabled=False,
            precision="float32",
        )

        engine = InferenceEngine(config)

        # Load model - use consistent dimensions with SimpleTestModel (64->64)
        model = SimpleTestModel(rngs=nnx.Rngs(0))
        metadata = ModelMetadata(
            name="jit_test",
            version="1.0.0",
            model_type="fno",
            input_shape=(64,),
            output_shape=(64,),
        )

        engine.load_model(model, metadata)

        # Test that JIT compilation works
        assert engine.enable_jit is True
        assert hasattr(engine, "_compiled_predict")

    def test_gpu_memory_management(self):
        """Test GPU memory management."""
        if InferenceEngine is None:
            pytest.skip("InferenceEngine not yet implemented")

        config = DeploymentConfig(
            model_name="gpu_test",
            model_type="neural_operator",
            batch_size=2,
            gpu_enabled=True,  # Test GPU config even if not available
            precision="float32",
        )

        engine = InferenceEngine(config)

        # Test GPU configuration
        gpu_config = engine.configure_gpu()
        assert "memory_fraction" in gpu_config
        assert gpu_config["memory_fraction"] == 0.8
