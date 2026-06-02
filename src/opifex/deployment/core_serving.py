"""
Core model serving infrastructure for Opifex framework.

This module provides the foundational components for serving Opifex models in
production, including model loading, inference serving, deployment configuration,
and model registry. Follows JAX/Flax NNX best practices for scientific computing
workloads.
"""

import importlib
import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from enum import Enum
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from flax import nnx


# Configure logging
logger = logging.getLogger(__name__)


class ServingStatus(Enum):
    """Model server status enumeration."""

    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass(frozen=True, slots=True, kw_only=True)
class DeploymentConfig:
    """Configuration for model deployment."""

    model_name: str
    model_type: str
    serving_port: int = 8080
    batch_size: int = 32
    gpu_enabled: bool = True
    precision: str = "float32"
    max_concurrent_requests: int = 100
    timeout_seconds: int = 30

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        # Validate port range
        if not (1024 <= self.serving_port <= 65535):
            raise ValueError("Port must be between 1024 and 65535")

        # Validate batch size
        if self.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        # Validate precision
        if self.precision not in ["float16", "float32", "float64"]:
            raise ValueError("Precision must be one of: float16, float32, float64")

    def get_jax_dtype(self) -> jnp.dtype:
        """Get JAX dtype from precision string."""
        dtype_map = {
            "float16": jnp.float16,
            "float32": jnp.float32,
            "float64": jnp.float64,
        }
        return dtype_map[self.precision]


@dataclass(frozen=True, slots=True, kw_only=True)
class ModelMetadata:
    """Metadata for registered models."""

    name: str
    version: str
    model_type: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    parameters_count: int | None = None
    training_dataset: str | None = None
    accuracy_metrics: dict[str, float] | None = None
    created_at: str | None = None
    description: str | None = None

    def __post_init__(self) -> None:
        """Set default values and validate."""
        if self.created_at is None:
            object.__setattr__(self, "created_at", datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create metadata from dictionary."""
        return cls(**data)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ModelTemplate:
    """In-memory reconstruction template for a registered model.

    Pairs the module structure (``graphdef``) with an abstract, shape/dtype
    copy of its weight state (``abstract_state``). The latter is the restore
    target handed to Orbax; merging the restored state with ``graphdef``
    yields the original concrete module.
    """

    graphdef: nnx.GraphDef[nnx.Module]
    abstract_state: nnx.State


class ModelRegistry:
    """Registry for managing model versions and metadata.

    Model weights are serialized to disk via Orbax (the registry persists
    ``nnx.state(model)``). Reconstruction is **cross-process**: alongside the
    weight state the registry persists the module's fully-qualified
    ``module:class`` reference plus the keyword arguments needed to build an
    *abstract* (shape/dtype-only) copy of the module. On
    :meth:`get_model` the class is imported, an abstract module is built via
    :func:`flax.nnx.eval_shape`, and the persisted weights are restored into
    it — so a model registered by one process is reproduced exactly by any
    other process pointed at the same storage directory.

    An in-memory ``GraphDef`` template is also retained per ``model_id`` as a
    fast path within the registering process; it is never required for
    correctness. The on-disk ``module:class`` reference is the cross-process
    contract. When the class cannot be imported (e.g. it was removed or
    renamed) reconstruction fails fast with an import error rather than
    fabricating a model.
    """

    #: Sub-directory (under each model's directory) holding the Orbax state.
    _STATE_DIRNAME = "state"

    def __init__(self, storage_path: str | Path) -> None:
        """Initialize model registry.

        Args:
            storage_path: Path to store model files and metadata
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Metadata storage
        self.metadata_file = self.storage_path / "registry.json"
        self._models = self._load_registry()

        # In-memory reconstruction templates keyed by model_id: the module
        # structure (``nnx.GraphDef``) and an abstract (shape/dtype) copy of
        # the weight state used as the Orbax restore target. Held in memory
        # because an arbitrary module's GraphDef captures local initializer
        # closures and is not disk-serializable.
        self._templates: dict[str, _ModelTemplate] = {}

    def _load_registry(self) -> dict[str, Any]:
        """Load registry from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                return json.load(f)
        return {"models": {}, "versions": {}}

    def _save_registry(self) -> None:
        """Save registry to disk."""
        with open(self.metadata_file, "w") as f:
            json.dump(self._models, f, indent=2)

    def register_model(
        self,
        model: nnx.Module,
        metadata: ModelMetadata,
        init_kwargs: dict[str, Any] | None = None,
    ) -> str:
        """Register a model with metadata.

        Args:
            model: The model to register.
            metadata: Model metadata.
            init_kwargs: Keyword arguments (excluding ``rngs``) needed to
                build an abstract copy of ``type(model)`` for cross-process
                reconstruction. Must be JSON-serializable. Required only when
                the module's ``__init__`` takes configuration beyond ``rngs``;
                defaults to empty. The values must reproduce the registered
                module's *structure* (shapes/dtypes), not its weights.

        Returns:
            model_id: Unique identifier for the registered model.
        """
        model_id = str(uuid.uuid4())
        reconstruction_kwargs = dict(init_kwargs) if init_kwargs else {}

        # Create model directory
        model_dir = self.storage_path / model_id
        model_dir.mkdir(exist_ok=True)

        # Serialize the real model: split into structure (graphdef) + weights
        # (state). The weights are persisted to disk via Orbax; the structural
        # template is held in memory because it is not generically
        # disk-serializable (NNX captures local initializer closures).
        graphdef, state = nnx.split(model)
        abstract_state = jax.tree_util.tree_map(
            lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), state
        )
        self._templates[model_id] = _ModelTemplate(graphdef=graphdef, abstract_state=abstract_state)
        state_dir = (model_dir / self._STATE_DIRNAME).resolve()
        with ocp.StandardCheckpointer() as checkpointer:
            checkpointer.save(state_dir, state)

        # Persist the cross-process reconstruction contract: the
        # fully-qualified ``module:class`` reference plus the abstract-init
        # kwargs. Any process can import the class, rebuild an abstract module
        # via ``nnx.eval_shape``, and restore the persisted weights into it.
        model_info_path = model_dir / "model_info.json"
        model_class_info = {
            "model_qualname": f"{model.__class__.__module__}:{model.__class__.__qualname__}",
            "model_class_name": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "init_kwargs": reconstruction_kwargs,
            "model_id": model_id,
            "state_path": str(state_dir),
            "input_shape": metadata.input_shape,
            "output_shape": metadata.output_shape,
        }

        with open(model_info_path, "w") as f:
            json.dump(model_class_info, f, indent=2)

        # Save metadata
        metadata_path = model_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        # Update registry
        self._models["models"][model_id] = {
            "name": metadata.name,
            "version": metadata.version,
            "model_type": metadata.model_type,
            "model_path": str(model_dir),
            "metadata_path": str(metadata_path),
            "state_path": str(state_dir),
            "registered_at": datetime.now(UTC).isoformat(),
        }

        # Update version tracking
        model_name = metadata.name
        if model_name not in self._models["versions"]:
            self._models["versions"][model_name] = []

        self._models["versions"][model_name].append(
            {
                "version": metadata.version,
                "model_id": model_id,
                "created_at": metadata.created_at,
            }
        )

        # Sort by version (simple string comparison)
        self._models["versions"][model_name].sort(key=lambda x: x["version"], reverse=True)

        self._save_registry()
        logger.info("Registered model %s v%s with ID %s", metadata.name, metadata.version, model_id)

        return model_id

    @staticmethod
    def _resolve_model_class(qualname: str) -> type[nnx.Module]:
        """Import and return the ``nnx.Module`` subclass named by ``qualname``.

        Args:
            qualname: A ``module:Class`` reference (dotted attribute access is
                supported after the colon, e.g. ``pkg.mod:Outer.Inner``).

        Returns:
            The resolved class object.

        Raises:
            ValueError: If ``qualname`` is malformed.
            ImportError: If the module cannot be imported.
            AttributeError: If the class is absent from the module.
            TypeError: If the resolved object is not an ``nnx.Module`` subclass.
        """
        if ":" not in qualname:
            raise ValueError(f"Malformed model qualname {qualname!r}; expected 'module:Class'")
        module_name, _, attr_path = qualname.partition(":")
        module = importlib.import_module(module_name)
        obj: Any = module
        for part in attr_path.split("."):
            obj = getattr(obj, part)
        if not (isinstance(obj, type) and issubclass(obj, nnx.Module)):
            raise TypeError(f"Resolved object {qualname!r} is not an nnx.Module subclass")
        return obj

    def _build_abstract_template(self, model_info: dict[str, Any]) -> _ModelTemplate:
        """Reconstruct an abstract structural template from persisted info.

        The module's class is imported from the persisted ``module:class``
        reference and an abstract (shape/dtype-only) instance is built via
        :func:`flax.nnx.eval_shape`, using the persisted ``init_kwargs`` and a
        fresh ``nnx.Rngs``. No weights are materialized — the result is the
        Orbax restore target.

        Raises:
            ImportError, AttributeError, ValueError, TypeError: If the class
                cannot be resolved or instantiated abstractly. Reconstruction
                fails fast rather than fabricating a model.
        """
        model_path = Path(model_info["model_path"])
        info_path = model_path / "model_info.json"
        with open(info_path) as f:
            class_info = json.load(f)

        qualname = class_info["model_qualname"]
        init_kwargs = class_info.get("init_kwargs", {})
        model_class = self._resolve_model_class(qualname)

        # ``model_class`` is resolved dynamically; NNX modules conventionally
        # accept an ``rngs`` keyword for parameter initialization (supplied here
        # only to build the abstract structure — no weights are materialized).
        def _build_abstract() -> nnx.Module:
            return model_class(rngs=nnx.Rngs(0), **init_kwargs)  # type: ignore[call-arg]

        abstract_model = nnx.eval_shape(_build_abstract)
        graphdef, abstract_state = nnx.split(abstract_model)
        return _ModelTemplate(graphdef=graphdef, abstract_state=abstract_state)

    def get_model(self, model_id: str) -> tuple[nnx.Module, ModelMetadata]:
        """Retrieve the registered model and its metadata by ID.

        The model's weights are restored from the Orbax state persisted at
        registration and merged into a structural template. The template is
        taken from the in-memory cache when available (fast path within the
        registering process); otherwise it is rebuilt cross-process by
        importing the persisted ``module:class`` reference and constructing an
        abstract module via :func:`flax.nnx.eval_shape`. The returned module
        reproduces the registered model exactly — calling it on the same input
        yields the same outputs.

        Args:
            model_id: Model identifier.

        Returns:
            Tuple of ``(model, metadata)`` where ``model`` is the
            reconstructed :class:`flax.nnx.Module`.

        Raises:
            ValueError: If ``model_id`` is not registered or its persisted
                class reference is malformed.
            ImportError: If the persisted module cannot be imported.
            AttributeError: If the persisted class is absent from its module.
            TypeError: If the persisted reference is not an ``nnx.Module``.
        """
        if model_id not in self._models["models"]:
            raise ValueError(f"Model ID {model_id} not found")

        model_info = self._models["models"][model_id]

        # Load metadata first.
        metadata_path = model_info["metadata_path"]
        with open(metadata_path) as f:
            metadata_dict = json.load(f)
        metadata = ModelMetadata.from_dict(metadata_dict)

        # Prefer the in-memory template (fast path); otherwise reconstruct the
        # abstract structure cross-process from the persisted class reference.
        template = self._templates.get(model_id)
        if template is None:
            template = self._build_abstract_template(model_info)

        # Restore the persisted weights into the abstract state template,
        # then merge with the graphdef to obtain the concrete registered model.
        with ocp.StandardCheckpointer() as checkpointer:
            restored_state = checkpointer.restore(
                Path(model_info["state_path"]), target=template.abstract_state
            )
        model = nnx.merge(template.graphdef, restored_state)

        return model, metadata

    def list_models(self) -> list[dict[str, Any]]:
        """List all registered models.

        Returns:
            List of model information dictionaries
        """
        models = []
        for model_id, model_info in self._models["models"].items():
            models.append(
                {
                    "model_id": model_id,
                    "name": model_info["name"],
                    "version": model_info["version"],
                    "model_type": model_info["model_type"],
                    "registered_at": model_info["registered_at"],
                }
            )
        return models

    def get_latest_version(self, model_name: str) -> ModelMetadata:
        """Get metadata for the latest version of a model.

        Args:
            model_name: Name of the model

        Returns:
            ModelMetadata for the latest version
        """
        if model_name not in self._models["versions"]:
            raise ValueError(f"Model {model_name} not found")

        latest_version_info = self._models["versions"][model_name][0]
        model_id = latest_version_info["model_id"]
        _, metadata = self.get_model(model_id)

        return metadata

    def get_model_by_version(self, model_name: str, version: str) -> ModelMetadata:
        """Get metadata for a specific version of a model.

        Args:
            model_name: Name of the model
            version: Version string

        Returns:
            ModelMetadata for the specified version
        """
        if model_name not in self._models["versions"]:
            raise ValueError(f"Model {model_name} not found")

        for version_info in self._models["versions"][model_name]:
            if version_info["version"] == version:
                model_id = version_info["model_id"]
                _, metadata = self.get_model(model_id)
                return metadata

        raise ValueError(f"Version {version} not found for model {model_name}")


class InferenceEngine:
    """High-performance inference engine for Opifex models."""

    def __init__(self, config: DeploymentConfig) -> None:
        """Initialize inference engine.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.model: nnx.Module | None = None
        self.metadata: ModelMetadata | None = None
        self.is_initialized = False
        self.enable_jit = True

        # Performance tracking
        self._total_requests = 0
        self._total_latency = 0.0
        self._start_time = time.time()

        # JAX configuration
        if config.gpu_enabled:
            self.configure_gpu()

    def configure_gpu(self) -> dict[str, Any]:
        """Configure GPU settings for JAX."""
        # Set memory fraction to avoid OOM
        import os

        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")

        return {"memory_fraction": 0.8, "preallocate": False}

    def load_model(self, model: nnx.Module, metadata: ModelMetadata) -> None:
        """Load model for inference.

        Args:
            model: The model to load
            metadata: Model metadata
        """
        self.model = model
        self.metadata = metadata

        # JIT compile the model for performance
        if self.enable_jit:
            dummy_input = jnp.ones(
                (self.config.batch_size, *metadata.input_shape),
                dtype=self.config.get_jax_dtype(),
            )
            # Compile the model with dummy input
            self._compiled_predict = jax.jit(self._predict_fn)
            # Warm up JIT compilation
            _ = self._compiled_predict(dummy_input)

        self.is_initialized = True
        logger.info("Loaded model %s v%s", metadata.name, metadata.version)

    def _predict_fn(self, x: jax.Array) -> jax.Array:
        """Internal prediction function."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        # Flax NNX modules are callable
        return self.model(x)  # type: ignore[operator]

    def predict(self, input_data: jax.Array) -> jax.Array:
        """Perform inference on input data.

        Args:
            input_data: Input tensor for prediction

        Returns:
            Model predictions
        """
        if not self.is_initialized:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if self.metadata is None:
            raise RuntimeError("Model metadata not available")

        start_time = time.time()

        # Validate input shape
        expected_shape = (input_data.shape[0], *self.metadata.input_shape)
        if input_data.shape != expected_shape:
            raise ValueError(f"Expected input shape {expected_shape}, got {input_data.shape}")

        # Ensure correct dtype
        input_data = jnp.asarray(input_data)

        # Perform prediction
        if self.enable_jit and hasattr(self, "_compiled_predict"):
            predictions = self._compiled_predict(input_data)
        else:
            predictions = self._predict_fn(input_data)

        # Update performance metrics
        latency = time.time() - start_time
        self._total_requests += 1
        self._total_latency += latency

        return predictions

    def get_performance_metrics(self) -> dict[str, float]:
        """Get performance metrics."""
        uptime = time.time() - self._start_time
        avg_latency = self._total_latency / max(self._total_requests, 1)
        throughput = self._total_requests / max(uptime, 1)

        return {
            "total_requests": self._total_requests,
            "average_latency": avg_latency,
            "total_throughput": throughput,
            "uptime_seconds": uptime,
        }


class ModelServer:
    """HTTP server for model serving."""

    def __init__(self, config: DeploymentConfig) -> None:
        """Initialize model server.

        Args:
            config: Deployment configuration
        """
        self.config = config
        self.status = ServingStatus.STOPPED
        self.inference_engine: InferenceEngine | None = None
        self._start_time: float | None = None

    def start(self) -> None:
        """Start the model server."""
        logger.info("Starting model server on port %s", self.config.serving_port)
        self.status = ServingStatus.STARTING

        try:
            self._initialize_endpoints()
            self.status = ServingStatus.RUNNING
            self._start_time = time.time()

            # In a real implementation, this would start uvicorn
            # For testing, we just set the status
            logger.info("Model server started successfully")

        except Exception:
            logger.exception("Failed to start server")
            self.status = ServingStatus.ERROR
            raise

    def _initialize_endpoints(self) -> None:
        """Initialize server endpoints (FastAPI routes)."""
        # This would set up FastAPI routes in a real implementation

    def health_check(self) -> dict[str, Any]:
        """Health check endpoint.

        Returns:
            Health status information
        """
        uptime = time.time() - self._start_time if self._start_time else 0

        return {
            "status": self.status.value,
            "timestamp": datetime.now(UTC).isoformat(),
            "uptime_seconds": uptime,
            "model_name": self.config.model_name,
            "model_loaded": self.inference_engine is not None
            and self.inference_engine.is_initialized,
        }

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Prediction endpoint.

        Args:
            input_data: Input data dictionary with 'data' key

        Returns:
            Prediction results
        """
        if self.inference_engine is None or not self.inference_engine.is_initialized:
            raise RuntimeError("Model not loaded")

        # Extract data from request
        if "data" not in input_data:
            raise ValueError("Input must contain 'data' field")

        data = jnp.array(input_data["data"])

        # Perform inference
        predictions = self.inference_engine.predict(data)

        # Format response
        return {
            "predictions": predictions.tolist(),
            "metadata": {
                "model_name": self.config.model_name,
                "batch_size": data.shape[0],
                "input_shape": data.shape,
                "output_shape": predictions.shape,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }

    def stop(self) -> None:
        """Stop the model server."""
        logger.info("Stopping model server")
        self.status = ServingStatus.STOPPED
