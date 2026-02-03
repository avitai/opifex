"""
Structured Logging System for Opifex Framework.

Enterprise-grade structured logging with performance metrics and
scientific computing context.
"""

import json
import logging
import logging.config
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any


# Optional dependencies with proper type checking
try:
    import psutil  # type: ignore[import-untyped]

    has_psutil = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    has_psutil = False

try:
    import jax  # type: ignore[import-untyped]

    has_jax = True
except ImportError:
    jax = None  # type: ignore[assignment]
    has_jax = False

# Constants set once at module load
HAS_PSUTIL = has_psutil
HAS_JAX = has_jax


@dataclass
class LogContext:
    """Context information for structured logging."""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str | None = None
    session_id: str | None = None
    experiment_id: str | None = None
    job_id: str | None = None
    model_name: str | None = None
    version: str = "1.0.0"
    service: str = "opifex"
    component: str = "unknown"
    environment: str = "development"

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LogEntry:
    """Structured log entry for scientific computing workloads."""

    timestamp: str
    level: str
    message: str
    context: LogContext
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int

    # Performance metrics
    duration_ms: float | None = None
    memory_usage_mb: float | None = None
    gpu_memory_mb: float | None = None
    cpu_percent: float | None = None

    # Scientific computing specific
    operation_type: str | None = None
    model_parameters: int | None = None
    batch_size: int | None = None
    epoch: int | None = None
    step: int | None = None
    loss_value: float | None = None
    accuracy: float | None = None
    convergence_status: str | None = None

    # Error information
    error_type: str | None = None
    error_message: str | None = None
    stack_trace: str | None = None

    # Additional metadata
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization."""
        data = asdict(self)
        data["context"] = self.context.to_dict()
        # Remove None values for cleaner logs
        return {k: v for k, v in data.items() if v is not None}


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""

    def __init__(self, context: LogContext):
        super().__init__()
        self.context = context

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        # Get performance metrics if available
        duration_ms = getattr(record, "duration_ms", None)
        memory_usage_mb = None
        gpu_memory_mb = None
        cpu_percent = None

        if HAS_PSUTIL and psutil is not None:
            try:
                process = psutil.Process()
                memory_usage_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()
            except Exception:
                pass

        if HAS_JAX and jax is not None:
            try:
                # Get GPU memory usage
                devices = jax.devices()
                if devices:
                    gpu_memory_mb = (
                        sum(
                            device.memory_stats().get("bytes_in_use", 0)
                            for device in devices
                            if hasattr(device, "memory_stats")
                        )
                        / 1024
                        / 1024
                    )
            except Exception:
                pass

        # Extract scientific computing metadata
        operation_type = getattr(record, "operation_type", None)
        model_parameters = getattr(record, "model_parameters", None)
        batch_size = getattr(record, "batch_size", None)
        epoch = getattr(record, "epoch", None)
        step = getattr(record, "step", None)
        loss_value = getattr(record, "loss_value", None)
        accuracy = getattr(record, "accuracy", None)
        convergence_status = getattr(record, "convergence_status", None)

        # Extract error information
        error_type = None
        error_message = None
        stack_trace = None

        if record.exc_info:
            error_type = record.exc_info[0].__name__ if record.exc_info[0] else None
            error_message = str(record.exc_info[1]) if record.exc_info[1] else None
            stack_trace = self.formatException(record.exc_info)

        # Build log entry
        log_entry = LogEntry(
            timestamp=datetime.now(UTC).isoformat(),
            level=record.levelname,
            message=record.getMessage(),
            context=self.context,
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread or 0,
            process_id=record.process or 0,
            duration_ms=duration_ms,
            memory_usage_mb=memory_usage_mb,
            gpu_memory_mb=gpu_memory_mb,
            cpu_percent=cpu_percent,
            operation_type=operation_type,
            model_parameters=model_parameters,
            batch_size=batch_size,
            epoch=epoch,
            step=step,
            loss_value=loss_value,
            accuracy=accuracy,
            convergence_status=convergence_status,
            error_type=error_type,
            error_message=error_message,
            stack_trace=stack_trace,
            extra=getattr(record, "extra", {}),
        )

        return json.dumps(log_entry.to_dict(), default=str)


class StructuredLogger:
    """Enterprise-grade structured logger for scientific computing."""

    def __init__(
        self,
        name: str = "opifex",
        context: LogContext | None = None,
        level: str = "INFO",
        handlers: list[logging.Handler] | None = None,
        elk_config: dict[str, Any] | None = None,
    ):
        """
        Initialize structured logger.

        Args:
            name: Logger name
            context: Log context for correlation
            level: Logging level
            handlers: Custom handlers (defaults to console and file)
            elk_config: ELK stack configuration
        """
        self.context = context or LogContext()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Clear existing handlers
        self.logger.handlers.clear()

        # Set up handlers
        if handlers is None:
            handlers = self._create_default_handlers()

        for handler in handlers:
            handler.setFormatter(JsonFormatter(self.context))
            self.logger.addHandler(handler)

        # Configure ELK stack if provided
        if elk_config:
            self._configure_elk(elk_config)

    def _create_default_handlers(self) -> list[logging.Handler]:
        """Create default console and file handlers."""
        handlers: list[logging.Handler] = []

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        handlers.append(console_handler)

        # File handler for persistent logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)

        file_handler = logging.FileHandler(
            log_dir / f"opifex-{datetime.now(UTC).strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)

        return handlers

    def _configure_elk(self, elk_config: dict[str, Any]) -> None:
        """Configure ELK stack integration."""
        try:
            # Example ELK configuration (would require additional dependencies)
            # This serves as an integration point for ELK stack configuration
            self.elk_enabled = True
            self.elk_config = elk_config

            # In production, you would configure:
            # - Logstash handler for centralized logging
            # - Elasticsearch integration
            # - Kibana dashboard configuration

            self.info("ELK stack integration configured", operation_type="elk_setup")

        except Exception as e:
            self.warning(f"Failed to configure ELK stack: {e}")
            self.elk_enabled = False

    def update_context(self, **kwargs) -> None:
        """Update logging context."""
        for key, value in kwargs.items():
            if hasattr(self.context, key):
                setattr(self.context, key, value)

    def _log(
        self,
        level: str,
        message: str,
        operation_type: str | None = None,
        duration_ms: float | None = None,
        **kwargs,
    ) -> None:
        """Internal logging method with performance tracking."""
        # Create a new record with scientific computing metadata
        extra = {"operation_type": operation_type, "duration_ms": duration_ms, **kwargs}

        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra)

    def debug(self, message: str, **kwargs) -> None:
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """Log info message."""
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """Log warning message."""
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """Log error message."""
        self._log("ERROR", message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """Log critical message."""
        self._log("CRITICAL", message, **kwargs)

    def exception(self, message: str, **kwargs) -> None:
        """Log exception with traceback."""
        self.logger.exception(message, extra=kwargs)

    @contextmanager
    def timed_operation(self, operation_name: str, **context_kwargs):
        """
        Context manager for timing operations with automatic logging.

        Args:
            operation_name: Name of the operation being timed
            **context_kwargs: Additional context information
        """
        start_time = time.time()
        operation_id = str(uuid.uuid4())

        self.info(
            f"Starting {operation_name}",
            operation_type=operation_name,
            operation_id=operation_id,
            **context_kwargs,
        )

        try:
            yield
            duration_ms = (time.time() - start_time) * 1000

            self.info(
                f"Completed {operation_name}",
                operation_type=operation_name,
                operation_id=operation_id,
                duration_ms=duration_ms,
                status="success",
                **context_kwargs,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            self.error(
                f"Failed {operation_name}: {e!s}",
                operation_type=operation_name,
                operation_id=operation_id,
                duration_ms=duration_ms,
                status="error",
                error_type=type(e).__name__,
                **context_kwargs,
            )
            raise

    @contextmanager
    def training_context(self, model_name: str, epoch: int, batch_size: int, **kwargs):
        """Context manager for training operations."""
        original_context = {
            "model_name": self.context.model_name,
            "experiment_id": self.context.experiment_id,
        }

        # Update context for training
        self.update_context(
            model_name=model_name,
            experiment_id=kwargs.get("experiment_id", self.context.experiment_id),
        )

        try:
            with self.timed_operation(
                "training_epoch", epoch=epoch, batch_size=batch_size, **kwargs
            ):
                yield
        finally:
            # Restore original context
            self.update_context(**original_context)

    @contextmanager
    def inference_context(self, model_name: str, batch_size: int, **kwargs):
        """Context manager for inference operations."""
        original_model = self.context.model_name
        self.update_context(model_name=model_name)

        try:
            with self.timed_operation(
                "model_inference", batch_size=batch_size, **kwargs
            ):
                yield
        finally:
            self.update_context(model_name=original_model)


class LoggingConfig:
    """Configuration management for structured logging."""

    @staticmethod
    def get_development_config() -> dict[str, Any]:
        """Get development logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "()": JsonFormatter,
                    "context": LogContext(environment="development"),
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "structured",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.FileHandler",
                    "level": "DEBUG",
                    "formatter": "structured",
                    "filename": "logs/opifex-dev.log",
                },
            },
            "loggers": {
                "opifex": {
                    "level": "DEBUG",
                    "handlers": ["console", "file"],
                    "propagate": False,
                }
            },
        }

    @staticmethod
    def get_production_config() -> dict[str, Any]:
        """Get production logging configuration."""
        return {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "structured": {
                    "()": JsonFormatter,
                    "context": LogContext(environment="production"),
                }
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "level": "WARNING",
                    "formatter": "structured",
                    "stream": "ext://sys.stdout",
                },
                "file": {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "INFO",
                    "formatter": "structured",
                    "filename": "logs/opifex-prod.log",
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 10,
                },
            },
            "loggers": {
                "opifex": {
                    "level": "INFO",
                    "handlers": ["console", "file"],
                    "propagate": False,
                }
            },
        }


def get_logger(
    name: str = "opifex",
    context: LogContext | None = None,
    environment: str = "development",
    elk_config: dict[str, Any] | None = None,
) -> StructuredLogger:
    """
    Get a configured structured logger.

    Args:
        name: Logger name
        context: Optional log context
        environment: Environment (development/production)
        elk_config: ELK stack configuration

    Returns:
        Configured StructuredLogger instance
    """
    if context is None:
        context = LogContext(environment=environment)

    # Configure logging based on environment
    if environment == "production":
        config = LoggingConfig.get_production_config()
    else:
        config = LoggingConfig.get_development_config()

    # Ensure log directory exists
    Path("logs").mkdir(exist_ok=True)

    # Apply configuration
    logging.config.dictConfig(config)

    return StructuredLogger(
        name=name,
        context=context,
        level="INFO" if environment == "production" else "DEBUG",
        elk_config=elk_config,
    )


# Global logger instance
_global_logger: StructuredLogger | None = None


def setup_global_logger(
    environment: str = "development",
    context: LogContext | None = None,
    elk_config: dict[str, Any] | None = None,
) -> StructuredLogger:
    """Set up global logger instance."""
    global _global_logger  # noqa: PLW0603
    _global_logger = get_logger(
        context=context, environment=environment, elk_config=elk_config
    )
    return _global_logger


def get_global_logger() -> StructuredLogger:
    """Get global logger instance."""
    global _global_logger  # noqa: PLW0603
    if _global_logger is None:
        _global_logger = get_logger()
    return _global_logger


# Convenience functions for global logging
def log_training_start(model_name: str, epoch: int, batch_size: int, **kwargs) -> None:
    """Log training start event."""
    logger = get_global_logger()
    logger.info(
        f"Starting training: {model_name}",
        operation_type="training_start",
        model_name=model_name,
        epoch=epoch,
        batch_size=batch_size,
        **kwargs,
    )


def log_training_step(
    epoch: int, step: int, loss_value: float, accuracy: float | None = None, **kwargs
) -> None:
    """Log training step metrics."""
    logger = get_global_logger()
    logger.info(
        f"Training step {step}",
        operation_type="training_step",
        epoch=epoch,
        step=step,
        loss_value=loss_value,
        accuracy=accuracy,
        **kwargs,
    )


def log_inference_request(
    model_name: str, batch_size: int, input_shape: tuple | None = None, **kwargs
) -> None:
    """Log inference request."""
    logger = get_global_logger()
    logger.info(
        f"Inference request: {model_name}",
        operation_type="inference_request",
        model_name=model_name,
        batch_size=batch_size,
        input_shape=str(input_shape) if input_shape else None,
        **kwargs,
    )


def log_model_load(
    model_name: str, model_size_mb: float | None = None, **kwargs
) -> None:
    """Log model loading event."""
    logger = get_global_logger()
    logger.info(
        f"Loading model: {model_name}",
        operation_type="model_load",
        model_name=model_name,
        model_size_mb=model_size_mb,
        **kwargs,
    )


def log_error(error: Exception, operation_type: str = "unknown", **kwargs) -> None:
    """Log error with context."""
    logger = get_global_logger()
    logger.exception(
        f"Error in {operation_type}: {error!s}",
        operation_type=operation_type,
        error_type=type(error).__name__,
        **kwargs,
    )
