"""Opifex Model Serving Server.

Production-ready FastAPI server for Opifex model deployment.
"""

import logging
import os
import signal
import sys
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .core_serving import (
    DeploymentConfig,
    InferenceEngine,
    ModelRegistry,
)


# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("OPIFEX_LOG_LEVEL", "INFO")),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AppState:
    """Application state container."""

    def __init__(self):
        self.config: DeploymentConfig | None = None
        self.inference_engine: InferenceEngine | None = None
        self.model_registry: ModelRegistry | None = None


# Global application state
app = FastAPI(
    title="Opifex Model Serving API",
    description="Production-ready API for Opifex framework model serving",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Application state
app_state = AppState()


def initialize_components():
    """Initialize global components."""
    # Create default configuration
    app_state.config = DeploymentConfig(
        model_name=os.getenv("OPIFEX_MODEL_NAME", "default"),
        model_type=os.getenv("OPIFEX_MODEL_TYPE", "neural_operator"),
        serving_port=int(os.getenv("OPIFEX_PORT", "8080")),
        batch_size=int(os.getenv("OPIFEX_BATCH_SIZE", "32")),
        gpu_enabled=os.getenv("JAX_PLATFORM_NAME", "cpu") == "gpu",
        precision=os.getenv("OPIFEX_PRECISION", "float32"),
    )

    # Initialize components
    app_state.inference_engine = InferenceEngine(app_state.config)
    app_state.model_registry = ModelRegistry(
        storage_path=os.getenv("OPIFEX_MODEL_REGISTRY", "./models")
    )

    logger.info("Opifex components initialized successfully")


@app.on_event("startup")
async def startup_event():
    """Application startup event."""
    logger.info("Starting Opifex Model Serving API")
    initialize_components()


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown event."""
    logger.info("Shutting down Opifex Model Serving API")


def _raise_service_unavailable(detail: str) -> None:
    """Raise HTTP 503 Service Unavailable exception."""
    raise HTTPException(status_code=503, detail=detail)


def _raise_bad_request(detail: str) -> None:
    """Raise HTTP 400 Bad Request exception."""
    raise HTTPException(status_code=400, detail=detail)


@app.get("/health")
async def health_check():
    """Health check endpoint.

    Returns:
        Health status information
    """
    try:
        uptime = 0.0  # Calculate actual uptime in production

        return {
            "status": "healthy",
            "service": "opifex-model-serving",
            "version": "1.0.0",
            "uptime_seconds": uptime,
            "components": {
                "inference_engine": app_state.inference_engine is not None,
                "model_registry": app_state.model_registry is not None,
                "model_loaded": (
                    app_state.inference_engine is not None
                    and app_state.inference_engine.is_initialized
                ),
            },
        }
    except Exception as e:
        logger.exception("Health check failed")
        raise HTTPException(status_code=503, detail="Service unhealthy") from e


@app.get("/")
async def root():
    """Root endpoint with API information.

    Returns:
        API information
    """
    return {
        "service": "Opifex Model Serving API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "models": "/models",
    }


@app.get("/models")
async def list_models():
    """List available models.

    Returns:
        List of available models
    """
    try:
        registry = app_state.model_registry
        if registry is None:
            _raise_service_unavailable("Model registry not available")
            return None  # This will never execute but helps type checker

        models = registry.list_models()
        return {"models": models}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to list models")
        raise HTTPException(status_code=500, detail="Failed to list models") from e


@app.post("/predict")
async def predict(input_data: dict[str, Any]):
    """Prediction endpoint.

    Args:
        input_data: Input data for prediction

    Returns:
        Prediction results
    """
    try:
        engine = app_state.inference_engine
        if engine is None:
            _raise_service_unavailable("Inference engine not available")
            return None  # This will never execute but helps type checker

        if not engine.is_initialized:
            _raise_service_unavailable("Model not loaded")

        # Validate input
        if "data" not in input_data:
            _raise_bad_request("Input must contain 'data' field")

        # Perform prediction using core serving logic
        import jax.numpy as jnp

        data = jnp.array(input_data["data"])
        predictions = engine.predict(data)

        # Format response
        return {
            "predictions": predictions.tolist(),
            "metadata": {
                "model_name": app_state.config.model_name
                if app_state.config
                else "unknown",
                "batch_size": data.shape[0],
                "input_shape": list(data.shape),
                "output_shape": list(predictions.shape),
            },
        }

    except HTTPException:
        raise
    except ValueError as e:
        logger.exception("Validation error")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from e


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics.

    Returns:
        Performance metrics
    """
    try:
        engine = app_state.inference_engine
        if engine is None:
            _raise_service_unavailable("Inference engine not available")
            return None  # This will never execute but helps type checker

        metrics = engine.get_performance_metrics()
        return {"metrics": metrics}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get metrics")
        raise HTTPException(status_code=500, detail="Failed to get metrics") from e


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info("Received signal %s, shutting down...", signum)
    sys.exit(0)


def main():
    """Main entry point for the server."""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Get configuration from environment
    host = os.getenv(
        "OPIFEX_HOST", "127.0.0.1"
    )  # Use localhost instead of 0.0.0.0 for security
    port = int(os.getenv("OPIFEX_PORT", "8080"))
    workers = int(os.getenv("OPIFEX_WORKERS", "1"))
    log_level = os.getenv("OPIFEX_LOG_LEVEL", "info").lower()

    logger.info("Starting Opifex server on %s:%s", host, port)

    # Start server
    uvicorn.run(
        "opifex.deployment.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
        reload=False,  # Disable for production
    )


if __name__ == "__main__":
    main()
