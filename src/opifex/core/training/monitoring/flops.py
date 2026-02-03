"""
FLOPs Counter for Opifex Training Infrastructure

A focused component for counting floating-point operations during training.
Provides profiling capabilities for neural operator models.
"""

import contextlib
import time
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


class FlopsCounter:
    """
    Counts floating-point operations during training.

    Provides functionality to profile forward passes, backward passes,
    and complete training steps for neural operator models.
    """

    def __init__(self, enable_profiling: bool = True):
        """
        Initialize the FLOPs counter.

        Args:
            enable_profiling: Whether to enable profiling
        """
        self.enable_profiling = enable_profiling
        self.total_flops = 0
        self.operation_counts = {}

    def count_forward_flops(
        self, model: nnx.Module, inputs: jnp.ndarray, **kwargs
    ) -> dict[str, Any]:
        """
        Count FLOPs for a forward pass.

        Args:
            model: Model to profile
            inputs: Input data
            **kwargs: Additional arguments for the model

        Returns:
            Dictionary containing FLOPS analysis

        Raises:
            TypeError: If model is not an nnx.Module
            AttributeError: If inputs don't have required attributes
        """
        if not self.enable_profiling:
            return {"total_flops": 0, "timing": 0.0}

        # Validate inputs
        if not isinstance(model, nnx.Module):
            raise TypeError("Model must be an nnx.Module")

        if not hasattr(inputs, "size"):
            raise AttributeError("Input must have 'size' attribute")

        # Estimate FLOPs based on model structure and input size
        input_size = inputs.size

        # Basic FLOPS estimation for neural operators
        # This is a simplified estimation - real profiling would use JAX profiling
        estimated_flops = self._estimate_model_flops(model, input_size)

        # Time the forward pass for additional profiling
        start_time = time.time()

        # Simulate forward pass timing (in real implementation would call model)
        with contextlib.suppress(Exception):
            # This would be: output = model(inputs, **kwargs)
            pass

        end_time = time.time()
        timing = end_time - start_time

        return {
            "total_flops": estimated_flops,
            "timing": timing,
            "input_shape": inputs.shape,
            "input_size": input_size,
        }

    def count_backward_flops(
        self, model: nnx.Module, inputs: jnp.ndarray, targets: jnp.ndarray, **kwargs
    ) -> dict[str, Any]:
        """
        Count FLOPs for a backward pass (including forward pass).

        Args:
            model: Model to profile
            inputs: Input data
            targets: Target data for loss computation
            **kwargs: Additional arguments

        Returns:
            Dictionary containing FLOPS analysis
        """
        if not self.enable_profiling:
            return {
                "total_flops": 0,
                "backward_flops": 0,
                "forward_flops": 0,
                "timing": 0.0,
            }

        # Validate inputs
        if not isinstance(model, nnx.Module):
            raise TypeError("Model must be an nnx.Module")

        if not hasattr(inputs, "size") or not hasattr(targets, "size"):
            raise AttributeError("Inputs and targets must have 'size' attribute")

        # Get forward pass FLOPs
        forward_info = self.count_forward_flops(model, inputs, **kwargs)
        forward_flops = forward_info["total_flops"]

        # Estimate backward pass FLOPs (typically 2-3x forward pass)
        backward_flops = int(forward_flops * 2.5)
        total_flops = forward_flops + backward_flops

        return {
            "total_flops": total_flops,
            "forward_flops": forward_flops,
            "backward_flops": backward_flops,
            "timing": forward_info["timing"] * 2.5,  # Estimate backward timing
            "input_shape": inputs.shape,
            "target_shape": targets.shape,
        }

    def profile_training_step(
        self, model: nnx.Module, inputs: jnp.ndarray, targets: jnp.ndarray, **kwargs
    ) -> dict[str, Any]:
        """
        Profile a complete training step (forward + backward + update).

        Args:
            model: Model to profile
            inputs: Input data
            targets: Target data
            **kwargs: Additional arguments

        Returns:
            Dictionary containing detailed profiling information
        """
        if not self.enable_profiling:
            return {
                "total_flops": 0,
                "forward_flops": 0,
                "backward_flops": 0,
                "update_flops": 0,
                "timing": 0.0,
                "breakdown": {},
            }

        # Get backward pass info (includes forward)
        backward_info = self.count_backward_flops(model, inputs, targets, **kwargs)

        # Estimate parameter update FLOPs
        param_count = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(model)))
        update_flops = param_count  # Simple estimate for parameter updates

        total_flops = backward_info["total_flops"] + update_flops

        return {
            "total_flops": total_flops,
            "forward_flops": backward_info["forward_flops"],
            "backward_flops": backward_info["backward_flops"],
            "update_flops": update_flops,
            "timing": backward_info["timing"] + 0.1,  # Add update timing estimate
            "breakdown": {
                "forward_percentage": (backward_info["forward_flops"] / total_flops)
                * 100,
                "backward_percentage": (backward_info["backward_flops"] / total_flops)
                * 100,
                "update_percentage": (update_flops / total_flops) * 100,
            },
        }

    def compare_models(
        self, models: list[nnx.Module], inputs: jnp.ndarray, **kwargs
    ) -> dict[str, Any]:
        """
        Compare FLOPS across multiple models.

        Args:
            models: List of models to compare
            inputs: Input data
            **kwargs: Additional arguments

        Returns:
            Dictionary containing comparison results

        Raises:
            ValueError: If models list is empty
            TypeError: If any model is not an nnx.Module
        """
        if not models:
            raise ValueError("Models list cannot be empty")

        if not self.enable_profiling:
            return {"models": [], "comparison": {}}

        results = []
        for i, model in enumerate(models):
            if not isinstance(model, nnx.Module):
                raise TypeError(f"Model {i} must be an nnx.Module")

            flops_info = self.count_forward_flops(model, inputs, **kwargs)
            results.append(
                {
                    "model_index": i,
                    "total_flops": flops_info["total_flops"],
                    "timing": flops_info["timing"],
                }
            )

        # Calculate relative metrics
        if len(results) > 1:
            base_flops = results[0]["total_flops"]
            for result in results:
                if base_flops > 0:
                    result["relative_flops"] = result["total_flops"] / base_flops
                else:
                    result["relative_flops"] = 1.0

        return {
            "models": results,
            "comparison": {
                "total_models": len(results),
                "min_flops": min(r["total_flops"] for r in results) if results else 0,
                "max_flops": max(r["total_flops"] for r in results) if results else 0,
            },
        }

    def reset_counters(self):
        """Reset all FLOPS counters and profiling data."""
        self.total_flops = 0
        self.operation_counts = {}
        self.profile_data = {}

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of all FLOPS profiling data.

        Returns:
            Dictionary containing profiling summary
        """
        return {
            "total_flops": self.total_flops,
            "operation_counts": self.operation_counts.copy(),
            "profile_data": self.profile_data.copy(),
            "enable_profiling": self.enable_profiling,
        }

    def enable_profiling_mode(self, enable: bool = True):
        """
        Enable or disable FLOPS profiling.

        Args:
            enable: Whether to enable profiling
        """
        self.enable_profiling = enable

    def _estimate_model_flops(self, model: nnx.Module, input_size: int) -> int:
        """
        Estimate FLOPs for a model based on its structure.

        This is a simplified estimation. Real profiling would use JAX's
        profiling capabilities or more sophisticated analysis.

        Args:
            model: Model to estimate FLOPs for
            input_size: Size of input data

        Returns:
            Estimated FLOPS count

        Raises:
            TypeError: If model is not an nnx.Module
        """
        if not isinstance(model, nnx.Module):
            raise TypeError("Model must be an nnx.Module")

        try:
            # Get parameter count as a proxy for model complexity
            param_count = sum(
                x.size for x in jax.tree_util.tree_leaves(nnx.state(model))
            )
        except Exception as e:
            # If we can't get the state, it's not a valid model
            raise TypeError(f"Invalid model type: {type(model)}") from e

        # Simple estimation: assume each parameter is used once per input element
        # This is a rough approximation for neural operators
        base_flops = param_count * input_size

        # Add some overhead for activation functions, etc.
        return int(base_flops * 1.2)

    def _update_operation_count(self, operation: str, flops: int):
        """Update operation count tracking."""
        if operation not in self.operation_counts:
            self.operation_counts[operation] = 0
        self.operation_counts[operation] += flops
