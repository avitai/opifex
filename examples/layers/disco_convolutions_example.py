# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
"""
# DISCO Convolutions for Neural Operators

| Metadata | Value |
|----------|-------|
| **Level** | Intermediate |
| **Runtime** | ~5 min (CPU) |
| **Prerequisites** | JAX, Flax NNX, Convolution basics |
| **Format** | Python + Jupyter |

## Overview

Discrete-Continuous (DISCO) convolutions generalize standard convolutions to work on
both structured and unstructured grids. Unlike standard convolutions that require regular
grid spacing, DISCO convolutions can operate on arbitrary point distributions, making them
ideal for scientific applications with irregular meshes.

This example reproduces and extends the classic 'Einstein' demo from the NeuralOperator
library, demonstrating basic DISCO convolution, equidistant optimization for regular grids,
and encoder-decoder architectures built with DISCO layers.

## Learning Goals

1. Apply `DiscreteContinuousConv2d` for general convolution on arbitrary grids
2. Use `EquidistantDiscreteContinuousConv2d` for optimized regular grid processing
3. Build encoder-decoder architectures with `create_disco_encoder` and `create_disco_decoder`
4. Compare performance between regular and equidistant DISCO variants
"""

# %%
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

# Import Opifex layers
from opifex.neural.operators.specialized.disco import (
    create_disco_decoder,
    create_disco_encoder,
    DiscreteContinuousConv2d,
    EquidistantDiscreteContinuousConv2d,
)


# %% [markdown]
"""
## Utilities

First, we define a helper function to create the test image (Einstein-like).
"""


# %%
def create_einstein_test_image(size: int = 64) -> jnp.ndarray:
    """Create a simplified Einstein-like test image for DISCO demonstration.

    Args:
        size: Image size (square)

    Returns:
        Test image array of shape (size, size)
    """
    # Create coordinate grids
    x = jnp.linspace(-1, 1, size)
    y = jnp.linspace(-1, 1, size)
    X, Y = jnp.meshgrid(x, y)

    # Create Einstein-like face features
    # Head (circle)
    head = jnp.exp(-3 * (X**2 + Y**2))

    # Eyes (two smaller circles)
    left_eye = 0.7 * jnp.exp(-15 * ((X + 0.3) ** 2 + (Y + 0.2) ** 2))
    right_eye = 0.7 * jnp.exp(-15 * ((X - 0.3) ** 2 + (Y + 0.2) ** 2))

    # Mouth (curved line)
    mouth = 0.5 * jnp.exp(-20 * (X**2 + (Y - 0.4) ** 2)) * (jnp.abs(X) < 0.4)

    # Hair (top region with some texture)
    hair = 0.6 * jnp.exp(-2 * (X**2 + (Y + 0.7) ** 2)) * (Y > 0.1)

    # Combine features
    image = head + left_eye + right_eye + mouth + hair

    # Add some noise for texture
    key = jax.random.PRNGKey(42)
    noise = 0.1 * jax.random.normal(key, (size, size))
    image = image + noise

    # Normalize to [0, 1]
    return (image - image.min()) / (image.max() - image.min())


# %% [markdown]
"""
## 1. Basic DISCO Convolution

We demonstrate `DiscreteContinuousConv2d` which works for general convolutions.
"""


# %%
def demonstrate_disco_convolution_basic(
    image_size: int = 32,
    in_channels: int = 1,
    out_channels: int = 4,
    kernel_size: int = 3,
) -> dict[str, Any]:
    """Demonstrate basic DiSCo convolution on 2D images."""
    print()
    print("Basic DISCO Convolution Demonstration")
    print(f"   Image Size: {image_size}x{image_size}")
    print(f"   Channels: {in_channels} -> {out_channels}")
    print(f"   Kernel Size: {kernel_size}x{kernel_size}")

    # Create test image
    test_image = create_einstein_test_image(image_size)

    # Prepare input tensor (batch, height, width, channels)
    input_tensor = test_image[None, :, :, None]  # Add batch and channel dims
    if in_channels > 1:
        # Replicate across channels
        input_tensor = jnp.repeat(input_tensor, in_channels, axis=-1)

    # Initialize DISCO convolution layer
    disco_conv = DiscreteContinuousConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        activation=jax.nn.gelu,
        rngs=nnx.Rngs(42),
    )

    # Apply convolution
    start_time = time.time()
    output_tensor = disco_conv(input_tensor)
    conv_time = time.time() - start_time

    print(f"   Input Shape: {input_tensor.shape}")
    print(f"   Output Shape: {output_tensor.shape}")
    print(f"   Convolution Time: {conv_time * 1000:.2f} ms")

    return {
        "timestamp": datetime.now(UTC).isoformat(),
        "input_image": test_image,
        "input_tensor": input_tensor,
        "output_tensor": output_tensor,
        "disco_conv": disco_conv,
        "conv_time": conv_time,
        "output_shape": output_tensor.shape,
    }


# %% [markdown]
"""
## 2. Equidistant Optimization

`EquidistantDiscreteContinuousConv2d` provides optimized performance for regular grids.
"""


# %%
def demonstrate_equidistant_optimization(
    image_size: int = 48,
    in_channels: int = 2,
    out_channels: int = 3,
    grid_spacing: float = 0.1,
) -> dict[str, Any]:
    """Demonstrate DiSCo convolution with equidistant grid optimization."""
    print()
    print("Equidistant DISCO Convolution Demonstration")
    print(f"   Image Size: {image_size}x{image_size}")

    # Create test data
    test_image = create_einstein_test_image(image_size)
    input_tensor = jnp.stack([test_image, test_image * 0.5], axis=-1)[None, :, :, :]

    # Regular DISCO convolution
    regular_conv = DiscreteContinuousConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        rngs=nnx.Rngs(43),
    )

    # Equidistant DISCO convolution
    equidistant_conv = EquidistantDiscreteContinuousConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        grid_spacing=grid_spacing,
        rngs=nnx.Rngs(44),
    )

    # Performance comparison
    start_time = time.time()
    regular_output = regular_conv(input_tensor)
    regular_time = time.time() - start_time

    start_time = time.time()
    equidistant_output = equidistant_conv(input_tensor)
    equidistant_time = time.time() - start_time

    speedup = regular_time / equidistant_time

    print(f"   Regular DISCO Time: {regular_time * 1000:.2f} ms")
    print(f"   Equidistant DISCO Time: {equidistant_time * 1000:.2f} ms")
    print(f"   Speedup Factor: {speedup:.2f}x")

    return {
        "regular_output": regular_output,
        "equidistant_output": equidistant_output,
        "regular_time": regular_time,
        "equidistant_time": equidistant_time,
        "speedup_factor": speedup,
    }


# %% [markdown]
"""
## 3. Encoder-Decoder Architecture

Building a full autoencoder using DISCO layers.
"""


# %%
def demonstrate_encoder_decoder_architecture(
    image_size: int = 32,
    in_channels: int = 1,
    latent_channels: int = 64,
) -> dict[str, Any]:
    """Demonstrate encoder-decoder architecture with DiSCo convolutions."""
    print()
    print("DISCO Encoder-Decoder Architecture Demonstration")

    # Create test image
    test_image = create_einstein_test_image(image_size)
    input_tensor = test_image[None, :, :, None]

    # Encoder: downsamples
    encoder = create_disco_encoder(
        in_channels=in_channels,
        hidden_channels=(16, 32, latent_channels),
        kernel_size=3,
        use_equidistant=True,
        rngs=nnx.Rngs(45),
    )

    # Decoder: upsamples
    decoder = create_disco_decoder(
        hidden_channels=(latent_channels, 32, 16),
        out_channels=in_channels,
        kernel_size=3,
        use_equidistant=True,
        rngs=nnx.Rngs(46),
    )

    # Forward pass
    start_time = time.time()
    encoded = encoder(input_tensor)
    reconstructed = decoder(encoded)
    total_time = time.time() - start_time

    # Compute error
    reconstruction_error = jnp.mean((input_tensor - reconstructed) ** 2)

    print(f"   Encoded Shape: {encoded.shape}")
    print(f"   Reconstructed Shape: {reconstructed.shape}")
    print(f"   Reconstruction Error: {reconstruction_error:.6f}")

    return {
        "original_image": test_image,
        "encoded": encoded,
        "reconstructed": reconstructed,
        "reconstruction_error": reconstruction_error,
    }


# %% [markdown]
"""
## Visualization

Let's visualize the results from all demonstrations.
"""


# %%
def visualize_results(basic, equidistant, enc_dec):
    """Visualize results from all DiSCo convolution demonstrations."""
    _fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    # Basic
    axes[0, 0].imshow(basic["input_image"], cmap="gray")
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(basic["output_tensor"][0, :, :, 0], cmap="viridis")
    axes[0, 1].set_title("Basic Output")
    axes[0, 1].axis("off")

    # Equidistant
    axes[0, 2].imshow(equidistant["regular_output"][0, :, :, 0], cmap="plasma")
    axes[0, 2].set_title("Regular Output")
    axes[0, 2].axis("off")

    axes[0, 3].imshow(equidistant["equidistant_output"][0, :, :, 0], cmap="plasma")
    axes[0, 3].set_title(f"Equidistant (Speedup: {equidistant['speedup_factor']:.1f}x)")
    axes[0, 3].axis("off")

    # Encoder-Decoder
    axes[1, 0].imshow(enc_dec["original_image"], cmap="gray")
    axes[1, 0].set_title("Original")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(enc_dec["encoded"][0, :, :, 0], cmap="magma")
    axes[1, 1].set_title("Latent")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(enc_dec["reconstructed"][0, :, :, 0], cmap="gray")
    axes[1, 2].set_title(f"Reconstructed (Err: {enc_dec['reconstruction_error']:.2e})")
    axes[1, 2].axis("off")

    # Summary
    axes[1, 3].text(
        0.5, 0.5, "DISCO Demo\nComplete", ha="center", va="center", fontsize=16
    )
    axes[1, 3].axis("off")

    plt.tight_layout()

    # Save figure
    output_dir = Path("docs/assets/examples/disco_convolutions")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "disco_convolutions_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"   Visualization saved to: {output_path}")


# %% [markdown]
"""
## Results Summary

| Component | Description | Performance |
|-----------|-------------|-------------|
| Basic DISCO Conv | General convolution on arbitrary grids | Baseline |
| Equidistant DISCO | Optimized for regular grids | Speedup varies by hardware |
| Encoder-Decoder | Full autoencoder with DISCO layers | Reconstruction error shown |

## Next Steps

### Experiments to Try

1. Increase `out_channels` and observe feature extraction quality
2. Try different `kernel_size` values (5, 7) for broader receptive fields
3. Apply DISCO convolutions to real scientific data on irregular meshes

### Related Examples

- [Grid Embeddings](grid_embeddings_example.md) - Spatial coordinate injection for neural operators
- [Fourier Continuation](fourier_continuation_example.md) - Boundary handling for spectral methods
- [FNO Darcy Comprehensive](../models/fno_darcy_comprehensive.md) - Full neural operator training

### API Reference

- [`DiscreteContinuousConv2d`](../../api/neural.md) - General DISCO convolution
- [`EquidistantDiscreteContinuousConv2d`](../../api/neural.md) - Optimized regular grid DISCO
- [`create_disco_encoder`](../../api/neural.md) - DISCO encoder factory
- [`create_disco_decoder`](../../api/neural.md) - DISCO decoder factory
"""


# %%
def main():
    """Run all DISCO convolution demonstrations."""
    print("=" * 60)
    print("DISCO CONVOLUTIONS FOR NEURAL OPERATORS")
    print("=" * 60)

    basic_results = demonstrate_disco_convolution_basic(
        image_size=32, in_channels=1, out_channels=4
    )
    equidistant_results = demonstrate_equidistant_optimization()
    encoder_decoder_results = demonstrate_encoder_decoder_architecture()
    visualize_results(basic_results, equidistant_results, encoder_decoder_results)

    print()
    print("=" * 60)
    print("DISCO convolution demonstrations complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
