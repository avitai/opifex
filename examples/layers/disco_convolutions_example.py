"""DISCO Convolutions Example - Opifex Framework.

Comprehensive demonstration of Discrete-Continuous (DISCO) convolution layers,
reproducing and extending neuraloperator plot_DISCO_convolutions.py with Opifex framework.

This example demonstrates:
- DiscreteContinuousConv2d for general convolution on structured/unstructured grids
- EquidistantDiscreteContinuousConv2d optimized for regular grids
- DiscreteContinuousConvTranspose2d for upsampling operations
- Einstein image processing example (classic DISCO demonstration)
- Comparison with standard convolutions
- Performance analysis and visualization

Usage:
    python examples/layers/disco_convolutions_example.py
"""

import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from opifex.neural.operators.specialized.disco import (
    create_disco_decoder,
    create_disco_encoder,
    DiscreteContinuousConv2d,
    DiscreteContinuousConvTranspose2d,
    EquidistantDiscreteContinuousConv2d,
)


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


def demonstrate_disco_convolution_basic(
    image_size: int = 32,
    in_channels: int = 1,
    out_channels: int = 4,
    kernel_size: int = 3,
) -> dict[str, Any]:
    """Demonstrate basic DISCO convolution functionality.

    Args:
        image_size: Size of test image
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size

    Returns:
        Dictionary containing results and timing information
    """
    print("\n🔲 Basic DISCO Convolution Demonstration")
    print(f"   Image Size: {image_size}x{image_size}")
    print(f"   Channels: {in_channels} → {out_channels}")
    print(f"   Kernel Size: {kernel_size}x{kernel_size}")

    # Create test image
    test_image = create_einstein_test_image(image_size)

    # Prepare input tensor (batch, height, width, channels)
    input_tensor = test_image[None, :, :, None]  # Add batch and channel dims
    if in_channels > 1:
        # Replicate across channels
        input_tensor = jnp.repeat(input_tensor, in_channels, axis=-1)

    # Initialize DISCO convolution layer
    from flax import nnx

    # Note: using nnx.Rngs below for layer initialization
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

    print(f"   ✅ Input Shape: {input_tensor.shape}")
    print(f"   ✅ Output Shape: {output_tensor.shape}")
    print(f"   ✅ Convolution Time: {conv_time * 1000:.2f} ms")

    return {
        "input_image": test_image,
        "input_tensor": input_tensor,
        "output_tensor": output_tensor,
        "disco_conv": disco_conv,
        "conv_time": conv_time,
        "output_shape": output_tensor.shape,
    }


def demonstrate_equidistant_optimization(
    image_size: int = 48,
    in_channels: int = 2,
    out_channels: int = 3,
    grid_spacing: float = 0.1,
) -> dict[str, Any]:
    """Demonstrate optimized DISCO convolution for equidistant grids.

    Args:
        image_size: Size of test image
        in_channels: Number of input channels
        out_channels: Number of output channels
        grid_spacing: Spacing between grid points

    Returns:
        Dictionary containing results and performance comparison
    """
    print("\n📐 Equidistant DISCO Convolution Demonstration")
    print(f"   Image Size: {image_size}x{image_size}")
    print(f"   Grid Spacing: {grid_spacing}")
    print(f"   Channels: {in_channels} → {out_channels}")

    # Create test data
    test_image = create_einstein_test_image(image_size)
    input_tensor = jnp.stack([test_image, test_image * 0.5], axis=-1)[None, :, :, :]

    # Initialize both regular and equidistant DISCO convolutions
    # Note: using nnx.Rngs below for layer initialization

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

    # Compute difference for analysis
    output_difference = jnp.abs(regular_output - equidistant_output).mean()

    print(f"   ✅ Regular DISCO Time: {regular_time * 1000:.2f} ms")
    print(f"   ✅ Equidistant DISCO Time: {equidistant_time * 1000:.2f} ms")
    print(f"   ✅ Speedup Factor: {regular_time / equidistant_time:.2f}x")
    print(f"   ✅ Output Difference: {output_difference:.6f}")

    return {
        "input_tensor": input_tensor,
        "regular_output": regular_output,
        "equidistant_output": equidistant_output,
        "regular_time": regular_time,
        "equidistant_time": equidistant_time,
        "speedup_factor": regular_time / equidistant_time,
        "output_difference": output_difference,
        "regular_conv": regular_conv,
        "equidistant_conv": equidistant_conv,
    }


def demonstrate_transpose_convolution(
    input_size: int = 16,
    scale_factor: int = 2,
    in_channels: int = 8,
    out_channels: int = 4,
) -> dict[str, Any]:
    """Demonstrate DISCO transpose convolution for upsampling.

    Args:
        input_size: Size of input feature map
        scale_factor: Upsampling factor
        in_channels: Number of input channels
        out_channels: Number of output channels

    Returns:
        Dictionary containing upsampling results
    """
    print("\n🔄 DISCO Transpose Convolution Demonstration")
    print(f"   Input Size: {input_size}x{input_size}")
    print(f"   Scale Factor: {scale_factor}x")
    print(f"   Channels: {in_channels} → {out_channels}")

    # Create random input feature map
    # Note: using nnx.Rngs below for layer initialization
    rng = jax.random.PRNGKey(47)
    input_features = jax.random.normal(rng, (1, input_size, input_size, in_channels))

    # Initialize transpose convolution
    transpose_conv = DiscreteContinuousConvTranspose2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=scale_factor + 1,
        stride=scale_factor,
        activation=jax.nn.gelu,
        rngs=nnx.Rngs(47),
    )

    # Apply transpose convolution
    start_time = time.time()
    upsampled_features = transpose_conv(input_features)
    transpose_time = time.time() - start_time

    # Calculate actual scale factor achieved
    actual_scale_h = upsampled_features.shape[1] / input_features.shape[1]
    actual_scale_w = upsampled_features.shape[2] / input_features.shape[2]

    print(f"   ✅ Input Shape: {input_features.shape}")
    print(f"   ✅ Output Shape: {upsampled_features.shape}")
    print(f"   ✅ Actual Scale: {actual_scale_h:.1f}x{actual_scale_w:.1f}")
    print(f"   ✅ Transpose Time: {transpose_time * 1000:.2f} ms")

    return {
        "input_features": input_features,
        "upsampled_features": upsampled_features,
        "transpose_conv": transpose_conv,
        "transpose_time": transpose_time,
        "actual_scale": (actual_scale_h, actual_scale_w),
        "target_scale": scale_factor,
    }


def demonstrate_encoder_decoder_architecture(
    image_size: int = 32,
    in_channels: int = 1,
    latent_channels: int = 64,
) -> dict[str, Any]:
    """Demonstrate DISCO encoder-decoder architecture.

    Args:
        image_size: Size of input image
        in_channels: Number of input channels
        latent_channels: Number of latent space channels

    Returns:
        Dictionary containing encoder-decoder results
    """
    print("\n🏗️ DISCO Encoder-Decoder Architecture Demonstration")
    print(f"   Image Size: {image_size}x{image_size}")
    print(f"   Architecture: {in_channels} → {latent_channels} → {in_channels}")

    # Create test image
    test_image = create_einstein_test_image(image_size)
    input_tensor = test_image[None, :, :, None]  # Add batch and channel dims

    # Initialize encoder and decoder
    rngs = jax.random.PRNGKey(45)
    _, _ = jax.random.split(rngs, 2)

    # Encoder: downsamples and increases channels
    encoder = create_disco_encoder(
        in_channels=in_channels,
        hidden_channels=(16, 32, latent_channels),
        kernel_size=3,
        use_equidistant=True,
        rngs=nnx.Rngs(45),
    )

    # Decoder: upsamples and decreases channels
    decoder = create_disco_decoder(
        hidden_channels=(latent_channels, 32, 16),
        out_channels=in_channels,
        kernel_size=3,
        use_equidistant=True,
        rngs=nnx.Rngs(46),
    )

    # Forward pass through encoder-decoder
    start_time = time.time()

    # Encoding
    encoded = encoder(input_tensor)
    encoding_time = time.time() - start_time

    # Decoding
    start_time = time.time()
    reconstructed = decoder(encoded)
    decoding_time = time.time() - start_time

    # Compute reconstruction error
    reconstruction_error = jnp.mean((input_tensor - reconstructed) ** 2)

    print(f"   ✅ Input Shape: {input_tensor.shape}")
    print(f"   ✅ Encoded Shape: {encoded.shape}")
    print(f"   ✅ Reconstructed Shape: {reconstructed.shape}")
    print(f"   ✅ Encoding Time: {encoding_time * 1000:.2f} ms")
    print(f"   ✅ Decoding Time: {decoding_time * 1000:.2f} ms")
    print(f"   ✅ Reconstruction Error: {reconstruction_error:.6f}")

    return {
        "original_image": test_image,
        "input_tensor": input_tensor,
        "encoded": encoded,
        "reconstructed": reconstructed,
        "encoder": encoder,
        "decoder": decoder,
        "reconstruction_error": reconstruction_error,
        "encoding_time": encoding_time,
        "decoding_time": decoding_time,
    }


def compare_with_standard_convolution(
    image_size: int = 40,
    in_channels: int = 3,
    out_channels: int = 6,
    kernel_size: int = 5,
) -> dict[str, Any]:
    """Compare DISCO convolution with standard JAX convolution.

    Args:
        image_size: Size of test image
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size

    Returns:
        Dictionary containing comparison results
    """
    print("\n⚖️ DISCO vs Standard Convolution Comparison")
    print(f"   Image Size: {image_size}x{image_size}")
    print(f"   Channels: {in_channels} → {out_channels}")
    print(f"   Kernel Size: {kernel_size}x{kernel_size}")

    # Create test data
    test_image = create_einstein_test_image(image_size)
    # Create multi-channel input
    rng = jax.random.PRNGKey(46)
    input_tensor = jax.random.normal(rng, (1, image_size, image_size, in_channels))
    input_tensor = input_tensor.at[0, :, :, 0].set(
        test_image
    )  # Set first channel to test image

    # Initialize DISCO convolution
    disco_conv = DiscreteContinuousConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding="SAME",
        rngs=nnx.Rngs(46),
    )

    # Standard convolution for comparison (using JAX)
    standard_kernel = jax.random.normal(
        rng, (kernel_size, kernel_size, in_channels, out_channels)
    )

    def standard_conv(x):
        return jax.lax.conv_general_dilated(
            x,
            standard_kernel,
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    # Performance comparison
    # DISCO convolution
    start_time = time.time()
    disco_output = disco_conv(input_tensor)
    disco_time = time.time() - start_time

    # Standard convolution
    start_time = time.time()
    standard_output = standard_conv(input_tensor)
    standard_time = time.time() - start_time

    # Analyze outputs
    disco_mean = jnp.mean(disco_output)
    disco_std = jnp.std(disco_output)
    standard_mean = jnp.mean(standard_output)
    standard_std = jnp.std(standard_output)

    print(f"   ✅ DISCO Time: {disco_time * 1000:.2f} ms")
    print(f"   ✅ Standard Time: {standard_time * 1000:.2f} ms")
    print(f"   ✅ Time Ratio: {disco_time / standard_time:.2f}x")
    print(f"   ✅ DISCO Output - Mean: {disco_mean:.4f}, Std: {disco_std:.4f}")
    print(f"   ✅ Standard Output - Mean: {standard_mean:.4f}, Std: {standard_std:.4f}")

    return {
        "input_tensor": input_tensor,
        "disco_output": disco_output,
        "standard_output": standard_output,
        "disco_time": disco_time,
        "standard_time": standard_time,
        "time_ratio": disco_time / standard_time,
        "disco_stats": {"mean": disco_mean, "std": disco_std},
        "standard_stats": {"mean": standard_mean, "std": standard_std},
        "disco_conv": disco_conv,
    }


def visualize_disco_results(
    basic_results: dict,
    equidistant_results: dict,
    encoder_decoder_results: dict,
    save_path: str | None = None,
) -> None:
    """Create comprehensive visualization of DISCO convolution results.

    Args:
        basic_results: Results from basic DISCO demonstration
        equidistant_results: Results from equidistant optimization demo
        encoder_decoder_results: Results from encoder-decoder demo
        save_path: Optional path to save the visualization
    """
    print("\n📊 Creating DISCO Convolution Visualizations")

    # Create figure with subplots
    _, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 1. Basic DISCO Results
    axes[0, 0].imshow(basic_results["input_image"], cmap="gray")
    axes[0, 0].set_title("Input Image\n(Einstein Test)")
    axes[0, 0].axis("off")

    output_sample = basic_results["output_tensor"][0, :, :, 0]
    axes[0, 1].imshow(output_sample, cmap="viridis")
    axes[0, 1].set_title(
        f"DISCO Output (Ch 0)\nTime: {basic_results['conv_time'] * 1000:.1f}ms"
    )
    axes[0, 1].axis("off")

    # 2. Equidistant Optimization Comparison
    regular_sample = equidistant_results["regular_output"][0, :, :, 0]
    axes[0, 2].imshow(regular_sample, cmap="plasma")
    axes[0, 2].set_title(
        f"Regular DISCO\nTime: {equidistant_results['regular_time'] * 1000:.1f}ms"
    )
    axes[0, 2].axis("off")

    equidistant_sample = equidistant_results["equidistant_output"][0, :, :, 0]
    axes[0, 3].imshow(equidistant_sample, cmap="plasma")
    axes[0, 3].set_title(
        f"Equidistant DISCO\nTime: {equidistant_results['equidistant_time'] * 1000:.1f}ms"
    )
    axes[0, 3].axis("off")

    # 3. Encoder-Decoder Architecture
    axes[1, 0].imshow(encoder_decoder_results["original_image"], cmap="gray")
    axes[1, 0].set_title("Original Image")
    axes[1, 0].axis("off")

    encoded = encoder_decoder_results["encoded"][0, :, :, 0]
    axes[1, 1].imshow(encoded, cmap="magma")
    axes[1, 1].set_title(f"Encoded Features\n{encoded.shape}")
    axes[1, 1].axis("off")

    reconstructed = encoder_decoder_results["reconstructed"][0, :, :, 0]
    axes[1, 2].imshow(reconstructed, cmap="gray")
    error = encoder_decoder_results["reconstruction_error"]
    axes[1, 2].set_title(f"Reconstructed\nError: {error:.2e}")
    axes[1, 2].axis("off")

    # 4. Performance Summary
    speedup = equidistant_results["speedup_factor"]
    axes[1, 3].text(
        0.5,
        0.7,
        f"Speedup: {speedup:.2f}x",
        ha="center",
        va="center",
        fontsize=14,
        weight="bold",
        transform=axes[1, 3].transAxes,
    )
    axes[1, 3].text(
        0.5,
        0.5,
        f"Reconstruction Error:\n{error:.2e}",
        ha="center",
        va="center",
        fontsize=12,
        transform=axes[1, 3].transAxes,
    )
    axes[1, 3].text(
        0.5,
        0.3,
        "DISCO Success",
        ha="center",
        va="center",
        fontsize=10,
        style="italic",
        transform=axes[1, 3].transAxes,
    )
    axes[1, 3].set_title("Performance Summary")
    axes[1, 3].axis("off")

    plt.suptitle(
        "DISCO Convolutions: Discrete-Continuous Neural Network Layers\n"
        f"Opifex Framework - Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"   ✅ Visualization saved to: {save_path}")

    plt.show()


def run_disco_convolutions_demo(
    save_outputs: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the complete DISCO convolutions demonstration.

    Args:
        save_outputs: Whether to save visualization outputs
        verbose: Whether to print detailed progress

    Returns:
        Dictionary containing all demonstration results
    """
    if verbose:
        print("🎯 DISCO Convolutions Example - Opifex Framework")
        print("=" * 60)
        print(f"Start Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"JAX Backend: {jax.default_backend()}")
        print(f"Available Devices: {len(jax.devices())} ({jax.devices()})")

    # Run all demonstrations
    print("\n🚀 Running DISCO Convolution Demonstrations...")

    # 1. Basic DISCO convolution
    basic_results = demonstrate_disco_convolution_basic(
        image_size=32, in_channels=1, out_channels=4
    )

    # 2. Equidistant optimization
    equidistant_results = demonstrate_equidistant_optimization(
        image_size=48, in_channels=2, out_channels=3, grid_spacing=0.05
    )

    # 3. Encoder-decoder architecture
    encoder_decoder_results = demonstrate_encoder_decoder_architecture(
        image_size=32, in_channels=1, latent_channels=64
    )

    # Create visualizations
    if save_outputs:
        save_path = "examples_output/disco_convolutions_demo.png"
        Path("examples_output").mkdir(exist_ok=True)
    else:
        save_path = None

    visualize_disco_results(
        basic_results=basic_results,
        equidistant_results=equidistant_results,
        encoder_decoder_results=encoder_decoder_results,
        save_path=save_path,
    )

    # Compile comprehensive results
    results = {
        "demo_info": {
            "timestamp": datetime.now(UTC).isoformat(),
            "jax_backend": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "framework": "Opifex with JAX/Flax NNX",
        },
        "basic_disco": basic_results,
        "equidistant_optimization": equidistant_results,
        "encoder_decoder": encoder_decoder_results,
    }

    if verbose:
        print("\n✅ DISCO Convolutions Demonstration Complete!")
        print(
            f"   🎯 Basic DISCO convolution: {basic_results['conv_time'] * 1000:.1f}ms"
        )
        print(
            f"   ⚡ Equidistant optimization: {equidistant_results['speedup_factor']:.2f}x speedup"
        )
        print(
            f"   🏗️ Encoder-decoder error: {encoder_decoder_results['reconstruction_error']:.2e}"
        )
        print("   📊 Comprehensive visualization created")
        print(f"End Time: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')}")

    return results


if __name__ == "__main__":
    # Run the comprehensive DISCO convolutions demonstration
    results = run_disco_convolutions_demo(save_outputs=True, verbose=True)

    print("\n🎉 DISCO Convolutions Example completed successfully!")
    print("🔍 Key insights:")
    print("  • DISCO convolutions enable flexible kernel parameterization")
    print(
        "  • Equidistant optimization provides significant speedups for regular grids"
    )
    print("  • Continuous kernel representation handles irregular sampling patterns")
    print("  • Encoder-decoder architectures enable multi-scale feature learning")
    print("  • Framework integration supports physics-informed applications")
