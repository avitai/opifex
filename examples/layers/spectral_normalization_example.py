"""Spectral Normalization Examples - Neural Operator Stability Enhancement.

This example demonstrates the various spectral normalization techniques available
in Opifex for enhancing neural operator stability and controlling Lipschitz constants.
Spectral normalization is particularly important for PDE-solving neural operators
where stability and convergence are critical.

Key Features Demonstrated:
- SpectralLinear: Spectral normalized linear layers
- SpectralNormalizedConv: Spectral normalized convolution layers
- SpectralMultiHeadAttention: Spectral normalized attention mechanisms
- AdaptiveSpectralNorm: Adaptive spectral bounds for flexible control
- PowerIteration: Core algorithm for efficient spectral norm estimation
- create_spectral_neural_operator: Complete spectral normalized architectures
- Stability analysis and Lipschitz constant control demonstrations
"""

import time

import jax
import jax.numpy as jnp
from flax import nnx

# Note: SpectralConvolution here is for spectral NORMALIZATION (different from FNO SpectralConvolution)
# Complete spectral neural operators are in FNO spectral module
from opifex.neural.operators.fno.spectral import create_spectral_neural_operator
from opifex.neural.operators.specialized.spectral_normalization import (
    AdaptiveSpectralNorm,
    PowerIteration,
    spectral_norm_summary,
    SpectralLinear,
    SpectralMultiHeadAttention,
    SpectralNormalizedConv,
)


def create_test_problems():
    """Create test problems for demonstrating spectral normalization benefits."""
    # 1D Function approximation problem
    x_1d = jnp.linspace(-2, 2, 100)
    y_1d = jnp.sin(2 * jnp.pi * x_1d) + 0.5 * jnp.cos(4 * jnp.pi * x_1d)

    # 2D Image denoising problem
    x = jnp.linspace(-1, 1, 32)
    y = jnp.linspace(-1, 1, 32)
    X, Y = jnp.meshgrid(x, y)
    clean_image = jnp.exp(-(X**2 + Y**2)) * jnp.sin(3 * X) * jnp.cos(3 * Y)
    noise = 0.1 * jax.random.normal(jax.random.PRNGKey(42), clean_image.shape)
    noisy_image = clean_image + noise

    # PDE solution problem (heat equation)
    nx, nt = 64, 50
    x_pde = jnp.linspace(0, 1, nx)
    t_pde = jnp.linspace(0, 0.1, nt)

    # Initial condition: Gaussian pulse
    initial_temp = jnp.exp(-50 * (x_pde - 0.5) ** 2)

    return {
        "function_1d": (x_1d, y_1d),
        "image_denoising": (noisy_image, clean_image),
        "pde_initial": (x_pde, t_pde, initial_temp),
    }


def demonstrate_basic_spectral_layers():
    """Demonstrate basic spectral normalization layers."""
    print("🔬 BASIC SPECTRAL NORMALIZATION LAYERS")
    print("=" * 50)

    rngs = nnx.Rngs(42)

    # Linear layer comparison
    print("\n📊 Linear Layer Comparison:")
    regular_linear = nnx.Linear(10, 5, rngs=rngs)
    spectral_linear = SpectralLinear(10, 5, power_iterations=5, rngs=rngs)

    # Test input
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (8, 10))

    # Regular forward pass
    y_regular = regular_linear(x)
    print(f"   Regular Linear: {x.shape} -> {y_regular.shape}")

    # Spectral normalized forward pass
    y_spectral = spectral_linear(x, training=True)
    print(f"   Spectral Linear: {x.shape} -> {y_spectral.shape}")

    # Analyze spectral norms
    regular_spectral_norm = jnp.linalg.norm(
        jnp.linalg.svd(regular_linear.kernel.value, compute_uv=False), ord=2
    )
    spectral_norm_estimate, _ = spectral_linear.power_iter(
        spectral_linear.linear.kernel.value, training=False
    )

    print(f"   Regular kernel spectral norm: {regular_spectral_norm:.3f}")
    print(f"   Spectral normalized estimate: {spectral_norm_estimate:.3f}")

    # Convolution layer comparison
    print("\n🖼️ Convolution Layer Comparison:")
    regular_conv = nnx.Conv(3, 16, kernel_size=3, rngs=rngs)
    spectral_conv = SpectralNormalizedConv(
        3, 16, kernel_size=3, power_iterations=3, rngs=rngs
    )

    # Test input
    x_img = jax.random.normal(key, (4, 32, 32, 3))

    y_regular_conv = regular_conv(x_img)
    y_spectral_conv = spectral_conv(x_img, training=True)

    print(f"   Regular Conv: {x_img.shape} -> {y_regular_conv.shape}")
    print(f"   Spectral Conv: {x_img.shape} -> {y_spectral_conv.shape}")


def demonstrate_spectral_attention():
    """Demonstrate spectral normalized attention mechanisms."""
    print("\n🧠 SPECTRAL NORMALIZED ATTENTION")
    print("=" * 50)

    rngs = nnx.Rngs(42)

    # Create spectral normalized attention
    spectral_attention = SpectralMultiHeadAttention(
        num_heads=8, in_features=64, power_iterations=3, rngs=rngs
    )

    print("🔧 Attention configuration:")
    print(f"   Number of heads: {spectral_attention.num_heads}")
    print(f"   Feature dimension: {spectral_attention.qkv_features}")
    print(f"   Head dimension: {spectral_attention.head_dim}")

    # Test sequence data (like neural operator coordinates)
    key = jax.random.PRNGKey(0)
    batch_size, seq_len, features = 2, 32, 64
    x = jax.random.normal(key, (batch_size, seq_len, features))

    print(f"\n📈 Processing sequence: {x.shape}")

    # Forward pass
    start_time = time.time()
    output = spectral_attention(x, training=True)
    end_time = time.time()

    print(f"   Output shape: {output.shape}")
    print(f"   Forward pass time: {(end_time - start_time) * 1000:.2f} ms")

    # Test with causal mask
    mask = jnp.tril(
        jnp.ones((batch_size, spectral_attention.num_heads, seq_len, seq_len))
    )
    output_masked = spectral_attention(x, mask=mask, training=True)

    print(f"   Masked output shape: {output_masked.shape}")
    print(f"   Attention mask applied: {mask.shape}")


def demonstrate_adaptive_spectral_norm():
    """Demonstrate adaptive spectral normalization with learnable bounds."""
    print("\n⚙️ ADAPTIVE SPECTRAL NORMALIZATION")
    print("=" * 50)

    rngs = nnx.Rngs(42)

    # Create different adaptive configurations
    configs = {
        "Fixed bound (1.0)": {"initial_bound": 1.0, "learnable_bound": False},
        "Fixed bound (0.5)": {"initial_bound": 0.5, "learnable_bound": False},
        "Learnable bound": {"initial_bound": 1.0, "learnable_bound": True},
        "Learnable relaxed": {"initial_bound": 2.0, "learnable_bound": True},
    }

    models = {}
    for name, config in configs.items():
        base_linear = nnx.Linear(16, 8, rngs=rngs)
        adaptive_layer = AdaptiveSpectralNorm(
            base_linear, power_iterations=5, rngs=rngs, **config
        )
        models[name] = adaptive_layer

        print(f"📋 {name}:")
        print(f"   Initial bound: {config['initial_bound']}")
        print(f"   Learnable: {config['learnable_bound']}")

    # Test with sample data
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (5, 16))

    print(f"\n🧪 Testing with input shape: {x.shape}")

    for name, model in models.items():
        output = model(x, training=True)
        bound_value = model.bound.value
        print(f"   {name}: bound = {bound_value:.3f}, output shape = {output.shape}")


def demonstrate_power_iteration_algorithm():
    """Demonstrate the core power iteration algorithm."""
    print("\n⚡ POWER ITERATION ALGORITHM")
    print("=" * 50)

    rngs = nnx.Rngs(42)

    # Test matrices with known properties
    test_matrices = {
        "Identity": jnp.eye(4),
        "Diagonal": jnp.diag(jnp.array([3.0, 2.0, 1.0, 0.5])),
        "Random": jax.random.normal(jax.random.PRNGKey(42), (6, 4)),
        "Large Random": jax.random.normal(jax.random.PRNGKey(123), (128, 64)),
    }

    # Test different iteration counts
    iteration_counts = [1, 3, 5, 10]

    for matrix_name, matrix in test_matrices.items():
        print(f"\n📊 Matrix: {matrix_name} (shape: {matrix.shape})")

        # True spectral norm via SVD
        true_spectral_norm = jnp.max(jnp.linalg.svd(matrix, compute_uv=False))
        print(f"   True spectral norm (SVD): {true_spectral_norm:.6f}")

        for num_iter in iteration_counts:
            power_iter = PowerIteration(num_iterations=num_iter, rngs=rngs)

            start_time = time.time()
            estimated_norm, _ = power_iter(matrix, training=True)
            end_time = time.time()

            error = abs(estimated_norm - true_spectral_norm)
            print(
                f"   {num_iter:2d} iterations: {estimated_norm:.6f} "
                f"(error: {error:.6f}, time: {(end_time - start_time) * 1000:.2f} ms)"
            )


def demonstrate_complete_neural_operators():
    """Demonstrate complete spectral normalized neural operators."""
    print("\n🏗️ COMPLETE SPECTRAL NEURAL OPERATORS")
    print("=" * 50)

    rngs = nnx.Rngs(42)

    # Create different neural operator architectures
    architectures = {
        "Small FNO-style": {
            "input_dim": 32,
            "output_dim": 32,
            "hidden_dims": (64, 64),
            "num_heads": 4,
            "power_iterations": 1,
        },
        "Medium PDE solver": {
            "input_dim": 64,
            "output_dim": 64,
            "hidden_dims": (128, 128, 64),
            "num_heads": 8,
            "power_iterations": 3,
        },
        "Large Multi-scale": {
            "input_dim": 128,
            "output_dim": 64,
            "hidden_dims": (256, 192, 128, 96),
            "num_heads": 16,
            "power_iterations": 5,
        },
    }

    models = {}
    for name, config in architectures.items():
        print(f"\n🔧 Creating {name}:")

        start_time = time.time()
        model = create_spectral_neural_operator(rngs=rngs, **config)
        end_time = time.time()

        models[name] = model

        print(f"   Input/Output dims: {config['input_dim']} -> {config['output_dim']}")
        print(f"   Hidden layers: {config['hidden_dims']}")
        print(f"   Attention heads: {config['num_heads']}")
        print(f"   Creation time: {(end_time - start_time) * 1000:.2f} ms")

    # Test forward passes
    print("\n🧪 Testing forward passes:")

    for name, model in models.items():
        config = architectures[name]

        # Create test input
        key = jax.random.PRNGKey(0)
        batch_size = 4
        x = jax.random.normal(key, (batch_size, config["input_dim"]))

        # Timed forward pass
        start_time = time.time()
        output = model(x, training=True)
        end_time = time.time()

        print(
            f"   {name}: {x.shape} -> {output.shape} "
            f"({(end_time - start_time) * 1000:.2f} ms)"
        )


def demonstrate_stability_analysis():
    """Demonstrate stability analysis and Lipschitz constant control."""
    print("\n📈 STABILITY ANALYSIS & LIPSCHITZ CONTROL")
    print("=" * 50)

    rngs = nnx.Rngs(42)

    # Create regular vs spectral normalized networks
    input_dim, output_dim = 16, 8

    regular_model = nnx.Sequential(
        nnx.Linear(input_dim, 32, rngs=rngs),
        nnx.relu,
        nnx.Linear(32, 16, rngs=rngs),
        nnx.relu,
        nnx.Linear(16, output_dim, rngs=rngs),
    )

    spectral_model = nnx.Sequential(
        SpectralLinear(input_dim, 32, power_iterations=5, rngs=rngs),
        nnx.relu,
        SpectralLinear(32, 16, power_iterations=5, rngs=rngs),
        nnx.relu,
        SpectralLinear(16, output_dim, power_iterations=5, rngs=rngs),
    )

    print("🔧 Network configurations:")
    print("   Regular: Linear layers with standard weights")
    print("   Spectral: SpectralLinear layers with spectral normalization")

    # Lipschitz constant estimation
    print("\n📊 Lipschitz constant estimation:")

    num_samples = 100
    lipschitz_estimates_regular = []
    lipschitz_estimates_spectral = []

    key = jax.random.PRNGKey(0)

    for i in range(num_samples):
        # Generate random input pairs
        x1 = jax.random.normal(jax.random.split(key)[0], (1, input_dim))
        x2 = x1 + 0.1 * jax.random.normal(jax.random.split(key)[1], (1, input_dim))

        # Forward passes
        y1_regular = regular_model(x1)
        y2_regular = regular_model(x2)

        # Stabilize spectral model first
        if i == 0:
            for _ in range(5):  # Warm up spectral normalization
                _ = spectral_model(x1, training=True)

        y1_spectral = spectral_model(x1, training=False)
        y2_spectral = spectral_model(x2, training=False)

        # Compute Lipschitz estimates
        input_diff = jnp.linalg.norm(x2 - x1)
        output_diff_regular = jnp.linalg.norm(y2_regular - y1_regular)
        output_diff_spectral = jnp.linalg.norm(y2_spectral - y1_spectral)

        lipschitz_regular = output_diff_regular / (input_diff + 1e-8)
        lipschitz_spectral = output_diff_spectral / (input_diff + 1e-8)

        lipschitz_estimates_regular.append(float(lipschitz_regular))
        lipschitz_estimates_spectral.append(float(lipschitz_spectral))

        key = jax.random.split(key)[0]

    # Statistical analysis
    regular_stats = {
        "mean": jnp.mean(jnp.array(lipschitz_estimates_regular)),
        "std": jnp.std(jnp.array(lipschitz_estimates_regular)),
        "max": jnp.max(jnp.array(lipschitz_estimates_regular)),
    }

    spectral_stats = {
        "mean": jnp.mean(jnp.array(lipschitz_estimates_spectral)),
        "std": jnp.std(jnp.array(lipschitz_estimates_spectral)),
        "max": jnp.max(jnp.array(lipschitz_estimates_spectral)),
    }

    print("   Regular network:")
    print(
        f"     Mean Lipschitz: {regular_stats['mean']:.3f} ± {regular_stats['std']:.3f}"
    )
    print(f"     Max Lipschitz: {regular_stats['max']:.3f}")

    print("   Spectral normalized network:")
    print(
        f"     Mean Lipschitz: {spectral_stats['mean']:.3f} ± {spectral_stats['std']:.3f}"
    )
    print(f"     Max Lipschitz: {spectral_stats['max']:.3f}")

    # Spectral norm analysis
    print("\n🔍 Spectral norm analysis:")
    summary = spectral_norm_summary(spectral_model)

    if "num_layers" in summary:
        print(f"   Spectral normalized layers: {summary['num_layers']}")
        print(f"   Mean spectral norm: {summary['mean_spectral_norm']:.3f}")
        print(f"   Max spectral norm: {summary['max_spectral_norm']:.3f}")
        print(f"   Min spectral norm: {summary['min_spectral_norm']:.3f}")


def demonstrate_jax_transformations():
    """Demonstrate JAX transformations compatibility."""
    print("\n⚡ JAX TRANSFORMATIONS COMPATIBILITY")
    print("=" * 50)

    rngs = nnx.Rngs(42)

    # Create spectral normalized layer
    layer = SpectralLinear(12, 6, power_iterations=3, rngs=rngs)

    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (8, 12))

    # Test JIT compilation
    @jax.jit
    def jit_forward(x_input):
        return layer(x_input, training=True)

    start_time = time.time()
    output_jit = jit_forward(x)
    end_time = time.time()
    print(
        f"✅ JIT compilation: {x.shape} -> {output_jit.shape} "
        f"({(end_time - start_time) * 1000:.2f} ms)"
    )

    # Test gradient computation
    def loss_function(x_input):
        output = layer(x_input, training=True)
        return jnp.sum(output**2)

    grad_fn = jax.grad(loss_function)
    start_time = time.time()
    gradients = grad_fn(x)
    end_time = time.time()

    print(
        f"✅ Gradient computation: gradient shape {gradients.shape}, "
        f"norm = {jnp.linalg.norm(gradients):.3f} "
        f"({(end_time - start_time) * 1000:.2f} ms)"
    )

    # Test vectorized mapping (vmap)
    batch_x = jax.random.normal(key, (16, 4, 12))  # (batch, mini_batch, features)

    vectorized_forward = jax.vmap(
        lambda x_single: layer(x_single, training=True), in_axes=0
    )

    start_time = time.time()
    batch_output = vectorized_forward(batch_x)
    end_time = time.time()

    print(
        f"✅ Vectorized mapping (vmap): {batch_x.shape} -> {batch_output.shape} "
        f"({(end_time - start_time) * 1000:.2f} ms)"
    )

    # Test higher-order transformations
    hessian_fn = jax.hessian(loss_function)
    small_x = x[:2, :]  # Smaller input for Hessian computation

    start_time = time.time()
    hessian = hessian_fn(small_x)
    end_time = time.time()

    print(
        f"✅ Hessian computation: shape {hessian.shape} "
        f"({(end_time - start_time) * 1000:.2f} ms)"
    )


def run_performance_benchmark():
    """Run performance benchmarks comparing spectral vs regular layers."""
    print("\n🏃 PERFORMANCE BENCHMARKS")
    print("=" * 50)

    rngs = nnx.Rngs(42)

    # Benchmark configurations
    configs = [
        {"name": "Small", "input_dim": 32, "output_dim": 16, "batch_size": 64},
        {"name": "Medium", "input_dim": 128, "output_dim": 64, "batch_size": 32},
        {"name": "Large", "input_dim": 512, "output_dim": 256, "batch_size": 8},
    ]

    for config in configs:
        print(f"\n📊 {config['name']} benchmark:")
        print(f"   Dimensions: {config['input_dim']} -> {config['output_dim']}")
        print(f"   Batch size: {config['batch_size']}")

        # Create layers
        regular_layer = nnx.Linear(config["input_dim"], config["output_dim"], rngs=rngs)
        spectral_layer = SpectralLinear(
            config["input_dim"], config["output_dim"], power_iterations=3, rngs=rngs
        )

        # Create test data
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (config["batch_size"], config["input_dim"]))

        # JIT compile with proper closure capture
        @jax.jit
        def regular_forward(x_input, layer=regular_layer):
            return layer(x_input)

        @jax.jit
        def spectral_forward(x_input, layer=spectral_layer):
            return layer(x_input, training=True)

        # Warm up
        _ = regular_forward(x)
        _ = spectral_forward(x)

        # Benchmark regular layer
        num_runs = 100
        times_regular = []

        for _ in range(num_runs):
            start = time.time()
            _ = regular_forward(x)
            end = time.time()
            times_regular.append((end - start) * 1000)

        # Benchmark spectral layer
        times_spectral = []

        for _ in range(num_runs):
            start = time.time()
            _ = spectral_forward(x)
            end = time.time()
            times_spectral.append((end - start) * 1000)

        # Results
        mean_regular = jnp.mean(jnp.array(times_regular))
        std_regular = jnp.std(jnp.array(times_regular))
        mean_spectral = jnp.mean(jnp.array(times_spectral))
        std_spectral = jnp.std(jnp.array(times_spectral))

        overhead = (mean_spectral - mean_regular) / mean_regular * 100

        print(f"   Regular layer: {mean_regular:.2f} ± {std_regular:.2f} ms")
        print(f"   Spectral layer: {mean_spectral:.2f} ± {std_spectral:.2f} ms")
        print(f"   Overhead: {overhead:.1f}%")


def create_visualization_demo():
    """Create visualizations demonstrating spectral normalization effects."""
    print("\n📊 VISUALIZATION DEMONSTRATIONS")
    print("=" * 50)

    rngs = nnx.Rngs(42)

    # Test on simple 2D function
    x = jnp.linspace(-2, 2, 100)
    y_true = jnp.sin(3 * x) * jnp.exp(-(x**2))

    # Add noise
    noise = 0.1 * jax.random.normal(jax.random.PRNGKey(42), y_true.shape)
    y_noisy = y_true + noise

    # Create models
    regular_model = nnx.Sequential(
        nnx.Linear(1, 32, rngs=rngs),
        nnx.tanh,
        nnx.Linear(32, 32, rngs=rngs),
        nnx.tanh,
        nnx.Linear(32, 1, rngs=rngs),
    )

    spectral_model = nnx.Sequential(
        SpectralLinear(1, 32, power_iterations=5, rngs=rngs),
        nnx.tanh,
        SpectralLinear(32, 32, power_iterations=5, rngs=rngs),
        nnx.tanh,
        SpectralLinear(32, 1, power_iterations=5, rngs=rngs),
    )

    # Simple training simulation (just a few steps for demonstration)
    x_input = x.reshape(-1, 1)
    y_target = y_noisy.reshape(-1, 1)

    print("🎯 Function approximation demonstration:")
    print(f"   Training data: {x_input.shape} -> {y_target.shape}")
    print("   True function: sin(3x) * exp(-x²)")
    print("   Noise level: 10%")

    # Quick "training" simulation
    for i in range(5):
        # Regular model prediction
        y_pred_regular = regular_model(x_input)

        # Spectral model prediction (stabilize first)
        if i == 0:
            for _ in range(3):
                _ = spectral_model(x_input, training=True)
        y_pred_spectral = spectral_model(x_input, training=False)

        if i % 2 == 0:
            mse_regular = jnp.mean((y_pred_regular - y_target) ** 2)
            mse_spectral = jnp.mean((y_pred_spectral - y_target) ** 2)

            print(
                f"   Step {i}: Regular MSE = {mse_regular:.6f}, "
                f"Spectral MSE = {mse_spectral:.6f}"
            )

    print(
        "   📊 Note: In practice, spectral normalization provides more stable training"
    )
    print("         and better generalization, especially for longer training periods.")


def main():
    """Run all spectral normalization demonstrations."""
    print("🌟 SPECTRAL NORMALIZATION FOR NEURAL OPERATORS")
    print("=" * 60)
    print("Comprehensive demonstrations of spectral normalization techniques")
    print("for enhancing neural operator stability and controlling Lipschitz constants")
    print("=" * 60)

    # Run all demonstrations
    demonstrate_basic_spectral_layers()
    demonstrate_spectral_attention()
    demonstrate_adaptive_spectral_norm()
    demonstrate_power_iteration_algorithm()
    demonstrate_complete_neural_operators()
    demonstrate_stability_analysis()
    demonstrate_jax_transformations()
    run_performance_benchmark()
    create_visualization_demo()

    print("\n" + "=" * 60)
    print("🎉 SPECTRAL NORMALIZATION DEMONSTRATIONS COMPLETE")
    print("=" * 60)
    print("\n📋 Key Takeaways:")
    print("• Spectral normalization helps control Lipschitz constants")
    print("• Power iteration provides efficient spectral norm estimation")
    print("• Adaptive bounds allow flexible control over normalization")
    print("• JAX transformations work seamlessly with spectral layers")
    print("• Modest performance overhead (~10-30%) for improved stability")
    print("• Particularly beneficial for PDE-solving neural operators")
    print("\n🔧 Usage Recommendations:")
    print("• Use SpectralLinear for critical stability layers")
    print("• Apply SpectralNormalizedConv for spatial neural operators")
    print("• Consider AdaptiveSpectralNorm for layer-specific tuning")
    print("• Increase power_iterations for better spectral norm accuracy")
    print("• Monitor spectral norms using spectral_norm_summary()")


if __name__ == "__main__":
    main()
