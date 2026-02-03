"""Fourier Continuation Examples - Signal Extension and Boundary Handling.

This example demonstrates the various Fourier continuation methods available
in Opifex for extending signals beyond their boundaries. These methods are
essential for neural operators that need to handle boundary conditions.

Key Features Demonstrated:
- FourierContinuationExtender: Core signal extension
- PeriodicContinuation: Periodic boundary extension
- SymmetricContinuation: Symmetric/mirror boundary extension
- SmoothContinuation: Smooth tapering to zero at boundaries
- FourierBoundaryHandler: Intelligent neural boundary handling
- Multi-dimensional signal extension capabilities
"""

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.operators.specialized.fourier_continuation import (
    create_continuation_pipeline,
    FourierBoundaryHandler,
    FourierContinuationExtender,
    PeriodicContinuation,
    SmoothContinuation,
    SymmetricContinuation,
)


def create_test_signals():
    """Create various test signals to demonstrate continuation methods."""
    x = jnp.linspace(0, 2 * jnp.pi, 32)

    signals = {
        "sine_wave": jnp.sin(x),
        "cosine_wave": jnp.cos(x),
        "step_function": jnp.where(x < jnp.pi, 1.0, -1.0),
        "ramp": x / (2 * jnp.pi),
        "gaussian": jnp.exp(-((x - jnp.pi) ** 2) / 0.5),
    }

    return x, signals


def demonstrate_basic_continuation_methods():
    """Demonstrate basic Fourier continuation methods."""
    print("🔬 BASIC FOURIER CONTINUATION METHODS")
    print("=" * 50)

    _, signals = create_test_signals()
    extension_length = 16

    # Test different continuation methods
    methods = {
        "Periodic": PeriodicContinuation(extension_length=extension_length),
        "Symmetric": SymmetricContinuation(extension_length=extension_length),
        "Smooth": SmoothContinuation(extension_length=extension_length),
        "Zero Padding": FourierContinuationExtender(
            extension_type="zero", extension_length=extension_length
        ),
    }

    for signal_name, signal in signals.items():
        print(f"\n📊 Signal: {signal_name}")
        print(f"   Original length: {len(signal)}")

        for method_name, extender in methods.items():
            extended = extender(signal)
            print(
                f"   {method_name:12}: {len(extended)} -> "
                f"Shape preservation: {extended.shape}"
            )

            # Verify extension properties
            original_part = extended[extension_length:-extension_length]
            similarity = jnp.mean((original_part - signal) ** 2)
            print(f"   {'':<12}  Original signal preserved: MSE = {similarity:.2e}")


def demonstrate_intelligent_boundary_handling():
    """Demonstrate intelligent boundary handling with neural networks."""
    print("\n🧠 INTELLIGENT BOUNDARY HANDLING")
    print("=" * 50)

    # Create RNG for neural network initialization
    rngs = nnx.Rngs(42)

    # Create intelligent boundary handler
    handler = FourierBoundaryHandler(
        continuation_methods=["periodic", "symmetric", "smooth"],
        extension_length=16,
        hidden_dim=32,
        rngs=rngs,
    )

    _, signals = create_test_signals()

    print("🔧 Handler configuration:")
    print(f"   Methods: {handler.continuation_methods}")
    print(f"   Extension length: {handler.extension_length}")
    print(f"   Neural network: {len(handler.continuation_methods)} method weights")

    for signal_name, signal in signals.items():
        print(f"\n📈 Processing: {signal_name}")

        # Extract signal features for decision making
        features = handler._extract_signal_features(signal)
        print(
            f"   Signal features: mean={features[0]:.3f}, "
            f"std={features[1]:.3f}, boundary_grad={features[2]:.3f}, "
            f"periodicity={features[3]:.3f}"
        )

        # Apply intelligent boundary handling
        extended = handler(signal)
        print(f"   Extended from {len(signal)} to {len(extended)} points")

        # Verify extension quality
        original_part = extended[handler.extension_length : -handler.extension_length]
        preservation_error = jnp.mean((original_part - signal) ** 2)
        print(f"   Signal preservation error: {preservation_error:.2e}")


def demonstrate_2d_extension():
    """Demonstrate 2D signal extension capabilities."""
    print("\n🗂️ 2D SIGNAL EXTENSION")
    print("=" * 50)

    # Create 2D test signal
    x = jnp.linspace(-1, 1, 16)
    y = jnp.linspace(-1, 1, 12)
    X, Y = jnp.meshgrid(x, y)
    signal_2d = jnp.exp(-(X**2 + Y**2))  # 2D Gaussian

    print(f"📐 Original 2D signal shape: {signal_2d.shape}")

    # Test 2D extension
    extender = FourierContinuationExtender(
        extension_type="symmetric",
        extension_length=8,
    )

    extended_2d = extender(signal_2d)
    print(f"📐 Extended 2D signal shape: {extended_2d.shape}")

    # Verify preservation of original signal
    h_ext, w_ext = extender.extension_length, extender.extension_length
    recovered = extended_2d[h_ext:-h_ext, w_ext:-w_ext]
    reconstruction_error = jnp.mean((recovered - signal_2d) ** 2)
    print(f"🎯 2D signal preservation error: {reconstruction_error:.2e}")


def demonstrate_jax_transformations():
    """Demonstrate JAX transformations compatibility."""
    print("\n⚡ JAX TRANSFORMATIONS COMPATIBILITY")
    print("=" * 50)

    extender = FourierContinuationExtender(
        extension_type="periodic",
        extension_length=8,
    )

    _, signals = create_test_signals()
    test_signal = signals["sine_wave"]

    # Test JIT compilation
    @jax.jit
    def extend_signal(signal):
        return extender(signal)

    extended_jit = extend_signal(test_signal)
    print(f"✅ JIT compilation: Extended {len(test_signal)} -> {len(extended_jit)}")

    # Test gradient computation
    def signal_loss(signal):
        extended = extender(signal)
        return jnp.sum(extended**2)

    grad_fn = jax.grad(signal_loss)
    gradients = grad_fn(test_signal)
    print(f"✅ Gradient computation: Gradient shape {gradients.shape}")
    print(f"   Gradient norm: {jnp.linalg.norm(gradients):.3f}")

    # Test vectorized mapping (vmap)
    batch_signals = jnp.stack([signals[name] for name in ["sine_wave", "cosine_wave"]])

    vectorized_extend = jax.vmap(extender, in_axes=0)
    batch_extended = vectorized_extend(batch_signals)
    print(
        f"✅ Vectorized mapping: Batch {batch_signals.shape} -> {batch_extended.shape}"
    )


def demonstrate_pipeline_creation():
    """Demonstrate pipeline creation utilities."""
    print("\n🏗️ CONTINUATION PIPELINE CREATION")
    print("=" * 50)

    # Create simple pipeline
    simple_pipeline = create_continuation_pipeline(
        methods=["symmetric"],
        extension_length=12,
        use_intelligent_handler=False,
    )

    print(f"🔧 Simple pipeline: {type(simple_pipeline).__name__}")
    print(f"   Extension type: {simple_pipeline.extension_type}")
    print(f"   Extension length: {simple_pipeline.extension_length}")

    # Create intelligent pipeline
    rngs = nnx.Rngs(123)
    intelligent_pipeline = create_continuation_pipeline(
        methods=["periodic", "symmetric", "smooth"],
        extension_length=12,
        use_intelligent_handler=True,
        rngs=rngs,
    )

    print(f"\n🧠 Intelligent pipeline: {type(intelligent_pipeline).__name__}")
    print(f"   Available methods: {intelligent_pipeline.continuation_methods}")
    print(f"   Extension length: {intelligent_pipeline.extension_length}")

    # Test both pipelines
    _, signals = create_test_signals()
    test_signal = signals["gaussian"]

    simple_result = simple_pipeline(test_signal)
    intelligent_result = intelligent_pipeline(test_signal)

    print("\n📊 Pipeline comparison:")
    print(f"   Input signal length: {len(test_signal)}")
    print(f"   Simple pipeline output: {len(simple_result)}")
    print(f"   Intelligent pipeline output: {len(intelligent_result)}")


def run_performance_benchmark():
    """Run performance benchmarks for different methods."""
    print("\n⏱️ PERFORMANCE BENCHMARKS")
    print("=" * 50)

    _, signals = create_test_signals()
    test_signal = signals["sine_wave"]

    methods = {
        "Periodic": PeriodicContinuation(extension_length=16),
        "Symmetric": SymmetricContinuation(extension_length=16),
        "Smooth": SmoothContinuation(extension_length=16),
    }

    # Warm up JIT compilation
    for method in methods.values():
        _ = method(test_signal)

    import time

    n_iterations = 1000

    for name, method in methods.items():
        # Compile the method
        compiled_method = jax.jit(method)

        # Timing
        start_time = time.time()
        for _ in range(n_iterations):
            _ = compiled_method(test_signal).block_until_ready()
        end_time = time.time()

        avg_time = (end_time - start_time) / n_iterations * 1000
        print(f"⚡ {name:10}: {avg_time:.3f} ms per call (JIT compiled)")


def main():
    """Run all Fourier continuation examples."""
    print("🎯 FOURIER CONTINUATION EXAMPLES")
    print("Opifex Neural Operators - Signal Extension & Boundary Handling")
    print("=" * 70)

    try:
        demonstrate_basic_continuation_methods()
        demonstrate_intelligent_boundary_handling()
        demonstrate_2d_extension()
        demonstrate_jax_transformations()
        demonstrate_pipeline_creation()
        run_performance_benchmark()

        print("\n✅ ALL FOURIER CONTINUATION EXAMPLES COMPLETED SUCCESSFULLY!")
        print("\n📚 Key Takeaways:")
        print(
            "   • Multiple continuation methods available (periodic, symmetric, smooth, zero)"
        )
        print("   • Intelligent boundary handling with neural networks")
        print("   • Support for 1D and 2D signal extension")
        print("   • Full JAX compatibility (JIT, grad, vmap)")
        print("   • High performance with compiled execution")
        print("   • Easy pipeline creation utilities")

    except Exception as e:
        print(f"\n❌ Error during example execution: {e}")
        raise


if __name__ == "__main__":
    main()
