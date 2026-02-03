"""
Darcy Flow FNO - Opifex Framework Implementation.

==============================================

Reproduces the neuraloperator Darcy Flow example using Opifex framework.
This implements training an FNO on the Darcy Flow equation (2D elliptic PDE).

Equivalent to: neuraloperator/examples/models/plot_FNO_darcy.py
"""

from datetime import datetime, UTC
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx

from opifex.core.training.config import TrainingConfig
from opifex.core.training.trainer import Trainer

# Opifex imports
from opifex.neural.operators.foundations import FourierNeuralOperator


def create_results_directory():
    """Create a timestamped results directory for this run."""
    # Create base output directory if it doesn't exist
    base_dir = Path("examples_output")
    base_dir.mkdir(exist_ok=True)

    # Create timestamped subdirectory for this run
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    results_dir = base_dir / f"darcy_fno_run_{timestamp}"
    results_dir.mkdir(exist_ok=True)

    print(f"Results will be saved to: {results_dir}")
    return results_dir


def load_darcy_flow_data(  # noqa: PLR0915
    n_train: int = 1000, n_test: int = 200, resolution: int = 64
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Load or generate Darcy Flow dataset using JAX transformations for efficiency.

    Darcy's equation: -∇·(a(x)∇u(x)) = f(x) in Ω
    with homogeneous Dirichlet boundary conditions.

    Args:
        n_train: Number of training samples
        n_test: Number of test samples
        resolution: Grid resolution

    Returns:
        Tuple of (train_input, train_output, test_input, test_output)
    """
    print(f"Generating Darcy Flow dataset with resolution {resolution}x{resolution}")

    # Create spatial grid (coordinates not used in current implementation)
    # x = jnp.linspace(0, 1, resolution)
    # y = jnp.linspace(0, 1, resolution)
    # X, Y = jnp.meshgrid(x, y)  # Spatial coordinates (not used in current implementation)

    key = jax.random.PRNGKey(42)

    def generate_coefficient_field(key):
        """Generate random permeability coefficient field"""
        # Generate smooth random field using Fourier modes
        modes = 8
        coeffs_real = jax.random.normal(key, (modes, modes)) * 0.5
        coeffs_imag = jax.random.normal(key, (modes, modes)) * 0.5

        # Create frequency grid (coordinates not used directly in current implementation)
        # kx = jnp.fft.fftfreq(resolution, d=1 / resolution)[:modes]
        # ky = jnp.fft.fftfreq(resolution, d=1 / resolution)[:modes]
        # KX, KY = jnp.meshgrid(kx, ky)  # Frequency coordinates (not used directly)

        # Generate field in Fourier space
        field_fft = jnp.zeros((resolution, resolution))
        field_fft = field_fft.at[:modes, :modes].set(coeffs_real + 1j * coeffs_imag)

        # Transform to physical space
        field = jnp.real(jnp.fft.ifft2(field_fft))

        # Ensure positive permeability
        return jnp.exp(field - jnp.mean(field) + 1.0)

    def solve_darcy_vectorized(a_field):
        """Solve Darcy equation using vectorized finite difference operations"""
        dx = 1.0 / (resolution - 1)

        # Create right-hand side (forcing term)
        f = jnp.ones((resolution, resolution))

        # Initialize solution
        u = jnp.zeros((resolution, resolution))

        def update_step(u):
            """Single vectorized update step for Gauss-Seidel iteration"""
            # Extract interior points only (avoiding boundaries)
            interior_slice = slice(1, -1)

            # Get neighbor values for interior points
            u_right = u[interior_slice, 2:]  # u[i, j+1] for interior points
            u_left = u[interior_slice, :-2]  # u[i, j-1] for interior points
            u_up = u[2:, interior_slice]  # u[i+1, j] for interior points
            u_down = u[:-2, interior_slice]  # u[i-1, j] for interior points

            # Get coefficient values for interior points
            a_center = a_field[interior_slice, interior_slice]  # a[i, j]
            a_right = a_field[interior_slice, 2:]  # a[i, j+1]
            a_left = a_field[interior_slice, :-2]  # a[i, j-1]
            a_up = a_field[2:, interior_slice]  # a[i+1, j]
            a_down = a_field[:-2, interior_slice]  # a[i-1, j]

            # Compute averaged coefficients (vectorized)
            a_avg_x = 0.5 * (a_right + a_center)  # average in x+ direction
            a_avg_x_m = 0.5 * (a_center + a_left)  # average in x- direction
            a_avg_y = 0.5 * (a_up + a_center)  # average in y+ direction
            a_avg_y_m = 0.5 * (a_center + a_down)  # average in y- direction

            # Compute numerator and denominator (vectorized)
            numerator = (
                a_avg_x * u_right
                + a_avg_x_m * u_left
                + a_avg_y * u_up
                + a_avg_y_m * u_down
                - dx**2 * f[interior_slice, interior_slice]
            )

            denominator = a_avg_x + a_avg_x_m + a_avg_y + a_avg_y_m

            # Update interior points (vectorized)
            u_new_interior = numerator / denominator

            # Update the full array (keeping boundary conditions as zero)
            return u.at[interior_slice, interior_slice].set(u_new_interior)

        # Iterative solver using lax.fori_loop for better performance
        def body_fun(i, u):
            return update_step(u)

        # Run iterations
        return jax.lax.fori_loop(0, 500, body_fun, u)

    # Vectorized data generation using vmap
    def generate_single_sample(key):
        """Generate a single coefficient field and solution pair"""
        a_field = generate_coefficient_field(key)
        u_field = solve_darcy_vectorized(a_field)
        return a_field, u_field

    # Use vmap to generate multiple samples in parallel
    generate_batch = jax.vmap(generate_single_sample)

    # Generate training data in batches for better memory efficiency
    print("Generating training data...")
    batch_size = 100  # Process in smaller batches to avoid memory issues
    train_inputs_batches = []
    train_outputs_batches = []

    for i in range(0, n_train, batch_size):
        current_batch_size = min(batch_size, n_train - i)
        keys = jax.random.split(key, current_batch_size + 1)
        key = keys[0]
        batch_keys = keys[1:]

        # Generate batch using vmap
        batch_inputs, batch_outputs = generate_batch(batch_keys)
        train_inputs_batches.append(batch_inputs)
        train_outputs_batches.append(batch_outputs)

        print(f"Generated {i + current_batch_size}/{n_train} training samples")

    # Generate test data in batches
    print("Generating test data...")
    test_inputs_batches = []
    test_outputs_batches = []

    for i in range(0, n_test, batch_size):
        current_batch_size = min(batch_size, n_test - i)
        keys = jax.random.split(key, current_batch_size + 1)
        key = keys[0]
        batch_keys = keys[1:]

        # Generate batch using vmap
        batch_inputs, batch_outputs = generate_batch(batch_keys)
        test_inputs_batches.append(batch_inputs)
        test_outputs_batches.append(batch_outputs)

        print(f"Generated {i + current_batch_size}/{n_test} test samples")

    # Concatenate all batches
    train_inputs = jnp.concatenate(train_inputs_batches, axis=0)
    train_outputs = jnp.concatenate(train_outputs_batches, axis=0)
    test_inputs = jnp.concatenate(test_inputs_batches, axis=0)
    test_outputs = jnp.concatenate(test_outputs_batches, axis=0)

    # Add channel dimension
    train_inputs = train_inputs[:, None, :, :]  # (n_train, 1, H, W)
    train_outputs = train_outputs[:, None, :, :]  # (n_train, 1, H, W)
    test_inputs = test_inputs[:, None, :, :]  # (n_test, 1, H, W)
    test_outputs = test_outputs[:, None, :, :]  # (n_test, 1, H, W)

    print("Dataset shapes:")
    print(f"  Train input: {train_inputs.shape}")
    print(f"  Train output: {train_outputs.shape}")
    print(f"  Test input: {test_inputs.shape}")
    print(f"  Test output: {test_outputs.shape}")

    return train_inputs, train_outputs, test_inputs, test_outputs


def create_fno_model(resolution: int = 64) -> FourierNeuralOperator:
    """Create FNO model for Darcy Flow."""
    rngs = nnx.Rngs(jax.random.PRNGKey(42))

    return FourierNeuralOperator(
        in_channels=1,
        out_channels=1,
        hidden_channels=32,
        modes=16,  # Number of Fourier modes
        num_layers=4,
        activation=nnx.gelu,
        rngs=rngs,
    )


def train_darcy_fno(results_dir="."):  # noqa: PLR0915
    """Main training function"""
    print("=" * 60)
    print("Training FNO on Darcy Flow - Opifex Implementation")
    print("=" * 60)

    # Parameters
    resolution = 64
    n_train = 1000
    n_test = 200
    n_epochs = 20
    batch_size = 32
    learning_rate = 8e-3

    # Load data
    train_inputs, train_outputs, test_inputs, test_outputs = load_darcy_flow_data(
        n_train=n_train, n_test=n_test, resolution=resolution
    )

    # Create model
    print("\nCreating FNO model...")
    model = create_fno_model(resolution)

    # Count parameters
    def count_params(model):
        return sum(x.size for x in jax.tree.leaves(nnx.state(model)))

    n_params = count_params(model)
    print(f"Model has {n_params:,} parameters")

    # Create training configuration
    config = TrainingConfig(
        num_epochs=n_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        validation_frequency=3,
        checkpoint_frequency=50,
    )

    # Update optimizer configuration for better performance
    config.optimization_config.optimizer = "adamw"
    config.optimization_config.weight_decay = 1e-4

    # Create trainer
    trainer = Trainer(model=model, config=config)

    # Training data generator using vmap for batch processing
    def create_batched_data_generator(inputs, outputs, batch_size, key):
        """Create batches of training data using vectorized operations"""
        n_samples = inputs.shape[0]
        indices = jax.random.permutation(key, n_samples)

        # Reshape data into batches
        n_batches = n_samples // batch_size
        if n_batches * batch_size < n_samples:
            # Pad to make even batches
            pad_size = n_batches * batch_size + batch_size - n_samples
            indices = jnp.concatenate([indices, indices[:pad_size]])
            n_batches += 1

        # Reshape indices into batches
        batch_indices = indices[: n_batches * batch_size].reshape(n_batches, batch_size)

        # Use vmap to create all batches at once
        def get_batch(batch_idx):
            return inputs[batch_idx], outputs[batch_idx]

        # Vectorize batch creation
        batched_inputs, batched_outputs = jax.vmap(get_batch)(batch_indices)

        return batched_inputs, batched_outputs

    # Training loop with optimized batch processing
    print(f"\nStarting training for {n_epochs} epochs...")
    train_losses = []
    test_losses = []

    key = jax.random.PRNGKey(123)

    for epoch in range(n_epochs):
        key, subkey = jax.random.split(key)

        # Create all batches for this epoch using vectorized operations
        batched_inputs, batched_outputs = create_batched_data_generator(
            train_inputs, train_outputs, batch_size, subkey
        )

        # Process batches sequentially
        epoch_losses = []

        for batch_idx in range(batched_inputs.shape[0]):
            batch_input = batched_inputs[batch_idx]
            batch_output = batched_outputs[batch_idx]

            # Use regular training step (no JIT to avoid trace level conflicts)
            loss = trainer.training_step(batch_input, batch_output)
            epoch_losses.append(loss)

        avg_train_loss = jnp.mean(jnp.array(epoch_losses))
        train_losses.append(avg_train_loss)

        # Validation every 3 epochs with vectorized operations
        if epoch % 3 == 0:
            # Vectorized prediction and loss computation
            test_pred = model(test_inputs[:50])  # First 50 test samples
            test_loss = jnp.mean((test_pred - test_outputs[:50]) ** 2)
            test_losses.append(test_loss)

            print(
                f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, Test Loss = {test_loss:.6f}"
            )
        else:
            print(f"Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}")

    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)

    # Final evaluation
    print("\nEvaluating trained model...")
    final_pred = model(test_inputs)
    final_test_loss = jnp.mean((final_pred - test_outputs) ** 2)
    print(f"Final test MSE: {final_test_loss:.6f}")

    # Visualize results
    visualize_results(
        test_inputs, test_outputs, final_pred, n_samples=3, results_dir=results_dir
    )

    return model, train_losses, test_losses


def visualize_results(inputs, targets, predictions, n_samples=3, results_dir="."):
    """Visualize input, target, and prediction."""
    _, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))

    for i in range(n_samples):
        # Input (permeability field)
        im1 = axes[i, 0].imshow(inputs[i, 0], cmap="viridis")
        axes[i, 0].set_title(f"Input Permeability {i + 1}")
        axes[i, 0].set_xlabel("x")
        axes[i, 0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[i, 0])

        # Target solution
        im2 = axes[i, 1].imshow(targets[i, 0], cmap="plasma")
        axes[i, 1].set_title(f"Target Solution {i + 1}")
        axes[i, 1].set_xlabel("x")
        axes[i, 1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[i, 1])

        # FNO prediction
        im3 = axes[i, 2].imshow(predictions[i, 0], cmap="plasma")
        axes[i, 2].set_title(f"FNO Prediction {i + 1}")
        axes[i, 2].set_xlabel("x")
        axes[i, 2].set_ylabel("y")
        plt.colorbar(im3, ax=axes[i, 2])

    plt.tight_layout()
    results_file = Path(results_dir) / "darcy_fno_results.png"
    plt.savefig(results_file, dpi=150, bbox_inches="tight")
    plt.show()

    # Calculate and print error statistics
    errors = jnp.abs(predictions - targets)
    relative_errors = errors / (jnp.abs(targets) + 1e-8)

    print("\nError Statistics:")
    print(f"  Mean Absolute Error: {jnp.mean(errors):.6f}")
    print(f"  Max Absolute Error: {jnp.max(errors):.6f}")
    print(f"  Mean Relative Error: {jnp.mean(relative_errors):.6f}")
    print(f"  Max Relative Error: {jnp.max(relative_errors):.6f}")

    # Save error statistics to file
    stats_file = Path(results_dir) / "error_statistics.txt"
    with open(stats_file, "w") as f:
        f.write("Darcy Flow FNO - Error Statistics\n")
        f.write("=" * 35 + "\n\n")
        f.write(f"Mean Absolute Error: {jnp.mean(errors):.6f}\n")
        f.write(f"Max Absolute Error: {jnp.max(errors):.6f}\n")
        f.write(f"Mean Relative Error: {jnp.mean(relative_errors):.6f}\n")
        f.write(f"Max Relative Error: {jnp.max(relative_errors):.6f}\n")

    print(f"Results saved to: {results_file}")
    print(f"Error statistics saved to: {stats_file}")


if __name__ == "__main__":
    # Create results directory for this run
    results_dir = create_results_directory()

    # Run the training
    model, train_losses, test_losses = train_darcy_fno(str(results_dir))

    # Plot training curves
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")

    plt.subplot(1, 2, 2)
    epochs_test = range(0, len(train_losses), 3)
    plt.plot(epochs_test, test_losses)
    plt.title("Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")

    plt.tight_layout()
    training_curves_file = Path(results_dir) / "darcy_training_curves.png"
    plt.savefig(training_curves_file, dpi=150, bbox_inches="tight")
    plt.show()

    # Save training data
    training_data_file = Path(results_dir) / "training_data.txt"
    with open(training_data_file, "w") as f:
        f.write("Darcy Flow FNO - Training Data\n")
        f.write("=" * 30 + "\n\n")
        f.write("Training Losses (by epoch):\n")
        for i, loss in enumerate(train_losses):
            f.write(f"Epoch {i}: {loss:.6f}\n")
        f.write("\nTest Losses (every 3 epochs):\n")
        for i, loss in enumerate(test_losses):
            f.write(f"Epoch {i * 3}: {loss:.6f}\n")

    print("\nOpifex Darcy Flow FNO training complete!")
    print(f"All results saved to: {results_dir}")
    print("Files created:")
    print("  - darcy_fno_results.png")
    print("  - darcy_training_curves.png")
    print("  - error_statistics.txt")
    print("  - training_data.txt")
