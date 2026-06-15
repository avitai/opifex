# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # GNO on Darcy Flow
#
# This example demonstrates training a Graph Neural Operator (GNO) on the
# Darcy flow problem. GNO uses message passing neural networks to learn
# operators on irregular domains, making it suitable for problems with
# complex geometries or unstructured meshes.
#
# Unlike spectral operators (FNO, UNO), GNO samples local neighborhoods and is
# genuinely harder to train on a regular Darcy grid. We therefore follow the
# standard operator-learning recipe that makes it converge: ~1000 training
# samples, **Gaussian normalization** of the input/output fields, the
# scale-invariant **relative-L2 loss**, and mini-batched optimization. The raw
# Darcy pressure is tiny (~0.05 scale), so without normalization the loss
# gradients are negligible and the model never learns.

# %%
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import optax
from flax import nnx

from opifex.data.loaders import create_darcy_loader
from opifex.neural.operators.graph import (
    graph_to_grid,
    GraphNeuralOperator,
    grid_to_graph_data,
)


# %%
# Configuration
print("=" * 70)
print("Opifex Example: GNO on Darcy Flow")
print("=" * 70)
print(f"JAX backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Problem configuration
RESOLUTION = 16  # Smaller resolution for GNO (graph scales quadratically)
N_TRAIN = 1000
N_TEST = 100
BATCH_SIZE = 32
EPOCHS = 150
LEARNING_RATE = 1e-3
SEED = 42

# Model configuration
HIDDEN_DIM = 64
NUM_LAYERS = 4
CONNECTIVITY = 8  # 8-neighbor connectivity includes diagonals

print(f"Resolution: {RESOLUTION}x{RESOLUTION}")
print(f"Training samples: {N_TRAIN}, Test samples: {N_TEST}")
print(f"Batch size: {BATCH_SIZE}, Epochs: {EPOCHS}")
print(f"GNO config: hidden_dim={HIDDEN_DIM}, layers={NUM_LAYERS}")
print(f"Graph connectivity: {CONNECTIVITY}-neighbor")

# %% [markdown]
# ## Data Loading
#
# Generate Darcy flow data. We collect *all* samples from the loader (not just
# the first batch) so the model sees the full training set.

# %%
print()
print("Generating Darcy flow data...")

train_loader = create_darcy_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=True,
    seed=SEED,
    worker_count=0,
)

test_loader = create_darcy_loader(
    n_samples=N_TEST,
    batch_size=N_TEST,
    resolution=RESOLUTION,
    shuffle=False,
    seed=SEED + 1000,
    worker_count=0,
)


def collect_split(loader: object) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate every batch from a Grain loader into input/output arrays."""
    inputs, outputs = [], []
    for batch in loader:  # type: ignore[attr-defined]
        inputs.append(np.asarray(batch["input"]))
        outputs.append(np.asarray(batch["output"]))
    return np.concatenate(inputs, axis=0), np.concatenate(outputs, axis=0)


X_train_np, Y_train_np = collect_split(train_loader)
X_test_np, Y_test_np = collect_split(test_loader)

# Ensure channel dimension exists (NCHW format)
if X_train_np.ndim == 3:
    X_train_np = X_train_np[:, None, :, :]  # Add channel dimension
    Y_train_np = Y_train_np[:, None, :, :]
    X_test_np = X_test_np[:, None, :, :]
    Y_test_np = Y_test_np[:, None, :, :]

print(f"Grid data: X={X_train_np.shape}, Y={Y_train_np.shape}")

# %% [markdown]
# ## Normalization
#
# Neural operators train best on standardized fields. The raw Darcy pressure
# has a tiny magnitude (~0.05), so we fit Gaussian statistics on the **training**
# split, normalize all splits, and un-normalize predictions before computing the
# physical-space relative-L2 error.

# %%
x_mean = float(X_train_np.mean())
x_std = float(X_train_np.std())
y_mean = float(Y_train_np.mean())
y_std = float(Y_train_np.std())

X_train_n = (X_train_np - x_mean) / x_std
Y_train_n = (Y_train_np - y_mean) / y_std
X_test_n = (X_test_np - x_mean) / x_std

print(f"Input mean/std:  {x_mean:.4f} / {x_std:.4f}")
print(f"Output mean/std: {y_mean:.6f} / {y_std:.6f}")

# Keep the physical-scale targets for evaluation/visualization.
X_test = jnp.array(X_test_np)
Y_test = jnp.array(Y_test_np)

# %% [markdown]
# ## Graph Conversion
#
# Convert the (normalized) 2D grids to graph representation for GNO. Each node
# carries the normalized field value plus its `(x, y)` position; edges connect
# neighboring grid points with relative-position features.

# %%
print()
print("Converting grids to graphs...")

train_nodes, train_edges, train_edge_feats = grid_to_graph_data(
    jnp.array(X_train_n), connectivity=CONNECTIVITY
)
test_nodes, test_edges, test_edge_feats = grid_to_graph_data(
    jnp.array(X_test_n), connectivity=CONNECTIVITY
)

# Convert normalized targets to node format (value channel is what we predict).
train_targets, _, _ = grid_to_graph_data(jnp.array(Y_train_n), connectivity=CONNECTIVITY)

print(f"Node features shape: {train_nodes.shape}")
print(f"Edge indices shape:  {train_edges.shape}")
print(f"Edge features shape: {train_edge_feats.shape}")
print(f"Num nodes per graph: {train_nodes.shape[1]} ({RESOLUTION}x{RESOLUTION})")
print(f"Num edges per graph: {train_edges.shape[1]}")

# %% [markdown]
# ## Model Creation
#
# Create a GraphNeuralOperator for the Darcy flow problem.

# %%
print()
print("Creating GNO model...")

gno = GraphNeuralOperator(
    node_dim=train_nodes.shape[-1],
    hidden_dim=HIDDEN_DIM,
    num_layers=NUM_LAYERS,
    edge_dim=train_edge_feats.shape[-1],
    rngs=nnx.Rngs(SEED),
)

# Count parameters
gno_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(gno, nnx.Param)))
print(f"GNO parameters: {gno_params:,}")

# %% [markdown]
# ## Training
#
# Train the GNO with the Adam optimizer using the **relative-L2 loss** on the
# node value channel. Relative L2 is the standard operator-learning objective:
# it normalizes each sample by its own field magnitude, so the tiny Darcy
# pressure scale no longer starves the gradients. We mini-batch over the full
# training set every epoch.


# %%
def relative_l2_loss(pred_values: jax.Array, target_values: jax.Array) -> jax.Array:
    """Mean per-sample relative-L2 error ``||pred - y|| / ||y||``.

    Args:
        pred_values: Predicted node values [batch, num_nodes].
        target_values: Target node values [batch, num_nodes].

    Returns:
        Scalar mean relative-L2 loss.
    """
    diff = jnp.linalg.norm(pred_values - target_values, axis=1)
    denom = jnp.linalg.norm(target_values, axis=1) + 1e-8
    return jnp.mean(diff / denom)


def train_model(
    model: GraphNeuralOperator,
    nodes: jax.Array,
    edges: jax.Array,
    edge_feats: jax.Array,
    targets: jax.Array,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    model_name: str = "GNO",
) -> list[float]:
    """Train a GNO with mini-batched relative-L2 loss on the value channel.

    Args:
        model: The GraphNeuralOperator to train (updated in place).
        nodes: Node features [n_samples, num_nodes, node_dim].
        edges: Edge indices [n_samples, num_edges, 2].
        edge_feats: Edge features [n_samples, num_edges, edge_dim].
        targets: Target node features [n_samples, num_nodes, node_dim].
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        learning_rate: Adam learning rate.
        seed: Seed for the shuffling RNG.
        model_name: Label used in progress logging.

    Returns:
        Per-epoch mean training loss.
    """
    opt = nnx.Optimizer(model, optax.adam(learning_rate), wrt=nnx.Param)

    @nnx.jit
    def train_step(
        model: GraphNeuralOperator,
        opt: nnx.Optimizer,
        nodes: jax.Array,
        edges: jax.Array,
        edge_feats: jax.Array,
        targets: jax.Array,
    ) -> jax.Array:
        def loss_fn(model: GraphNeuralOperator) -> jax.Array:
            pred = model(nodes, edges, edge_feats)
            # Compare the value channel only (column 0), not the position encoding.
            return relative_l2_loss(pred[:, :, 0], targets[:, :, 0])

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    print(f"Training {model_name}...")
    n_samples = nodes.shape[0]
    rng = np.random.default_rng(seed)
    losses: list[float] = []

    for epoch in range(epochs):
        perm = rng.permutation(n_samples)
        epoch_losses: list[float] = []
        for start in range(0, n_samples, batch_size):
            idx = perm[start : start + batch_size]
            loss = train_step(
                model,
                opt,
                nodes[idx],
                edges[idx],
                edge_feats[idx],
                targets[idx],
            )
            epoch_losses.append(float(loss))

        mean_loss = float(np.mean(epoch_losses))
        losses.append(mean_loss)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: loss={mean_loss:.6f}")

    return losses


# %%
print()
gno_losses = train_model(
    gno,
    train_nodes,
    train_edges,
    train_edge_feats,
    train_targets,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    seed=SEED,
    model_name="GNO",
)
print(f"Final GNO loss: {gno_losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation
#
# Run the GNO over the test set in batches, un-normalize the predicted value
# channel back to physical pressure, and report the relative-L2 error against
# the (physical-scale) ground truth.


# %%
def predict_value_grid(
    model: GraphNeuralOperator,
    nodes: jax.Array,
    edges: jax.Array,
    edge_feats: jax.Array,
    *,
    height: int,
    width: int,
    batch_size: int = 128,
) -> jax.Array:
    """Predict node values in batches and reshape them back to a grid.

    Args:
        model: Trained GraphNeuralOperator.
        nodes: Node features [n_samples, num_nodes, node_dim].
        edges: Edge indices [n_samples, num_edges, 2].
        edge_feats: Edge features [n_samples, num_edges, edge_dim].
        height: Output grid height.
        width: Output grid width.
        batch_size: Forward-pass batch size to bound memory.

    Returns:
        Predicted (normalized) value grid [n_samples, 1, height, width].
    """
    grids = []
    for start in range(0, nodes.shape[0], batch_size):
        stop = start + batch_size
        pred = model(nodes[start:stop], edges[start:stop], edge_feats[start:stop])
        grids.append(graph_to_grid(pred, height=height, width=width, channels=1))
    return jnp.concatenate(grids, axis=0)


def evaluate_model(
    pred_grid_physical: jax.Array,
    target_grid_physical: jax.Array,
    model_name: str = "GNO",
) -> tuple[float, float, float, float]:
    """Report MSE and relative-L2 statistics on physical-scale grids.

    Args:
        pred_grid_physical: Un-normalized predictions [n_samples, 1, H, W].
        target_grid_physical: Ground-truth pressure [n_samples, 1, H, W].
        model_name: Label used in logging.

    Returns:
        Tuple of (mse, rel_l2_mean, rel_l2_min, rel_l2_max).
    """
    mse = float(jnp.mean((pred_grid_physical - target_grid_physical) ** 2))

    pred_flat = pred_grid_physical.reshape(pred_grid_physical.shape[0], -1)
    target_flat = target_grid_physical.reshape(target_grid_physical.shape[0], -1)
    rel_l2_per_sample = jnp.linalg.norm(pred_flat - target_flat, axis=1) / (
        jnp.linalg.norm(target_flat, axis=1) + 1e-8
    )
    rel_l2_mean = float(jnp.mean(rel_l2_per_sample))
    rel_l2_min = float(jnp.min(rel_l2_per_sample))
    rel_l2_max = float(jnp.max(rel_l2_per_sample))

    print(f"{model_name} Results:")
    print(f"  Test MSE:         {mse:.6e}")
    print(f"  Relative L2:      {rel_l2_mean:.6f} (min={rel_l2_min:.6f}, max={rel_l2_max:.6f})")

    return mse, rel_l2_mean, rel_l2_min, rel_l2_max


# %%
print()
print("Running evaluation...")
pred_grid_n = predict_value_grid(
    gno,
    test_nodes,
    test_edges,
    test_edge_feats,
    height=RESOLUTION,
    width=RESOLUTION,
)
# Un-normalize predictions back to physical pressure before scoring.
pred_grid = pred_grid_n * y_std + y_mean

gno_mse, gno_rel_l2, gno_rel_l2_min, gno_rel_l2_max = evaluate_model(pred_grid, Y_test, "GNO")

# %% [markdown]
# ## Visualization
#
# Visualize predictions and compare with ground truth.

# %%
# Create output directory
output_dir = Path("docs/assets/examples/gno_darcy")
output_dir.mkdir(parents=True, exist_ok=True)

# Plot predictions for a sample
mpl.use("Agg")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for sample_idx in range(2):
    row = sample_idx

    # Input permeability
    axes[row, 0].imshow(np.array(X_test[sample_idx, 0]), cmap="viridis")
    axes[row, 0].set_title(f"Sample {sample_idx + 1}: Input (Permeability)")
    axes[row, 0].axis("off")

    # Ground truth pressure
    axes[row, 1].imshow(np.array(Y_test[sample_idx, 0]), cmap="RdBu_r")
    axes[row, 1].set_title("Ground Truth (Pressure)")
    axes[row, 1].axis("off")

    # GNO prediction
    axes[row, 2].imshow(np.array(pred_grid[sample_idx, 0]), cmap="RdBu_r")
    axes[row, 2].set_title("GNO Prediction")
    axes[row, 2].axis("off")

    # Error
    error = np.abs(np.array(pred_grid[sample_idx, 0] - Y_test[sample_idx, 0]))
    im = axes[row, 3].imshow(error, cmap="hot")
    axes[row, 3].set_title(f"Error (max={error.max():.4f})")
    axes[row, 3].axis("off")
    plt.colorbar(im, ax=axes[row, 3], fraction=0.046)

plt.tight_layout()
plt.savefig(output_dir / "predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Predictions saved to {output_dir / 'predictions.png'}")

# %%
# Training loss plot
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
axes[0].semilogy(gno_losses, linewidth=2)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Relative L2 Loss")
axes[0].set_title("GNO Training Loss")
axes[0].grid(True, alpha=0.3)

# Graph visualization (first sample)
ax = axes[1]
# Show input as background
ax.imshow(np.array(X_test[0, 0]), cmap="viridis", alpha=0.5, extent=[0, 1, 0, 1])

# Draw edges (sample of edges for visibility)
node_pos = test_nodes[0, :, 1:3]  # [num_nodes, 2] positions
edges = test_edges[0]  # [num_edges, 2]
num_edges_to_show = min(100, edges.shape[0])
for i in range(0, edges.shape[0], edges.shape[0] // num_edges_to_show):
    src, dst = int(edges[i, 0]), int(edges[i, 1])
    x1, y1 = float(node_pos[src, 0]), float(node_pos[src, 1])
    x2, y2 = float(node_pos[dst, 0]), float(node_pos[dst, 1])
    ax.plot([x1, x2], [y1, y2], "k-", alpha=0.2, linewidth=0.5)

# Draw nodes
ax.scatter(
    np.array(node_pos[:, 0]),
    np.array(node_pos[:, 1]),
    c=np.array(test_nodes[0, :, 0]),
    cmap="viridis",
    s=20,
    edgecolors="white",
    linewidths=0.5,
)
ax.set_title(f"Graph Structure ({CONNECTIVITY}-connectivity)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(output_dir / "training.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Training plot saved to {output_dir / 'training.png'}")

# %%
# Summary
print()
print("=" * 70)
print("GNO Darcy Flow example completed")
print("=" * 70)
print()
print("Results Summary:")
print(f"  GNO:        MSE={gno_mse:.6e}, Rel L2={gno_rel_l2:.4f}, Params={gno_params:,}")
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
