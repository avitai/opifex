# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
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
N_TRAIN = 200
N_TEST = 50
BATCH_SIZE = 16
EPOCHS = 30

# Model configuration
HIDDEN_DIM = 32
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
# Generate Darcy flow data and convert to graph representation.

# %%
print()
print("Generating Darcy flow data...")

train_loader = create_darcy_loader(
    n_samples=N_TRAIN,
    batch_size=BATCH_SIZE,
    resolution=RESOLUTION,
    shuffle=True,
    seed=42,
    worker_count=0,
)

test_loader = create_darcy_loader(
    n_samples=N_TEST,
    batch_size=N_TEST,
    resolution=RESOLUTION,
    shuffle=False,
    seed=123,
    worker_count=0,
)

# Get data as arrays
train_batch = next(iter(train_loader))
X_train = jnp.array(train_batch["input"])
Y_train = jnp.array(train_batch["output"])

test_batch = next(iter(test_loader))
X_test = jnp.array(test_batch["input"])
Y_test = jnp.array(test_batch["output"])

# Ensure channel dimension exists (NCHW format)
if X_train.ndim == 3:
    X_train = X_train[:, None, :, :]  # Add channel dimension
    Y_train = Y_train[:, None, :, :]
    X_test = X_test[:, None, :, :]
    Y_test = Y_test[:, None, :, :]

print(f"Grid data: X={X_train.shape}, Y={Y_train.shape}")

# %% [markdown]
# ## Graph Conversion
#
# Convert 2D grid data to graph representation for GNO.

# %%
print()
print("Converting grids to graphs...")

# Convert to graph format
train_nodes, train_edges, train_edge_feats = grid_to_graph_data(
    X_train, connectivity=CONNECTIVITY
)
test_nodes, test_edges, test_edge_feats = grid_to_graph_data(
    X_test, connectivity=CONNECTIVITY
)

# Also convert target outputs
train_targets, _, _ = grid_to_graph_data(Y_train, connectivity=CONNECTIVITY)
test_targets, _, _ = grid_to_graph_data(Y_test, connectivity=CONNECTIVITY)

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
    rngs=nnx.Rngs(42),
)

# Count parameters
gno_params = sum(x.size for x in jax.tree_util.tree_leaves(nnx.state(gno, nnx.Param)))
print(f"GNO parameters: {gno_params:,}")

# %% [markdown]
# ## Training
#
# Train the GNO with Adam optimizer using MSE loss on node features.


# %%
def train_model(model, train_data, epochs, lr=1e-3, model_name="GNO"):
    """Train a model with MSE loss."""
    nodes, edges, edge_feats, targets = train_data
    opt = nnx.Optimizer(model, optax.adam(lr), wrt=nnx.Param)

    @nnx.jit
    def train_step(model, opt, nodes, edges, edge_feats, targets):
        def loss_fn(model):
            pred = model(nodes, edges, edge_feats)
            # Only compare value channel (first column), not position encoding
            return jnp.mean((pred[:, :, 0] - targets[:, :, 0]) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    print(f"Training {model_name}...")
    losses = []

    for epoch in range(epochs):
        loss = train_step(model, opt, nodes, edges, edge_feats, targets)
        losses.append(float(loss))

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: loss={loss:.6f}")

    return losses


# %%
print()
train_data = (train_nodes, train_edges, train_edge_feats, train_targets)
gno_losses = train_model(gno, train_data, EPOCHS, model_name="GNO")
print(f"Final GNO loss: {gno_losses[-1]:.6e}")

# %% [markdown]
# ## Evaluation
#
# Evaluate GNO on test data.


# %%
def evaluate_model(model, test_data, model_name="GNO"):
    """Evaluate model on test data."""
    nodes, edges, edge_feats, targets = test_data

    predictions = model(nodes, edges, edge_feats)

    # MSE on value channel only
    pred_values = predictions[:, :, 0]
    target_values = targets[:, :, 0]

    mse = float(jnp.mean((pred_values - target_values) ** 2))

    # Relative L2 error per sample
    rel_l2_per_sample = jnp.sqrt(
        jnp.sum((pred_values - target_values) ** 2, axis=1)
        / jnp.sum(target_values**2, axis=1)
    )
    rel_l2_mean = float(jnp.mean(rel_l2_per_sample))
    rel_l2_min = float(jnp.min(rel_l2_per_sample))
    rel_l2_max = float(jnp.max(rel_l2_per_sample))

    print(f"{model_name} Results:")
    print(f"  Test MSE:         {mse:.6f}")
    print(
        f"  Relative L2:      {rel_l2_mean:.6f} (min={rel_l2_min:.6f}, max={rel_l2_max:.6f})"
    )

    return predictions, mse, rel_l2_mean


# %%
print()
print("Running evaluation...")
test_data = (test_nodes, test_edges, test_edge_feats, test_targets)
gno_pred, gno_mse, gno_rel_l2 = evaluate_model(gno, test_data, "GNO")

# %% [markdown]
# ## Visualization
#
# Visualize predictions and compare with ground truth.

# %%
# Create output directory
output_dir = Path("docs/assets/examples/gno_darcy")
output_dir.mkdir(parents=True, exist_ok=True)

# Convert predictions back to grid format for visualization
pred_grid = graph_to_grid(gno_pred, height=RESOLUTION, width=RESOLUTION, channels=1)

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
axes[0].set_ylabel("MSE Loss")
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
print(
    f"  GNO:        MSE={gno_mse:.6f}, Rel L2={gno_rel_l2:.4f}, Params={gno_params:,}"
)
print()
print(f"Results saved to: {output_dir}")
print("=" * 70)
