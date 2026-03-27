#!/usr/bin/env python3
"""PDEBench competitor comparison benchmark.

Runs Opifex neural operators on actual PDEBench HDF5 data and compares
against published results from the PDEBench paper (Takamoto et al. 2022).

Published baseline results (normalized RMSE from PDEBench Table 2):
  2D Darcy Flow (beta=1.0): FNO = 7.22e-3, U-Net = 1.05e-2
  1D Burgers (Nu=1.0):      FNO = 3.07e-3, U-Net = 5.44e-3

Usage:
    source activate.sh
    uv run python scripts/run_pdebench_comparison.py
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from opifex.neural.operators.fno.base import FourierNeuralOperator


PDEBENCH_BASELINES = {
    "2D_DarcyFlow_FNO": 7.22e-3,
    "2D_DarcyFlow_UNet": 1.05e-2,
    "1D_Burgers_FNO": 3.07e-3,
    "1D_Burgers_UNet": 5.44e-3,
}

DATA_DIR = Path("example_data/pdebench")

PDEBENCH_URLS = {
    "2D_DarcyFlow_beta1.0_Train.hdf5": "https://darus.uni-stuttgart.de/api/access/datafile/133219",
    "1D_Burgers_Sols_Nu1.0.hdf5": "https://darus.uni-stuttgart.de/api/access/datafile/281365",
}


def _ensure_pdebench_data(filename: str) -> Path:
    """Download PDEBench HDF5 file if not present."""
    filepath = DATA_DIR / filename
    if filepath.exists():
        return filepath

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = PDEBENCH_URLS[filename]
    print(f"  Downloading {filename} from DaRUS...")

    import urllib.request

    urllib.request.urlretrieve(url, filepath)  # noqa: S310
    print(f"  Downloaded: {filepath} ({filepath.stat().st_size / 1e6:.0f} MB)")
    return filepath


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Result from a single benchmark run."""

    dataset: str
    operator: str
    n_train: int
    n_test: int
    n_epochs: int
    relative_l2: float
    mse: float
    training_time_s: float
    parameters: int


def _train_and_evaluate(
    dataset_name: str,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    x_test: jnp.ndarray,
    y_test: jnp.ndarray,
    n_epochs: int,
    batch_size: int,
    modes: int,
    hidden: int,
    lr: float,
    seed: int,
    domain_padding: int = 0,
) -> BenchmarkResult:
    """Train FNO and evaluate — shared logic for all datasets (DRY)."""
    import sys

    print(f"  Train: {x_train.shape} -> {y_train.shape}")
    sys.stdout.flush()
    print(f"  Test:  {x_test.shape} -> {y_test.shape}")

    rngs = nnx.Rngs(seed)
    model = FourierNeuralOperator(
        in_channels=x_train.shape[1],
        out_channels=y_train.shape[1],
        modes=modes,
        hidden_channels=hidden,
        num_layers=4,
        domain_padding=domain_padding,
        rngs=rngs,
    )
    # StepLR: halve every 100 epochs (matching PDEBench config)
    n_batches_est = max(x_train.shape[0] // batch_size, 1)
    boundaries = {100 * n_batches_est * i: 0.5 for i in range(1, 6)}
    schedule = optax.piecewise_constant_schedule(lr, boundaries)
    # PDEBench uses torch.optim.Adam (L2 reg), not AdamW (decoupled weight decay)
    optimizer = optax.chain(
        optax.add_decayed_weights(weight_decay=1e-4),  # L2 regularization
        optax.adam(schedule),
    )
    opt = nnx.Optimizer(model, optimizer, wrt=nnx.Param)
    n_params = sum(p.size for p in jax.tree_util.tree_leaves(nnx.state(model, nnx.Param)))
    print(f"  Parameters: {n_params}")

    @nnx.jit
    def train_step(m, o, x, y):
        def loss_fn(m_inner):
            return jnp.mean((m_inner(x) - y) ** 2)

        loss, grads = nnx.value_and_grad(loss_fn)(m)
        o.update(m, grads)
        return loss

    # Shuffle data each epoch (matches PyTorch DataLoader shuffle=True)
    rng = np.random.default_rng(seed)
    n_samples = x_train.shape[0]

    start = time.perf_counter()
    n_batches = max(n_samples // batch_size, 1)
    log_interval = max(n_epochs // 20, 1)  # Log ~20 times during training
    for epoch in range(n_epochs):
        # Random permutation each epoch (matching PDEBench DataLoader)
        perm = rng.permutation(n_samples)
        epoch_loss = 0.0
        for i in range(n_batches):
            idx = perm[i * batch_size : (i + 1) * batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            epoch_loss += train_step(model, opt, xb, yb)
        if (epoch + 1) % log_interval == 0:
            elapsed = time.perf_counter() - start
            avg_loss = epoch_loss / n_batches
            print(f"  Epoch {epoch + 1}/{n_epochs}: loss={avg_loss:.6f} [{elapsed:.0f}s]")
            sys.stdout.flush()
    train_time = time.perf_counter() - start

    # Evaluate in batches to avoid OOM on large test sets
    eval_bs = min(batch_size, x_test.shape[0])
    all_preds = []
    for i in range(0, x_test.shape[0], eval_bs):
        all_preds.append(model(x_test[i : i + eval_bs]))
    preds = jnp.concatenate(all_preds, axis=0)

    mse = float(jnp.mean((preds - y_test) ** 2))
    rel_l2 = float(
        jnp.mean(
            jnp.linalg.norm((preds - y_test).reshape(preds.shape[0], -1), axis=1)
            / jnp.linalg.norm(y_test.reshape(y_test.shape[0], -1), axis=1)
        )
    )
    print(f"  MSE: {mse:.6f}, Relative L2: {rel_l2:.6f}, Time: {train_time:.1f}s")

    return BenchmarkResult(
        dataset=dataset_name,
        operator="FNO",
        n_train=x_train.shape[0],
        n_test=x_test.shape[0],
        n_epochs=n_epochs,
        relative_l2=rel_l2,
        mse=mse,
        training_time_s=train_time,
        parameters=n_params,
    )


def benchmark_darcy_pdebench(
    n_train: int = 9000,
    n_test: int = 1000,
    n_epochs: int = 500,
    batch_size: int = 5,
    modes: int = 12,
    hidden: int = 20,
    lr: float = 1e-3,
    seed: int = 42,
) -> BenchmarkResult:
    """Benchmark FNO on PDEBench 2D Darcy Flow (actual HDF5 data).

    Matches PDEBench config exactly: batch_size=5, modes=12, width=20,
    StepLR(step=100, gamma=0.5), domain_padding=2, Adam(weight_decay=1e-4),
    grid coordinates appended as input channels (like reference FNO2d).
    """
    import sys

    print("=" * 60)
    print("Benchmark: FNO on PDEBench 2D Darcy Flow")
    print(f"  Config: {n_train} train, {n_test} test, {n_epochs} epochs")
    print(f"  Architecture: modes={modes}, hidden={hidden}, padding=2")
    print("=" * 60)
    sys.stdout.flush()

    hdf5_path = _ensure_pdebench_data("2D_DarcyFlow_beta1.0_Train.hdf5")

    with h5py.File(hdf5_path, "r") as f:
        x_all = np.array(f["nu"][: n_train + n_test])  # (N, 128, 128)
        y_all = np.array(f["tensor"][: n_train + n_test])  # (N, 1, 128, 128)

    H, W = x_all.shape[1], x_all.shape[2]

    # Use PDEBench's grid coordinates from the HDF5 file (cell-centered)
    with h5py.File(hdf5_path, "r") as f:
        gx = np.array(f["x-coordinate"], dtype=np.float32)
        gy = np.array(f["y-coordinate"], dtype=np.float32)
    grid_x, grid_y = np.meshgrid(gx, gy, indexing="ij")
    grid = np.stack([grid_x, grid_y], axis=0)  # (2, H, W)

    def _add_grid(x_raw):
        """Concat permeability + grid coords: (N,128,128) -> (N,3,128,128)."""
        n = x_raw.shape[0]
        x_ch = x_raw[:, np.newaxis, :, :]  # (N, 1, H, W)
        g = np.broadcast_to(grid[np.newaxis], (n, 2, H, W))
        return jnp.array(np.concatenate([x_ch, g], axis=1))

    x_train = _add_grid(x_all[:n_train])
    y_train = jnp.array(y_all[:n_train])
    x_test = _add_grid(x_all[n_train:])
    y_test = jnp.array(y_all[n_train:])

    return _train_and_evaluate(
        "2D_DarcyFlow",
        x_train,
        y_train,
        x_test,
        y_test,
        n_epochs,
        batch_size,
        modes,
        hidden,
        lr,
        seed,
        domain_padding=2,
    )


def benchmark_burgers_pdebench(
    n_train: int = 9000,
    n_test: int = 1000,
    n_epochs: int = 500,
    batch_size: int = 5,
    modes: int = 16,
    hidden: int = 64,
    lr: float = 1e-3,
    seed: int = 42,
) -> BenchmarkResult | None:
    """Benchmark FNO on PDEBench 1D Burgers (actual HDF5 data).

    Matches PDEBench FNO1d config: batch_size=5, modes=16, width=64,
    StepLR(step=100, gamma=0.5), domain_padding=8, Adam(weight_decay=1e-4).

    Note: PDEBench trains Burgers autoregressively (sliding window of 10
    timesteps). For simplicity, we use single-step prediction (t=0 -> t=-1)
    which is a harder task. Results may differ from published baselines.
    """
    import sys

    print("=" * 60)
    print("Benchmark: FNO on PDEBench 1D Burgers")
    print(f"  Config: {n_train} train, {n_test} test, {n_epochs} epochs")
    print(f"  Architecture: modes={modes}, hidden={hidden}, padding=8")
    print("=" * 60)
    sys.stdout.flush()

    hdf5_path = _ensure_pdebench_data("1D_Burgers_Sols_Nu1.0.hdf5")

    with h5py.File(hdf5_path, "r") as f:
        # tensor shape: (N, T, X) — time-series data
        data = np.array(f["tensor"][: n_train + n_test])

    # Input: initial condition (t=0), Target: final state (t=-1)
    x_all = data[:, 0, :]  # (N, X)
    y_all = data[:, -1, :]  # (N, X)

    # Add grid coordinates as second channel (matching PDEBench FNO1d)
    n_x = x_all.shape[1]
    with h5py.File(hdf5_path, "r") as f:
        if "x-coordinate" in f:
            grid_x = np.array(f["x-coordinate"], dtype=np.float32)
        else:
            grid_x = np.linspace(0, 1, n_x, dtype=np.float32)

    # Channel-first: (N, 2, X) = [u(x,t=0), x_coord]
    n_total = x_all.shape[0]
    grid_broadcast = np.broadcast_to(grid_x[np.newaxis, np.newaxis, :], (n_total, 1, n_x))
    x_with_grid = np.concatenate([x_all[:, np.newaxis, :], grid_broadcast], axis=1)

    x_train = jnp.array(x_with_grid[:n_train])  # (N, 2, X)
    y_train = jnp.array(y_all[:n_train, np.newaxis, :])
    x_test = jnp.array(x_with_grid[n_train:])
    y_test = jnp.array(y_all[n_train:, np.newaxis, :])

    return _train_and_evaluate(
        "1D_Burgers",
        x_train,
        y_train,
        x_test,
        y_test,
        n_epochs,
        batch_size,
        modes,
        hidden,
        lr,
        seed,
        domain_padding=8,
    )


def print_comparison(results: list[BenchmarkResult]) -> None:
    """Print comparison table with PDEBench baselines."""
    print()
    print("=" * 80)
    print("COMPARISON WITH PDEBENCH BASELINES (Takamoto et al. 2022, NeurIPS)")
    print("=" * 80)
    print()
    print(f"{'Dataset':<20} {'Opifex FNO':>12} {'PDEBench FNO':>14} {'PDEBench UNet':>14}")
    print("-" * 62)
    for r in results:
        fno = PDEBENCH_BASELINES.get(f"{r.dataset}_FNO", float("nan"))
        unet = PDEBENCH_BASELINES.get(f"{r.dataset}_UNet", float("nan"))
        print(f"{r.dataset:<20} {r.relative_l2:>12.6f} {fno:>14.6f} {unet:>14.6f}")
    print()


def main() -> None:
    """Run benchmarks on actual PDEBench data."""
    print(f"JAX backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")
    print()

    results = []

    # Darcy Flow — exact PDEBench config (modes=12, width=20)
    darcy = benchmark_darcy_pdebench()
    results.append(darcy)
    print()

    # Burgers — exact PDEBench config (modes=12, width=20)
    burgers = benchmark_burgers_pdebench()
    if burgers is not None:
        results.append(burgers)
    print()

    print_comparison(results)

    out = Path("benchmark-data")
    out.mkdir(exist_ok=True)
    with (out / "pdebench_comparison.json").open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to: {out / 'pdebench_comparison.json'}")


if __name__ == "__main__":
    main()
