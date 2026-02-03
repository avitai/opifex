"""
Field plotting utilities for visualizing PDE solutions and neural operator outputs.

Provides comprehensive plotting tools for 2D fields, error analysis,
spectral content visualization, and comparative analysis.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_field(
    field: jax.Array,
    title: str = "Field Visualization",
    xlabel: str = "X",
    ylabel: str = "Y",
    colormap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    figsize: tuple[int, int] = (8, 6),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a 2D field with customizable options.

    Args:
        field: 2D array to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        colormap: Matplotlib colormap name
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        figsize: Figure size (width, height)
        save_path: Optional path to save the plot

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy for matplotlib
    field = jnp.asarray(field)

    # Handle 3D input (take first channel if multi-channel)
    if field.ndim == 3:
        field = field[..., 0]
    elif field.ndim != 2:
        raise ValueError("Field must be 2D or 3D (with channels)")

    fig, ax = plt.subplots(figsize=figsize)

    # Set color limits
    if vmin is None:
        vmin = float(np.min(field))
    if vmax is None:
        vmax = float(np.max(field))

    # Create plot
    im = ax.imshow(
        field, cmap=colormap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto"
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Field Value")

    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_field_comparison(
    ground_truth: jax.Array,
    prediction: jax.Array,
    titles: list[str] | None = None,
    colormap: str = "viridis",
    error_colormap: str = "Reds",
    figsize: tuple[int, int] = (15, 5),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot side-by-side comparison of ground truth, prediction, and error.

    Args:
        ground_truth: Ground truth field
        prediction: Predicted field
        titles: Optional titles for subplots
        colormap: Colormap for fields
        error_colormap: Colormap for error plot
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    # Convert to numpy
    gt = np.array(ground_truth)
    pred = np.array(prediction)

    # Handle channel dimension
    if gt.ndim == 3:
        gt = gt[..., 0]
    if pred.ndim == 3:
        pred = pred[..., 0]

    # Calculate error
    error = np.abs(gt - pred)

    # Set default titles
    if titles is None:
        titles = ["Ground Truth", "Prediction", "Absolute Error"]

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Shared color scale for GT and prediction
    vmin = min(float(np.min(gt)), float(np.min(pred)))
    vmax = max(float(np.max(gt)), float(np.max(pred)))

    # Ground truth
    im1 = axes[0].imshow(
        gt, cmap=colormap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto"
    )
    axes[0].set_title(titles[0])
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    plt.colorbar(im1, ax=axes[0], shrink=0.8)

    # Prediction
    im2 = axes[1].imshow(
        pred, cmap=colormap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto"
    )
    axes[1].set_title(titles[1])
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    plt.colorbar(im2, ax=axes[1], shrink=0.8)

    # Error
    im3 = axes[2].imshow(error, cmap=error_colormap, origin="lower", aspect="auto")
    axes[2].set_title(titles[2])
    axes[2].set_xlabel("X")
    axes[2].set_ylabel("Y")
    plt.colorbar(im3, ax=axes[2], shrink=0.8)

    # Add statistics
    mse = float(np.mean((gt - pred) ** 2))
    mae = float(np.mean(np.abs(gt - pred)))
    max_error = float(np.max(error))

    fig.suptitle(f"MSE: {mse:.6f} | MAE: {mae:.6f} | Max Error: {max_error:.6f}")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_spectral_analysis(
    field: jax.Array,
    title: str = "Spectral Analysis",
    log_scale: bool = True,
    figsize: tuple[int, int] = (12, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot spectral analysis of a 2D field including power spectral density.

    Args:
        field: 2D field to analyze
        title: Plot title
        log_scale: Whether to use log scale for PSD
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    field = jnp.asarray(field)

    # Handle channel dimension
    if field.ndim == 3:
        field = field[..., 0]

    # Compute 2D FFT
    field_fft = np.fft.fft2(field)
    field_fft_shifted = np.fft.fftshift(field_fft)

    # Power spectral density
    psd = np.abs(field_fft_shifted) ** 2

    # Frequency grids
    ny, nx = field.shape
    freq_x = np.fft.fftfreq(nx)
    freq_y = np.fft.fftfreq(ny)
    freq_x_shifted = np.fft.fftshift(freq_x)
    freq_y_shifted = np.fft.fftshift(freq_y)

    # Radial average for 1D spectrum
    center_x, center_y = nx // 2, ny // 2
    y, x = np.ogrid[:ny, :nx]
    r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    r = np.asarray(r, dtype=np.int_)

    # Compute radial average
    tbin = np.bincount(r.ravel(), psd.ravel())
    nr = np.bincount(r.ravel())
    radial_spectrum = tbin / nr

    # Create subplot layout
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

    # Original field
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(field, cmap="viridis", origin="lower", aspect="auto")
    ax1.set_title("Original Field")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    plt.colorbar(im1, ax=ax1, shrink=0.8)

    # 2D Power Spectral Density
    ax2 = fig.add_subplot(gs[0, 1])
    if log_scale:
        psd_plot = np.log10(psd + 1e-10)  # Add small value to avoid log(0)
        im2 = ax2.imshow(psd_plot, cmap="hot", origin="lower", aspect="auto")
        ax2.set_title("Log Power Spectral Density")
    else:
        im2 = ax2.imshow(psd, cmap="hot", origin="lower", aspect="auto")
        ax2.set_title("Power Spectral Density")

    ax2.set_xlabel("Frequency X")
    ax2.set_ylabel("Frequency Y")
    plt.colorbar(im2, ax=ax2, shrink=0.8)

    # Radial spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    k = np.arange(len(radial_spectrum))
    if log_scale:
        ax3.loglog(k[1:], radial_spectrum[1:], "b-", linewidth=2)
        ax3.set_ylabel("Log Power")
    else:
        ax3.semilogy(k[1:], radial_spectrum[1:], "b-", linewidth=2)
        ax3.set_ylabel("Power")

    ax3.set_xlabel("Wavenumber")
    ax3.set_title("Radial Spectrum")
    ax3.grid(True, alpha=0.3)

    # Spectral statistics
    ax4 = fig.add_subplot(gs[1, :])

    # Compute spectral moments
    total_energy = float(np.sum(psd))
    dominant_freq_idx = np.unravel_index(np.argmax(psd), psd.shape)
    dominant_freq = (
        freq_x_shifted[dominant_freq_idx[1]],
        freq_y_shifted[dominant_freq_idx[0]],
    )

    # Energy in different frequency bands
    low_freq_energy = float(
        np.sum(
            psd[
                ny // 2 - ny // 8 : ny // 2 + ny // 8,
                nx // 2 - nx // 8 : nx // 2 + nx // 8,
            ]
        )
    )
    high_freq_energy = total_energy - low_freq_energy

    low_freq_pct = 100 * low_freq_energy / total_energy
    high_freq_pct = 100 * high_freq_energy / total_energy

    analysis_text = f"""
    Spectral Analysis Results:

    Total Energy: {total_energy:.2e}
    Dominant Frequency: ({dominant_freq[0]:.3f}, {dominant_freq[1]:.3f})
    Low Freq Energy: {low_freq_energy:.2e} ({low_freq_pct:.1f}%)
    High Freq Energy: {high_freq_energy:.2e} ({high_freq_pct:.1f}%)
    """

    ax4.text(
        0.1,
        0.5,
        analysis_text,
        transform=ax4.transAxes,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
    )
    ax4.axis("off")

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_field_evolution(
    field_sequence: jax.Array,
    time_points: jax.Array | None = None,
    n_snapshots: int = 6,
    colormap: str = "viridis",
    figsize: tuple[int, int] = (15, 10),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot evolution of a field over time with multiple snapshots.

    Args:
        field_sequence: Array of shape (time, height, width) or
            (time, height, width, channels)
        time_points: Optional time values for each snapshot
        n_snapshots: Number of snapshots to show
        colormap: Matplotlib colormap
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    field_sequence_np = np.array(field_sequence)

    # Handle channel dimension
    if field_sequence_np.ndim == 4:
        field_sequence_np = field_sequence_np[..., 0]

    n_times = field_sequence_np.shape[0]

    # Select time indices
    time_indices = np.linspace(
        0, n_times - 1, n_snapshots
    )  # Remove explicit dtype casting

    # Set up subplots
    rows = 2
    cols = (n_snapshots + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_snapshots > 1 else [axes]

    # Global color scale
    vmin = float(np.min(field_sequence_np))
    vmax = float(np.max(field_sequence_np))

    for i, time_idx in enumerate(time_indices):
        if i >= len(axes):
            break

        ax = axes[i]
        time_idx_int = int(time_idx)
        field = field_sequence_np[time_idx_int]

        im = ax.imshow(
            field, cmap=colormap, vmin=vmin, vmax=vmax, origin="lower", aspect="auto"
        )

        title = (
            f"t = {float(time_points[time_idx_int]):.3f}"
            if time_points is not None
            else f"Step {time_idx_int}"
        )

        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")

        # Add colorbar to every other plot to avoid crowding
        if i % 2 == 0:
            plt.colorbar(im, ax=ax, shrink=0.8)

    # Hide unused subplots
    for i in range(n_snapshots, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_vector_field(
    u: jax.Array,
    v: jax.Array,
    x: jax.Array | None = None,
    y: jax.Array | None = None,
    title: str = "Vector Field",
    subsample: int = 1,
    figsize: tuple[int, int] = (10, 8),
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a 2D vector field using quiver plot.

    Args:
        u: X-component of vector field
        v: Y-component of vector field
        x: Optional X coordinates
        y: Optional Y coordinates
        title: Plot title
        subsample: Subsampling factor for arrows
        figsize: Figure size
        save_path: Optional save path

    Returns:
        matplotlib Figure object
    """
    u = jnp.asarray(u)
    v = jnp.asarray(v)

    # Create coordinate grids if not provided
    if x is None or y is None:
        ny, nx = u.shape
        x = jnp.arange(nx)
        y = jnp.arange(ny)
        x, y = jnp.meshgrid(x, y)
    else:
        x = jnp.asarray(x)
        y = jnp.asarray(y)

    # Subsample for cleaner visualization
    x_sub = x[::subsample, ::subsample]
    y_sub = y[::subsample, ::subsample]
    u_sub = u[::subsample, ::subsample]
    v_sub = v[::subsample, ::subsample]

    # Calculate magnitude for color coding
    magnitude = jnp.sqrt(u_sub**2 + v_sub**2)

    fig, ax = plt.subplots(figsize=figsize)

    # Create quiver plot
    Q = ax.quiver(
        x_sub,
        y_sub,
        u_sub,
        v_sub,
        magnitude,
        cmap="viridis",
        scale_units="xy",
        angles="xy",
    )

    # Add colorbar
    plt.colorbar(Q, ax=ax, label="Magnitude", shrink=0.8)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.set_aspect("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig
