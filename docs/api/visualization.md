# Visualization API Reference
```python
from jax import Array
import jax.numpy as jnp
from typing import Optional, List, Tuple
```

The `opifex.visualization` package provides comprehensive visualization tools for scientific computing applications, including field plotting, animation, and performance analysis.

## Overview

The visualization module offers:

- **Field Plotting**: 2D/3D field visualizations with multiple plotting modes
- **Animation**: Create physics-based animations of time-dependent solutions
- **Performance Visualization**: Plot FLOPS, memory usage, and model complexity
- **Spectral Analysis**: Visualize frequency-domain representations
- **Vector Fields**: Streamline and quiver plots for vector data

All visualization functions are designed to work seamlessly with JAX arrays and support both interactive and publication-quality output.

## Field Plotting

### plot_2d_field

Plot 2D scalar fields with various visualization modes.

```python
from opifex.visualization import plot_2d_field

def plot_2d_field(
    field: Array,
    coordinates: Optional[Array] = None,
    title: str = "2D Field",
    cmap: str = "viridis",
    show_colorbar: bool = True,
    levels: Optional[int] = None,
    mode: str = "contourf",
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> plt.Figure:
    """
    Plot 2D scalar field with multiple visualization modes.

    Args:
        field: 2D array of field values, shape (nx, ny)
        coordinates: Optional coordinate grid, shape (nx, ny, 2)
            If None, uses uniform grid [0, nx] × [0, ny]
        title: Plot title
        cmap: Matplotlib colormap name
        show_colorbar: Whether to display colorbar
        levels: Number of contour levels (for contour/contourf modes)
        mode: Visualization mode:
            - 'contourf': Filled contours (default)
            - 'contour': Line contours
            - 'pcolormesh': Pseudocolor plot
            - 'imshow': Image plot
        ax: Matplotlib axes (creates new if None)
        **kwargs: Additional arguments passed to plotting function

    Returns:
        matplotlib Figure object

    Example:
        >>> import jax.numpy as jnp
        >>> x = jnp.linspace(-1, 1, 100)
        >>> y = jnp.linspace(-1, 1, 100)
        >>> X, Y = jnp.meshgrid(x, y)
        >>> field = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * Y)
        >>> fig = plot_2d_field(field, title="Standing Wave")
    """
```

### plot_field_evolution

Visualize the temporal evolution of a field as a sequence of subplots.

```python
from opifex.visualization import plot_field_evolution

def plot_field_evolution(
    trajectory: Array,
    times: Optional[Array] = None,
    num_snapshots: int = 6,
    title: str = "Field Evolution",
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[int, int] = (15, 10)
) -> plt.Figure:
    """
    Plot time evolution of field as subplot grid.

    Args:
        trajectory: Time-dependent field, shape (nt, nx, ny) or (nt, nx)
        times: Time values for each snapshot, shape (nt,)
            If None, uses indices
        num_snapshots: Number of snapshots to display
        title: Overall figure title
        cmap: Colormap name
        vmin, vmax: Color scale limits (auto if None)
        figsize: Figure size in inches

    Returns:
        matplotlib Figure object

    Example:
        >>> # Visualize PDE solution evolution
        >>> trajectory = burgers_solution  # Shape: (100, 256, 256)
        >>> times = jnp.linspace(0, 1, 100)
        >>> fig = plot_field_evolution(
        ...     trajectory,
        ...     times=times,
        ...     num_snapshots=6,
        ...     title="Burgers Equation Evolution"
        ... )
    """
```

### plot_field_comparison

Compare multiple fields side-by-side (e.g., ground truth vs. prediction).

```python
from opifex.visualization import plot_field_comparison

def plot_field_comparison(
    fields: List[Array],
    titles: List[str],
    suptitle: str = "Field Comparison",
    cmap: str = "viridis",
    show_difference: bool = True,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Compare multiple 2D fields side-by-side.

    Args:
        fields: List of 2D arrays to compare
        titles: Title for each field
        suptitle: Overall figure title
        cmap: Colormap name
        show_difference: If True and 2 fields, show difference plot
        figsize: Figure size (auto-computed if None)

    Returns:
        matplotlib Figure object

    Example:
        >>> # Compare model prediction with ground truth
        >>> fields = [ground_truth, prediction]
        >>> titles = ["Ground Truth", "Neural Operator Prediction"]
        >>> fig = plot_field_comparison(
        ...     fields, titles,
        ...     suptitle="FNO Performance",
        ...     show_difference=True
        ... )
    """
```

### plot_vector_field

Visualize 2D vector fields using streamlines or quiver plots.

```python
from opifex.visualization import plot_vector_field

def plot_vector_field(
    u: Array,
    v: Array,
    coordinates: Optional[Tuple[Array, Array]] = None,
    mode: str = "streamplot",
    density: float = 1.0,
    color: Optional[Array] = None,
    title: str = "Vector Field",
    ax: Optional[plt.Axes] = None
) -> plt.Figure:
    """
    Plot 2D vector field.

    Args:
        u: x-component of vector field, shape (nx, ny)
        v: y-component of vector field, shape (nx, ny)
        coordinates: Optional (X, Y) mesh grids
        mode: Visualization mode:
            - 'streamplot': Streamlines (default)
            - 'quiver': Arrow plot
        density: Streamline/arrow density
        color: Optional scalar field for coloring, shape (nx, ny)
        title: Plot title
        ax: Matplotlib axes

    Returns:
        matplotlib Figure object

    Example:
        >>> # Visualize fluid velocity field
        >>> u = jnp.cos(X) * jnp.sin(Y)  # x-velocity
        >>> v = -jnp.sin(X) * jnp.cos(Y)  # y-velocity
        >>> magnitude = jnp.sqrt(u**2 + v**2)
        >>> fig = plot_vector_field(
        ...     u, v,
        ...     mode="streamplot",
        ...     color=magnitude,
        ...     title="Velocity Field"
        ... )
    """
```

### plot_spectral_analysis

Visualize frequency-domain representation of fields.

```python
from opifex.visualization import plot_spectral_analysis

def plot_spectral_analysis(
    field: Array,
    axis: int = -1,
    title: str = "Spectral Analysis",
    show_phase: bool = False,
    log_scale: bool = True
) -> plt.Figure:
    """
    Plot spectral (Fourier) analysis of field.

    Args:
        field: Input field array
        axis: Axis along which to compute FFT
        title: Plot title
        show_phase: Whether to show phase plot
        log_scale: Use logarithmic scale for magnitude

    Returns:
        matplotlib Figure object

    Example:
        >>> # Analyze frequency content of solution
        >>> fig = plot_spectral_analysis(
        ...     solution,
        ...     axis=-1,
        ...     title="Frequency Spectrum",
        ...     log_scale=True
        ... )
    """
```

## Animation

### create_physics_animation

Create animated visualizations of time-dependent physics simulations.

```python
from opifex.visualization import create_physics_animation

def create_physics_animation(
    trajectory: Array,
    times: Optional[Array] = None,
    interval: int = 50,
    cmap: str = "viridis",
    title: str = "Physics Animation",
    save_path: Optional[str] = None,
    fps: int = 30,
    writer: str = "pillow"
) -> animation.FuncAnimation:
    """
    Create animation of time-dependent field evolution.

    Args:
        trajectory: Time-dependent field, shape (nt, nx, ny) or (nt, nx)
        times: Time values, shape (nt,)
        interval: Delay between frames in milliseconds
        cmap: Colormap name
        title: Animation title
        save_path: If provided, save animation to this path
            Supports .gif, .mp4, .avi formats
        fps: Frames per second for saved video
        writer: Animation writer ('pillow', 'ffmpeg', 'imagemagick')

    Returns:
        matplotlib FuncAnimation object

    Example:
        >>> # Create and save animation
        >>> trajectory = burgers_evolution  # Shape: (200, 256, 256)
        >>> times = jnp.linspace(0, 2, 200)
        >>> anim = create_physics_animation(
        ...     trajectory,
        ...     times=times,
        ...     save_path="burgers_evolution.gif",
        ...     fps=30,
        ...     title="Burgers Equation"
        ... )
        >>> # Display in Jupyter
        >>> from IPython.display import HTML
        >>> HTML(anim.to_html5_video())
    """
```

### Advanced Animation Features

```python
# Create multi-panel animations
def create_comparison_animation(
    trajectories: List[Array],
    titles: List[str],
    times: Optional[Array] = None,
    **kwargs
) -> animation.FuncAnimation:
    """
    Animate multiple fields side-by-side for comparison.

    Args:
        trajectories: List of time-dependent fields
        titles: Title for each panel
        times: Shared time values
        **kwargs: Additional arguments passed to create_physics_animation

    Returns:
        matplotlib FuncAnimation object
    """

# Add overlays (e.g., sensor locations, boundaries)
def create_annotated_animation(
    trajectory: Array,
    annotations: Dict[str, Any],
    **kwargs
) -> animation.FuncAnimation:
    """
    Create animation with custom annotations.

    Args:
        trajectory: Time-dependent field
        annotations: Dictionary specifying overlays:
            - 'points': Array of (x, y) coordinates
            - 'lines': List of line segments
            - 'text': List of text labels
        **kwargs: Additional arguments

    Returns:
        matplotlib FuncAnimation object
    """
```

## Performance Visualization

### plot_flops_analysis

Visualize computational complexity (FLOPs) analysis.

```python
from opifex.visualization import plot_flops_analysis

def plot_flops_analysis(
    flops_data: Dict[str, int],
    title: str = "FLOPS Analysis",
    log_scale: bool = True,
    show_breakdown: bool = True
) -> plt.Figure:
    """
    Plot FLOPS analysis for model or computation.

    Args:
        flops_data: Dictionary mapping operation names to FLOP counts
            Example: {'forward': 1e9, 'backward': 2e9, 'total': 3e9}
        title: Plot title
        log_scale: Use logarithmic scale
        show_breakdown: Show breakdown by operation type

    Returns:
        matplotlib Figure object

    Example:
        >>> from opifex.training import FlopsCounter
        >>> counter = FlopsCounter(model)
        >>> flops = counter.count(sample_input)
        >>> fig = plot_flops_analysis(
        ...     flops,
        ...     title="FNO Computational Cost"
        ... )
    """
```

### plot_memory_usage

Visualize memory consumption over time or by component.

```python
from opifex.visualization import plot_memory_usage

def plot_memory_usage(
    memory_data: Array,
    timestamps: Optional[Array] = None,
    title: str = "Memory Usage",
    show_peak: bool = True,
    unit: str = "GB"
) -> plt.Figure:
    """
    Plot memory usage over time.

    Args:
        memory_data: Memory usage values
        timestamps: Time points (or iteration numbers)
        title: Plot title
        show_peak: Highlight peak memory usage
        unit: Memory unit ('GB', 'MB', 'KB')

    Returns:
        matplotlib Figure object

    Example:
        >>> # Monitor memory during training
        >>> from opifex.training import MemoryMonitor
        >>> monitor = MemoryMonitor()
        >>> # ... training loop ...
        >>> fig = plot_memory_usage(
        ...     monitor.memory_history,
        ...     timestamps=monitor.timestamps,
        ...     title="Training Memory Profile"
        ... )
    """
```

### plot_model_complexity_comparison

Compare complexity metrics across multiple models.

```python
from opifex.visualization import plot_model_complexity_comparison

def plot_model_complexity_comparison(
    models: Dict[str, Any],
    metrics: List[str] = ['params', 'flops', 'memory'],
    normalize: bool = True
) -> plt.Figure:
    """
    Compare computational complexity of multiple models.

    Args:
        models: Dictionary mapping model names to model objects
        metrics: List of metrics to compare:
            - 'params': Number of parameters
            - 'flops': Floating point operations
            - 'memory': Memory footprint
            - 'inference_time': Inference latency
        normalize: Normalize to smallest model

    Returns:
        matplotlib Figure object

    Example:
        >>> models = {
        ...     'FNO-Small': fno_small,
        ...     'FNO-Large': fno_large,
        ...     'DeepONet': deeponet,
        ...     'U-Net': unet
        ... }
        >>> fig = plot_model_complexity_comparison(
        ...     models,
        ...     metrics=['params', 'flops', 'memory']
        ... )
    """
```

## Integration with Training

### Training Progress Visualization

```python
from opifex.visualization import plot_training_curves

def plot_training_curves(
    history: Dict[str, List[float]],
    metrics: Optional[List[str]] = None,
    smoothing: float = 0.0,
    log_scale: bool = False
) -> plt.Figure:
    """
    Plot training and validation curves.

    Args:
        history: Dictionary mapping metric names to value lists
            Example: {'train_loss': [...], 'val_loss': [...]}
        metrics: Specific metrics to plot (None = all)
        smoothing: Exponential smoothing factor (0 = none, 1 = max)
        log_scale: Use logarithmic scale for y-axis

    Returns:
        matplotlib Figure object

    Example:
        >>> from opifex.training import BasicTrainer
        >>> trainer = BasicTrainer(model, dataset)
        >>> history = trainer.train(epochs=100)
        >>> fig = plot_training_curves(
        ...     history,
        ...     metrics=['train_loss', 'val_loss'],
        ...     smoothing=0.6
        ... )
    """
```

## Customization and Styling

### Publication-Quality Plots

```python
from opifex.visualization import set_publication_style

def set_publication_style(style: str = 'default'):
    """
    Configure matplotlib for publication-quality figures.

    Args:
        style: Style preset:
            - 'default': Opifex default style
            - 'nature': Nature journal style
            - 'ieee': IEEE style
            - 'thesis': Thesis/dissertation style

    Example:
        >>> set_publication_style('nature')
        >>> fig = plot_2d_field(field)
        >>> fig.savefig('figure.pdf', dpi=300, bbox_inches='tight')
    """
```

### Custom Colormaps

```python
from opifex.visualization import create_custom_colormap

def create_custom_colormap(
    name: str,
    colors: List[str],
    n_bins: int = 256
) -> mcolors.LinearSegmentedColormap:
    """
    Create custom colormap for specific visualization needs.

    Args:
        name: Colormap name
        colors: List of color specifications (hex, RGB, or names)
        n_bins: Number of discrete color levels

    Returns:
        matplotlib colormap object

    Example:
        >>> # Create physics-specific colormap
        >>> cmap = create_custom_colormap(
        ...     'pressure',
        ...     ['#0000FF', '#FFFFFF', '#FF0000'],
        ...     n_bins=256
        ... )
        >>> fig = plot_2d_field(pressure_field, cmap=cmap)
    """
```

## Performance Considerations

### Large Dataset Visualization

```python
# For large fields, downsample before plotting
from opifex.visualization import downsample_field

def downsample_field(
    field: Array,
    target_size: Tuple[int, int],
    method: str = 'mean'
) -> Array:
    """
    Downsample field for faster visualization.

    Args:
        field: High-resolution field
        target_size: Target (nx, ny) for visualization
        method: Downsampling method ('mean', 'max', 'min')

    Returns:
        Downsampled field
    """

# Example usage
high_res_field = solution  # Shape: (4096, 4096)
vis_field = downsample_field(high_res_field, (512, 512))
fig = plot_2d_field(vis_field)
```

### Batch Visualization

```python
from opifex.visualization import plot_batch_grid

def plot_batch_grid(
    batch: Array,
    num_samples: int = 16,
    titles: Optional[List[str]] = None,
    **kwargs
) -> plt.Figure:
    """
    Visualize multiple samples from a batch in grid layout.

    Args:
        batch: Batch of fields, shape (batch_size, nx, ny)
        num_samples: Number of samples to display
        titles: Optional title for each sample
        **kwargs: Additional arguments passed to plot_2d_field

    Returns:
        matplotlib Figure object

    Example:
        >>> # Visualize batch of predictions
        >>> predictions = model(test_batch)  # Shape: (64, 128, 128)
        >>> fig = plot_batch_grid(
        ...     predictions,
        ...     num_samples=16,
        ...     cmap='viridis'
        ... )
    """
```

## Integration Examples

### Complete Workflow Example

```python
import jax
import jax.numpy as jnp
from opifex.data.loaders import create_burgers_loader
from opifex.neural.operators.fno import FourierNeuralOperator
from opifex.training import BasicTrainer, TrainingConfig
from opifex.visualization import (
    plot_field_comparison,
    create_physics_animation,
    plot_training_curves,
    set_publication_style
)

# Setup data loader
train_loader = create_burgers_loader(
    n_samples=1000,
    batch_size=32,
    resolution=256,
    seed=42,
)

# Train model
model = FourierNeuralOperator(
    in_channels=1,
    out_channels=1,
    hidden_channels=32,
    modes=12,
    num_layers=4,
    rngs=nnx.Rngs(42),
)
config = TrainingConfig(num_epochs=100, learning_rate=1e-3)
trainer = BasicTrainer(model, config)
trained_model, history = trainer.train(train_loader)

# Visualize training progress
set_publication_style('default')
fig1 = plot_training_curves(history)
fig1.savefig('training_curves.pdf')

# Compare predictions
test_sample = dataset[0]
prediction = model(test_sample['input'])
ground_truth = test_sample['output'][-1]  # Final time

fig2 = plot_field_comparison(
    [ground_truth, prediction],
    titles=['Ground Truth', 'FNO Prediction'],
    show_difference=True
)
fig2.savefig('comparison.pdf')

# Create animation
trajectory_pred = model.predict_trajectory(test_sample['input'], steps=100)
anim = create_physics_animation(
    trajectory_pred,
    save_path='prediction.gif',
    title='FNO Prediction'
)
```

## See Also

- [Data API](data.md): Dataset classes and preprocessing
- [Training API](training.md): Training infrastructure
- [Neural API](neural.md): Neural network architectures
- [Examples](../examples/index.md): Complete usage examples
