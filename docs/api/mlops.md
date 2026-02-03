# MLOps API

```python
from typing import Optional, List, Dict, Any
```
 Reference

The `opifex.mlops` package provides unified experiment tracking and model lifecycle management optimized for scientific machine learning workflows.

## Overview

The MLOps module offers:

- **Experiment Tracking**: Track experiments across different physics domains
- **Model Versioning**: Version models with rich metadata
- **Metrics Logging**: Domain-specific metrics for PINNs, neural operators, L2O, etc.
- **Backend Agnostic**: Support for MLflow and extensible to other backends
- **Physics-Aware**: Specialized tracking for scientific ML workflows

##! Experiment Management

### ExperimentTracker

Main interface for experiment tracking in Opifex.

```python
from opifex.mlops import ExperimentTracker

class ExperimentTracker:
    """
    Unified experiment tracking for scientific ML workflows.

    Provides a backend-agnostic interface for logging experiments,
    metrics, models, and artifacts with physics-domain awareness.

    Args:
        backend: Backend name ('mlflow' or custom)
        experiment_name: Name of experiment group
        tracking_uri: URI for tracking server (optional)
        auto_log: Automatically log common metrics

    Example:
        >>> tracker = ExperimentTracker(
        ...     backend='mlflow',
        ...     experiment_name='darcy-flow-operators',
        ...     tracking_uri='http://localhost:5000'
        ... )
    """

    def __init__(
        self,
        backend: str = 'mlflow',
        experiment_name: str = 'default',
        tracking_uri: Optional[str] = None,
        auto_log: bool = True
    ):
        """Initialize experiment tracker."""
```

#### Methods

##### `start_run(run_name, config) -> Run`

Start a new experiment run.

```python
def start_run(
    self,
    run_name: Optional[str] = None,
    config: Optional[ExperimentConfig] = None,
    tags: Optional[Dict[str, str]] = None
) -> Run:
    """
    Start new experiment run.

    Args:
        run_name: Human-readable run name
        config: Experiment configuration
        tags: Additional tags for organization

    Returns:
        Run object for logging metrics and artifacts

    Example:
        >>> config = ExperimentConfig(
        ...     framework=Framework.JAX,
        ...     domain=PhysicsDomain.NEURAL_OPERATORS,
        ...     model_type='FNO',
        ...     learning_rate=1e-3,
        ...     batch_size=32
        ... )
        >>> run = tracker.start_run(
        ...     run_name='fno-baseline',
        ...     config=config,
        ...     tags={'dataset': 'darcy', 'resolution': '64x64'}
        ... )
    """
```

##### `log_metrics(metrics, step)`

Log metrics for current run.

```python
def log_metrics(
    self,
    metrics: Dict[str, float],
    step: Optional[int] = None,
    timestamp: Optional[int] = None
) -> None:
    """
    Log metrics to current run.

    Args:
        metrics: Dictionary of metric names to values
        step: Training step/epoch number
        timestamp: Unix timestamp (auto-generated if None)

    Example:
        >>> # Log training metrics
        >>> run.log_metrics({
        ...     'train/loss': 0.045,
        ...     'train/relative_l2': 0.023,
        ...     'val/loss': 0.052,
        ...     'val/relative_l2': 0.028
        ... }, step=100)
    """
```

##### `log_physics_metrics(metrics, step)`

Log physics-domain specific metrics.

```python
def log_physics_metrics(
    self,
    metrics: Union[PINNMetrics, NeuralOperatorMetrics, L2OMetrics, NeuralDFTMetrics],
    step: Optional[int] = None
) -> None:
    """
    Log domain-specific physics metrics.

    Args:
        metrics: Physics-specific metrics object
        step: Training step

    Example:
        >>> # For neural operators
        >>> metrics = NeuralOperatorMetrics(
        ...     operator_error=0.012,
        ...     pointwise_error=0.034,
        ...     conservation_error=1.2e-5,
        ...     stability_metric=0.998
        ... )
        >>> run.log_physics_metrics(metrics, step=500)
        >>>
        >>> # For PINNs
        >>> pinn_metrics = PINNMetrics(
        ...     pde_residual=2.3e-4,
        ...     bc_violation=1.1e-5,
        ...     ic_violation=8.7e-6,
        ...     total_loss=0.045
        ... )
        >>> run.log_physics_metrics(pinn_metrics, step=500)
    """
```

##### `log_model(model, artifact_path)`

Log model weights and architecture.

```python
def log_model(
    self,
    model: nnx.Module,
    artifact_path: str = "model",
    metadata: Optional[Dict] = None,
    save_optimizer_state: bool = False
) -> None:
    """
    Log model to experiment tracking.

    Args:
        model: Flax NNX model
        artifact_path: Path within run artifacts
        metadata: Additional model metadata
        save_optimizer_state: Include optimizer state

    Example:
        >>> from opifex.neural.operators.fno import FNO
        >>> model = FNO(modes=12, width=64)
        >>> # After training...
        >>> run.log_model(
        ...     model,
        ...     artifact_path="final_model",
        ...     metadata={'val_loss': 0.045}
        ... )
    """
```

##### `log_artifact(path, artifact_type)`

Log arbitrary artifacts (plots, data, etc.).

```python
def log_artifact(
    self,
    path: str,
    artifact_type: Optional[str] = None,
    description: Optional[str] = None
) -> None:
    """
    Log artifact to run.

    Args:
        path: Path to artifact file/directory
        artifact_type: Type hint ('plot', 'data', 'config', etc.)
        description: Human-readable description

    Example:
        >>> # Log visualization
        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(history['loss'])
        >>> fig.savefig('loss_curve.png')
        >>> run.log_artifact(
        ...     'loss_curve.png',
        ...     artifact_type='plot',
        ...     description='Training loss curve'
        ... )
    """
```

##### `end_run(status)`

End current run.

```python
def end_run(
    self,
    status: str = 'FINISHED'
) -> None:
    """
    End current experiment run.

    Args:
        status: Run status ('FINISHED', 'FAILED', 'KILLED')

    Example:
        >>> try:
        ...     # Training code
        ...     run.end_run(status='FINISHED')
        ... except Exception as e:
        ...     run.log_param('error', str(e))
        ...     run.end_run(status='FAILED')
    """
```

## Configuration

### ExperimentConfig

Configuration object for experiments.

```python
from opifex.mlops import ExperimentConfig, Framework, PhysicsDomain

@dataclass
class ExperimentConfig:
    """
    Configuration for scientific ML experiments.

    Attributes:
        framework: ML framework (JAX, PyTorch, TensorFlow)
        domain: Physics domain
        model_type: Model architecture name
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of training epochs
        optimizer: Optimizer name
        loss_function: Loss function specification
        regularization: Regularization config
        data_config: Dataset configuration
        hardware: Hardware configuration (GPU/TPU)
        seed: Random seed for reproducibility
    """

    framework: Framework
    domain: PhysicsDomain
    model_type: str
    learning_rate: float
    batch_size: int
    num_epochs: int
    optimizer: str = "adam"
    loss_function: str = "mse"
    regularization: Optional[Dict] = None
    data_config: Optional[Dict] = None
    hardware: Optional[str] = None
    seed: int = 42
```

### Framework Enum

Supported ML frameworks.

```python
from enum import Enum

class Framework(str, Enum):
    """Supported ML frameworks."""
    JAX = "jax"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
```

### PhysicsDomain Enum

Physics domains for specialized tracking.

```python
class PhysicsDomain(str, Enum):
    """Physics domains for scientific ML."""
    NEURAL_OPERATORS = "neural-operators"
    PINNS = "pinn"
    L2O = "l2o"
    NEURAL_DFT = "neural-dft"
    QUANTUM_COMPUTING = "quantum-computing"
```

## Physics-Specific Metrics

### PINNMetrics

Metrics for Physics-Informed Neural Networks.

```python
from opifex.mlops import PINNMetrics

@dataclass
class PINNMetrics:
    """
    Metrics specific to Physics-Informed Neural Networks.

    Attributes:
        pde_residual: PDE equation residual loss
        bc_violation: Boundary condition violation
        ic_violation: Initial condition violation
        total_loss: Combined loss
        data_loss: Supervised data fitting loss (if applicable)
        gradient_norm: Gradient norm for stability monitoring
    """

    pde_residual: float
    bc_violation: float
    ic_violation: float
    total_loss: float
    data_loss: Optional[float] = None
    gradient_norm: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for logging."""
        return {
            'pinn/pde_residual': self.pde_residual,
            'pinn/bc_violation': self.bc_violation,
            'pinn/ic_violation': self.ic_violation,
            'pinn/total_loss': self.total_loss,
            'pinn/data_loss': self.data_loss or 0.0,
            'pinn/gradient_norm': self.gradient_norm or 0.0
        }
```

### NeuralOperatorMetrics

Metrics for neural operators.

```python
from opifex.mlops import NeuralOperatorMetrics

@dataclass
class NeuralOperatorMetrics:
    """
    Metrics for neural operator learning.

    Attributes:
        operator_error: Operator approximation error
        pointwise_error: Pointwise prediction error
        conservation_error: Conservation law violation
        stability_metric: Stability measure
        relative_l2: Relative L2 error
        spectral_error: Error in frequency domain
    """

    operator_error: float
    pointwise_error: float
    conservation_error: Optional[float] = None
    stability_metric: Optional[float] = None
    relative_l2: Optional[float] = None
    spectral_error: Optional[float] = None
```

### L2OMetrics

Metrics for Learn-to-Optimize algorithms.

```python
from opifex.mlops import L2OMetrics

@dataclass
class L2OMetrics:
    """
    Metrics for learn-to-optimize meta-learning.

    Attributes:
        meta_loss: Meta-learning objective value
        inner_loss: Inner optimization loss
        outer_loss: Outer optimization loss
        optimization_steps: Number of inner steps taken
        convergence_rate: Rate of convergence
        final_accuracy: Final task accuracy
    """

    meta_loss: float
    inner_loss: float
    outer_loss: float
    optimization_steps: int
    convergence_rate: Optional[float] = None
    final_accuracy: Optional[float] = None
```

### NeuralDFTMetrics

Metrics for Neural Density Functional Theory.

```python
from opifex.mlops import NeuralDFTMetrics

@dataclass
class NeuralDFTMetrics:
    """
    Metrics for neural DFT calculations.

    Attributes:
        total_energy_error: Total energy prediction error
        density_error: Electron density error
        xc_energy_error: Exchange-correlation energy error
        scf_iterations: Self-consistent field iterations
        convergence_achieved: Whether SCF converged
        forces_mae: Mean absolute error in forces
    """

    total_energy_error: float
    density_error: float
    xc_energy_error: float
    scf_iterations: int
    convergence_achieved: bool
    forces_mae: Optional[float] = None
```

## Backend Integration

### MLflow Backend

MLflow integration for experiment tracking.

```python
from opifex.mlops.backends import MLflowBackend, MLFLOW_AVAILABLE

if MLFLOW_AVAILABLE:
    backend = MLflowBackend(
        tracking_uri='http://localhost:5000',
        experiment_name='my-experiment'
    )

    # Use with ExperimentTracker
    tracker = ExperimentTracker(backend=backend)
```

### Custom Backends

Implement custom tracking backends.

```python
from opifex.mlops.backends import BackendInterface

class CustomBackend(BackendInterface):
    """Custom experiment tracking backend."""

    def start_run(self, run_name, config):
        """Start new run in custom system."""
        pass

    def log_metrics(self, metrics, step):
        """Log metrics to custom system."""
        pass

    def log_model(self, model, path):
        """Log model to custom system."""
        pass

    # Implement other required methods...

# Use custom backend
tracker = ExperimentTracker(backend=CustomBackend())
```

## Integration Examples

### Complete Training Workflow

```python
import jax
from opifex.mlops import (
    ExperimentTracker,
    ExperimentConfig,
    Framework,
    PhysicsDomain,
    NeuralOperatorMetrics
)
from opifex.neural.operators.fno import FNO
from opifex.training import BasicTrainer
from opifex.data.loaders import create_darcy_loader

# Initialize experiment tracker
tracker = ExperimentTracker(
    backend='mlflow',
    experiment_name='darcy-flow-benchmark',
    tracking_uri='./mlruns'
)

# Configure experiment
config = ExperimentConfig(
    framework=Framework.JAX,
    domain=PhysicsDomain.NEURAL_OPERATORS,
    model_type='FNO',
    learning_rate=1e-3,
    batch_size=32,
    num_epochs=100,
    optimizer='adam',
    seed=42
)

# Start run
run = tracker.start_run(
    run_name='fno-modes12-width64',
    config=config,
    tags={
        'dataset': 'darcy-flow',
        'resolution': '64x64',
        'experiment_type': 'baseline'
    }
)

try:
    # Create data loader and model
    train_loader = create_darcy_loader(
        n_samples=1000,
        batch_size=config.batch_size,
        resolution=64,
        seed=config.seed,
    )
    model = FNO(modes=12, width=64, depth=4)

    # Train with logging
    trainer = BasicTrainer(model, TrainingConfig(
        num_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
    ))

    for epoch in range(config.num_epochs):
        # Training step
        for batch in train_loader:
            train_loss = trainer.train_step(batch)

        # Validation
        val_loss, val_predictions = trainer.validate()

        # Compute physics-specific metrics
        operator_metrics = NeuralOperatorMetrics(
            operator_error=val_loss,
            pointwise_error=compute_pointwise_error(val_predictions),
            conservation_error=compute_conservation_error(val_predictions),
            relative_l2=compute_relative_l2(val_predictions)
        )

        # Log all metrics
        run.log_metrics({
            'train/loss': train_loss,
            'val/loss': val_loss
        }, step=epoch)

        run.log_physics_metrics(operator_metrics, step=epoch)

        # Log learning rate schedule
        run.log_metrics({
            'train/learning_rate': trainer.current_lr
        }, step=epoch)

    # Log final model
    run.log_model(
        model,
        artifact_path='final_model',
        metadata={
            'final_val_loss': val_loss,
            'final_operator_error': operator_metrics.operator_error
        }
    )

    # Log training curve plot
    fig = plot_training_curves(trainer.history)
    fig.savefig('training_curves.png')
    run.log_artifact(
        'training_curves.png',
        artifact_type='plot',
        description='Training and validation curves'
    )

    run.end_run(status='FINISHED')

except Exception as e:
    print(f"Training failed: {e}")
    run.log_param('error_message', str(e))
    run.end_run(status='FAILED')
    raise
```

### Hyperparameter Sweeps

```python
from itertools import product

# Define hyperparameter grid
param_grid = {
    'modes': [8, 12, 16],
    'width': [32, 64, 128],
    'learning_rate': [1e-4, 1e-3, 1e-2]
}

# Run grid search
for modes, width, lr in product(*param_grid.values()):
    config = ExperimentConfig(
        framework=Framework.JAX,
        domain=PhysicsDomain.NEURAL_OPERATORS,
        model_type='FNO',
        learning_rate=lr,
        batch_size=32,
        num_epochs=50
    )

    run = tracker.start_run(
        run_name=f'fno-m{modes}-w{width}-lr{lr}',
        config=config,
        tags={'sweep': 'grid-search-v1'}
    )

    # Train and log...
    model = FNO(modes=modes, width=width)
    # ... training code ...

    run.end_run()

# Query best run
best_run = tracker.get_best_run(
    metric='val/operator_error',
    minimize=True
)
print(f"Best config: {best_run.config}")
```

### Model Comparison

```python
# Compare multiple architectures
architectures = ['FNO', 'DeepONet', 'U-Net']

for arch_name in architectures:
    run = tracker.start_run(
        run_name=f'{arch_name}-baseline',
        tags={'comparison': 'architecture-study'}
    )

    model = create_model(arch_name)  # Your model factory
    # ... training ...

    # Log architecture-specific metrics
    run.log_metrics({
        'model/num_parameters': count_parameters(model),
        'model/memory_mb': estimate_memory(model),
        'model/inference_time_ms': benchmark_inference(model)
    })

    run.end_run()

# Analyze results
comparison_df = tracker.compare_runs(
    tags={'comparison': 'architecture-study'},
    metrics=['val/loss', 'model/num_parameters', 'model/inference_time_ms']
)
print(comparison_df)
```

## Advanced Features

### Auto-logging

Automatic logging of framework-specific information.

```python
# Enable auto-logging
tracker = ExperimentTracker(
    backend='mlflow',
    auto_log=True  # Automatically log system metrics, git info, etc.
)

# With auto-log enabled:
# - Git commit hash
# - System metrics (CPU, memory, GPU)
# - Environment info (Python version, package versions)
# - Training time
# All logged automatically
```

### Nested Runs

Organize related experiments hierarchically.

```python
# Parent run for entire experiment
with tracker.start_run('multi-task-experiment') as parent_run:

    for task in ['task1', 'task2', 'task3']:
        # Child run for each task
        with tracker.start_run(f'{task}-training', parent=parent_run) as run:
            # Train on specific task
            model = train_task(task)
            run.log_model(model)
```

## See Also

- [Platform API](platform.md): Model registry and versioning
- [Training API](training.md): Training infrastructure
- [Deployment API](deployment.md): Model serving
- [Benchmarking API](benchmarking.md): Performance benchmarking
