# Platform API Reference

The `opifex.platform` package provides community platform infrastructure for storing, versioning, and sharing neural functionals and scientific ML models.

## Overview

The platform module offers:

- **Neural Functional Registry**: Store and version neural network weights and architectures
- **Model Search**: Semantic search over registered models
- **Version Control**: Track model evolution and lineage
- **Validation**: Ensure model compatibility and correctness
- **Collaboration** *(planned)*: Team collaboration features
- **Dashboard** *(planned)*: Analytics and monitoring

## Registry Core

### NeuralFunctionalRegistry

Central registry for neural functionals with versioning and metadata.

```python
from opifex.platform.registry import RegistryService

class NeuralFunctionalRegistry:
    """
    Registry for neural functionals with version control and metadata.

    Neural functionals are parameterized neural networks that solve
    specific scientific computing tasks (PDEs, operators, etc.).

    Args:
        storage_path: Path to registry storage directory
        enable_caching: Whether to cache frequently accessed models
        cache_size: Maximum number of cached models

    Attributes:
        models: Dictionary of registered models
        metadata_store: Model metadata database
        version_graph: DAG of model versions
    """

    def __init__(
        self,
        storage_path: str = "~/.opifex/registry",
        enable_caching: bool = True,
        cache_size: int = 100
    ):
        """Initialize registry with storage configuration."""
```

#### Methods

##### `register(model, metadata) -> str`

Register a new neural functional.

```python
def register(
    self,
    model: nnx.Module,
    metadata: ModelMetadata,
    version: Optional[str] = None,
    parent_id: Optional[str] = None
) -> str:
    """
    Register neural functional in the registry.

    Args:
        model: Flax NNX model to register
        metadata: Model metadata (name, description, etc.)
        version: Version string (auto-generated if None)
        parent_id: ID of parent model (for versioning)

    Returns:
        Unique model ID

    Example:
        >>> from opifex.neural.operators.fno import FNO
        >>> model = FNO(modes=12, width=32)
        >>> metadata = ModelMetadata(
        ...     name="darcy-flow-fno",
        ...     description="FNO for Darcy flow prediction",
        ...     task="operator-learning",
        ...     domain="fluid-dynamics",
        ...     tags=["pde", "elliptic", "darcy"]
        ... )
        >>> model_id = registry.register(model, metadata)
        >>> print(f"Registered as: {model_id}")
    """
```

##### `load(model_id, version='latest') -> nnx.Module`

Load registered model from registry.

```python
def load(
    self,
    model_id: str,
    version: str = "latest",
    device: Optional[str] = None
) -> nnx.Module:
    """
    Load model from registry.

    Args:
        model_id: Unique model identifier
        version: Version to load ('latest', specific version string)
        device: Target device ('cpu', 'cuda', 'tpu')

    Returns:
        Loaded Flax NNX model

    Example:
        >>> model = registry.load("darcy-flow-fno", version="latest")
        >>> # Use for inference
        >>> prediction = model(test_input)
    """
```

##### `update(model_id, model, metadata) -> str`

Update existing model with new version.

```python
def update(
    self,
    model_id: str,
    model: nnx.Module,
    metadata: Optional[ModelMetadata] = None,
    version_note: str = ""
) -> str:
    """
    Create new version of existing model.

    Args:
        model_id: ID of model to update
        model: Updated model
        metadata: Updated metadata (None = keep existing)
        version_note: Description of changes

    Returns:
        New version ID

    Example:
        >>> # Fine-tune existing model
        >>> model = registry.load("darcy-flow-fno")
        >>> # ... training ...
        >>> new_version = registry.update(
        ...     "darcy-flow-fno",
        ...     model,
        ...     version_note="Fine-tuned on high-Reynolds data"
        ... )
    """
```

##### `search(query, filters) -> List[str]`

Search for models using semantic query and filters.

```python
def search(
    self,
    query: str = "",
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 10,
    sort_by: str = "relevance"
) -> List[ModelSearchResult]:
    """
    Search registry for relevant models.

    Args:
        query: Free-text search query
        filters: Filter criteria:
            - 'task': Task type (operator-learning, pinn, dft)
            - 'domain': Physics domain (fluid-dynamics, quantum)
            - 'tags': List of required tags
            - 'min_accuracy': Minimum validation accuracy
        limit: Maximum number of results
        sort_by: Sort criterion (relevance, accuracy, date)

    Returns:
        List of search results with model IDs and metadata

    Example:
        >>> # Find fluid dynamics models
        >>> results = registry.search(
        ...     query="fluid flow prediction",
        ...     filters={
        ...         'domain': 'fluid-dynamics',
        ...         'task': 'operator-learning',
        ...         'min_accuracy': 0.95
        ...     },
        ...     limit=5
        ... )
        >>> for result in results:
        ...     print(f"{result.name}: {result.score:.2f}")
    """
```

##### `list_versions(model_id) -> List[str]`

List all versions of a model.

```python
def list_versions(
    self,
    model_id: str,
    include_metadata: bool = True
) -> List[VersionInfo]:
    """
    Get version history for model.

    Args:
        model_id: Model identifier
        include_metadata: Include full metadata for each version

    Returns:
        List of version information

    Example:
        >>> versions = registry.list_versions("darcy-flow-fno")
        >>> for v in versions:
        ...     print(f"{v.version}: {v.timestamp} - {v.note}")
    """
```

##### `delete(model_id, version=None)`

Delete model or specific version.

```python
def delete(
    self,
    model_id: str,
    version: Optional[str] = None,
    cascade: bool = False
) -> None:
    """
    Delete model or version from registry.

    Args:
        model_id: Model to delete
        version: Specific version (None = delete all versions)
        cascade: If True, also delete dependent models

    Example:
        >>> # Delete specific version
        >>> registry.delete("darcy-flow-fno", version="v0.1.0")
        >>>
        >>> # Delete entire model
        >>> registry.delete("old-model", cascade=True)
    """
```

## Model Metadata

### ModelMetadata

Structured metadata for neural functionals.

```python
from opifex.platform.registry import ModelMetadata

@dataclass
class ModelMetadata:
    """
    Metadata for registered neural functionals.

    Attributes:
        name: Human-readable model name
        description: Detailed description
        task: Task type (operator-learning, pinn, neural-dft, etc.)
        domain: Physics domain (fluid-dynamics, quantum, etc.)
        tags: List of descriptive tags
        architecture: Architecture name (FNO, DeepONet, etc.)
        input_shape: Expected input shape
        output_shape: Expected output shape
        performance_metrics: Validation metrics
        training_config: Training configuration used
        created_at: Creation timestamp
        updated_at: Last update timestamp
        author: Model author/organization
        license: Software license
        paper_url: Link to associated paper
        code_url: Link to training code
    """

    name: str
    description: str
    task: str
    domain: str
    tags: List[str] = field(default_factory=list)
    architecture: Optional[str] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    training_config: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    author: Optional[str] = None
    license: str = "MIT"
    paper_url: Optional[str] = None
    code_url: Optional[str] = None
```

### Example Usage

```python
metadata = ModelMetadata(
    name="burgers-fno-large",
    description="Large FNO trained on Burgers equation dataset",
    task="operator-learning",
    domain="fluid-dynamics",
    architecture="FNO",
    tags=["pde", "nonlinear", "burgers", "turbulence"],
    input_shape=(256,),
    output_shape=(100, 256),
    performance_metrics={
        'val_mse': 1.2e-4,
        'val_relative_l2': 0.023,
        'inference_time_ms': 5.2
    },
    training_config={
        'epochs': 500,
        'batch_size': 32,
        'learning_rate': 1e-3,
        'optimizer': 'adam'
    },
    author="Opifex Team",
    license="MIT",
    paper_url="https://arxiv.org/abs/2010.08895"
)
```

## Model Search

### Semantic Search

Advanced search capabilities using embeddings and metadata.

```python
from opifex.platform.registry import SemanticSearch

class SemanticSearch:
    """
    Semantic search over model registry using embeddings.

    Uses neural embeddings of model descriptions and metadata
    to find semantically similar models.

    Args:
        registry: NeuralFunctionalRegistry instance
        embedding_model: Model for computing embeddings
        index_type: Search index type ('faiss', 'annoy', 'nmslib')
    """

    def __init__(
        self,
        registry: NeuralFunctionalRegistry,
        embedding_model: str = "sentence-transformers",
        index_type: str = "faiss"
    ):
        """Initialize semantic search engine."""

    def search(
        self,
        query: str,
        k: int = 10,
        filters: Optional[Dict] = None
    ) -> List[SearchResult]:
        """
        Perform semantic search.

        Args:
            query: Natural language query
            k: Number of results to return
            filters: Optional metadata filters

        Returns:
            Ranked list of search results

        Example:
            >>> search = SemanticSearch(registry)
            >>> results = search.search(
            ...     "neural operator for solving Navier-Stokes equations",
            ...     k=5
            ... )
            >>> for r in results:
            ...     print(f"{r.model_id}: {r.similarity:.3f}")
        """
```

## Model Validation

### ValidationFramework

Ensure model compatibility and correctness.

```python
from opifex.platform.registry import ValidationFramework

class ValidationFramework:
    """
    Validation framework for registered models.

    Validates:
    - Input/output shapes
    - Numerical correctness
    - Performance benchmarks
    - API compatibility
    """

    def validate_model(
        self,
        model: nnx.Module,
        validation_suite: str = "standard"
    ) -> ValidationReport:
        """
        Run validation suite on model.

        Args:
            model: Model to validate
            validation_suite: Validation level:
                - 'basic': Shape and type checks
                - 'standard': + numerical correctness
                - 'comprehensive': + performance benchmarks

        Returns:
            Validation report with pass/fail status

        Example:
            >>> validator = ValidationFramework()
            >>> report = validator.validate_model(
            ...     model,
            ...     validation_suite="comprehensive"
            ... )
            >>> if report.passed:
            ...     print("All validations passed!")
            >>> else:
            ...     print(f"Failed: {report.failures}")
        """
```

## Version Control

### Model Lineage

Track model evolution and relationships.

```python
from opifex.platform.registry import VersionControl

class VersionControl:
    """
    Version control system for neural functionals.

    Tracks model lineage, branching, and merging.
    """

    def get_lineage(
        self,
        model_id: str,
        max_depth: Optional[int] = None
    ) -> nx.DiGraph:
        """
        Get model lineage graph.

        Args:
            model_id: Model to trace
            max_depth: Maximum ancestor depth

        Returns:
            NetworkX directed graph of model lineage

        Example:
            >>> vc = VersionControl(registry)
            >>> lineage = vc.get_lineage("darcy-flow-fno")
            >>> # Visualize lineage
            >>> import matplotlib.pyplot as plt
            >>> nx.draw(lineage, with_labels=True)
        """

    def compare_versions(
        self,
        model_id: str,
        version1: str,
        version2: str
    ) -> VersionDiff:
        """
        Compare two model versions.

        Args:
            model_id: Model identifier
            version1, version2: Versions to compare

        Returns:
            Diff object with changes

        Example:
            >>> diff = vc.compare_versions(
            ...     "darcy-flow-fno",
            ...     "v1.0.0",
            ...     "v2.0.0"
            ... )
            >>> print(f"Parameter changes: {diff.param_changes}")
            >>> print(f"Architecture changes: {diff.arch_changes}")
        """
```

## Integration Examples

### Complete Registry Workflow

```python
import jax
from opifex.platform.registry import (
    NeuralFunctionalRegistry,
    ModelMetadata
)
from opifex.neural.operators.fno import FNO
from opifex.training import BasicTrainer
from opifex.data.loaders import create_darcy_loader

# Initialize registry
registry = NeuralFunctionalRegistry(storage_path="./models")

# Train model
train_loader = create_darcy_loader(
    n_samples=1000,
    batch_size=32,
    resolution=64,
    seed=42,
)
model = FNO(modes=12, width=64, depth=4)

config = TrainingConfig(num_epochs=100, learning_rate=1e-3)
trainer = BasicTrainer(model, config)
trained_model, history = trainer.train(train_loader)

# Register model
metadata = ModelMetadata(
    name="darcy-fno-v1",
    description="FNO for Darcy flow operator learning",
    task="operator-learning",
    domain="fluid-dynamics",
    architecture="FNO",
    tags=["pde", "elliptic", "darcy"],
    performance_metrics={
        'val_loss': history['val_loss'][-1],
        'val_relative_error': 0.015
    }
)

model_id = registry.register(model, metadata)
print(f"Model registered: {model_id}")

# Later: Load and use model
loaded_model = registry.load(model_id)
prediction = loaded_model(test_input)

# Update after fine-tuning
# ... fine-tuning code ...
new_version = registry.update(
    model_id,
    model,
    version_note="Fine-tuned on challenging cases"
)
```

### Team Collaboration

```python
# Team member 1: Register model
registry = NeuralFunctionalRegistry()
model_id = registry.register(my_model, metadata)

# Team member 2: Search and load
results = registry.search(
    query="darcy flow high accuracy",
    filters={'domain': 'fluid-dynamics'}
)
best_model_id = results[0].model_id
model = registry.load(best_model_id)

# Team member 3: Create variant
base_model = registry.load(best_model_id)
# ... modify architecture ...
variant_id = registry.register(
    modified_model,
    variant_metadata,
    parent_id=best_model_id
)
```

## Performance and Scalability

### Caching

```python
# Enable caching for frequently accessed models
registry = NeuralFunctionalRegistry(
    enable_caching=True,
    cache_size=50  # Keep 50 models in memory
)

# Cached load (much faster for repeated access)
model = registry.load("popular-model")  # Cached after first load
```

### Distributed Registry

```python
# Connect to remote registry
from opifex.platform.registry import RemoteRegistry

remote_registry = RemoteRegistry(
    url="https://registry.opifex.io",
    api_key=os.getenv("OPIFEX_API_KEY")
)

# Push local model to remote
local_id = local_registry.register(model, metadata)
remote_id = remote_registry.push(local_id)

# Pull remote model
remote_registry.pull(remote_id, target_path="./models")
```

## See Also

- [MLOps API](mlops.md): Experiment tracking and model lifecycle
- [Deployment API](deployment.md): Model serving and deployment
- [Training API](training.md): Training infrastructure
- [Neural API](neural.md): Neural network architectures
