# Technology Stack

Opifex is built on a carefully curated technology stack that provides high performance, reliability, and modern development practices for scientific machine learning applications.

## 🛠️ Core Technologies

### Core JAX Ecosystem

The foundation of Opifex is built on the JAX ecosystem, providing high-performance numerical computing with automatic differentiation and GPU acceleration.

- **JAX 0.11.1+**: Core framework with CUDA support

    - Automatic differentiation for gradient-based optimization
    - Just-in-time (JIT) compilation for performance
    - Vectorization and parallelization support
    - GPU and TPU acceleration

- **Flax 0.12.10+**: NNX neural network framework (exclusive)
    - Stateful neural network transformations
    - Modular and composable neural network components
    - Integration with JAX transformations
    - Type-safe parameter handling

- **Optax 0.2.8+**: Optimization algorithms and line searches

    - Gradient-based and quasi-Newton optimization algorithms
    - Zoom and backtracking line searches
    - Gradient clipping and normalization
    - Composable optimization transformations

- **BlackJAX 1.2.5+**: MCMC sampling

    - Hamiltonian Monte Carlo (HMC)
    - No-U-Turn Sampler (NUTS)
    - Metropolis-Hastings algorithms
    - Bayesian inference support

- **Calibrax 0.1.23+ & Substrax 0.1.37+**: Calibration, statistics, profiling, and NNX foundations

    - Exact conformal order statistics, survival estimators, and scaling analysis
    - Compilation and execution profiling harnesses
    - Flax NNX RNG, module conformance, and directional gradient verification

### Quantum Chemistry Stack

Specialized components for quantum mechanical calculations and molecular systems.

- **Neural DFT**: Chemical accuracy (<1 kcal/mol) quantum calculations

    - Neural exchange-correlation functionals
    - Self-consistent field (SCF) acceleration
    - Density functional theory implementations
    - Chemical accuracy validation

- **Molecular Systems**: 3D geometry with periodic boundary conditions

    - Molecular structure representation
    - Periodic boundary condition handling
    - Symmetry operations and point groups
    - Force field integration

- **Electronic Structure**: Quantum mechanical problem definitions

    - Wavefunction representations
    - Basis set management
    - Quantum mechanical operators
    - Many-body theory support

- **Physics Constraints**: Conservation laws and quantum mechanical principles

    - Particle number conservation
    - Energy conservation
    - Symmetry enforcement
    - Quantum mechanical constraints

### Advanced Training Stack

Infrastructure for physics-aware training and meta-optimization.

- **Physics-Informed Losses**: Multi-physics composition with adaptive weighting

    - Hierarchical loss composition
    - Adaptive weight scheduling
    - Conservation law enforcement
    - Multi-physics problem support

- **Conservation Law Enforcement**: Mass, momentum, energy, quantum conservation

    - Automatic conservation law detection
    - Soft and hard constraint enforcement
    - Physics-aware regularization
    - Constraint satisfaction monitoring

- **Meta-Optimization**: L2O algorithms with neural meta-learning

    - Learn-to-optimize (L2O) framework
    - Meta-learning algorithms (MAML, Reptile)
    - Multi-objective optimization
    - Reinforcement learning for optimization

- **Adaptive Scheduling**: Performance-based learning rate and weight adaptation

    - Performance monitoring
    - Adaptive learning rate scheduling
    - Dynamic weight adjustment
    - Convergence detection

- **Quantum-Aware Training**: SCF acceleration and quantum constraint handling

    - SCF convergence acceleration
    - Quantum constraint enforcement
    - Chemical accuracy monitoring
    - Quantum mechanical validation

### Development Infrastructure

Modern development tools ensuring code quality, testing, and documentation.

- **uv**: Package management (exclusive)
    - Fast Python package installation
    - Dependency resolution
    - Virtual environment management
    - Lock file generation

- **ruff + pyright**: Code quality (exclusive)
    - Fast Python linting with ruff
    - Type checking with pyright
    - Code formatting and style enforcement
    - Import sorting and organization

- **pytest**: Testing framework
    - Parametrized testing
    - Fixture management
    - Coverage reporting

- **MkDocs**: Documentation system
    - Markdown-based documentation
    - Material theme for modern UI
    - Mathematical notation support
    - API documentation generation

## 🔧 Supporting Libraries

### Numerical Computing

- **SciPy 1.10+**: Reference special functions and sparse utilities
- **Distrax 0.1.0+**: Probabilistic distributions
- **Orbax 0.11.13+**: Checkpointing system

### Data Management

- **NumPy**: Numerical array operations
- **SQLAlchemy**: Database integration with type safety (optional, in `platform` extras)
- **HDF5** (h5py): Large-scale data storage (optional, in `neural-dft`/`scientific-data` extras)
- **Pandas**: Data manipulation and analysis (optional, in `pdebench` extras)

### Visualization

- **Matplotlib**: Scientific plotting
- **Plotly**: Interactive visualizations

### Security & Quality

- **Bandit**: Security analysis
- **pydocstyle**: Documentation standards (via pre-commit hook, not a direct dependency)
- **pre-commit**: Git hook management
- **pyright**: Static type checking

## 🚀 Performance Characteristics

### Computational Performance

- **GPU Acceleration**: Native CUDA support through JAX
- **JIT Compilation**: Automatic optimization of computational graphs
- **Vectorization**: SIMD operations for array computations
- **Memory Efficiency**: Optimized memory usage patterns

### Scalability

- **Distributed Computing**: Multi-GPU and multi-node support
- **Batch Processing**: Efficient batch operations
- **Streaming**: Large dataset processing
- **Cloud Integration**: Kubernetes-native deployment

### Quality Metrics

- **Test Coverage**: Extensive test coverage
- **Type Safety**: Full type annotation coverage
- **Documentation**: Full API documentation
- **Security**: Zero known vulnerabilities

## 🔄 Version Management

### Dependency Pinning

All dependencies are pinned to specific versions to ensure reproducibility and stability across different environments.

### Compatibility Matrix

| Component | Version | Python | CUDA | Notes |
|-----------|---------|--------|------|-------|
| JAX | 0.11.1+ | 3.12+ | 12.0+ | Core framework |
| Flax | 0.12.10+ | 3.12+ | - | Neural networks (NNX) |
| Optax | 0.2.8+ | 3.12+ | - | Optimization & line search |
| BlackJAX | 1.2.5+ | 3.12+ | - | MCMC sampling |
| Calibrax | 0.1.23+ | 3.12+ | - | Metrics, statistics & profiling |
| Substrax | 0.1.37+ | 3.12+ | - | NNX & testing utilities |

### Update Policy

- **Major versions**: Carefully evaluated for breaking changes
- **Minor versions**: Regular updates for new features
- **Patch versions**: Automatic updates for bug fixes
- **Security updates**: Immediate updates for security patches

## Learn More

- [Installation Guide](getting-started/installation.md) - Setup instructions
- [Development Setup](development/gpu-development.md) - Development environment
- [Architecture Overview](architecture.md) - System architecture
- [Performance Benchmarks](benchmarks.md) - Performance characteristics
