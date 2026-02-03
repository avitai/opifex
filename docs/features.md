# Features

Opifex provides extensive support for modern scientific machine learning paradigms, offering research-grade implementations designed for experimentation and development.

## 🧪 Supported Opifex Paradigms

### Neural Operators

**Discrete-Continuous Architectures**:

- **Discrete-Continuous (DISCO) Convolutions**: Continuous kernel convolutions for structured/unstructured grids
- **Grid Embeddings**: Coordinate injection and positional encoding for enhanced spatial awareness

**Core Architectures**:

- **Fourier Neural Operators (FNO)**: Spectral convolution for operator learning
- **Deep Operator Networks (DeepONet)**: Branch-trunk architecture for function-to-function mapping
- **Graph Neural Operators**: Message passing for irregular domains and unstructured meshes

**FNO Variants**:

- **Tensorized FNO (TFNO)**: Memory-efficient tensor decomposition (10-20x compression)
- **U-Fourier Neural Operator (U-FNO)**: Multi-scale encoder-decoder architecture
- **Spherical FNO (SFNO)**: Global climate and planetary science applications
- **Local FNO**: Hybrid global-local processing for wave propagation
- **Amortized FNO (AM-FNO)**: High-frequency problems with neural kernel networks

**Specialized Operators**:

- **Geometry-Informed Neural Operator (GINO)**: Complex geometries and CAD domains
- **Multipole Graph Neural Operator (MGNO)**: Molecular dynamics and particle systems
- **Uncertainty Quantification Neural Operator (UQNO)**: Applications requiring uncertainty estimates

**Classical Architectures**:

- **Multi-Scale Fourier Neural Operators (MS-FNO)**: Hierarchical resolution handling for multi-scale physics
- **Latent Neural Operators (LNO)**: Attention-based compression with learnable latent representations
- **Wavelet Neural Operators (WNO)**: Multi-scale wavelet decomposition for time-frequency localization
- **Transform-Based Layers**: Spectral convolution with FFT integration and factorization

### Physics-Informed Neural Networks

- **Standard PINNs**: Physics-constrained neural networks
- **Variants**: XPINNs, VPINNs, cPINNs, Fourier PINNs
- **Multi-Physics Composition**: Hierarchical loss composition with adaptive weighting
- **Conservation Laws**: Mass, momentum, energy, and quantum conservation enforcement

### Neural Density Functional Theory (Neural DFT)

- **Neural Exchange-Correlation Functionals**: DM21-style equivariant functionals
- **ML-Accelerated SCF Methods**: Neural convergence acceleration
- **Hybrid Classical-Neural DFT**: Multi-fidelity quantum mechanical models
- **Chemical Accuracy**: Sub-kcal/mol precision for molecular energies

### Training Infrastructure

- **ModularTrainer**: Component-based training architecture with pluggable components for flexible composition
- **BasicTrainer**: Training framework with physics-informed capabilities and PINN integration
- **ErrorRecoveryManager**: Robust error handling with gradient stability, NaN detection, and loss explosion recovery
- **FlexibleOptimizerFactory**: Advanced optimizer creation (Adam, AdamW, SGD) with cosine, exponential, and linear scheduling
- **AdvancedMetricsCollector**: Physics-aware metrics with convergence tracking, chemical accuracy monitoring, and SCF diagnostics
- **TrainingComponentBase**: Base class for extensible training component development
- **TrainingConfig**: Configuration management for quantum-aware training, loss configuration, and checkpointing
- **TrainingState**: Enhanced state management with physics metrics, conservation violations, and recovery tracking
- **TrainingMetrics**: Extensive metrics tracking including physics losses, chemical accuracy, and SCF convergence

### Optimization

- **Learn-to-Optimize (L2O)**: Neural meta-learning framework with 158/158 tests passing
  - **Parametric Programming Solver**: Neural optimization with constraint handling
  - **L2O Engine**: Unified meta-optimization with problem encoding
  - **Meta-Learning**: MAML, Reptile, and gradient-based algorithms for few-shot adaptation
  - **Multi-Objective Optimization**: Pareto frontier approximation with learned scalarization
  - **Reinforcement Learning**: DQN-based optimization strategy selection with experience replay
- **Adaptive Learning Rates**: Performance-aware scheduling with convergence monitoring
- **Meta-Optimizers**: Learned optimization strategies with 100x+ potential speedup
- **Performance Monitoring**: Thorough tracking and analytics with quality indicators

### Benchmarking System

- **Domain-Specific Benchmarking**: 6 specialized components with physics-aware validation
  - **BenchmarkRegistry**: Configuration management with domain-specific settings
  - **ValidationFramework**: Chemical accuracy assessment and conservation law validation
  - **AnalysisEngine**: Multi-operator comparison with statistical significance testing
  - **ResultsManager**: Publication-ready output with LaTeX/HTML table generation
  - **BenchmarkRunner**: End-to-end workflow orchestration with component integration
- **Statistical Analysis**: Bootstrap confidence intervals and permutation significance testing
- **Publication Pipeline**: Automated generation of publication-ready figures and tables
- **Chemical Accuracy Validation**: <1 kcal/mol energy accuracy for quantum chemistry applications

### MLOps Integration

- **Multi-Backend Experiment Tracking**: MLflow, Weights & Biases, Neptune, and custom Opifex backend
- **Physics-Informed Metadata**: Domain-specific tracking for scientific computing applications
  - **Neural Operator Metrics**: Spectral accuracy, physics compliance, and conservation error tracking
  - **L2O Metrics**: Meta-learning performance, adaptation loss, and generalization metrics
  - **Neural DFT Metrics**: Chemical accuracy, SCF convergence, and density optimization tracking
  - **PINN Metrics**: Physics loss components, boundary condition compliance, and solution accuracy
  - **Quantum Metrics**: State fidelity, circuit depth, and quantum advantage measurements
- **Authentication Support**: Keycloak authentication with role-based access control
- **Deployment Infrastructure**: Kubernetes-native MLOps infrastructure for scalable experiments
- **Unified API**: Vendor-independent experiment tracking with comparative analysis capabilities

### Probabilistic Numerics

- **Uncertainty Quantification**: Multi-source uncertainty aggregation with adaptive weighting strategies
- **Epistemic Uncertainty**: Ensemble disagreement methods and predictive diversity computation
- **Aleatoric Uncertainty**: Distributional uncertainty for Gaussian, Laplace, and mixture distributions
- **Calibration Framework**: Physics-aware temperature scaling with constraint enforcement
- **Physics-Aware Constraints**: Energy conservation, mass conservation, positivity, and boundedness enforcement
- **Physics-Informed Priors**: Conservation law constraints and boundary condition enforcement
- **Domain-Specific Priors**: Quantum chemistry, fluid dynamics, and materials science parameter distributions
- **Hierarchical Bayesian Framework**: Multi-level uncertainty modeling with adaptive propagation
- **Physics-Aware Uncertainty Propagation**: Constraint-preserving uncertainty propagation
- **Uncertainty Quality Assessment**: Coverage probability, calibration metrics, and reliability estimation
- **Bayesian Inference**: Parameter estimation with BlackJAX MCMC integration
- **Multi-fidelity Methods**: Combining different model accuracies with uncertainty propagation
- **Robust Optimization**: Optimization under uncertainty with calibrated confidence intervals

## Learn More

- [Neural Operators Tutorial](methods/neural-operators.md) - Detailed guide to neural operator implementations
- [Physics-Informed Networks](methods/pinns.md) - PINNs documentation and examples
- [Neural DFT Guide](methods/neural-dft.md) - Quantum chemistry with neural networks
- [L2O Framework](methods/l2o.md) - Learn-to-optimize meta-learning
- [API Reference](api/core.md) - Technical documentation
