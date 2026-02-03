# Examples

Every example listed here runs successfully and produces the documented output.
Start with **Getting Started** if you're new, or jump to the category that matches your problem.

## Prerequisites

```bash
source ./activate.sh
python -c "import opifex; print('Opifex imported successfully')"
```

---

## Getting Started

New to Opifex? Start here with minimal, self-contained examples.

| Example | Time | Description |
|---------|------|-------------|
| [Your First Neural Operator](getting-started/first-neural-operator.md) | 5 min | Train an FNO on Darcy flow in ~50 lines |
| [Your First PINN](getting-started/first-pinn.md) | 5 min | Solve a Poisson equation with PINNs in ~40 lines |

---

## Neural Operators

Data-driven operator learning: map input functions to output functions.

| Example | Architecture | Dataset | Description |
|---------|-------------|---------|-------------|
| [FNO on Darcy Flow](neural-operators/fno-darcy.md) | FNO + GridEmbedding2D | Darcy Flow 64x64 | Standard FNO benchmark |
| [FNO on Burgers](neural-operators/fno-burgers.md) | FNO | Burgers 1D | Time-dependent PDE |
| [SFNO Climate (Simple)](neural-operators/sfno-climate-simple.md) | Spherical FNO | Shallow Water 32x32 | Quick start for spherical data |
| [SFNO Climate (Comprehensive)](neural-operators/sfno-climate-comprehensive.md) | Spherical FNO + Conservation | Shallow Water 64x64 | Conservation-aware training |
| [U-FNO on Turbulence](neural-operators/ufno-turbulence.md) | U-FNO + Energy Loss | 2D Burgers 64x64 | Multi-scale architecture |
| [UNO on Darcy Flow](neural-operators/uno-darcy.md) | UNO + Super-Resolution | Darcy Flow 32x32 | Zero-shot super-resolution |
| [TFNO on Darcy Flow](neural-operators/tfno-darcy.md) | Tensorized FNO | Darcy Flow 64x64 | Memory-efficient decomposition |
| [Local FNO on Darcy](neural-operators/local-fno-darcy.md) | Local FNO | Darcy Flow 64x64 | Local + global frequency mixing |
| [GNO on Darcy Flow](neural-operators/gno-darcy.md) | Graph Neural Operator | Darcy Flow 32x32 | Irregular mesh support |
| [DeepONet on Darcy](neural-operators/deeponet-darcy.md) | DeepONet | Darcy Flow 64x64 | Branch-trunk architecture |
| [DeepONet Antiderivative](neural-operators/deeponet-antiderivative.md) | DeepONet | Antiderivative | Classic DeepONet benchmark |
| [PINO on Burgers](neural-operators/pino-burgers.md) | Physics-Informed NO | Burgers 1D | Hybrid data + physics loss |
| [Operator Comparison Tour](neural-operators/operator-tour.md) | All architectures | Multiple | Overview of all 26 operators |

```bash
python examples/neural-operators/fno_darcy.py
```

---

## PINNs

Solve PDEs from governing equations using physics-informed neural networks.

| Example | PDE Type | Description |
|---------|----------|-------------|
| [Poisson Equation](pinns/poisson.md) | Elliptic | Classic Laplace equation benchmark |
| [Heat Equation](pinns/heat-equation.md) | Parabolic | 2D diffusion with time evolution |
| [Burgers Equation](pinns/burgers.md) | Nonlinear | Shock formation and viscosity |
| [Wave Equation](pinns/wave.md) | Hyperbolic | 1D wave propagation |
| [Helmholtz Equation](pinns/helmholtz.md) | Oscillatory | Frequency-domain wave equation |
| [Advection Equation](pinns/advection.md) | Hyperbolic | Transport phenomena |
| [Allen-Cahn Equation](pinns/allen-cahn.md) | Reaction-Diffusion | Phase-field dynamics |
| [Diffusion-Reaction](pinns/diffusion-reaction.md) | Coupled | Multi-physics systems |
| [Navier-Stokes](pinns/navier-stokes.md) | Fluid Dynamics | Kovasznay flow benchmark |
| [Euler Beam](pinns/euler-beam.md) | Structural | 4th-order ODE for beam deflection |
| [Inverse Diffusion](pinns/inverse-diffusion.md) | Inverse Problem | Parameter discovery from data |

```bash
python examples/pinns/poisson.py
```

---

## Domain Decomposition

Parallel and decomposed PINNs for large-scale problems.

| Example | Method | Description |
|---------|--------|-------------|
| [FBPINN on Poisson](domain-decomposition/fbpinn-poisson.md) | Finite Basis PINN | Overlapping subdomains with window functions |
| [XPINN on Helmholtz](domain-decomposition/xpinn-helmholtz.md) | Extended PINN | Non-overlapping domain decomposition |
| [CPINN on Advection-Diffusion](domain-decomposition/cpinn-advection-diffusion.md) | Conservative PINN | Flux conservation at interfaces |

```bash
python examples/domain-decomposition/fbpinn_poisson.py
```

---

## Advanced Training

Techniques for improving PINN training dynamics and convergence.

| Example | Technique | Description |
|---------|-----------|-------------|
| [NTK Analysis](advanced-training/ntk-analysis.md) | Neural Tangent Kernel | Eigenvalue spectrum and spectral bias detection |
| [GradNorm](advanced-training/gradnorm.md) | Adaptive Loss Balancing | Automatic loss weight adjustment |
| [Adaptive Sampling](advanced-training/adaptive-sampling.md) | RAR-D Refinement | Residual-based collocation point refinement |

```bash
python examples/advanced-training/ntk_analysis.py
```

---

## Optimization

Learn-to-optimize and meta-learning for PDE solving.

| Example | Method | Description |
|---------|--------|-------------|
| [Learn to Optimize](optimization/learn-to-optimize.md) | L2O | Learned optimizers for parametric PDEs |
| [Meta-Optimization](optimization/meta-optimization.md) | MAML/Reptile | Fast adaptation across PDE families |

```bash
python examples/optimization/learn_to_optimize.py
```

---

## Uncertainty Quantification

Calibration, Bayesian inference, and uncertainty-aware operators.

| Example | Method | Description |
|---------|--------|-------------|
| [Calibration Methods](uncertainty/calibration.md) | Platt, Isotonic, Conformal | Post-hoc calibration and conformal prediction |
| [UQNO on Darcy](uncertainty/uqno-darcy.md) | Bayesian Spectral Conv | Uncertainty-aware neural operator |
| [Bayesian FNO](uncertainty/bayesian-fno.md) | Variational Framework | Amortized variational inference |

```bash
python examples/uncertainty/calibration.py
```

---

## Quantum Chemistry

Neural approaches to density functional theory.

| Example | Method | Description |
|---------|--------|-------------|
| [Neural DFT](quantum-chemistry/neural-dft.md) | Neural SCF Solver | H2 molecule ground state |
| [Neural XC Functional](quantum-chemistry/neural-xc-functional.md) | Learned Exchange-Correlation | Training on LDA reference |

```bash
python examples/quantum-chemistry/neural_dft.py
```

---

## Layers & Components

Individual neural operator building blocks demonstrated in isolation.

| Example | Component | Description |
|---------|-----------|-------------|
| [DISCO Convolutions](layers/disco-convolutions.md) | `DiscoConv2D` | Discrete-continuous convolutions with 6x+ speedup |
| [Grid Embeddings](layers/grid-embeddings.md) | `GridEmbedding2D` | Coordinate injection and positional encoding |
| [Fourier Continuation](layers/fourier-continuation.md) | `FourierContinuation` | Boundary handling for non-periodic domains |
| [Spectral Normalization](layers/spectral-normalization.md) | `SpectralNormalization` | Training stability for deep operators |

```bash
python examples/layers/disco_convolutions_example.py
```

---

## Data & Analysis

Explore and validate the synthetic datasets used by neural operator examples.

| Example | Focus |
|---------|-------|
| [Darcy Flow Analysis](data/darcy-flow-analysis.md) | FNO prediction validation, error analysis |
| [Spectral Analysis](data/spectral-analysis.md) | Power spectrum and mode analysis |

```bash
python examples/data/darcy_flow_analysis.py
```

---

## Benchmarking

Performance comparisons and GPU optimization guides.

| Example | Focus | Runtime |
|---------|-------|---------|
| [Neural Operator Benchmark](benchmarking/operator-benchmark.md) | UNO, FNO, SFNO across resolutions | ~15 min |
| [GPU Profiling & Optimization](benchmarking/gpu-profiling.md) | Memory pools, mixed precision, JIT analysis | ~5 min |

```bash
python examples/benchmarking/operator_benchmark.py
```

---

## Troubleshooting

**Import errors:** Ensure the environment is activated with `source ./activate.sh`.

**GPU availability:** Check with `python -c "import jax; print(jax.default_backend(), jax.devices())"`.

**Memory issues:** Reduce `BATCH_SIZE` or `N_TRAIN` in the configuration section of the example.

## Additional Resources

- [Neural Operators Guide](../methods/neural-operators.md) -- Theory and architecture details
- [PINNs Guide](../methods/pinns.md) -- Physics-informed methods
- [Training Guide](../user-guide/training.md) -- `Trainer` API reference
- [Example Documentation Design](../development/example_documentation_design.md) -- Contributing guidelines
