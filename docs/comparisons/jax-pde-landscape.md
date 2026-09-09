# The JAX PDE learning landscape

Where opifex sits among the libraries that learn solutions and solution operators
of PDEs, restricted to what each project states about itself. Nothing here is a
benchmark: throughput, accuracy and training cost were not measured.

| Library | Framework layer | What it provides | Scope relative to opifex |
| --- | --- | --- | --- |
| **opifex** | JAX, Flax NNX | The registered operator architectures the README lists (`opifex.neural.operators.OPERATOR_REGISTRY`), PINNs with domain decomposition (FBPINN, XPINN, CPINN), uncertainty quantification adapters, atomistic potentials, differentiable DFT and benchmarking | The reference for this table |
| [jNO](https://github.com/FhG-IISB/jNO) (Fraunhofer IISB; arXiv:2605.10159; EPL-2.0; `pip install jax-numerical-operators`) | JAX | A tracing system in which domains, model calls, residuals, supervised losses and diagnostics are one symbolic language compiled into one optimisation pipeline; data-driven and physics-informed training of the architectures its companion library foundax ships (MLPs, transformers, DeepONet, FNO, PROSE) | Overlaps on operator regression and PINN losses; opifex has no symbolic residual language, jNO has no uncertainty, atomistic or DFT layers |
| [jinns](https://gitlab.com/mia_jinns/jinns) (arXiv:2412.14132; Apache-2.0) | JAX, Equinox, Optax | Forward, inverse and meta-model PINNs; separable PINNs, HyperPINNs, PPINNs; second-order optimisers | PINN-only; no neural operators. opifex's PINN family is narrower on PINN variants and wider elsewhere |
| [PINNx](https://github.com/chaobrain/pinnx) (LGPL-2.1) | JAX, brainstate, brainunit, braintools | A rewrite of DeepXDE on JAX with explicit variables and physical units for PINNs | PINN-only with unit checking, which opifex does not have |
| [DeepXDE](https://github.com/lululxvi/deepxde) (LGPL-2.1) | TensorFlow 1 and 2, PyTorch, JAX, PaddlePaddle backends | PINNs, fPINNs, NN-aPC, DeepONet, MIONet, Fourier-DeepONet, physics-informed DeepONet, multifidelity networks | The broadest PINN toolbox; JAX is one backend among five, while opifex is JAX-only and builds on Flax NNX modules |
| [NeuralPDE.jl](https://github.com/SciML/NeuralPDE.jl) (SciML) | Julia | PINN solvers for PDEs written as ModelingToolkit symbolic systems, plus neural SDE methods | A different language and ecosystem; the comparison point is symbolic PDE specification, which opifex does not offer |

Two families of the wider landscape are deliberately outside this table:
[neuraloperator](https://github.com/neuraloperator/neuraloperator), the reference
PyTorch library whose architectures several opifex operators follow, and the
Equinox-based [neojax](https://github.com/paulgekeler/neojax) (FNO, Tucker FNO,
DeepONet), both of which lack the physics-informed training the other rows share.

The operator catalogue in the first row is measured, not asserted:
`scripts/derive_status.py --check` renders the README bullet from the registry and
fails the quality checks when the two disagree.
