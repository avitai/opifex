# Neural Network API Reference

The `opifex.neural` package provides the building blocks for scientific machine learning models, built on top of Flax NNX.

## Base Architectures

### Standard MLP

::: opifex.neural.base.StandardMLP
    options:
        show_root_heading: true
        show_source: true

### Quantum MLP {: #quantum-networks }

::: opifex.neural.base.QuantumMLP
    options:
        show_root_heading: true
        show_source: true

## Neural Quantum {: #opifex.neural.quantum }

::: opifex.neural.quantum
    options:
        show_root_heading: true
        show_source: false

## Neural Operators

::: opifex.neural.operators
    options:
        show_root_heading: true
        show_source: false

## Bayesian Networks {: #bayesian-networks }

::: opifex.neural.bayesian
    options:
        show_root_heading: true
        show_source: false

## Domain Decomposition PINNs {: #domain-decomposition }

Domain decomposition methods for physics-informed neural networks, enabling efficient training on complex geometries.

### Base Classes

::: opifex.neural.pinns.domain_decomposition.base
    options:
        show_root_heading: true
        show_source: false
        members:
            - Subdomain
            - Interface
            - DomainDecompositionPINN
            - SubdomainNetwork
            - uniform_partition

### XPINN (Extended PINN)

::: opifex.neural.pinns.domain_decomposition.xpinn
    options:
        show_root_heading: true
        show_source: false
        members:
            - XPINN
            - XPINNConfig

### FBPINN (Finite Basis PINN)

::: opifex.neural.pinns.domain_decomposition.fbpinn
    options:
        show_root_heading: true
        show_source: false
        members:
            - FBPINN
            - FBPINNConfig
            - WindowFunction
            - CosineWindow
            - GaussianWindow

### CPINN (Conservative PINN)

::: opifex.neural.pinns.domain_decomposition.cpinn
    options:
        show_root_heading: true
        show_source: false
        members:
            - CPINN
            - CPINNConfig

### APINN (Augmented PINN)

::: opifex.neural.pinns.domain_decomposition.apinn
    options:
        show_root_heading: true
        show_source: false
        members:
            - APINN
            - APINNConfig
            - GatingNetwork

For usage examples and best practices, see the [Domain Decomposition PINNs Guide](../methods/domain-decomposition-pinns.md).

## Activations

::: opifex.neural.activations
    options:
        show_root_heading: true
        show_source: false
