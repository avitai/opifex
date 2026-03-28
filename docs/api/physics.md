# Physics API Reference

The `opifex.physics` package provides JAX-native physics solvers and numerical methods for scientific computing applications.

## Overview

The physics module offers:

- **PDE Solvers**: Numerical solvers for common PDEs (Burgers, diffusion-advection, shallow water)
- **Spectral Methods**: Fourier-based PDE solvers and analysis tools
- **Numerical Schemes**: Finite difference, finite element, spectral methods
- **Conservation Laws**: Tools for enforcing physical constraints
- **Quantum Spectral**: Quantum chemistry spectral solvers

## PDE Solvers

### Burgers 2D Solver

::: opifex.physics.solvers.burgers.Burgers2DSolver
    options:
        show_root_heading: true
        show_source: false

### Diffusion-Advection Solver

::: opifex.physics.solvers.diffusion_advection.solve_diffusion_advection_2d
    options:
        show_root_heading: true
        show_source: false

### Navier-Stokes Solver

::: opifex.physics.solvers.navier_stokes.solve_navier_stokes_2d
    options:
        show_root_heading: true
        show_source: false

#### Initial Condition Factories

::: opifex.physics.solvers.navier_stokes.create_taylor_green_vortex
    options:
        show_root_heading: true
        show_source: false

::: opifex.physics.solvers.navier_stokes.create_lid_driven_cavity_ic
    options:
        show_root_heading: true
        show_source: false

::: opifex.physics.solvers.navier_stokes.create_double_shear_layer
    options:
        show_root_heading: true
        show_source: false

### Shallow Water Equations Solver

::: opifex.physics.solvers.shallow_water.solve_shallow_water_2d
    options:
        show_root_heading: true
        show_source: false

## Conservation Laws

`ConservationLaw` is an `Enum` in `opifex.core.physics.conservation` that defines all supported conservation law types.

::: opifex.core.physics.conservation.ConservationLaw
    options:
        show_root_heading: true
        show_source: false

## Neural Tangent Kernel (NTK) Analysis {: #ntk }

Tools for spectral analysis and training diagnostics via the Neural Tangent Kernel.

### NTK Wrapper

::: opifex.core.physics.ntk.wrapper
    options:
        show_root_heading: true
        show_source: false
        members:
            - NTKWrapper
            - NTKConfig

### Spectral Analysis

::: opifex.core.physics.ntk.spectral_analysis
    options:
        show_root_heading: true
        show_source: false
        members:
            - NTKSpectralAnalyzer
            - compute_effective_rank
            - estimate_convergence_rate
            - estimate_epochs_to_convergence
            - identify_slow_modes
            - detect_spectral_bias
            - compute_mode_convergence_rates

### Training Diagnostics

::: opifex.core.physics.ntk.diagnostics
    options:
        show_root_heading: true
        show_source: false
        members:
            - NTKDiagnostics
            - NTKDiagnosticsCallback

For detailed usage and theoretical background, see the [NTK Analysis Guide](../methods/ntk-analysis.md).

## GradNorm Loss Balancing {: #gradnorm }

Multi-task loss balancing through gradient magnitude normalization.

::: opifex.core.physics.gradnorm
    options:
        show_root_heading: true
        show_source: false
        members:
            - GradNormBalancer
            - GradNormConfig
            - compute_gradient_norms
            - compute_inverse_training_rates

For algorithm details and best practices, see the [GradNorm Guide](../methods/gradnorm.md).

## See Also

- [Core API](core.md): Problem definition and boundary conditions
- [Neural API](neural.md): Physics-informed neural networks
- [Data API](data.md): PDE datasets
- [Visualization API](visualization.md): Solution visualization
- [NTK Analysis Guide](../methods/ntk-analysis.md): Detailed NTK usage
- [GradNorm Guide](../methods/gradnorm.md): Multi-task loss balancing
