"""Cross-validate the spectral solver against the published exponax library.

``exponax`` (Felix Koehler, https://github.com/Ceyron/exponax) is an independent
JAX implementation of exponential-time-differencing spectral PDE solvers. Running
our :func:`solve_burgers_spectral` and exponax's Burgers stepper on an identical
configuration must agree to machine precision -- a reference-grounded check that
the ETDRK4 core, the convection nonlinearity, the dealiasing, and the Fourier
conventions are all correct. Requires double precision (see the package-level
``conftest`` fixture) and the optional ``exponax`` test dependency.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from opifex.physics.spectral.steppers import solve_burgers_spectral


exponax = pytest.importorskip("exponax")


def test_burgers_matches_exponax_etdrk4() -> None:
    """``solve_burgers_spectral`` matches exponax's Burgers ETDRK4 stepper.

    Same domain, resolution, time step, viscosity and convection scaling; exponax
    is configured single-channel (scalar 1D Burgers) at ETDRK order 4.
    """
    num_points, domain, viscosity, dt, num_steps = 128, 1.0, 0.05, 1e-3, 200
    x = np.linspace(0.0, domain, num_points, endpoint=False)
    ic = jnp.asarray(np.sin(2 * np.pi * x) + 0.3 * np.cos(4 * np.pi * x))

    ours = solve_burgers_spectral(
        ic,
        viscosity,
        domain_extent=domain,
        time_final=dt * num_steps,
        num_steps=num_steps,
        num_snapshots=1,
    )[-1]

    stepper = exponax.stepper.Burgers(
        1,
        domain,
        num_points,
        dt,
        diffusivity=viscosity,
        convection_scale=1.0,
        single_channel=True,
        order=4,
    )
    reference = exponax.rollout(stepper, num_steps)(ic[None])[-1, 0]

    assert float(jnp.max(jnp.abs(ours - reference))) < 1e-12
