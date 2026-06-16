"""Vmapped PDE-dataset generation for operator learning.

Clean, batched dataset generators that map a conditioning field to the
**final-time** solution of a PDE (a single output time, not a trajectory).
Every generator returns one uniform contract::

    {
        "input":  (n_samples, C_in,  *spatial)  float32,
        "output": (n_samples, C_out, *spatial)  float32,
    }

in channels-first (FNO) layout, where ``C`` counts the physical fields.

Per-sample randomness uses ``jax.random.PRNGKey(seed + i)`` so a given
``(seed, i)`` always yields the same sample. The four JAX-native PDEs
(Burgers, diffusion-advection, Navier-Stokes, shallow water) generate their
whole batch with a single ``jax.jit(jax.vmap(per_sample_fn))(keys)`` call —
initial-condition synthesis *and* the time integration run vmapped on-device.
Darcy is the exception: its steady solve is a ``scipy`` sparse factorization
(:func:`opifex.physics.solvers.darcy.solve_darcy_flow`) that cannot be vmapped,
so only the coefficient-field generation is vmapped and the solve runs in a
Python loop over samples.

The initial-condition / coefficient-field *physics* (the Gaussian/sine/step
bumps, the thresholded-GRF and smooth log-Fourier permeability, the
Taylor-Green / shear / random-vortex flows, the shallow-water ``h/u/v`` init)
is reused from the former per-sample Grain sources, recast here as pure
functions over a PRNG key.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

from opifex.physics.solvers.darcy import solve_darcy_flow
from opifex.physics.solvers.diffusion_advection import _solve_diffusion_advection_2d_jit
from opifex.physics.solvers.navier_stokes import (
    create_double_shear_layer,
    create_taylor_green_vortex,
    solve_navier_stokes_2d,
)
from opifex.physics.solvers.shallow_water import solve_shallow_water_2d
from opifex.physics.spectral.steppers import solve_burgers_spectral


__all__ = [
    "generate_burgers",
    "generate_darcy",
    "generate_diffusion",
    "generate_navier_stokes",
    "generate_shallow_water",
]


def _per_sample_keys(seed: int, n_samples: int) -> jax.Array:
    """Build the stack of per-sample keys ``PRNGKey(seed + i)`` for ``i < n``."""
    return jax.vmap(jax.random.PRNGKey)(seed + jnp.arange(n_samples))


def _stack_float32(batched: jax.Array) -> np.ndarray:
    """Move a device batch to a contiguous ``float32`` NumPy array."""
    return np.asarray(batched, dtype=np.float32)


def _run_vmapped(
    per_sample_fn: Callable[[jax.Array], jax.Array],
    seed: int,
    n_samples: int,
) -> np.ndarray:
    """Jit+vmap ``per_sample_fn`` over per-sample keys and return float32 NumPy."""
    keys = _per_sample_keys(seed, n_samples)
    batched = jax.jit(jax.vmap(per_sample_fn))(keys)
    return _stack_float32(batched)


# --------------------------------------------------------------------------- #
# Initial-condition / coefficient physics (pure, single-key functions).
# --------------------------------------------------------------------------- #


# Canonical 1D Burgers GRF parameters (Li et al., 2020, ``gen_burgers1.m``):
# covariance ``sigma^2 (-Delta + tau^2 I)^(-gamma)`` on the periodic domain.
_BURGERS_GRF_GAMMA = 2.5
_BURGERS_GRF_TAU = 7.0
_BURGERS_GRF_SIGMA = 49.0


def _burgers_ic(key: jax.Array, resolution: int) -> jax.Array:
    """Canonical FNO-benchmark Burgers IC: a periodic Gaussian random field.

    Sampled from the Gaussian measure with covariance
    ``sigma^2 (-Delta + tau^2 I)^(-gamma)`` (gamma=2.5, tau=7, sigma=49), the
    initial condition used by the FNO-paper 1D Burgers benchmark. The steep
    ``gamma`` concentrates the spectral energy in the lowest Fourier modes, so the
    field is smooth on the periodic domain ``[0, 1)`` -- well matched to the
    pseudo-spectral solver. Synthesised spectrally so it is exactly periodic and
    jit/vmap-compatible.
    """
    wavenumbers = jnp.arange(resolution // 2 + 1)
    sqrt_eigenvalues = (
        jnp.sqrt(2.0)
        * _BURGERS_GRF_SIGMA
        * ((2.0 * jnp.pi * wavenumbers) ** 2 + _BURGERS_GRF_TAU**2)
        ** (-_BURGERS_GRF_GAMMA / 2.0)
    )
    sqrt_eigenvalues = sqrt_eigenvalues.at[0].set(0.0)  # zero mean
    real_key, imag_key = jax.random.split(key)
    coefficients = (
        jax.random.normal(real_key, (wavenumbers.shape[0],))
        + 1j * jax.random.normal(imag_key, (wavenumbers.shape[0],))
    ) * sqrt_eigenvalues
    # ``irfft`` carries a 1/N normalisation; undo it for the O(1) amplitude of the
    # continuous Gaussian measure.
    return jnp.fft.irfft(coefficients, n=resolution) * resolution


def _darcy_coeff(
    key: jax.Array,
    resolution: int,
    coeff_range: tuple[float, float],
    field_type: str,
    grf_filter: jax.Array,
    grid: tuple[jax.Array, jax.Array],
) -> jax.Array:
    """Random Darcy permeability field (thresholded-GRF binary or smooth log-Fourier)."""
    low, high = coeff_range
    if field_type == "binary":
        white = jax.random.normal(key, (resolution, resolution))
        field = jnp.fft.ifft2(jnp.fft.fft2(white) * grf_filter).real
        return jnp.where(field >= 0.0, high, low)

    grid_x, grid_y = grid
    key_freq, key_phase, key_amp = jax.random.split(key, 3)
    n_modes = 12
    freqs = jax.random.randint(key_freq, (n_modes, 2), 1, 6)
    phases = jax.random.uniform(key_phase, (n_modes,), minval=0.0, maxval=2 * jnp.pi)
    amplitudes = jax.random.uniform(key_amp, (n_modes,), minval=0.1, maxval=1.0)

    modes = amplitudes[:, None, None] * jnp.sin(
        freqs[:, 0, None, None] * jnp.pi * grid_x[None]
        + freqs[:, 1, None, None] * jnp.pi * grid_y[None]
        + phases[:, None, None]
    )
    field = jnp.exp(1.0 + jnp.sum(modes, axis=0))
    span = jnp.max(field) - jnp.min(field) + 1e-10
    return low + (high - low) * (field - jnp.min(field)) / span


def _diffusion_ic(key: jax.Array, resolution: int) -> jax.Array:
    """Random 2D diffusion-advection initial condition (Gaussian / square / wave)."""
    coords = jnp.linspace(0.0, 1.0, resolution)
    grid_x, grid_y = jnp.meshgrid(coords, coords, indexing="ij")
    key_type, key_a, key_b = jax.random.split(key, 3)
    ic_type = jax.random.randint(key_type, (), 0, 3)

    def gaussian() -> jax.Array:
        cx = 0.3 + 0.4 * jax.random.uniform(key_a)
        cy = 0.3 + 0.4 * jax.random.uniform(key_b)
        width = 0.05 + 0.1 * jax.random.uniform(key_a)
        return jnp.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * width**2))

    def square() -> jax.Array:
        cx = 0.3 + 0.4 * jax.random.uniform(key_a)
        cy = 0.3 + 0.4 * jax.random.uniform(key_b)
        size = 0.1 + 0.2 * jax.random.uniform(key_a)
        return jnp.where(
            (jnp.abs(grid_x - cx) < size / 2) & (jnp.abs(grid_y - cy) < size / 2),
            1.0,
            0.0,
        )

    def wave() -> jax.Array:
        fx = 2 + 4 * jax.random.uniform(key_a)
        fy = 2 + 4 * jax.random.uniform(key_b)
        return jnp.sin(fx * jnp.pi * grid_x) * jnp.sin(fy * jnp.pi * grid_y)

    return jax.lax.switch(ic_type, [gaussian, square, wave])


def _navier_stokes_ic(key: jax.Array, resolution: int) -> tuple[jax.Array, jax.Array]:
    """Random Navier-Stokes velocity init (Taylor-Green / shear / random vortices)."""
    key_type, key_a, key_b = jax.random.split(key, 3)
    ic_type = jax.random.randint(key_type, (), 0, 3)

    def taylor_green() -> tuple[jax.Array, jax.Array]:
        amplitude = jax.random.uniform(key_a, (), minval=0.5, maxval=2.0)
        # The JAX-native IC builders accept traced scalars; their stubs only type floats.
        return create_taylor_green_vortex(resolution, amplitude)  # pyright: ignore[reportArgumentType]

    def shear() -> tuple[jax.Array, jax.Array]:
        perturbation = jax.random.uniform(key_a, (), minval=0.01, maxval=0.1)
        thickness = jax.random.uniform(key_b, (), minval=0.02, maxval=0.1)
        return create_double_shear_layer(resolution, thickness, perturbation)  # pyright: ignore[reportArgumentType]

    def random_vortices() -> tuple[jax.Array, jax.Array]:
        coords = jnp.linspace(0.0, 2 * jnp.pi, resolution, endpoint=False)
        grid_x, grid_y = jnp.meshgrid(coords, coords, indexing="ij")

        def add_vortex(
            state: tuple[jax.Array, jax.Array], index: jax.Array
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            u_acc, v_acc = state
            subkeys = jax.random.split(jax.random.fold_in(key_a, index), 4)
            cx = jax.random.uniform(subkeys[0], (), minval=1.0, maxval=5.0)
            cy = jax.random.uniform(subkeys[1], (), minval=1.0, maxval=5.0)
            strength = jax.random.uniform(subkeys[2], (), minval=-1.0, maxval=1.0)
            width = jax.random.uniform(subkeys[3], (), minval=0.3, maxval=1.0)
            decay = jnp.exp(-((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * width**2))
            return (
                u_acc - strength * (grid_y - cy) * decay,
                v_acc + strength * (grid_x - cx) * decay,
            ), None

        zeros = jnp.zeros((resolution, resolution))
        (u0, v0), _ = jax.lax.scan(add_vortex, (zeros, zeros), jnp.arange(2))
        return u0, v0

    return jax.lax.switch(ic_type, [taylor_green, shear, random_vortices])


def _shallow_water_ic(key: jax.Array, resolution: int) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Random shallow-water init: height bump plus sinusoidal ``u`` / ``v`` fields."""
    coords = jnp.linspace(0.0, 1.0, resolution)
    grid_x, grid_y = jnp.meshgrid(coords, coords, indexing="ij")
    key_h, key_u, key_v = jax.random.split(key, 3)

    cx_h = jax.random.uniform(key_h, (), minval=0.3, maxval=0.7)
    cy_h = jax.random.uniform(key_h, (), minval=0.3, maxval=0.7)
    height = 1.0 + 0.1 * jnp.exp(-((grid_x - cx_h) ** 2 + (grid_y - cy_h) ** 2) / 0.05)

    cx_u = jax.random.uniform(key_u, (), minval=0.3, maxval=0.7)
    u_velocity = 0.1 * jnp.sin(2 * jnp.pi * grid_x) * jnp.exp(-((grid_y - cx_u) ** 2) / 0.1)

    cy_v = jax.random.uniform(key_v, (), minval=0.3, maxval=0.7)
    v_velocity = 0.1 * jnp.sin(2 * jnp.pi * grid_y) * jnp.exp(-((grid_x - cy_v) ** 2) / 0.1)

    return height, u_velocity, v_velocity


# --------------------------------------------------------------------------- #
# Public generators.
# --------------------------------------------------------------------------- #


def generate_burgers(
    *,
    n_samples: int,
    resolution: int = 128,
    viscosity_range: tuple[float, float] = (0.1, 0.1),
    time_final: float = 1.0,
    num_steps: int = 200,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate the canonical 1D Burgers operator dataset (IC -> final solution).

    Initial conditions are drawn from the FNO-benchmark Gaussian random field on
    the periodic domain ``[0, 1)`` and evolved to ``time_final`` with the
    pseudo-spectral ETDRK4 solver
    (:func:`opifex.physics.spectral.steppers.solve_burgers_spectral`), so the
    operator ``u(x, 0) -> u(x, time_final)`` matches the published benchmark.

    Args:
        n_samples: Number of samples to generate.
        resolution: Spatial grid resolution.
        viscosity_range: ``(low, high)`` uniform range for the per-sample viscosity.
        time_final: Integration time; the solution there is the label.
        num_steps: Number of spectral time steps (accuracy/cost trade-off).
        seed: Base seed; sample ``i`` uses ``PRNGKey(seed + i)``.

    Returns:
        ``{"input": (n, 1, res), "output": (n, 1, res)}`` float32 arrays.
    """
    low, high = viscosity_range

    def per_sample(key: jax.Array) -> jax.Array:
        ic_key, visc_key = jax.random.split(key)
        ic = _burgers_ic(ic_key, resolution)
        viscosity = jax.random.uniform(visc_key, (), minval=low, maxval=high)
        final = solve_burgers_spectral(
            ic,
            viscosity,
            domain_extent=1.0,
            time_final=time_final,
            num_steps=num_steps,
            num_snapshots=1,
        )[-1]
        return jnp.stack([ic, final])  # (2, res): [input, output]

    stacked = _run_vmapped(per_sample, seed, n_samples)  # (n, 2, res)
    return {"input": stacked[:, 0:1], "output": stacked[:, 1:2]}


def generate_darcy(
    *,
    n_samples: int,
    resolution: int = 64,
    coeff_range: tuple[float, float] = (0.1, 1.0),
    field_type: str = "smooth",
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a Darcy-flow operator dataset (permeability -> steady solution).

    The coefficient-field synthesis is vmapped on-device, but the steady solve
    (:func:`solve_darcy_flow`) is a ``scipy`` sparse direct factorization that is
    **not** jittable/vmappable, so it runs in a Python loop over samples.

    Args:
        n_samples: Number of samples to generate.
        resolution: Spatial grid resolution.
        coeff_range: ``(low, high)`` permeability bounds (smooth scaling range, or
            the two discrete values for ``field_type='binary'``).
        field_type: ``'smooth'`` log-Fourier field, or ``'binary'`` thresholded GRF.
        seed: Base seed; sample ``i`` uses ``PRNGKey(seed + i)``.

    Returns:
        ``{"input": (n, 1, res, res), "output": (n, 1, res, res)}`` float32 arrays.

    Raises:
        ValueError: If ``field_type`` is not ``'smooth'`` or ``'binary'``.
    """
    if field_type not in ("smooth", "binary"):
        raise ValueError(f"field_type must be 'smooth' or 'binary', got {field_type!r}")

    coords = jnp.linspace(0.0, 1.0, resolution)
    grid_x, grid_y = jnp.meshgrid(coords, coords, indexing="ij")
    grid = (grid_x, grid_y)

    # Isotropic GRF power-spectrum filter (covariance ~ (-Δ + τ²I)^{-1}); DC zeroed
    # so the binary threshold splits ~50/50.
    freqs = jnp.fft.fftfreq(resolution) * resolution
    kx, ky = jnp.meshgrid(freqs, freqs, indexing="ij")
    grf_filter = ((kx**2 + ky**2 + 9.0) ** (-1.0)).at[0, 0].set(0.0)

    keys = _per_sample_keys(seed, n_samples)
    coeff_fields = jax.jit(
        jax.vmap(lambda k: _darcy_coeff(k, resolution, coeff_range, field_type, grf_filter, grid))
    )(keys)

    # scipy sparse direct solve is not vmappable -> loop over samples.
    solutions = [solve_darcy_flow(coeff_fields[i], resolution) for i in range(n_samples)]

    coeff_array = _stack_float32(coeff_fields)[:, None]
    solution_array = _stack_float32(jnp.stack(solutions))[:, None]
    return {"input": coeff_array, "output": solution_array}


def generate_diffusion(
    *,
    n_samples: int,
    resolution: int = 64,
    diffusion_range: tuple[float, float] = (0.01, 0.1),
    advection_range: tuple[float, float] = (-1.0, 1.0),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a 2D diffusion-advection dataset (IC -> final state).

    Args:
        n_samples: Number of samples to generate.
        resolution: Spatial grid resolution.
        diffusion_range: ``(low, high)`` uniform range for the diffusion coefficient.
        advection_range: ``(low, high)`` uniform range for each advection velocity.
        seed: Base seed; sample ``i`` uses ``PRNGKey(seed + i)``.

    Returns:
        ``{"input": (n, 1, res, res), "output": (n, 1, res, res)}`` float32 arrays.
    """
    diff_low, diff_high = diffusion_range
    adv_low, adv_high = advection_range
    grid_spacing = 1.0 / resolution
    dt = 0.01
    n_steps = 100

    def per_sample(key: jax.Array) -> jax.Array:
        ic_key, key_diff, key_vx, key_vy = jax.random.split(key, 4)
        ic = _diffusion_ic(ic_key, resolution)
        diffusion = jax.random.uniform(key_diff, (), minval=diff_low, maxval=diff_high)
        vx = jax.random.uniform(key_vx, (), minval=adv_low, maxval=adv_high)
        vy = jax.random.uniform(key_vy, (), minval=adv_low, maxval=adv_high)
        # Use the jitted core solver directly: the public wrapper does Python-level
        # isinstance validation on concrete floats, which fails under vmap tracers.
        final = _solve_diffusion_advection_2d_jit(
            ic, diffusion, (vx, vy), n_steps, grid_spacing, dt
        )
        return jnp.stack([ic, final])

    stacked = _run_vmapped(per_sample, seed, n_samples)  # (n, 2, res, res)
    return {"input": stacked[:, 0:1], "output": stacked[:, 1:2]}


def generate_navier_stokes(
    *,
    n_samples: int,
    resolution: int = 64,
    viscosity_range: tuple[float, float] = (0.001, 0.01),
    time_range: tuple[float, float] = (0.0, 1.0),
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a 2D incompressible Navier-Stokes dataset ([u0, v0] -> final [u, v]).

    Args:
        n_samples: Number of samples to generate.
        resolution: Spatial grid resolution.
        viscosity_range: ``(low, high)`` uniform range for the kinematic viscosity.
        time_range: ``(start, end)`` integration window; the end-time slice is the label.
        seed: Base seed; sample ``i`` uses ``PRNGKey(seed + i)``.

    Returns:
        ``{"input": (n, 2, res, res), "output": (n, 2, res, res)}`` float32 arrays
        with channels ordered ``[u, v]``.
    """
    low, high = viscosity_range

    def per_sample(key: jax.Array) -> jax.Array:
        ic_key, nu_key = jax.random.split(key)
        u0, v0 = _navier_stokes_ic(ic_key, resolution)
        nu = jax.random.uniform(nu_key, (), minval=low, maxval=high)
        u_traj, v_traj = solve_navier_stokes_2d(u0, v0, nu, time_range, 1, resolution)
        input_fields = jnp.stack([u0, v0])  # (2, res, res)
        output_fields = jnp.stack([u_traj[-1], v_traj[-1]])  # (2, res, res)
        return jnp.stack([input_fields, output_fields])  # (2, 2, res, res)

    stacked = _run_vmapped(per_sample, seed, n_samples)  # (n, 2, 2, res, res)
    return {"input": stacked[:, 0], "output": stacked[:, 1]}


def generate_shallow_water(
    *,
    n_samples: int,
    resolution: int = 64,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    """Generate a 2D shallow-water dataset ([h, u, v] init -> final [h, u, v]).

    Args:
        n_samples: Number of samples to generate.
        resolution: Spatial grid resolution.
        seed: Base seed; sample ``i`` uses ``PRNGKey(seed + i)``.

    Returns:
        ``{"input": (n, 3, res, res), "output": (n, 3, res, res)}`` float32 arrays
        with channels ordered ``[h, u, v]``.
    """
    grid_spacing = 1.0 / resolution

    def per_sample(key: jax.Array) -> jax.Array:
        height, u_velocity, v_velocity = _shallow_water_ic(key, resolution)
        h_final, u_final, v_final = solve_shallow_water_2d(
            height, u_velocity, v_velocity, 9.81, 0.001, 100, grid_spacing
        )
        input_fields = jnp.stack([height, u_velocity, v_velocity])
        output_fields = jnp.stack([h_final, u_final, v_final])
        return jnp.stack([input_fields, output_fields])  # (2, 3, res, res)

    stacked = _run_vmapped(per_sample, seed, n_samples)  # (n, 2, 3, res, res)
    return {"input": stacked[:, 0], "output": stacked[:, 1]}
