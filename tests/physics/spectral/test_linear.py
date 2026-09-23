"""Analytical tests for the exact Fourier solution of periodic advection-diffusion.

Ground truth is the closed form for a single Fourier mode: on ``[0, L)^d`` the
field ``sin(k . x)`` evolves under ``u_t + v . grad u = D lap u`` into
``exp(-D |k|^2 t) sin(k . (x - v t))``. Every property below follows from that
formula or from the structure of the PDE (mass conservation, the semigroup
property, L2 contraction), never from the implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.physics.spectral.linear import solve_advection_diffusion_exact


def _grid(num_points: int, domain: float, dims: int) -> list[np.ndarray]:
    """Cell-vertex coordinates of the periodic grid (right endpoint excluded)."""
    axis = np.linspace(0.0, domain, num_points, endpoint=False)
    return list(np.meshgrid(*([axis] * dims), indexing="ij"))


def _single_mode(
    coords: list[np.ndarray],
    modes: tuple[int, ...],
    domain: float,
    diffusivity: float,
    velocity: tuple[float, ...],
    time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Initial field ``sin(k . x)`` and its exact solution at ``time``."""
    wavevector = [2.0 * np.pi * m / domain for m in modes]
    phase0 = sum(k * x for k, x in zip(wavevector, coords, strict=True))
    shift = sum(k * v for k, v in zip(wavevector, velocity, strict=True)) * time
    decay = np.exp(-diffusivity * sum(k**2 for k in wavevector) * time)
    return np.sin(phase0), decay * np.sin(phase0 - shift)


class TestSingleModeExactness:
    """The solver reproduces the closed-form single-mode solution to round-off."""

    @pytest.mark.parametrize(
        ("dims", "num_points", "modes", "velocity"),
        [
            (1, 16, (3,), (0.7,)),
            (1, 15, (3,), (0.7,)),
            (2, 16, (1, 2), (0.4, -0.9)),
            (2, 16, (3, -1), (-1.0, 0.25)),
            (2, 17, (3, -1), (-1.0, 0.25)),
            (3, 16, (1, 1, 2), (0.3, -0.2, 0.5)),
        ],
    )
    def test_matches_closed_form(
        self,
        dims: int,
        num_points: int,
        modes: tuple[int, ...],
        velocity: tuple[float, ...],
    ) -> None:
        domain, diffusivity, time = 2.0, 0.03, 0.8
        coords = _grid(num_points, domain, dims)
        u0, expected = _single_mode(coords, modes, domain, diffusivity, velocity, time)
        result = solve_advection_diffusion_exact(
            jnp.asarray(u0),
            diffusivity,
            jnp.asarray(velocity),
            domain_extent=domain,
            time_final=time,
        )
        np.testing.assert_allclose(np.asarray(result), expected, rtol=0.0, atol=1e-12)

    def test_non_grid_multiple_shift_is_exact(self) -> None:
        """A translation by a fraction of a cell is exact, not interpolated."""
        coords = _grid(32, 1.0, 1)
        velocity = (0.37 / 32,)  # a shift of 0.37 cells over unit time
        u0, expected = _single_mode(coords, (5,), 1.0, 0.0, velocity, 1.0)
        result = solve_advection_diffusion_exact(jnp.asarray(u0), 0.0, jnp.asarray(velocity))
        np.testing.assert_allclose(np.asarray(result), expected, rtol=0.0, atol=1e-12)


class TestStructuralProperties:
    """Properties the PDE itself guarantees for any initial condition."""

    @staticmethod
    def _random_field(seed: int) -> jax.Array:
        return jax.random.normal(jax.random.key(seed), (24, 24))

    def test_mass_is_conserved(self) -> None:
        u0 = self._random_field(0) + 3.0
        result = solve_advection_diffusion_exact(u0, 0.05, jnp.array([0.6, -0.3]))
        np.testing.assert_allclose(float(result.mean()), float(u0.mean()), rtol=1e-14)

    def test_l2_norm_does_not_grow(self) -> None:
        u0 = self._random_field(1)
        result = solve_advection_diffusion_exact(u0, 0.01, jnp.array([1.0, 1.0]))
        assert float(jnp.sum(result**2)) < float(jnp.sum(u0**2))

    def test_pure_advection_preserves_l2_norm(self) -> None:
        """With ``D = 0`` the evolution is unitary on the resolved (non-Nyquist) modes."""
        u0 = self._random_field(2)
        u0_hat = jnp.fft.rfft2(u0).at[12, :].set(0.0).at[:, -1].set(0.0)
        u0 = jnp.fft.irfft2(u0_hat, s=u0.shape)
        result = solve_advection_diffusion_exact(u0, 0.0, jnp.array([0.31, -0.77]))
        np.testing.assert_allclose(float(jnp.sum(result**2)), float(jnp.sum(u0**2)), rtol=1e-13)

    def test_semigroup_property(self) -> None:
        """Solving to ``t1 + t2`` equals solving to ``t1`` and then for ``t2``."""
        u0 = self._random_field(3)
        diffusivity, velocity = 0.02, jnp.array([0.5, -0.8])
        direct = solve_advection_diffusion_exact(u0, diffusivity, velocity, time_final=0.7)
        halfway = solve_advection_diffusion_exact(u0, diffusivity, velocity, time_final=0.3)
        composed = solve_advection_diffusion_exact(halfway, diffusivity, velocity, time_final=0.4)
        np.testing.assert_allclose(np.asarray(composed), np.asarray(direct), atol=1e-13)

    def test_velocity_must_match_the_field_dimension(self) -> None:
        with pytest.raises(ValueError, match="velocity"):
            solve_advection_diffusion_exact(jnp.zeros((8, 8)), 0.1, jnp.array([1.0]))


class TestTransforms:
    """jit, vmap and grad through the solver agree with the closed form."""

    def test_jit_matches_eager(self) -> None:
        u0 = jax.random.normal(jax.random.key(4), (16, 16))
        velocity = jnp.array([0.2, 0.9])
        eager = solve_advection_diffusion_exact(u0, 0.04, velocity)
        jitted = jax.jit(solve_advection_diffusion_exact)(u0, 0.04, velocity)
        np.testing.assert_allclose(np.asarray(jitted), np.asarray(eager), atol=1e-14)

    def test_vmap_over_coefficients(self) -> None:
        coords = _grid(16, 1.0, 2)
        diffusivities = jnp.array([0.0, 0.01, 0.1])
        velocities = jnp.array([[0.0, 0.0], [0.5, -0.5], [-1.0, 0.3]])
        u0 = jnp.asarray(_single_mode(coords, (1, 2), 1.0, 0.0, (0.0, 0.0), 0.0)[0])
        batched = jax.vmap(solve_advection_diffusion_exact, in_axes=(None, 0, 0))(
            u0, diffusivities, velocities
        )
        for index in range(3):
            _, expected = _single_mode(
                coords,
                (1, 2),
                1.0,
                float(diffusivities[index]),
                tuple(float(v) for v in velocities[index]),
                1.0,
            )
            np.testing.assert_allclose(np.asarray(batched[index]), expected, atol=1e-12)

    def test_grad_with_respect_to_diffusivity(self) -> None:
        """``d/dD`` of the single-mode amplitude is ``-|k|^2 t`` times the solution."""
        coords = _grid(16, 1.0, 2)
        diffusivity, time = 0.02, 0.5
        u0, solution = _single_mode(coords, (2, 1), 1.0, diffusivity, (0.0, 0.0), time)
        probe = jnp.asarray(solution)

        def projection(d: jax.Array) -> jax.Array:
            field = solve_advection_diffusion_exact(
                jnp.asarray(u0), d, jnp.zeros(2), time_final=time
            )
            return jnp.sum(field * probe)

        k_squared = (2.0 * np.pi) ** 2 * (2**2 + 1**2)
        expected = -k_squared * time * float(np.sum(solution**2))
        np.testing.assert_allclose(
            float(jax.grad(projection)(jnp.asarray(diffusivity))), expected, rtol=1e-12
        )
