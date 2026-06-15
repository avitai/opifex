r"""Metropolis-Hastings Monte Carlo samplers for neural wavefunctions.

The walkers are drawn from the Born density :math:`|\psi(r)|^2`, i.e.
:math:`\log p(r) = 2 \log|\psi(r)|`. :class:`MetropolisHastingsSampler`
implements the FermiNet all-electron move with an *asymmetric, harmonic-mean*
proposal (``../ferminet`` ``mcmc.py``): the per-electron proposal width scales
with the harmonic mean of that electron's distances to the nuclei, so electrons
near a nucleus take small steps and valence electrons take large ones. The
accept/reject ratio includes the forward/reverse proposal densities to keep
detailed balance.

A fixed number of MCMC sweeps is fused into a single :func:`jax.lax.scan`, so the
whole sampling phase is one ``jit``-compiled GPU kernel (the scan-fusion pattern
used by the atomistic models). MALA (Langevin) is a documented future upgrade:
it would add a ``grad log|psi|`` drift term to the proposal mean here.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float  # noqa: TC002


def _harmonic_mean_width(
    walkers: Float[Array, "batch nelectron ndim"],
    atoms: Float[Array, "natom ndim"],
) -> Array:
    """Harmonic mean of each electron's distances to the nuclei.

    Returns an array of shape ``(batch, nelectron, 1)`` giving the proposal-width
    scale for each electron (FermiNet ``_harmonic_mean``).
    """
    displacement = walkers[:, :, None, :] - atoms[None, None, :, :]
    distance = jnp.linalg.norm(displacement, axis=-1)
    return 1.0 / jnp.mean(1.0 / distance, axis=-1, keepdims=True)


def _log_proposal_density(
    walkers: Float[Array, "batch nelectron ndim"],
    mean: Float[Array, "batch nelectron ndim"],
    width: Float[Array, "batch nelectron 1"],
) -> Array:
    """Log density of the diagonal-Gaussian proposal ``N(mean, width^2)``.

    Returns one value per walker (shape ``(batch,)``), summing over electrons
    and Cartesian components.
    """
    ndim = walkers.shape[-1]
    numerator = jnp.sum(-0.5 * (walkers - mean) ** 2 / width**2, axis=(1, 2))
    normaliser = ndim * jnp.sum(jnp.log(width[..., 0]), axis=1)
    return numerator - normaliser


@dataclass(frozen=True, slots=True)
class MetropolisHastingsSampler:
    """Harmonic-mean Metropolis-Hastings sampler for ``|psi|^2``.

    Args:
        atoms: Nuclear coordinates of shape ``(natom, ndim)``. Used to scale the
            per-electron proposal width by the harmonic mean of nuclear
            distances.
        steps: Number of MCMC sweeps per :meth:`sample` call (fused into one
            ``lax.scan``).
        step_size: Base proposal standard deviation (multiplied by the
            harmonic-mean width).
    """

    atoms: Float[Array, "natom ndim"]
    steps: int = 10
    step_size: float = 0.1

    def sample(
        self,
        log_abs: Callable[[Array], Array],
        walkers: Float[Array, "batch nelectron ndim"],
        key: Array,
    ) -> tuple[Array, Array]:
        r"""Advance the walkers by :attr:`steps` Metropolis-Hastings sweeps.

        Args:
            log_abs: ``positions -> log|psi|`` for a *single* walker. It is
                ``vmap``-ed over the batch internally.
            walkers: Current walker positions of shape
                ``(batch, nelectron, ndim)``.
            key: PRNG key.

        Returns:
            A ``(new_walkers, acceptance_fraction)`` tuple. The acceptance
            fraction is averaged over all walkers and sweeps.
        """
        atoms = jnp.asarray(self.atoms)
        batched_log_abs = jax.vmap(log_abs)

        def log_prob(positions: Array) -> Array:
            return 2.0 * batched_log_abs(positions)

        def sweep(
            carry: tuple[Array, Array, Array], _: None
        ) -> tuple[tuple[Array, Array, Array], Array]:
            current, current_log_prob, rng = carry
            rng, proposal_key, accept_key = jax.random.split(rng, 3)

            width = self.step_size * _harmonic_mean_width(current, atoms)
            noise = jax.random.normal(proposal_key, shape=current.shape)
            proposal = current + width * noise
            proposal_log_prob = log_prob(proposal)
            proposal_width = self.step_size * _harmonic_mean_width(proposal, atoms)

            forward = _log_proposal_density(proposal, current, width)
            reverse = _log_proposal_density(current, proposal, proposal_width)
            ratio = proposal_log_prob + reverse - current_log_prob - forward

            uniform = jnp.log(jax.random.uniform(accept_key, shape=ratio.shape))
            accept = ratio > uniform
            new_walkers = jnp.where(accept[:, None, None], proposal, current)
            new_log_prob = jnp.where(accept, proposal_log_prob, current_log_prob)
            return (new_walkers, new_log_prob, rng), jnp.mean(accept)

        init = (walkers, log_prob(walkers), key)
        (final_walkers, _, _), accept_history = jax.lax.scan(sweep, init, None, length=self.steps)
        return final_walkers, jnp.mean(accept_history)


__all__ = ["MetropolisHastingsSampler"]
