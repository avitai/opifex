r"""Metropolis-adjusted Langevin (MALA) sampler for neural wavefunctions.

The walkers are drawn from the Born density :math:`|\psi(r)|^2`, i.e.
:math:`\log p(r) = 2\,\log|\psi(r)|`. :class:`MALASampler` is a
*gradient-informed* drop-in alternative to the harmonic-mean
:class:`~opifex.neural.quantum.vmc.sampler.MetropolisHastingsSampler`: instead of
an isotropic random walk it proposes a Langevin move that drifts each walker up
the log-density gradient, then applies the exact Metropolis-Hastings correction
for the *asymmetric* proposal density. The drift typically decorrelates walkers
faster, so fewer sweeps are needed per effective sample.

For a step size :math:`\varepsilon` (``step_size``) and the half-step
:math:`\tau = \varepsilon^2 / 2` (``dt``), the proposal is

.. math::
   r' = r + \tau\,\nabla_r \log p(r) + \sqrt{2\tau}\,\eta,
   \qquad \eta \sim \mathcal{N}(0, I),

which is exactly :math:`r' = r + (\varepsilon^2/2)\,\nabla \log p(r) +
\varepsilon\,\eta`. The Gaussian proposal density is

.. math::
   \log q(r' \mid r) = -\frac{\lVert r' - r - \tau\,\nabla\log p(r)\rVert^2}
                              {4\tau} + \text{const},

and the move :math:`r \to r'` is accepted with probability
:math:`\min\bigl(1, \tfrac{p(r')\,q(r\mid r')}{p(r)\,q(r'\mid r)}\bigr)`. The
:math:`q(r\mid r')/q(r'\mid r)` factor is the asymmetric correction that keeps
:math:`|\psi|^2` invariant; dropping it (``asymmetric_correction=False``) yields a
biased, unadjusted Langevin walk and is provided only for diagnostics.

As with the Metropolis sampler, the fixed number of sweeps is fused into a single
:func:`jax.lax.scan`, so the whole sampling phase is one ``jit``-compiled kernel.

References:
    Roberts & Tweedie, "Exponential convergence of Langevin distributions and
    their discrete approximations", Bernoulli 2(4), 1996 (the MALA proposal and
    its Metropolis adjustment).
    Roberts & Rosenthal, "Optimal scaling of discrete approximations to Langevin
    diffusions", JRSS B 60(1), 1998 (the ~0.574 optimal acceptance).
    Implementations: ``../netket`` ``netket/sampler/rules/langevin.py``
    (``LangevinRule`` / ``_langevin_step``); ``../deepqmc``
    ``deepqmc/sampling/electron_samplers.py`` (``LangevinSampler``);
    ``../ferminet`` ``ferminet/mcmc.py`` (the asymmetric accept/reject pattern).
"""

from __future__ import annotations

import logging
from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray  # noqa: TC002


logger = logging.getLogger(__name__)


def langevin_drift(
    log_abs: Callable[[Array], Array],
    walkers: Float[Array, "batch nelectron ndim"],
) -> Float[Array, "batch nelectron ndim"]:
    r"""Per-walker drift :math:`\nabla_r \log p(r) = 2\,\nabla_r \log|\psi(r)|`.

    The single-walker ``log_abs`` is differentiated with :func:`jax.grad` and
    ``vmap``-ed over the batch, mirroring the per-walker evaluation contract of
    the :class:`~opifex.neural.quantum.vmc.protocols.Wavefunction` protocol.

    Args:
        log_abs: ``positions -> log|psi|`` for a *single* walker of shape
            ``(nelectron, ndim)``, returning a scalar.
        walkers: Walker positions of shape ``(batch, nelectron, ndim)``.

    Returns:
        The drift, same shape as ``walkers``.
    """
    grad_log_abs = jax.grad(lambda positions: jnp.sum(log_abs(positions)))
    return 2.0 * jax.vmap(grad_log_abs)(walkers)


def _log_proposal_density(
    target: Float[Array, "batch nelectron ndim"],
    source: Float[Array, "batch nelectron ndim"],
    drift: Float[Array, "batch nelectron ndim"],
    dt: Array,
) -> Float[Array, " batch"]:
    r"""Log Langevin proposal density :math:`\log q(\text{target}\mid\text{source})`.

    Computes ``-||target - (source + dt * drift)||^2 / (4 dt)`` summed over
    electrons and Cartesian components, dropping the per-walker-constant Gaussian
    normaliser (it is identical for the forward and reverse densities, so it
    cancels in the acceptance ratio).

    Args:
        target: Proposed-at positions, shape ``(batch, nelectron, ndim)``.
        source: Conditioned-on positions, shape ``(batch, nelectron, ndim)``.
        drift: Drift evaluated at ``source``, shape ``(batch, nelectron, ndim)``.
        dt: The Langevin half-step :math:`\tau = \varepsilon^2 / 2`.

    Returns:
        One log-density value per walker, shape ``(batch,)``.
    """
    mean = source + dt * drift
    return -jnp.sum((target - mean) ** 2, axis=(1, 2)) / (4.0 * dt)


@dataclass(frozen=True, slots=True)
class MALASampler:
    r"""Metropolis-adjusted Langevin sampler for the Born density ``|psi|^2``.

    Satisfies the :class:`~opifex.neural.quantum.vmc.protocols.Sampler` protocol,
    so it is interchangeable with
    :class:`~opifex.neural.quantum.vmc.sampler.MetropolisHastingsSampler`.

    Args:
        steps: Number of Langevin sweeps per :meth:`sample` call (fused into one
            ``lax.scan``).
        step_size: The Langevin step size :math:`\varepsilon` (a Python float or
            a traced JAX scalar, so it may be differentiated through). The
            proposal uses the half-step :math:`\tau = \varepsilon^2 / 2` for the
            drift and :math:`\varepsilon` for the noise, i.e.
            ``r' = r + (eps^2/2) * grad log p + eps * N(0, 1)``.
        asymmetric_correction: If ``True`` (default), accept with the exact MALA
            ratio including the ``q(r|r')/q(r'|r)`` term, leaving ``|psi|^2``
            invariant. If ``False``, accept with the symmetric ratio
            ``log p(r') - log p(r)`` -- a biased, unadjusted Langevin walk kept
            only for diagnostics.
        max_drift_norm: Optional per-walker cap on the Euclidean norm of the
            drift. ``None`` (default) leaves the drift unclipped. A positive value
            rescales any walker whose flattened drift norm exceeds the cap,
            bounding the proposal mean for numerical stability near nuclear cusps
            (the bounded-force idea of ``deepqmc`` ``clean_force``).
    """

    steps: int = 10
    step_size: float | Array = 0.1
    asymmetric_correction: bool = True
    max_drift_norm: float | None = None

    def _clip_drift(
        self, drift: Float[Array, "batch nelectron ndim"]
    ) -> Float[Array, "batch nelectron ndim"]:
        """Norm-clip each walker's drift to :attr:`max_drift_norm` if set."""
        if self.max_drift_norm is None:
            return drift
        flat = drift.reshape(drift.shape[0], -1)
        norm = jnp.linalg.norm(flat, axis=-1, keepdims=True)
        scale = jnp.minimum(1.0, self.max_drift_norm / jnp.clip(norm, a_min=1e-12))
        return (flat * scale).reshape(drift.shape)

    def _acceptance_ratio(
        self,
        current: Float[Array, "batch nelectron ndim"],
        proposal: Float[Array, "batch nelectron ndim"],
        current_log_prob: Float[Array, " batch"],
        proposal_log_prob: Float[Array, " batch"],
        current_drift: Float[Array, "batch nelectron ndim"],
        proposal_drift: Float[Array, "batch nelectron ndim"],
        dt: Array,
    ) -> Float[Array, " batch"]:
        r"""Per-walker log acceptance ratio for the move ``current -> proposal``.

        With :attr:`asymmetric_correction` the ratio is
        ``log p(r') + log q(r|r') - log p(r) - log q(r'|r)``; otherwise the
        symmetric ``log p(r') - log p(r)``.
        """
        density_ratio = proposal_log_prob - current_log_prob
        if not self.asymmetric_correction:
            return density_ratio
        forward = _log_proposal_density(proposal, current, current_drift, dt)
        reverse = _log_proposal_density(current, proposal, proposal_drift, dt)
        return density_ratio + reverse - forward

    def sample(
        self,
        log_abs: Callable[[Array], Array],
        walkers: Float[Array, "batch nelectron ndim"],
        key: PRNGKeyArray,
    ) -> tuple[Float[Array, "batch nelectron ndim"], Array]:
        r"""Advance the walkers by :attr:`steps` MALA sweeps.

        Args:
            log_abs: ``positions -> log|psi|`` for a *single* walker. It is
                differentiated and ``vmap``-ed over the batch internally.
            walkers: Current walker positions of shape
                ``(batch, nelectron, ndim)``.
            key: PRNG key (caller-owned; no hidden global key).

        Returns:
            A ``(new_walkers, acceptance_fraction)`` tuple. The acceptance
            fraction is averaged over all walkers and sweeps.
        """
        dt = 0.5 * jnp.asarray(self.step_size) ** 2
        noise_scale = jnp.sqrt(2.0 * dt)

        def log_prob(positions: Array) -> Array:
            return 2.0 * jax.vmap(log_abs)(positions)

        def drift(positions: Array) -> Array:
            return self._clip_drift(langevin_drift(log_abs, positions))

        def sweep(
            carry: tuple[Array, Array, Array, Array], _: None
        ) -> tuple[tuple[Array, Array, Array, Array], Array]:
            current, current_log_prob, current_drift, rng = carry
            rng, proposal_key, accept_key = jax.random.split(rng, 3)

            noise = jax.random.normal(proposal_key, shape=current.shape)
            proposal = current + dt * current_drift + noise_scale * noise
            proposal_log_prob = log_prob(proposal)
            proposal_drift = drift(proposal)

            ratio = self._acceptance_ratio(
                current,
                proposal,
                current_log_prob,
                proposal_log_prob,
                current_drift,
                proposal_drift,
                dt,
            )
            uniform = jnp.log(jax.random.uniform(accept_key, shape=ratio.shape))
            accept = ratio > uniform
            new_walkers = jnp.where(accept[:, None, None], proposal, current)
            new_log_prob = jnp.where(accept, proposal_log_prob, current_log_prob)
            new_drift = jnp.where(accept[:, None, None], proposal_drift, current_drift)
            return (new_walkers, new_log_prob, new_drift, rng), jnp.mean(accept)

        init = (walkers, log_prob(walkers), drift(walkers), key)
        (final_walkers, _, _, _), accept_history = jax.lax.scan(
            sweep, init, None, length=self.steps
        )
        return final_walkers, jnp.mean(accept_history)


__all__ = ["MALASampler", "langevin_drift"]
