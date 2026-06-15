r"""K-FAC natural-gradient preconditioner for variational Monte Carlo.

K-FAC (Kronecker-Factored Approximate Curvature; Martens & Grosse, *Optimizing
Neural Networks with Kronecker-factored Approximate Curvature*, ICML 2015,
arXiv:1503.05671) preconditions the gradient by an approximate inverse Fisher
information matrix. For each layer the Fisher block is approximated as a
Kronecker product :math:`A\otimes G` of the input second-moment ``A`` and the
output-gradient second-moment ``G``, which inverts cheaply as
:math:`(A\otimes G)^{-1}=A^{-1}\otimes G^{-1}`. For a VMC wavefunction the Fisher
of the log-amplitude *is* the quantum geometric tensor, so a K-FAC step is an
approximate quantum-natural-gradient (stochastic-reconfiguration) step -- the
same curvature that :mod:`~opifex.neural.quantum.vmc.optimizers` (MinSR / SPRING)
solves exactly in the sample space. K-FAC instead trades exactness for
:math:`O(\text{layer width}^3)` cost that is independent of the sample count,
which is what lets FermiNet scale to millions of parameters.

This module wraps the canonical ``kfac_jax`` library (the optimiser FermiNet
uses), exposing it behind the same ``init`` / ``step`` seam as the MinSR / SPRING
updates so it is a drop-in preconditioner choice for the VMC driver. The wiring
follows FermiNet exactly (``../ferminet/ferminet/train.py`` and
``../ferminet/ferminet/loss.py``):

* :func:`register_qmc_dense` tags the ansatz dense / attention layers (FermiNet's
  ``register_qmc``) so K-FAC recovers their Kronecker structure;
* :func:`register_log_amplitude` tags the network output as the mean of a normal
  predictive distribution (``kfac_jax.register_normal_predictive_distribution``),
  which makes K-FAC's Fisher equal the quantum geometric tensor;
* :func:`make_qmc_value_and_grad` builds the ``value_and_grad`` whose gradient is
  the FermiNet score-function estimator
  :math:`2\langle(E_{\mathrm{loc}}-\bar E)\,\nabla_\theta\log|\psi|\rangle`,
  injected through a ``custom_jvp`` exactly as FermiNet's loss does;
* :class:`KFACPreconditioner` wraps ``kfac_jax.Optimizer``, which manages its own
  ``jit`` internally.

References:
    Martens & Grosse, ICML 2015 (arXiv:1503.05671); ``kfac_jax`` 0.0.8
    (https://github.com/google-deepmind/kfac-jax); Pfau et al., FermiNet,
    *Phys. Rev. Research* 2, 033429 (2020) and ``../ferminet``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, cast

import jax
import jax.numpy as jnp
import kfac_jax
from jaxtyping import Array, Float, PyTree


logger = logging.getLogger(__name__)

# Type aliases for the QMC building blocks. ``Params`` is any array pytree (an
# ``nnx.State`` is one); the batched callables take all walkers at once.
Params = PyTree[Array]
LogAmplitudeFn = Callable[[Params, Float[Array, "batch nelectron ndim"]], Float[Array, " batch"]]
LocalEnergyFn = Callable[[Params, Float[Array, "batch nelectron ndim"]], Float[Array, " batch"]]
ValueAndGradFn = Callable[[Params, Array, Array], tuple[tuple[Array, dict[str, Any]], Params]]


def register_qmc_dense(
    y: Array,
    x: Array,
    w: Array,
    b: Array | None = None,
) -> Array:
    r"""Tag a dense / attention layer for K-FAC (FermiNet's ``register_qmc``).

    Annotates a linear map ``y = x @ w (+ b)`` so ``kfac_jax`` recovers its
    Kronecker-factored Fisher block. The output is returned unchanged -- the tag
    is a transparent identity at evaluation time and only affects curvature
    estimation. Apply it to every weight-bearing layer of the ansatz (FermiNet
    tags its dense streams and envelope this way).

    Args:
        y: The layer pre-activation ``x @ w (+ b)``.
        x: The layer input.
        w: The weight matrix.
        b: The optional bias vector.

    Returns:
        ``y`` unchanged (with a K-FAC dense layer tag attached).
    """
    return kfac_jax.register_dense(y, x, w, b)


def register_log_amplitude(log_amplitude: Float[Array, " batch"]) -> None:
    r"""Tag the log-amplitude as a normal-predictive mean (the QGT Fisher seam).

    Registering ``log|psi|`` as the mean of a unit-variance normal predictive
    distribution makes ``kfac_jax``'s Fisher equal the quantum geometric tensor
    of the wavefunction -- the metric of (quantum) natural gradient. This is the
    FermiNet ``kfac_jax.register_normal_predictive_distribution(psi[:, None])``
    call (``../ferminet/ferminet/loss.py``); it must be invoked inside the loss'
    ``custom_jvp`` so the tag is captured by K-FAC's graph tracer.

    Args:
        log_amplitude: Per-walker ``log|psi|`` of shape ``(batch,)``.
    """
    kfac_jax.register_normal_predictive_distribution(log_amplitude[:, None])


def _clip_to_mad(values: Array, window: float) -> Array:
    """Median-absolute-deviation clip the local energy (variance reduction).

    Mirrors the FermiNet / DeepQMC outlier-robust estimator used by the rest of
    the VMC stack: non-finite walkers are pulled to the median, then the energy
    is clipped to a ``window`` median-absolute-deviation band. ``window <= 0``
    disables the band but still sanitises non-finite values.
    """
    finite = jnp.isfinite(values)
    safe = jnp.where(finite, values, 0.0)
    centre = jnp.median(jnp.where(finite, safe, jnp.median(safe)))
    values = jnp.where(finite, values, centre)
    if window <= 0.0:
        return values
    deviation = jnp.mean(jnp.abs(values - centre))
    return jnp.clip(values, centre - window * deviation, centre + window * deviation)


def make_qmc_value_and_grad(
    log_abs: LogAmplitudeFn,
    local_energy: LocalEnergyFn,
    *,
    clip_local_energy: float = 5.0,
) -> ValueAndGradFn:
    r"""Build the K-FAC ``value_and_grad`` for the VMC energy expectation.

    The variational energy :math:`E=\langle E_{\mathrm{loc}}\rangle_{|\psi|^2}` is
    an expectation, so its gradient is *not* the gradient of the sampled mean.
    Following FermiNet (``../ferminet/ferminet/loss.py``), the value is the mean
    local energy and the gradient is injected through a ``custom_jvp`` as the
    score-function estimator

    .. math::

        \nabla_\theta E = 2\,\big\langle (E_{\mathrm{loc}}-\bar E)\,
        \nabla_\theta \log|\psi|\big\rangle,

    with the (clipped) local energy entering as a stop-gradient baseline. The
    ``custom_jvp`` is also where :func:`register_log_amplitude` runs, so K-FAC's
    Fisher becomes the quantum geometric tensor. The returned callable has the
    ``(params, rng, batch) -> ((loss, aux), grads)`` signature
    ``kfac_jax.Optimizer`` expects from ``jax.value_and_grad(..., has_aux=True)``.

    Args:
        log_abs: Batched ``(params, walkers) -> log|psi|`` of shape ``(batch,)``.
        local_energy: Batched ``(params, walkers) -> E_loc`` of shape
            ``(batch,)``.
        clip_local_energy: Median-absolute-deviation clipping window; ``0``
            disables it.

    Returns:
        A ``value_and_grad`` function returning ``((energy, aux), grads)`` where
        ``aux`` carries the (clipped) per-walker local energies.
    """

    @jax.custom_jvp
    def total_energy(params: Params, rng: Array, walkers: Array) -> tuple[Array, dict[str, Any]]:
        del rng
        energies = local_energy(params, walkers)
        clipped = _clip_to_mad(energies, clip_local_energy)
        return jnp.mean(clipped), {"e_loc": clipped}

    @total_energy.defjvp
    def total_energy_jvp(  # pyright: ignore[reportUnusedFunction]  # registered as the JVP rule
        primals: tuple, tangents: tuple
    ) -> tuple:
        params, rng, walkers = primals
        del rng
        energies = local_energy(params, walkers)
        clipped = _clip_to_mad(energies, clip_local_energy)
        loss = jnp.mean(clipped)
        diff = clipped - loss

        # JVP of log|psi| w.r.t. the parameter tangent only (walkers are fixed).
        log_psi, log_psi_tangent = jax.jvp(lambda p: log_abs(p, walkers), (params,), (tangents[0],))
        register_log_amplitude(log_psi)

        n_walkers = walkers.shape[0]
        primals_out = (loss, {"e_loc": clipped})
        # The directional derivative of the score-function estimator: the factor
        # of 2 makes <2 (E_loc - E) d log|psi|> the energy gradient.
        loss_tangent = 2.0 * jnp.dot(log_psi_tangent, diff) / n_walkers
        tangents_out = (loss_tangent, {"e_loc": clipped})
        return primals_out, tangents_out

    return jax.value_and_grad(total_energy, argnums=0, has_aux=True)


class KFACPreconditioner:
    r"""K-FAC natural-gradient preconditioner wrapping ``kfac_jax.Optimizer``.

    Exposes the MinSR / SPRING optimiser seam -- ``init`` to build the optimiser
    state and ``step`` to apply one preconditioned update -- so it is a drop-in
    preconditioner choice for the VMC driver. Unlike the pure-function MinSR /
    SPRING updates (which return a flat update vector for an external learning
    rate), K-FAC is stateful: it accumulates an exponential moving average of the
    curvature and amortises the Kronecker-factor inversions, so the optimiser
    owns the learning-rate / momentum / damping application and its own ``jit``.

    The construction mirrors FermiNet (``../ferminet/ferminet/train.py``):
    ``value_func_has_aux=True`` (the loss returns ``(energy, aux)``),
    ``value_func_has_rng=True`` (the loss takes an rng), single-device, no burn-in
    steps, and ``estimation_mode='fisher_exact'``.

    Args:
        value_and_grad: The ``(params, rng, batch) -> ((loss, aux), grads)``
            function (see :func:`make_qmc_value_and_grad`).
        learning_rate: Fixed step size applied to the preconditioned update.
        damping: Tikhonov damping added to the curvature before inversion (the
            K-FAC analogue of the MinSR ``diag_shift``).
        momentum: Heavy-ball momentum coefficient.
        l2_reg: L2 regularisation strength on the parameters.
        norm_constraint: Optional trust-region cap on the update norm (KL bound).
        curvature_ema: Decay of the curvature exponential moving average.
        inverse_update_period: Steps between Kronecker-factor re-inversions.
    """

    def __init__(
        self,
        value_and_grad: ValueAndGradFn,
        *,
        learning_rate: float = 0.05,
        damping: float = 1e-3,
        momentum: float = 0.0,
        l2_reg: float = 0.0,
        norm_constraint: float | None = 1e-3,
        curvature_ema: float = 0.95,
        inverse_update_period: int = 1,
    ) -> None:
        self._damping = jnp.asarray(damping)
        self._momentum = jnp.asarray(momentum)
        self._optimizer = kfac_jax.Optimizer(
            value_and_grad,
            l2_reg=l2_reg,
            value_func_has_aux=True,
            value_func_has_rng=True,
            learning_rate_schedule=lambda _step: jnp.asarray(learning_rate),
            norm_constraint=norm_constraint,
            curvature_ema=curvature_ema,
            inverse_update_period=inverse_update_period,
            min_damping=1e-8,
            num_burnin_steps=0,
            estimation_mode="fisher_exact",
            multi_device=False,
        )

    def init(self, params: Params, key: Array, batch: Any) -> Any:
        """Initialise the K-FAC optimiser state for the given parameters.

        Args:
            params: The ansatz parameters (any array pytree, e.g. ``nnx.State``).
            key: PRNG key.
            batch: A representative batch passed to the loss to trace the
                curvature graph (e.g. the walker positions).

        Returns:
            The opaque ``kfac_jax.Optimizer.State`` pytree.
        """
        return self._optimizer.init(params, key, batch)

    def step(
        self, params: Params, state: Any, key: Array, batch: Any
    ) -> tuple[Params, Any, Array, dict[str, Any]]:
        """Apply one K-FAC preconditioned update.

        Args:
            params: The current parameters.
            state: The current optimiser state.
            key: PRNG key for this step.
            batch: The batch (walker positions) for the loss / curvature.

        Returns:
            A ``(new_params, new_state, loss, aux)`` tuple, where ``loss`` is the
            mean local energy and ``aux`` carries the per-walker local energies.
        """
        # ``Optimizer.step`` returns a 4-tuple ``(params, state, func_state,
        # stats)`` only when ``value_func_has_state=True``; here that flag is
        # False, so it returns the 3-tuple ``(params, state, stats)`` -- narrow
        # the static union accordingly.
        result = cast(
            "tuple[Params, Any, dict[str, Any]]",
            self._optimizer.step(
                params,
                state,
                key,
                batch=batch,
                momentum=self._momentum,
                damping=self._damping,
            ),
        )
        new_params, new_state, stats = result
        return new_params, new_state, stats["loss"], stats["aux"]


__all__ = [
    "KFACPreconditioner",
    "make_qmc_value_and_grad",
    "register_log_amplitude",
    "register_qmc_dense",
]
