r"""Shared building blocks for the SBI neural estimators.

NPE / NLE / NRE share four pieces of machinery that this module owns once
(Rule 1 — DRY) so each estimator keeps only its estimator-specific loss /
log-posterior closure:

* :class:`_SBIFittedState` — the fitted-state container (``train_losses`` /
  ``num_simulations`` array fields + a static ``metadata`` tuple). The public
  ``NPEState`` / ``NLEState`` / ``NREState`` subclass it so they stay distinct
  ``isinstance`` types while reusing one body and one ``validate()``.
* :func:`_resolve_sbi_backend` / :func:`_trigger_optional_backend_import` —
  backend resolution against the router ``"flow"`` / ``"sampler"`` families
  (the already-shared pattern, hoisted here from ``posterior_estimation``).
* :func:`_build_conditional_flow` — the Artifex ``ConditionalRealNVP``
  constructor (NPE conditions on ``x``; NLE swaps the roles).
* :func:`_train_loop` — the generic ``nnx.value_and_grad`` + ``nnx.jit``
  training loop driven by an estimator-specific ``loss_fn`` closure.
* :func:`_mcmc_posterior_predictive` — the BlackJAX MCMC predictive block
  shared verbatim by NLE and NRE.
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — eager per opifex convention
from typing import Any, ClassVar, TypeVar

import jax
import jax.numpy as jnp
from artifex.generative_models.core.configuration.flow_config import (
    ConditionalFlowConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.models.flow.conditional import ConditionalRealNVP
from flax import nnx, struct

from opifex.uncertainty._predictive import sample_based_predictive
from opifex.uncertainty.inference_backends.blackjax import BlackJAXBackend
from opifex.uncertainty.inference_backends.router import BackendRouter
from opifex.uncertainty.sbi.simulators import sample_joint, Simulator
from opifex.uncertainty.types import (
    metadata_to_dict,
    MetadataItems,
    PredictiveDistribution,
)


_DEFAULT_BACKEND_NAME: str = "RealNVP"
_TRAIN_STREAMS: tuple[str, ...] = ("sbi_train", "params", "default")
_SAMPLE_STREAMS: tuple[str, ...] = ("sbi_sample", "sample", "default")
_SBI_BACKEND_FAMILIES: tuple[str, ...] = ("flow", "sampler")

# Model type for the generic training loop: each estimator passes its own
# concrete ``nnx.Module`` subclass (flow / classifier) plus a matching loss
# closure, so the loop must be parametric to keep the two type-aligned.
_ModelT = TypeVar("_ModelT", bound=nnx.Module)


@struct.dataclass(slots=True, kw_only=True)
class _SBIFittedState:
    """Shared fitted-state container for the SBI neural estimators (pattern (B)).

    Array fields ``train_losses`` and ``num_simulations`` flow through
    ``jit`` / ``vmap``; the ``metadata`` tuple is static aux_data.
    ``validate()`` is public and not called from ``__post_init__`` or the
    pytree unflatten path (GUIDE_ALIGNMENT item 7).

    Subclasses set the class-level :attr:`_method_label` to tailor the
    divergence-diagnostic message without re-implementing :meth:`validate`.
    """

    train_losses: jax.Array
    num_simulations: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    _method_label: ClassVar[str] = "SBI"

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate the fitted state.

        Raises:
            ValueError: When ``train_losses`` is empty / all-NaN.

        """
        if self.train_losses.ndim != 1 or self.train_losses.shape[0] == 0:
            raise ValueError(
                f"train_losses must be a 1-d array of length >= 1; "
                f"got shape={self.train_losses.shape}."
            )
        if bool(jnp.all(jnp.isnan(self.train_losses))):
            raise ValueError(
                f"train_losses is entirely NaN — {self._method_label} training diverged. "
                "Reduce learning_rate, increase num_simulations, or check the simulator."
            )


def _resolve_sbi_backend(backend: str) -> Any:
    """Validate ``backend`` against the SBI-eligible router families.

    SBI accepts backends from two router families:

    * ``"flow"`` — density-estimator backends (``RealNVP``, ``MAF``,
      ``IAF``, ``Glow``, ``NeuralSplineFlow``, ``ConditionalFlow``,
      ``MADE``, ``bijx``, ``FlowJAX``).
    * ``"sampler"`` — flow-enhanced samplers and SBI workflows
      (``sbiax``, ``flowMC``).

    Returns the matching :class:`OptionalBackendSpec`. Raises
    :class:`ValueError` when the name is registered in neither family.
    Does NOT call :meth:`OptionalBackendSpec.instantiate` — installation
    is checked at ``fit`` time.
    """
    router = BackendRouter()
    errors: list[str] = []
    for family in _SBI_BACKEND_FAMILIES:
        try:
            return router.resolve(family, name=backend)
        except ValueError as exc:
            errors.append(str(exc))
    raise ValueError(
        f"unknown backend {backend!r} for SBI families {_SBI_BACKEND_FAMILIES!r}; "
        f"router responses: {errors}."
    )


def _trigger_optional_backend_import(backend: str) -> None:
    """Trigger ``ImportError`` for an uninstalled optional backend.

    Default Artifex backends always probe ``True`` (they are bundled);
    only the install-gated specs (``bijx``, ``FlowJAX``, ``sbiax``,
    ``flowMC``) can raise here.
    """
    spec = _resolve_sbi_backend(backend)
    if not spec.probe():
        spec.instantiate()  # raises ImportError with canonical hint


def _simulate_training_data(
    *,
    simulator: Simulator,
    num_simulations: int,
    backend: str,
    rngs: nnx.Rngs,
) -> tuple[jax.Array, jax.Array]:
    """Gate the optional backend, draw joint pairs, and unpack ``(theta, x)``.

    Shared ``fit`` preamble for all three estimators: triggers the optional
    backend's ``ImportError`` when a non-default backend is requested, then
    draws ``num_simulations`` joint ``(theta, x)`` pairs via
    :func:`sample_joint` and returns the unpacked arrays.

    Args:
        simulator: Static simulator description.
        num_simulations: Number of ``(theta, x)`` pairs to draw.
        backend: Selected density-estimator backend name.
        rngs: Caller-owned ``nnx.Rngs`` carrying the ``sbi_simulate`` stream.

    Returns:
        The ``(theta, x)`` training tensors.

    Raises:
        ImportError: When the requested optional backend is not installed.
    """
    if backend != _DEFAULT_BACKEND_NAME:
        _trigger_optional_backend_import(backend)
    batch = sample_joint(simulator, num_simulations=num_simulations, rngs=rngs)
    data = batch.data.value
    return data["theta"], data["x"]


def _build_conditional_flow(
    *,
    name: str,
    input_dim: int,
    condition_dim: int,
    hidden_dim: int,
    num_coupling_layers: int,
    rngs: nnx.Rngs,
) -> ConditionalRealNVP:
    """Build an Artifex ``ConditionalRealNVP`` density estimator.

    Args:
        name: Prefix for the coupling/flow config names (e.g. ``"npe"``).
        input_dim: Dimension of the modelled variable (NPE: ``theta``;
            NLE: ``x``).
        condition_dim: Dimension of the conditioning variable (NPE: ``x``;
            NLE: ``theta``).
        hidden_dim: Width of the coupling MLP hidden layers.
        num_coupling_layers: Coupling-layer depth.
        rngs: Parameter RNG stream for initialisation.
    """
    coupling = CouplingNetworkConfig(
        name=f"{name}_coupling",
        hidden_dims=(hidden_dim, hidden_dim),
        activation="relu",
    )
    cfg = ConditionalFlowConfig(
        name=f"{name}_flow",
        coupling_network=coupling,
        input_dim=input_dim,
        condition_dim=condition_dim,
        num_coupling_layers=num_coupling_layers,
        mask_type="checkerboard",
    )
    return ConditionalRealNVP(cfg, rngs=rngs)


def _train_loop(
    *,
    model: _ModelT,
    optimizer: nnx.Optimizer,
    loss_fn: Callable[[_ModelT], jax.Array],
    num_steps: int,
) -> jax.Array:
    """Run ``num_steps`` of full-batch Adam on ``model`` and stack the losses.

    Generic over the concrete ``nnx.Module`` subclass so each estimator's
    ``loss_fn`` stays type-aligned with the ``model`` it trains. The inner
    step is ``nnx.jit``-compiled (Rule: JAX/NNX transform compatibility);
    ``loss_fn`` is the estimator-specific objective closing over the training
    tensors. Returns the per-step loss as a stacked array.
    """

    @nnx.jit
    def step(m: _ModelT, opt: nnx.Optimizer) -> jax.Array:
        """Run one gradient-descent training step and return the batch loss."""
        loss, grads = nnx.value_and_grad(loss_fn)(m)
        opt.update(m, grads)
        return loss

    losses = [step(model, optimizer) for _ in range(num_steps)]
    return jnp.stack(losses)


def _mcmc_posterior_predictive(
    *,
    log_posterior: Callable[[jax.Array], jax.Array],
    theta_dim: int,
    num_samples: int,
    mcmc_samples: int,
    mcmc_burnin: int,
    mcmc_method: str,
    mcmc_step_size: float,
    sample_key: jax.Array,
    metadata: MetadataItems,
) -> PredictiveDistribution:
    """Run posterior MCMC over ``log_posterior`` and build the predictive.

    Shared by NLE and NRE: both reduce to ``log r(theta) + log prior`` (or
    ``log q(x|theta) + log prior``) and sample it with the BlackJAX backend.
    The sampler starts at the prior mean (zeros for a centred prior), keeps
    ``max(num_samples, mcmc_samples)`` draws, and returns the last
    ``num_samples`` as the predictive samples (reduced to ``mean`` /
    ``variance`` by :func:`sample_based_predictive`).

    Args:
        log_posterior: Scalar unnormalised log-posterior at a single ``theta``.
        theta_dim: Parameter-space dimension (sets the initial state shape).
        num_samples: Number of trailing posterior draws to retain.
        mcmc_samples: Minimum posterior-sample budget for the sampler.
        mcmc_burnin: Warmup samples to discard.
        mcmc_method: BlackJAX sampler family (``"nuts"`` / ``"hmc"`` /
            ``"mala"``).
        mcmc_step_size: BlackJAX step size.
        sample_key: PRNG key forwarded to the backend's ``sample`` stream.
        metadata: Provenance tuple stamped onto the predictive.
    """
    init_state = jnp.zeros((theta_dim,))
    sampler = BlackJAXBackend(
        target_log_prob=log_posterior,
        init_state=init_state,
        n_samples=max(num_samples, mcmc_samples),
        n_burnin=mcmc_burnin,
        method=mcmc_method,
        step_size=mcmc_step_size,
    )
    # The MCMC backend creates its own RNG stream; forward the user's key so
    # it can pull ``sample`` / ``default``.
    result = sampler.fit(log_posterior, rngs=nnx.Rngs(sample=sample_key))
    samples = jnp.asarray(result.sampler_state)[-num_samples:]
    return sample_based_predictive(samples, metadata=metadata)


__all__ = [
    "_DEFAULT_BACKEND_NAME",
    "_SAMPLE_STREAMS",
    "_TRAIN_STREAMS",
    "_SBIFittedState",
    "_build_conditional_flow",
    "_mcmc_posterior_predictive",
    "_resolve_sbi_backend",
    "_simulate_training_data",
    "_train_loop",
    "_trigger_optional_backend_import",
]
