r"""Neural posterior estimation (NPE) for the SBI subsystem.

NPE fits a conditional density estimator :math:`q_\phi(\theta \mid x)`
on joint simulation samples ``(theta, x)`` drawn from the prior and
simulator. The fitted estimator yields posterior samples for any
observation by conditioning on ``x_obs``.

The default density-estimator backend wraps Artifex's NNX-native
:class:`artifex.generative_models.models.flow.conditional.ConditionalRealNVP`.
Optional backends (``bijx`` / ``sbiax`` / ``flowMC``) are routed through
:class:`opifex.uncertainty.inference_backends.router.BackendRouter` and
raise :class:`ImportError` with the canonical install hint when not
installed.

References
----------
* Greenberg, Nonnenmacher, Macke (2019) — Automatic Posterior
  Transformation for Likelihood-free Inference, ``arXiv:1905.07488``.
* Lueckmann+ (2017) — Flexible statistical inference for mechanistic
  models of neural dynamics, ``arXiv:1711.01861``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import optax
from artifex.generative_models.core.configuration.flow_config import (
    ConditionalFlowConfig,
    CouplingNetworkConfig,
)
from artifex.generative_models.core.rng import extract_rng_key
from artifex.generative_models.models.flow.conditional import ConditionalRealNVP
from flax import nnx, struct

from opifex.uncertainty.inference_backends.router import BackendRouter
from opifex.uncertainty.sbi.simulators import sample_joint, Simulator
from opifex.uncertainty.types import (
    metadata_to_dict,
    MetadataItems,
    PredictiveDistribution,
    require_fitted_state,
)


_DEFAULT_BACKEND_NAME: str = "RealNVP"
_TRAIN_STREAMS: tuple[str, ...] = ("sbi_train", "params", "default")
_SAMPLE_STREAMS: tuple[str, ...] = ("sbi_sample", "sample", "default")


@struct.dataclass(slots=True, kw_only=True)
class NPEState:
    """Fitted-state container for :class:`NeuralPosteriorEstimator` (pattern (B)).

    Array fields ``train_losses`` and ``num_simulations`` flow through
    ``jit`` / ``vmap``; the ``metadata`` tuple is static aux_data.
    ``validate()`` is public and not called from ``__post_init__`` or the
    pytree unflatten path (GUIDE_ALIGNMENT item 7).
    """

    train_losses: jax.Array
    num_simulations: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

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
                "train_losses is entirely NaN — NPE training diverged. "
                "Reduce learning_rate, increase num_simulations, or check the simulator."
            )


_SBI_BACKEND_FAMILIES: tuple[str, ...] = ("flow", "sampler")


def _resolve_sbi_backend(backend: str) -> Any:
    """Validate ``backend`` against the SBI-eligible families.

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


def _build_conditional_flow(
    *,
    theta_dim: int,
    x_dim: int,
    hidden_dim: int,
    num_coupling_layers: int,
    rngs: nnx.Rngs,
) -> ConditionalRealNVP:
    coupling = CouplingNetworkConfig(
        name="npe_coupling",
        hidden_dims=(hidden_dim, hidden_dim),
        activation="relu",
    )
    cfg = ConditionalFlowConfig(
        name="npe_flow",
        coupling_network=coupling,
        input_dim=theta_dim,
        condition_dim=x_dim,
        num_coupling_layers=num_coupling_layers,
        mask_type="checkerboard",
    )
    return ConditionalRealNVP(cfg, rngs=rngs)


@dataclasses.dataclass(slots=True)
class NeuralPosteriorEstimator:
    """Neural Posterior Estimator with a conditional-flow density estimator.

    Args:
        theta_dim: Parameter-space dimension.
        x_dim: Observation-space dimension (post-summary if a
            ``summary_fn`` is configured on the simulator).
        backend: Density-estimator backend name; must be one of the names
            registered in :class:`BackendRouter` ``"flow"`` family.
            Default ``"RealNVP"`` uses Artifex ``ConditionalRealNVP``.
        num_steps: Number of training steps (full-batch Adam).
        learning_rate: Adam learning rate.
        hidden_dim: Width of the coupling MLP hidden layers.
        num_coupling_layers: Coupling-layer depth in the conditional flow.

    Public state:

        ``state`` — fitted :class:`NPEState` after ``fit``; ``None`` otherwise.

    """

    theta_dim: int
    x_dim: int
    backend: str = _DEFAULT_BACKEND_NAME
    num_steps: int = 100
    learning_rate: float = 1e-3
    hidden_dim: int = 32
    num_coupling_layers: int = 4

    # Populated by ``fit``. Held as plain attributes (not @struct fields)
    # because the estimator is a Python-level orchestrator; the trained
    # array surface lives in ``state``.
    state: NPEState | None = dataclasses.field(default=None, init=False)
    _flow: ConditionalRealNVP | None = dataclasses.field(default=None, init=False)

    def __post_init__(self) -> None:
        """Resolve and validate the backend at construction time."""
        # Trigger ValueError for unknown backends eagerly; ImportError for
        # uninstalled-optional backends is deferred to ``fit`` so the
        # error timing matches the plan's TDD contract (importing the
        # missing backend happens only when the caller actually tries to
        # fit, mirroring lazy optional-dependency behaviour).
        _resolve_sbi_backend(self.backend)

    def fit(
        self,
        simulator: Simulator,
        num_simulations: int,
        *,
        rngs: nnx.Rngs,
    ) -> NeuralPosteriorEstimator:
        """Fit the conditional flow on ``num_simulations`` ``(theta, x)`` pairs.

        Deterministic given a fixed ``rngs``. Returns ``self`` with
        ``self.state`` populated.

        Raises:
            ImportError: When the requested optional backend is not installed.

        """
        if self.backend != _DEFAULT_BACKEND_NAME:
            _trigger_optional_backend_import(self.backend)

        # Simulate joint pairs.
        batch = sample_joint(simulator, num_simulations=num_simulations, rngs=rngs)
        data = batch.data.value
        theta = data["theta"]
        x = data["x"]

        # Build and train the conditional flow.
        train_key = extract_rng_key(
            rngs, streams=_TRAIN_STREAMS, context="NeuralPosteriorEstimator.fit"
        )
        flow = _build_conditional_flow(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            num_coupling_layers=self.num_coupling_layers,
            rngs=nnx.Rngs(params=train_key),
        )

        optimizer = nnx.Optimizer(flow, optax.adam(self.learning_rate), wrt=nnx.Param)
        losses = _train_conditional_flow(
            flow=flow, optimizer=optimizer, theta=theta, x=x, num_steps=self.num_steps
        )

        self._flow = flow
        self.state = NPEState(
            train_losses=losses,
            num_simulations=jnp.asarray(num_simulations),
            metadata=(("method", "npe"), ("backend", self.backend)),
        )
        return self

    def predict_distribution(
        self,
        observation: jax.Array,
        *,
        rngs: nnx.Rngs,
        num_samples: int,
    ) -> PredictiveDistribution:
        """Sample the fitted posterior ``q(theta | observation)``."""
        require_fitted_state(self.state, surface="NeuralPosteriorEstimator.predict_distribution")
        flow = require_fitted_state(
            self._flow, surface="NeuralPosteriorEstimator.predict_distribution"
        )
        sample_key = extract_rng_key(
            rngs, streams=_SAMPLE_STREAMS, context="NeuralPosteriorEstimator.predict_distribution"
        )
        samples = _sample_posterior(
            flow=flow, observation=observation, num_samples=num_samples, key=sample_key
        )
        mean = jnp.mean(samples, axis=0)
        variance = jnp.var(samples, axis=0)
        return PredictiveDistribution(
            mean=mean,
            variance=variance,
            samples=samples,
            metadata=(
                ("method", "npe"),
                ("backend", self.backend),
                ("num_samples", num_samples),
            ),
        )

    def refine_round(
        self,
        simulator: Simulator,
        *,
        observation: jax.Array,
        num_simulations: int,
        num_steps: int,
        rngs: nnx.Rngs,
    ) -> NeuralPosteriorEstimator:
        """Run one sequential SBI round centred on ``observation``.

        Sequential NPE narrows the proposal toward the current posterior:
        we draw the new round's parameter proposals from
        ``q(theta | observation)`` (the current flow) and train a fresh
        flow on the resulting ``(theta, x)`` pairs. With informative
        observations the posterior tightens across rounds.
        """
        require_fitted_state(self.state, surface="NeuralPosteriorEstimator.refine_round")
        current_flow = require_fitted_state(
            self._flow, surface="NeuralPosteriorEstimator.refine_round"
        )
        proposal_key = extract_rng_key(
            rngs,
            streams=("sbi_simulate", "sample", "default"),
            context="NeuralPosteriorEstimator.refine_round",
        )
        sim_key = extract_rng_key(
            rngs,
            streams=("sbi_simulate", "sample", "default"),
            context="NeuralPosteriorEstimator.refine_round.simulate",
        )
        theta = _sample_posterior(
            flow=current_flow,
            observation=observation,
            num_samples=num_simulations,
            key=proposal_key,
        )
        x = simulator.simulate_fn(sim_key, theta)
        if simulator.summary_fn is not None:
            x = simulator.summary_fn(x)

        train_key = extract_rng_key(
            rngs,
            streams=_TRAIN_STREAMS,
            context="NeuralPosteriorEstimator.refine_round.train",
        )
        flow = _build_conditional_flow(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            num_coupling_layers=self.num_coupling_layers,
            rngs=nnx.Rngs(params=train_key),
        )
        optimizer = nnx.Optimizer(flow, optax.adam(self.learning_rate), wrt=nnx.Param)
        losses = _train_conditional_flow(
            flow=flow, optimizer=optimizer, theta=theta, x=x, num_steps=num_steps
        )
        refined = NeuralPosteriorEstimator(
            theta_dim=self.theta_dim,
            x_dim=self.x_dim,
            backend=self.backend,
            num_steps=num_steps,
            learning_rate=self.learning_rate,
            hidden_dim=self.hidden_dim,
            num_coupling_layers=self.num_coupling_layers,
        )
        refined._flow = flow
        refined.state = NPEState(
            train_losses=losses,
            num_simulations=jnp.asarray(num_simulations),
            metadata=(("method", "npe"), ("backend", self.backend), ("sequential_round", 1)),
        )
        return refined


def _train_conditional_flow(
    *,
    flow: ConditionalRealNVP,
    optimizer: nnx.Optimizer,
    theta: jax.Array,
    x: jax.Array,
    num_steps: int,
) -> jax.Array:
    """Train ``flow`` to maximize ``log q(theta | x)`` for ``num_steps`` steps.

    Returns the per-step loss as a stacked array. Uses ``nnx.jit`` so the
    inner step is JIT-compiled (Rule: JAX/NNX transform compatibility).
    """

    @nnx.jit
    def step(model: ConditionalRealNVP, opt: nnx.Optimizer) -> jax.Array:
        def loss_fn(m: ConditionalRealNVP) -> jax.Array:
            return -jnp.mean(m.log_prob(theta, condition=x))

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        opt.update(model, grads)
        return loss

    losses = [step(flow, optimizer) for _ in range(num_steps)]
    return jnp.stack(losses)


def _sample_posterior(
    *,
    flow: ConditionalRealNVP,
    observation: jax.Array,
    num_samples: int,
    key: jax.Array,
) -> jax.Array:
    """Sample ``num_samples`` draws from the fitted conditional flow."""
    return flow.generate(
        n_samples=num_samples,
        condition=observation,
        rngs=nnx.Rngs(default=key),
    )


__all__ = ["NPEState", "NeuralPosteriorEstimator"]
