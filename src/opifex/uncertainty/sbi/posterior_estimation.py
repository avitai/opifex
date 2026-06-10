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

The fitted-state container, backend resolution, conditional-flow
construction, and training loop are shared with NLE / NRE via
:mod:`opifex.uncertainty.sbi._base`; NPE keeps only its posterior-flow
loss and sampling closures.

References
----------
* Greenberg, Nonnenmacher, Macke (2019) — Automatic Posterior
  Transformation for Likelihood-free Inference, ``arXiv:1905.07488``.
* Lueckmann+ (2017) — Flexible statistical inference for mechanistic
  models of neural dynamics, ``arXiv:1711.01861``.
"""

from __future__ import annotations

import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp
import optax
from artifex.generative_models.core.rng import extract_rng_key
from artifex.generative_models.models.flow.conditional import (  # noqa: TC002 — pyproject dep, kept eager
    ConditionalRealNVP,
)
from flax import nnx, struct

from opifex.uncertainty._predictive import sample_based_predictive
from opifex.uncertainty.sbi._base import (
    _build_conditional_flow,
    _DEFAULT_BACKEND_NAME,
    _resolve_sbi_backend,
    _SAMPLE_STREAMS,
    _SBIFittedState,
    _simulate_training_data,
    _train_loop,
    _TRAIN_STREAMS,
)
from opifex.uncertainty.sbi.simulators import Simulator  # noqa: TC001 — eager per opifex convention
from opifex.uncertainty.types import (
    PredictiveDistribution,
    require_fitted_state,
)


@struct.dataclass(slots=True, kw_only=True)
class NPEState(_SBIFittedState):
    """Fitted-state container for :class:`NeuralPosteriorEstimator` (pattern (B)).

    Shares its body and ``validate()`` with the other SBI estimators via
    :class:`opifex.uncertainty.sbi._base._SBIFittedState`; only the
    divergence-diagnostic label differs.
    """

    _method_label: ClassVar[str] = "NPE"


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

    def _build_flow(self, *, train_key: jax.Array) -> ConditionalRealNVP:
        """Construct the NPE conditional flow ``q(theta | x)``."""
        return _build_conditional_flow(
            name="npe",
            input_dim=self.theta_dim,
            condition_dim=self.x_dim,
            hidden_dim=self.hidden_dim,
            num_coupling_layers=self.num_coupling_layers,
            rngs=nnx.Rngs(params=train_key),
        )

    def _train_flow(
        self, *, flow: ConditionalRealNVP, theta: jax.Array, x: jax.Array, num_steps: int
    ) -> jax.Array:
        """Train ``flow`` to maximise ``log q(theta | x)`` and return the losses."""
        optimizer = nnx.Optimizer(flow, optax.adam(self.learning_rate), wrt=nnx.Param)

        def loss_fn(model: ConditionalRealNVP) -> jax.Array:
            """Negative mean log-posterior of ``theta`` under the flow conditioned on ``x``."""
            return -jnp.mean(model.log_prob(theta, condition=x))

        return _train_loop(model=flow, optimizer=optimizer, loss_fn=loss_fn, num_steps=num_steps)

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
        theta, x = _simulate_training_data(
            simulator=simulator,
            num_simulations=num_simulations,
            backend=self.backend,
            rngs=rngs,
        )

        # Build and train the conditional flow.
        train_key = extract_rng_key(
            rngs, streams=_TRAIN_STREAMS, context="NeuralPosteriorEstimator.fit"
        )
        flow = self._build_flow(train_key=train_key)
        losses = self._train_flow(flow=flow, theta=theta, x=x, num_steps=self.num_steps)

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
        return sample_based_predictive(
            samples,
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
        flow = self._build_flow(train_key=train_key)
        losses = self._train_flow(flow=flow, theta=theta, x=x, num_steps=num_steps)
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
