"""Coordinatewise learned optimisers (meta-learned update rules).

A :class:`LearnedOptimizer` carries meta-parameters ``theta`` and an ``opt_fn(theta)`` that
bakes ``theta`` into an :class:`~opifex.optimization.l2o.optimizers.Optimizer`. ``theta`` is a
plain pytree (the optimiser MLP's :class:`flax.nnx` state, obtained via ``nnx.split``), so it
is directly perturbable/vmappable for evolution-strategies meta-training (PES).

:class:`MLPLearnedOptimizer` is the per-parameter MLP design of Metz et al. 2020
(``arXiv:2009.11243``; "LOLv2", ``learned_optimization/learned_optimizers/mlp_lopt.py``): a tiny
MLP, shared across all scalar parameters, maps a 19-feature per-parameter vector to a
``(direction, magnitude)`` pair, and the update is
``step = direction * exp(magnitude * exp_mult) * step_mult``. The richer Adafactor-feature
variant (``adafac_mlp_lopt.py``) extends the same scaffolding and is added on top of this base.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx, struct

from opifex.optimization.l2o.features import (
    ADAFAC_DECAYS,
    ADAFAC_MOMENTUM_DECAYS,
    ADAFAC_RMS_DECAYS,
    init_adafactor_accum,
    init_ema,
    MOMENTUM_DECAYS,
    safe_rsqrt,
    second_moment_normalize,
    tanh_time_embedding,
    TANH_TIMESCALES,
    update_adafactor_accum,
    update_momentum,
    update_rms,
)
from opifex.optimization.l2o.optimizers import Optimizer


if TYPE_CHECKING:
    from jaxtyping import PyTree


# 8 normalised features (grad, param, 6-decay momentum) + 11 tanh time-embedding features.
_NUM_FEATURES = 8 + len(TANH_TIMESCALES)


class LearnedOptimizer(abc.ABC):
    """A meta-learned optimiser: ``init`` samples ``theta``; ``opt_fn(theta)`` applies it."""

    @abc.abstractmethod
    def init(self, key: jax.Array) -> nnx.State:
        """Sample the meta-parameters ``theta`` (the optimiser MLP state)."""

    @abc.abstractmethod
    def opt_fn(self, theta: nnx.State) -> Optimizer:
        """Return an :class:`Optimizer` whose update rule is parameterised by ``theta``."""


class MLPLOptState(struct.PyTreeNode):
    """Inner state for learned optimisers: optimisee params, momentum EMAs, iteration."""

    params: PyTree
    momentum: PyTree
    iteration: jax.Array


class _ScalarParam(nnx.Module):
    """Holds a single learnable scalar as an ``nnx.Param`` (e.g. a learnable log-lr)."""

    def __init__(self, value: float) -> None:
        """Store ``value`` as the lone meta-parameter."""
        self.value = nnx.Param(jnp.asarray(value, dtype=jnp.float32))


class _SGDInstance(Optimizer[MLPLOptState]):
    """SGD with a (meta-learned) learning rate; ``params -= exp(log_lr) * grad``."""

    def __init__(self, log_lr: jax.Array) -> None:
        """Close over the (traced) log learning rate."""
        self._log_lr = log_lr

    def init(
        self,
        params: PyTree,
        *,
        num_steps: int | None = None,  # noqa: ARG002 - uniform Optimizer interface
        key: jax.Array | None = None,  # noqa: ARG002 - uniform Optimizer interface
    ) -> MLPLOptState:
        """Initialise with empty momentum (unused) and a zero iteration counter."""
        return MLPLOptState(params=params, momentum=(), iteration=jnp.zeros((), jnp.int32))

    def update(
        self,
        state: MLPLOptState,
        grad: PyTree,
        *,
        loss: jax.Array | None = None,  # noqa: ARG002 - SGD ignores loss
    ) -> MLPLOptState:
        """Take one scaled-gradient step with learning rate ``exp(log_lr)``."""
        learning_rate = jnp.exp(self._log_lr)
        new_params = jax.tree.map(lambda p, g: p - learning_rate * g, state.params, grad)
        return MLPLOptState(
            params=new_params, momentum=state.momentum, iteration=state.iteration + 1
        )

    def get_params(self, state: MLPLOptState) -> PyTree:
        """Return the current parameters."""
        return state.params


class LearnableSGD(LearnedOptimizer):
    """SGD with a single learnable log learning rate (``learned_optimizers/base.LearnableSGD``).

    The simplest learned optimiser: ``theta`` is one scalar (``log_lr``). Useful as a
    meta-training smoke optimiser and for validating the meta-gradient estimator, since the
    full-unroll meta-gradient w.r.t. ``log_lr`` is analytically differentiable.
    """

    def __init__(self, initial_learning_rate: float = 0.1) -> None:
        """Capture the initial log-lr and the (static) scalar-param graph definition."""
        self._initial_log_lr = float(jnp.log(jnp.asarray(initial_learning_rate)))
        self._graphdef = nnx.split(_ScalarParam(self._initial_log_lr))[0]

    def init(self, key: jax.Array) -> nnx.State:  # noqa: ARG002 - deterministic init
        """Return ``theta`` = the initial log learning rate (deterministic)."""
        return nnx.split(_ScalarParam(self._initial_log_lr))[1]

    def opt_fn(self, theta: nnx.State) -> Optimizer:
        """Bake ``theta`` (the log-lr) into an SGD optimiser."""
        log_lr = nnx.merge(self._graphdef, theta).value.value
        return _SGDInstance(log_lr)


class _OptimizerMLP(nnx.Module):
    """Per-parameter MLP shared across all scalar parameters: features -> ``(dir, mag)``."""

    def __init__(
        self, num_features: int, hidden_size: int, hidden_layers: int, *, rngs: nnx.Rngs
    ) -> None:
        """Build an MLP ``num_features -> [hidden_size]*hidden_layers -> 2``."""
        dims = [num_features, *([hidden_size] * hidden_layers), 2]
        # nnx.List (not a plain list) so the sublayers are tracked as NNX state.
        self.layers = nnx.List(
            [nnx.Linear(dims[i], dims[i + 1], rngs=rngs) for i in range(len(dims) - 1)]
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        """Map a ``(..., num_features)`` feature batch to ``(..., 2)`` outputs."""
        for index, layer in enumerate(self.layers):
            x = layer(x)
            if index < len(self.layers) - 1:
                x = nnx.relu(x)
        return x


def _per_param_step(
    mlp: _OptimizerMLP,
    raw_features: jax.Array,
    time_embedding: jax.Array,
    *,
    exp_mult: float,
    step_mult: float,
) -> jax.Array:
    """Shared lopt update head: per-tensor-normalise, append time embedding, MLP, step rule.

    ``raw_features`` is ``(num_elements, num_raw_features)`` for one parameter tensor. The features
    are second-moment-normalised across the tensor (the un-normalised time embedding is appended
    afterwards), passed through the per-parameter ``mlp``, and mapped to a per-element step via
    ``direction * exp(magnitude * exp_mult) * step_mult`` (``mlp_lopt.py:176``). Returns the flat
    per-element step (the caller reshapes it to the parameter shape).
    """
    normalized = second_moment_normalize(raw_features, axis=0)
    time_block = jnp.broadcast_to(time_embedding, (normalized.shape[0], time_embedding.size))
    feature_vectors = jnp.concatenate([normalized, time_block], axis=-1)
    outputs = mlp(feature_vectors)
    return outputs[:, 0] * jnp.exp(outputs[:, 1] * exp_mult) * step_mult


class _MLPLearnedOptimizerInstance(Optimizer[MLPLOptState]):
    """An :class:`Optimizer` whose per-parameter update is the MLP defined by ``theta``."""

    def __init__(
        self,
        graphdef: nnx.GraphDef,
        theta: nnx.State,
        *,
        exp_mult: float,
        step_mult: float,
    ) -> None:
        """Close over the MLP ``graphdef`` (static) and meta-parameters ``theta``."""
        self._graphdef = graphdef
        self._theta = theta
        self._exp_mult = exp_mult
        self._step_mult = step_mult

    def init(
        self,
        params: PyTree,
        *,
        num_steps: int | None = None,  # noqa: ARG002 - uniform Optimizer interface
        key: jax.Array | None = None,  # noqa: ARG002 - uniform Optimizer interface
    ) -> MLPLOptState:
        """Zero the momentum EMAs and the iteration counter for ``params``."""
        momentum = jax.tree.map(lambda p: init_ema(p, MOMENTUM_DECAYS.size), params)
        return MLPLOptState(params=params, momentum=momentum, iteration=jnp.zeros((), jnp.int32))

    def update(
        self,
        state: MLPLOptState,
        grad: PyTree,
        *,
        loss: jax.Array | None = None,  # noqa: ARG002 - this lopt does not consume loss
    ) -> MLPLOptState:
        """Apply the per-parameter learned update across every parameter tensor."""
        mlp = nnx.merge(self._graphdef, self._theta)
        new_momentum = jax.tree.map(update_momentum, state.momentum, grad)
        time_embedding = tanh_time_embedding(state.iteration.astype(jnp.float32))

        def update_leaf(param: jax.Array, gradient: jax.Array, momentum: jax.Array) -> jax.Array:
            flat_param = param.reshape(-1)
            flat_grad = gradient.reshape(-1)
            flat_mom = momentum.reshape(-1, MOMENTUM_DECAYS.size)
            # Per-parameter features (grad, param, multi-decay momentum); normalised + stepped
            # by the shared head.
            raw = jnp.concatenate([flat_grad[:, None], flat_param[:, None], flat_mom], axis=-1)
            step = _per_param_step(
                mlp, raw, time_embedding, exp_mult=self._exp_mult, step_mult=self._step_mult
            )
            return param - step.reshape(param.shape)

        new_params = jax.tree.map(update_leaf, state.params, grad, new_momentum)
        return MLPLOptState(params=new_params, momentum=new_momentum, iteration=state.iteration + 1)

    def get_params(self, state: MLPLOptState) -> PyTree:
        """Return the current optimisee parameters."""
        return state.params


class MLPLearnedOptimizer(LearnedOptimizer):
    """Per-parameter MLP learned optimiser (LOLv2; Metz et al. 2020).

    ``theta`` is the MLP's ``nnx`` parameter state; ``opt_fn(theta)`` applies it coordinatewise
    with the ``direction * exp(magnitude * exp_mult) * step_mult`` update.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        hidden_layers: int = 2,
        exp_mult: float = 1e-3,
        step_mult: float = 1e-3,
    ) -> None:
        """Configure the optimiser MLP and capture its (static) graph definition."""
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.exp_mult = exp_mult
        self.step_mult = step_mult
        # Capture the config-determined graphdef once (init() supplies fresh theta).
        template = _OptimizerMLP(_NUM_FEATURES, hidden_size, hidden_layers, rngs=nnx.Rngs(0))
        self._graphdef = nnx.split(template)[0]

    def init(self, key: jax.Array) -> nnx.State:
        """Sample fresh MLP meta-parameters ``theta`` from ``key``."""
        mlp = _OptimizerMLP(_NUM_FEATURES, self.hidden_size, self.hidden_layers, rngs=nnx.Rngs(key))
        return nnx.split(mlp)[1]

    def opt_fn(self, theta: nnx.State) -> Optimizer:
        """Bake ``theta`` into a coordinatewise :class:`Optimizer`."""
        return _MLPLearnedOptimizerInstance(
            self._graphdef, theta, exp_mult=self.exp_mult, step_mult=self.step_mult
        )


# Adafactor per-element feature count: grad + param (2), momentum + m*rsqrt(rms) (2 * n_mom),
# rms + rsqrt(rms) (2 * n_rms), and 6 factored blocks (fac_g, row, col, rsqrt_row, rsqrt_col,
# fac_mom_mult) of n_adafactor each — see ``adafac_mlp_lopt._mod``.
_NUM_ADAFAC_FEATURES = (
    2 + 2 * ADAFAC_MOMENTUM_DECAYS.size + 2 * ADAFAC_RMS_DECAYS.size + 6 * ADAFAC_DECAYS.size
)
_NUM_ADAFAC_INPUTS = _NUM_ADAFAC_FEATURES + len(TANH_TIMESCALES)


class AdafacMLPLOptState(struct.PyTreeNode):
    """Inner state for the Adafactor-MLP lopt: params, momentum/RMS EMAs, factored accums, step.

    ``mom``/``rms`` are multi-decay EMA trees; ``v_row``/``v_col``/``v_diag`` are the Adafactor
    factored second-moment accumulators (one tree each, matching the params structure).
    """

    params: PyTree
    mom: PyTree
    rms: PyTree
    v_row: PyTree
    v_col: PyTree
    v_diag: PyTree
    iteration: jax.Array


class _AdafacMLPInstance(Optimizer[AdafacMLPLOptState]):
    """Adafactor-feature per-parameter learned optimiser parameterised by ``theta``."""

    def __init__(
        self, graphdef: nnx.GraphDef, theta: nnx.State, *, exp_mult: float, step_mult: float
    ) -> None:
        """Close over the MLP ``graphdef`` (static) and meta-parameters ``theta``."""
        self._graphdef = graphdef
        self._theta = theta
        self._exp_mult = exp_mult
        self._step_mult = step_mult

    def init(
        self,
        params: PyTree,
        *,
        num_steps: int | None = None,  # noqa: ARG002 - uniform Optimizer interface
        key: jax.Array | None = None,  # noqa: ARG002 - uniform Optimizer interface
    ) -> AdafacMLPLOptState:
        """Zero the momentum/RMS EMAs and the factored accumulators for ``params``."""
        mom = jax.tree.map(lambda p: init_ema(p, ADAFAC_MOMENTUM_DECAYS.size), params)
        rms = jax.tree.map(lambda p: init_ema(p, ADAFAC_RMS_DECAYS.size), params)
        v_row = jax.tree.map(lambda p: init_adafactor_accum(p, ADAFAC_DECAYS.size)[0], params)
        v_col = jax.tree.map(lambda p: init_adafactor_accum(p, ADAFAC_DECAYS.size)[1], params)
        v_diag = jax.tree.map(lambda p: init_adafactor_accum(p, ADAFAC_DECAYS.size)[2], params)
        return AdafacMLPLOptState(
            params=params,
            mom=mom,
            rms=rms,
            v_row=v_row,
            v_col=v_col,
            v_diag=v_diag,
            iteration=jnp.zeros((), jnp.int32),
        )

    def update(
        self,
        state: AdafacMLPLOptState,
        grad: PyTree,
        *,
        loss: jax.Array | None = None,  # noqa: ARG002 - this lopt does not consume loss
    ) -> AdafacMLPLOptState:
        """Apply the Adafactor-feature learned update across every parameter tensor."""
        mlp = nnx.merge(self._graphdef, self._theta)
        time_embedding = tanh_time_embedding(state.iteration.astype(jnp.float32))

        # Per-leaf processing in the original tensor shape (factored features need the 2D
        # structure); flatten/zip/unflatten keeps the parallel accumulator trees aligned.
        param_leaves, treedef = jax.tree.flatten(state.params)
        leaves = zip(
            param_leaves,
            jax.tree.leaves(grad),
            jax.tree.leaves(state.mom),
            jax.tree.leaves(state.rms),
            jax.tree.leaves(state.v_row),
            jax.tree.leaves(state.v_col),
            jax.tree.leaves(state.v_diag),
            strict=True,
        )

        new_params, new_mom, new_rms, new_v_row, new_v_col, new_v_diag = [], [], [], [], [], []
        for param, gradient, mom, rms, v_row, v_col, v_diag in leaves:
            next_mom = update_momentum(mom, gradient, ADAFAC_MOMENTUM_DECAYS)
            next_rms = update_rms(rms, gradient, ADAFAC_RMS_DECAYS)
            nvr, nvc, nvd, fac_g, row_feat, col_feat, factor = update_adafactor_accum(
                v_row, v_col, v_diag, gradient, ADAFAC_DECAYS
            )
            rsqrt_rms = jax.lax.rsqrt(next_rms + 1e-6)
            raw = jnp.concatenate(
                [
                    gradient[..., None],
                    param[..., None],
                    next_mom,
                    next_rms,
                    next_mom * rsqrt_rms,  # m * rsqrt(rms) (n_rms broadcasts over n_mom)
                    rsqrt_rms,
                    fac_g,
                    row_feat,
                    col_feat,
                    safe_rsqrt(row_feat + 1e-8),
                    safe_rsqrt(col_feat + 1e-8),
                    next_mom[..., -1:] * factor,  # slowest momentum, Adafactor-preconditioned
                ],
                axis=-1,
            )
            step = _per_param_step(
                mlp,
                raw.reshape(-1, _NUM_ADAFAC_FEATURES),
                time_embedding,
                exp_mult=self._exp_mult,
                step_mult=self._step_mult,
            )
            new_params.append(param - step.reshape(param.shape))
            new_mom.append(next_mom)
            new_rms.append(next_rms)
            new_v_row.append(nvr)
            new_v_col.append(nvc)
            new_v_diag.append(nvd)

        return AdafacMLPLOptState(
            params=treedef.unflatten(new_params),
            mom=treedef.unflatten(new_mom),
            rms=treedef.unflatten(new_rms),
            v_row=treedef.unflatten(new_v_row),
            v_col=treedef.unflatten(new_v_col),
            v_diag=treedef.unflatten(new_v_diag),
            iteration=state.iteration + 1,
        )

    def get_params(self, state: AdafacMLPLOptState) -> PyTree:
        """Return the current optimisee parameters."""
        return state.params


class AdafacMLPLearnedOptimizer(LearnedOptimizer):
    """Adafactor-feature per-parameter MLP learned optimiser (``adafac_mlp_lopt.py``).

    Extends :class:`MLPLearnedOptimizer` with Adafactor-style inputs — multi-decay RMS,
    ``m * rsqrt(rms)``, ``rsqrt(rms)``, and factored row/column second-moment features (Metz et al.
    2020). ``theta`` is the optimiser MLP's ``nnx`` state; the update head is the same
    ``direction * exp(magnitude * exp_mult) * step_mult`` rule.
    """

    def __init__(
        self,
        hidden_size: int = 32,
        hidden_layers: int = 2,
        exp_mult: float = 1e-3,
        step_mult: float = 1e-3,
    ) -> None:
        """Configure the optimiser MLP and capture its (static) graph definition."""
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.exp_mult = exp_mult
        self.step_mult = step_mult
        template = _OptimizerMLP(_NUM_ADAFAC_INPUTS, hidden_size, hidden_layers, rngs=nnx.Rngs(0))
        self._graphdef = nnx.split(template)[0]

    def init(self, key: jax.Array) -> nnx.State:
        """Sample fresh MLP meta-parameters ``theta`` from ``key``."""
        mlp = _OptimizerMLP(
            _NUM_ADAFAC_INPUTS, self.hidden_size, self.hidden_layers, rngs=nnx.Rngs(key)
        )
        return nnx.split(mlp)[1]

    def opt_fn(self, theta: nnx.State) -> Optimizer:
        """Bake ``theta`` into a coordinatewise Adafactor-feature :class:`Optimizer`."""
        return _AdafacMLPInstance(
            self._graphdef, theta, exp_mult=self.exp_mult, step_mult=self.step_mult
        )


__all__ = [
    "AdafacMLPLOptState",
    "AdafacMLPLearnedOptimizer",
    "LearnableSGD",
    "LearnedOptimizer",
    "MLPLOptState",
    "MLPLearnedOptimizer",
]
