"""Concrete optimisation tasks and task families for L2O meta-training/meta-test.

``QuadraticTask`` is the canonical L2O smoke task (a strictly convex quadratic with a known
optimum; cf. ``learned_optimization/tasks/quadratics.py``). ``QuadraticTaskFamily`` samples
quadratics with **varied conditioning** so a meta-trained optimiser must generalise across
loss landscapes of different curvature/scale — the diversity that makes a meta-test on
held-out tasks meaningful (Wichrowska et al. 2017, ``arXiv:1703.04813``).

``MLPTask`` is the canonical L2O *showcase* task: a small multilayer perceptron trained by the
inner optimiser. It mirrors the ``MLPTask`` used throughout Google's ``learned_optimization``
tutorials (``docs/notebooks/no_dependency_learned_optimizer``) — a genuinely non-convex
neural-network training objective, which is the regime where learned optimisers demonstrably
beat fixed-hyperparameter baselines (Metz et al. 2020, ``arXiv:2009.11243``). To stay
self-contained (no dataset dependency) the data is a synthetic teacher-student regression: a
random teacher MLP generates targets from Gaussian inputs and the student fits them with MSE, so
the global optimum is realisable yet the landscape is non-convex.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp

from opifex.optimization.l2o.core import Task, TaskFamily


# A per-layer MLP parameter pytree is a list of ``(weight, bias)`` tuples.
MLPParams: TypeAlias = list[tuple[jax.Array, jax.Array]]


def _init_mlp_params(key: jax.Array, layer_sizes: tuple[int, ...], scale: float) -> MLPParams:
    """Sample per-layer ``(weight, bias)`` pairs; weights ``~ scale * N(0, 1)``, biases zero.

    Mirrors the small-init convention of the reference ``MLPTask`` (weights scaled by a small
    constant; biases initialised to zero).
    """
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params: MLPParams = []
    for layer_key, fan_in, fan_out in zip(keys, layer_sizes[:-1], layer_sizes[1:], strict=True):
        weight = scale * jax.random.normal(layer_key, (fan_in, fan_out))
        bias = jnp.zeros((fan_out,))
        params.append((weight, bias))
    return params


def _mlp_forward(params: MLPParams, inputs: jax.Array) -> jax.Array:
    """Apply the MLP to ``inputs`` ``(n, input_dim)`` with ReLU on every hidden layer."""
    activations = inputs
    for index, (weight, bias) in enumerate(params):
        activations = activations @ weight + bias
        if index < len(params) - 1:
            activations = jax.nn.relu(activations)
    return activations


@dataclass(frozen=True)
class QuadraticTask(Task):
    """Strictly convex quadratic ``f(x) = 0.5 (x - x*)^T A (x - x*)`` with SPD ``A``.

    Shapes: ``matrix`` is ``(dim, dim)`` SPD, ``optimum`` (``x*``) is ``(dim,)``. The unique
    minimiser is ``optimum`` with ``f(x*) = 0``. The objective is deterministic, so the
    per-call ``key`` is unused.
    """

    matrix: jax.Array
    optimum: jax.Array
    init_scale: float = 1.0
    loss_scale: jax.Array | float = 1.0

    def init(self, key: jax.Array) -> jax.Array:
        """Sample a ``(dim,)`` starting point offset from the optimum by ``init_scale``."""
        return self.optimum + self.init_scale * jax.random.normal(key, self.optimum.shape)

    def loss(
        self,
        params: jax.Array,
        key: jax.Array,  # noqa: ARG002 - deterministic quadratic ignores the key
    ) -> jax.Array:
        """Evaluate the scalar quadratic objective at ``params`` ``(dim,)``."""
        delta = params - self.optimum
        return 0.5 * delta @ (self.matrix @ delta)

    def normalizer(self, loss: jax.Array) -> jax.Array:
        """Scale by the expected initial loss so normalised loss is O(1) across tasks."""
        return loss / self.loss_scale


@dataclass(frozen=True)
class QuadraticTaskFamily(TaskFamily):
    """Quadratics with random optima and random SPD curvature of varied conditioning.

    Each draw builds ``A = Q diag(eigs) Q^T`` with a random orthogonal ``Q`` and eigenvalues
    log-uniformly spread over a sampled condition number in
    ``[1, 10**max_log_condition]`` — so tasks differ in both scale and conditioning.
    """

    dim: int
    init_scale: float = 1.0
    max_log_condition: float = 3.0

    def sample(self, key: jax.Array) -> QuadraticTask:
        """Draw a quadratic task: random SPD ``A``, random optimum, matched loss scale."""
        key_q, key_cond, key_opt = jax.random.split(key, 3)

        # Random orthogonal basis via QR of a Gaussian matrix.
        basis, _ = jnp.linalg.qr(jax.random.normal(key_q, (self.dim, self.dim)))
        # Eigenvalues log-spread from 1 to a sampled condition number.
        log_cond = jax.random.uniform(key_cond, (), minval=0.0, maxval=self.max_log_condition)
        eigenvalues = jnp.logspace(0.0, log_cond, self.dim)
        matrix = (basis * eigenvalues) @ basis.T

        optimum = jax.random.normal(key_opt, (self.dim,))
        # E[initial loss] = 0.5 * init_scale^2 * trace(A) for a Gaussian start.
        loss_scale = 0.5 * self.init_scale**2 * jnp.sum(eigenvalues)
        return QuadraticTask(
            matrix=matrix,
            optimum=optimum,
            init_scale=self.init_scale,
            loss_scale=loss_scale,
        )


@dataclass(frozen=True)
class MLPTask(Task):
    """Minibatch MLP regression onto a per-task teacher-generated dataset.

    The student MLP (architecture ``layer_sizes``) is fit by MSE to ``targets`` produced by a
    random teacher MLP on ``inputs``. Each :meth:`loss` call draws a fresh ``batch_size``
    minibatch using the supplied ``key`` (mirroring the reference ``MLPTask``, which consumes a
    new minibatch per step), so the gradients are *stochastic* — the regime where a learned
    optimiser's implicit learning-rate schedule beats a fixed-step baseline. The objective is
    non-convex in the student weights; the optimum (loss 0) is realised at ``teacher_params``.
    """

    inputs: jax.Array
    targets: jax.Array
    teacher_params: MLPParams
    layer_sizes: tuple[int, ...]
    init_scale: float
    loss_scale: jax.Array
    batch_size: int

    def init(self, key: jax.Array) -> MLPParams:
        """Sample initial student parameters with the small-init convention."""
        return _init_mlp_params(key, self.layer_sizes, self.init_scale)

    def loss(self, params: MLPParams, key: jax.Array) -> jax.Array:
        """Mean-squared error on a fresh ``batch_size`` minibatch drawn with ``key``."""
        indices = jax.random.randint(key, (self.batch_size,), 0, self.inputs.shape[0])
        predictions = _mlp_forward(params, self.inputs[indices])
        return jnp.mean((predictions - self.targets[indices]) ** 2)

    def normalizer(self, loss: jax.Array) -> jax.Array:
        """Scale by the expected initial loss so normalised loss is O(1) across tasks."""
        return loss / self.loss_scale


@dataclass(frozen=True)
class MLPTaskFamily(TaskFamily):
    """Teacher-student MLP regression tasks: random teacher + synthetic Gaussian data per draw.

    Each draw samples a fresh teacher MLP and a fresh dataset, so a meta-trained optimiser must
    generalise across many non-convex training problems rather than memorise one (the diversity
    that makes the held-out meta-test meaningful). Gradients are stochastic minibatch gradients.
    """

    input_dim: int
    hidden_dim: int
    output_dim: int
    num_data: int = 512
    batch_size: int = 32
    init_scale: float = 0.02
    teacher_scale: float = 1.0

    @property
    def layer_sizes(self) -> tuple[int, ...]:
        """The student/teacher architecture ``(input, hidden, output)``."""
        return (self.input_dim, self.hidden_dim, self.output_dim)

    def sample(self, key: jax.Array) -> MLPTask:
        """Draw a teacher-student regression task with fresh teacher and inputs."""
        key_teacher, key_data = jax.random.split(key)
        teacher_params = _init_mlp_params(key_teacher, self.layer_sizes, self.teacher_scale)
        inputs = jax.random.normal(key_data, (self.num_data, self.input_dim))
        targets = _mlp_forward(teacher_params, inputs)
        # E[initial loss] ~ mean(targets^2) since the small-init student outputs ~ 0.
        loss_scale = jnp.mean(targets**2) + 1e-8
        return MLPTask(
            inputs=inputs,
            targets=targets,
            teacher_params=teacher_params,
            layer_sizes=self.layer_sizes,
            init_scale=self.init_scale,
            loss_scale=loss_scale,
            batch_size=self.batch_size,
        )


__all__ = ["MLPTask", "MLPTaskFamily", "QuadraticTask", "QuadraticTaskFamily"]
