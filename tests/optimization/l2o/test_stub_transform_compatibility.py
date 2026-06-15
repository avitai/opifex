"""JAX/Flax-NNX transform-compatibility tests for the refined compute paths.

Every transformable array computation introduced for the scientific-integrity
compute paths must remain ``jit``/``grad``/``vmap`` clean and agree with eager
execution. Each test below pins one such computation against its eager result.

Paths that are inherently Python-stateful (the RL optimization step, which
mutates per-instance iterate state, casts to ``float`` and branches on string
action types) are exercised with a smoke test instead, with the reason
documented in-line.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.optimization.l2o.advanced_meta_learning import refine_with_keep_best
from opifex.optimization.l2o.multi_objective import MultiObjectiveL2OEngine
from opifex.optimization.l2o.parametric_solver import OptimizationProblem
from opifex.optimization.l2o.rl_optimization import (
    RLOptimizationConfig,
    RLOptimizationEngine,
)
from opifex.optimization.scientific_integration import translation_invariance_error
from opifex.solvers.hybrid import relative_field_discrepancy


def _quadratic(x: jax.Array) -> jax.Array:
    return jnp.sum(x**2)


class TestTranslationInvarianceTransforms:
    """Transform compatibility for the translation-invariance check."""

    def test_translation_invariance_is_jit_vmap_compatible(self) -> None:
        """``jit`` and ``vmap`` over a field batch match eager evaluation."""
        fields = jnp.stack(
            [
                jnp.array([1.0, 1.0, 1.0, 1.0]),
                jnp.array([0.0, 1.0, 2.0, 3.0]),
            ]
        )
        shifts = (1,)
        axes = (0,)

        eager = jnp.stack([translation_invariance_error(f, shifts, axes) for f in fields])

        jitted = jax.jit(translation_invariance_error, static_argnums=(1, 2))
        jit_eager = jnp.stack([jitted(f, shifts, axes) for f in fields])

        batched = jax.vmap(lambda f: translation_invariance_error(f, shifts, axes))(fields)

        assert jnp.isfinite(batched).all()
        assert batched.shape == (2,)
        assert jnp.allclose(jit_eager, eager)
        assert jnp.allclose(batched, eager)
        # Constant field is translation invariant; structured field is not.
        assert float(eager[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(eager[1]) > 0.0

    def test_translation_invariance_is_grad_clean(self) -> None:
        """The discrepancy is differentiable w.r.t. the field."""
        field = jnp.array([0.0, 1.0, 2.0, 3.0])
        grad = jax.grad(lambda f: translation_invariance_error(f, (1,), (0,)))(field)
        assert grad.shape == field.shape
        assert jnp.isfinite(grad).all()


class TestHybridDiscrepancyTransforms:
    """Transform compatibility for the hybrid classical/neural discrepancy."""

    def test_hybrid_error_is_jit_grad_clean(self) -> None:
        """``jit`` and ``grad`` of the discrepancy match eager evaluation."""
        classical = jnp.array([1.0, 2.0, 3.0, 4.0])
        neural = jnp.array([1.1, 1.9, 3.2, 3.8])

        eager = relative_field_discrepancy(classical, neural)
        jitted = jax.jit(relative_field_discrepancy)(classical, neural)
        assert jnp.allclose(jitted, eager)
        assert jnp.isfinite(eager)

        grad = jax.grad(relative_field_discrepancy, argnums=1)(classical, neural)
        assert grad.shape == neural.shape
        assert jnp.isfinite(grad).all()

    def test_hybrid_error_is_vmap_compatible(self) -> None:
        """``vmap`` over a batch of field pairs matches eager evaluation."""
        classical = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        neural = jnp.array([[1.1, 1.9], [2.8, 4.2], [5.0, 6.0]])

        eager = jnp.stack(
            [relative_field_discrepancy(c, n) for c, n in zip(classical, neural, strict=True)]
        )
        batched = jax.vmap(relative_field_discrepancy)(classical, neural)
        assert batched.shape == (3,)
        assert jnp.allclose(batched, eager)


class TestObjectiveFeedbackTransforms:
    """Transform compatibility for the multi-objective feedback computation."""

    def test_objective_feedback_is_jit_vmap_compatible(self) -> None:
        """``jit`` and ``vmap`` over objective-value batches match eager."""
        feedback = MultiObjectiveL2OEngine._compute_objective_feedback

        objectives = jnp.array([[0.1, 2.0], [0.2, 1.5], [0.0, 3.0]])
        eager = feedback(objectives)
        jitted = jax.jit(feedback)(objectives)
        assert jnp.allclose(jitted, eager)
        assert eager.shape == (2,)
        # Feedback lies in (0, 1] and decreases with achieved objective magnitude.
        assert jnp.all((eager > 0.0) & (eager <= 1.0))

        batch = jnp.stack([objectives, objectives * 2.0])
        batched = jax.vmap(feedback)(batch)
        assert batched.shape == (2, 2)
        assert jnp.allclose(batched[0], eager)

    def test_objective_feedback_is_grad_clean(self) -> None:
        """The feedback is differentiable w.r.t. the achieved objectives."""
        feedback = MultiObjectiveL2OEngine._compute_objective_feedback
        objectives = jnp.array([[0.1, 2.0], [0.2, 1.5]])
        grad = jax.grad(lambda o: jnp.sum(feedback(o)))(objectives)
        assert grad.shape == objectives.shape
        assert jnp.isfinite(grad).all()


class TestGradientRefinementTransforms:
    """Transform compatibility for the keep-best gradient refinement guard."""

    def test_gradient_refinement_is_jit_grad_compatible(self) -> None:
        """The keep-best guard is ``jit`` and ``grad`` clean and never worsens.

        ``refine_with_keep_best`` is exercised with a pure gradient-descent
        refiner so the whole path is a differentiable array computation.
        """

        def refine(
            objective: Callable[[jax.Array], jax.Array],
            warm_start: jax.Array,
            steps: int,
        ) -> jax.Array:
            iterate = warm_start
            for _ in range(steps):
                iterate = iterate - 0.1 * jax.grad(objective)(iterate)
            return iterate

        warm_start = jnp.array([0.5, -0.3, 0.2])

        eager = refine_with_keep_best(_quadratic, warm_start, refine, 5)
        jitted = jax.jit(lambda w: refine_with_keep_best(_quadratic, w, refine, 5))(warm_start)
        assert jnp.allclose(jitted, eager)
        # Refinement never increases the objective.
        assert float(_quadratic(eager)) <= float(_quadratic(warm_start)) + 1e-6

        grad = jax.grad(lambda w: _quadratic(refine_with_keep_best(_quadratic, w, refine, 5)))(
            warm_start
        )
        assert grad.shape == warm_start.shape
        assert jnp.isfinite(grad).all()

    def test_keep_best_returns_warm_start_when_refinement_overshoots(self) -> None:
        """An overshooting refiner is rejected in favour of the warm start."""

        def overshoot(
            objective: Callable[[jax.Array], jax.Array],
            warm_start: jax.Array,
            steps: int,
        ) -> jax.Array:
            return warm_start + 100.0  # Drives the objective far higher.

        warm_start = jnp.array([0.01, 0.01])
        result = refine_with_keep_best(_quadratic, warm_start, overshoot, 5)
        assert jnp.allclose(result, warm_start)


class TestRLOptimizationStepSmoke:
    """Smoke coverage for the RL optimization step.

    The RL step is inherently Python-stateful: it mutates the per-instance
    optimization iterate, casts objective values to ``float`` for bookkeeping,
    and branches on string action types. It is therefore not a pure jittable
    function, so a ``jit`` test does not apply; the underlying gradient-descent
    update is already covered as a transformable path by the refinement and
    feedback tests above. This smoke test confirms the stateful step runs and
    actually decreases the objective.
    """

    def test_rl_optimization_step_runs_and_decreases_objective(self) -> None:
        """The stateful RL step lowers the default quadratic objective."""
        engine = RLOptimizationEngine(RLOptimizationConfig(), rngs=nnx.Rngs(0))
        problem = OptimizationProblem(problem_type="quadratic", dimension=4)
        engine._reset_optimization_iterate(problem.dimension)
        start_objective = engine._previous_objective

        result = engine._execute_optimization_step(
            problem, "continue_optimization", {}, iteration=0
        )

        assert jnp.isfinite(jnp.asarray(result["solution"])).all()
        assert result["solution"].shape == (4,)
        assert result["objective_value"] <= start_objective
