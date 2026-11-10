"""Batch-acquisition tests for Task 8.3.

* ``batch_bald`` returns a non-redundant batch and strictly beats top-k
  BALD on a controlled synthetic example where two pool points have the
  same per-point BALD but identical samples (so picking both is
  redundant). batch-BALD must pick a more diverse alternative.
* ``batch_mc_expected_improvement`` matches the reparameterised Monte Carlo q-EI
  formula (Wilson et al. 2017, arXiv:1712.00424).
* ``q_expected_hypervolume_improvement`` returns non-negative contributions
  whose sum equals the closed-form box-volume contributions for a known
  Pareto front.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from opifex.uncertainty.active import batch_active
from opifex.uncertainty.active.acquisition import bald
from opifex.uncertainty.active.batch_active import (
    _dominated_hypervolume_grid,
    _pareto_volume_2d,
    batch_bald,
    batch_mc_expected_improvement,
    dominated_hypervolume,
    q_expected_hypervolume_improvement,
)
from opifex.uncertainty.types import PredictiveDistribution


if TYPE_CHECKING:
    from jax.typing import DTypeLike


def _redundant_predictive() -> PredictiveDistribution:
    """Pool of 4 candidates with one redundant high-BALD pair.

    Uses K=4 ensemble members so the joint mixture-entropy estimator
    actually exercises the redundancy correction. Samples are designed
    so that:

    * Indices 0 and 1 are *perfectly correlated* across members
      (samples[:, 0] == samples[:, 1]). Observing one fully determines
      the other → no extra information about theta from observing both.
    * Index 2 has a DIFFERENT pattern across members so it adds new
      information not extractable from index 0 alone.
    * Index 3 has zero disagreement across members (low marginal BALD).
    """
    samples = jnp.array(
        [
            # member 0
            [-1.0, -1.0, 1.0, 0.0],
            # member 1
            [1.0, 1.0, -1.0, 0.0],
            # member 2: same disagreement on 0,1; OPPOSITE pattern on 2
            [-1.0, -1.0, -1.0, 0.0],
            # member 3
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    mean = jnp.mean(samples, axis=0)
    epistemic = jnp.var(samples, axis=0)
    aleatoric = jnp.full_like(epistemic, 0.05)
    total = epistemic + aleatoric
    return PredictiveDistribution(
        mean=mean,
        samples=samples,
        variance=total,
        epistemic=epistemic,
        aleatoric=aleatoric,
        total_uncertainty=total,
    )


class TestBatchBALD:
    def test_batch_bald_beats_top_k_on_redundant_pool(self) -> None:
        pd = _redundant_predictive()
        rngs = nnx.Rngs(active_bald=0)

        # Naive top-k BALD: indices 0 and 1 have identical marginal BALD
        # (their ensemble samples coincide), and both exceed the BALD of
        # the diverse-but-still-disagreeing point 2. So top-2-by-marginal
        # would pick BOTH redundant points {0, 1}.
        per_point = bald(pd, rngs=nnx.Rngs(active_bald=0))
        # Marginal BALD: indices 0 and 1 are highest (tied); index 2 next.
        assert float(per_point[0]) == pytest.approx(float(per_point[1]), abs=1e-7)
        assert float(per_point[0]) >= float(per_point[2])

        batch = batch_bald(pd, batch_size=2, rngs=rngs)

        # batch-BALD must break the redundancy: the chosen pair must
        # include the diverse point 2 (which adds independent information
        # about theta) rather than both redundant points {0, 1}.
        chosen = {int(i) for i in batch.indices}
        assert chosen != {0, 1}, "batch-BALD failed to break redundancy"
        assert 2 in chosen, "batch-BALD missed the diverse point 2"

    def test_batch_bald_deterministic_for_fixed_key(self) -> None:
        pd = _redundant_predictive()
        a = batch_bald(pd, batch_size=2, rngs=nnx.Rngs(active_bald=7))
        b = batch_bald(pd, batch_size=2, rngs=nnx.Rngs(active_bald=7))
        assert jnp.array_equal(a.indices, b.indices)


class TestBatchMCExpectedImprovement:
    def test_matches_mc_formula(self) -> None:
        """Matches the reparameterised Monte Carlo batch-EI estimator."""
        rngs = nnx.Rngs(active_acquire=0)
        # candidate batch: shape (q, d) — here q=3, d=1 trivial domain.
        # The predictive over the joint batch is N(mean, cov).
        mean = jnp.array([0.5, 0.7, 0.9])
        # diagonal covariance for simplicity.
        std = jnp.array([0.3, 0.2, 0.4])
        eta = 1.0
        num_samples = 8192
        # Manual Monte-Carlo ground truth.
        key = jax.random.PRNGKey(0)
        z = jax.random.normal(key, (num_samples, 3))
        samples = mean[None, :] + std[None, :] * z
        # per-batch best = min over batch dim.
        min_per_sample = jnp.min(samples, axis=-1)
        expected_ei = jnp.mean(jnp.maximum(eta - min_per_sample, 0.0))

        out = batch_mc_expected_improvement(
            mean=mean,
            std=std,
            best_value=eta,
            num_samples=num_samples,
            rngs=rngs,
        )
        assert jnp.allclose(out, expected_ei, rtol=5e-2)


class TestQEHVI:
    """q-EHVI against the hypervolume increment it estimates.

    Every tolerance below comes from :func:`_deterministic_limit_budget` or
    :func:`_reduction_budget`, defined under the accuracy section further down this
    module together with the derivation each rests on.
    """

    def test_qehvi_nonnegative_and_matches_known_pareto_volume(self) -> None:
        """Two-objective minimisation; non-dominated points (1,3),(2,2),(3,1).

        Reference point (4, 4). A near-deterministic predictive at (1.5, 2.5) must
        recover that candidate's exact hypervolume increment, which is the box
        ``(2 - 1.5) * (3 - 2.5) = 0.25``. The tolerance is the Lipschitz bound on the
        bias of using ``std = 1e-6`` in place of a point mass -- ``3e-5`` here, against
        the ``1e-2`` this assertion carried before, which admitted a 4% error in a
        quantity the indicator pins exactly.
        """
        rngs = nnx.Rngs(active_acquire=0)
        pareto_front = jnp.array(
            [
                [1.0, 3.0],
                [2.0, 2.0],
                [3.0, 1.0],
            ]
        )
        reference_point = jnp.array([4.0, 4.0])
        # Single candidate batch (q=1), 2 objectives, deterministic.
        candidate_mean = jnp.array([[1.5, 2.5]])
        standard_deviation = 1e-6
        candidate_std = jnp.full(candidate_mean.shape, standard_deviation)

        result = q_expected_hypervolume_improvement(
            candidate_mean=candidate_mean,
            candidate_std=candidate_std,
            pareto_front=pareto_front,
            reference_point=reference_point,
            num_samples=1024,
            rngs=rngs,
        )

        exact = _hypervolume_increment(pareto_front, candidate_mean, reference_point)
        assert exact == 0.25
        assert float(result) >= 0.0
        assert float(result) == pytest.approx(
            exact,
            abs=_deterministic_limit_budget(
                front=pareto_front,
                candidates=candidate_mean,
                reference_point=reference_point,
                standard_deviation=standard_deviation,
            ),
        )

    def test_qehvi_3d_matches_the_exact_increment(self) -> None:
        """Three-objective q-EHVI recovers the increment of its two candidates.

        Renamed from ``..._runs_and_is_finite``: it already asserted a value, and
        ``isfinite`` was the weakest claim it made. The expected 3.0 was hand-computed
        in a comment and is now taken from :func:`dominated_hypervolume`, whose own
        ``M = 3`` values are pinned against BoTorch's published 358.0.
        """
        rngs = nnx.Rngs(active_acquire=0)
        pareto_front = jnp.array(
            [
                [1.0, 1.0, 3.0],
                [3.0, 3.0, 1.0],
            ]
        )
        reference_point = jnp.array([4.0, 4.0, 4.0])
        # q=2 deterministic candidates dominating fresh volume.
        candidate_mean = jnp.array([[2.0, 2.0, 2.0], [1.5, 1.5, 3.5]])
        standard_deviation = 1e-6
        candidate_std = jnp.full(candidate_mean.shape, standard_deviation)

        result = q_expected_hypervolume_improvement(
            candidate_mean=candidate_mean,
            candidate_std=candidate_std,
            pareto_front=pareto_front,
            reference_point=reference_point,
            num_samples=512,
            rngs=rngs,
        )

        exact = _hypervolume_increment(pareto_front, candidate_mean, reference_point)
        assert exact == 3.0
        assert float(result) >= 0.0
        assert float(result) == pytest.approx(
            exact,
            abs=_deterministic_limit_budget(
                front=pareto_front,
                candidates=candidate_mean,
                reference_point=reference_point,
                standard_deviation=standard_deviation,
            ),
        )

    def test_qehvi_increases_with_more_dominated_volume(self) -> None:
        """A candidate dominating more volume yields a larger q-EHVI.

        The ordering was the whole of this test, and an ordering holds for any
        monotone function of the volume -- a q-EHVI off by a constant factor passed it.
        Both increments are exact: ``(4 - 1.9) ** 2 - 4 = 0.41`` and
        ``(4 - 0.5) ** 2 - 4 = 8.25``.
        """
        rngs_a = nnx.Rngs(active_acquire=0)
        rngs_b = nnx.Rngs(active_acquire=0)
        pareto_front = jnp.array([[2.0, 2.0]])
        reference_point = jnp.array([4.0, 4.0])
        standard_deviation = 1e-6
        std = jnp.full((1, 2), standard_deviation)
        near_reference = jnp.array([[1.9, 1.9]])
        near_ideal = jnp.array([[0.5, 0.5]])

        # Candidate near the reference dominates a tiny box.
        small = q_expected_hypervolume_improvement(
            candidate_mean=near_reference,
            candidate_std=std,
            pareto_front=pareto_front,
            reference_point=reference_point,
            num_samples=512,
            rngs=rngs_a,
        )
        # Candidate near the ideal point dominates a much larger box.
        large = q_expected_hypervolume_improvement(
            candidate_mean=near_ideal,
            candidate_std=std,
            pareto_front=pareto_front,
            reference_point=reference_point,
            num_samples=512,
            rngs=rngs_b,
        )
        assert float(large) > float(small)

        for measured, candidate in ((small, near_reference), (large, near_ideal)):
            exact = _hypervolume_increment(pareto_front, candidate, reference_point)
            assert float(measured) == pytest.approx(
                exact,
                abs=_deterministic_limit_budget(
                    front=pareto_front,
                    candidates=candidate,
                    reference_point=reference_point,
                    standard_deviation=standard_deviation,
                ),
            )

    def test_qehvi_is_jit_and_grad_differentiable(self) -> None:
        """q-EHVI must stay jit/grad/vmap-compatible for BO loops.

        Compatibility only: that each transform traces and returns the right shape.
        What the transforms *compute* is :class:`TestQEHVITransformsAgree`, which this
        test deliberately leaves alone -- a transform that returns a finite number of
        the right shape and the wrong value passes everything here.
        """
        pareto_front = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        reference_point = jnp.array([4.0, 4.0])
        candidate_std = jnp.array([[0.1, 0.1]])

        def acq(candidate_mean: jax.Array) -> jax.Array:
            return q_expected_hypervolume_improvement(
                candidate_mean=candidate_mean,
                candidate_std=candidate_std,
                pareto_front=pareto_front,
                reference_point=reference_point,
                num_samples=64,
                rngs=jax.random.PRNGKey(0),
            )

        jitted = jax.jit(acq)
        value = jitted(jnp.array([[1.5, 2.5]]))
        assert bool(jnp.isfinite(value))

        grad = jax.grad(lambda m: acq(m).sum())(jnp.array([[1.5, 2.5]]))
        assert grad.shape == (1, 2)
        assert bool(jnp.all(jnp.isfinite(grad)))

        batched = jax.vmap(acq)(jnp.array([[[1.5, 2.5]], [[2.5, 1.5]]]))
        assert batched.shape == (2,)
        assert bool(jnp.all(jnp.isfinite(batched)))


class TestGeneralMHypervolume:
    def test_general_grid_path_matches_exact_2d(self) -> None:
        """The general grid path equals the exact 2-D staircase formula."""
        front = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        reference = jnp.array([4.0, 4.0])
        exact = _pareto_volume_2d(front, reference)
        # Exercise the dimension-agnostic grid core directly so the
        # agreement is a genuine cross-check, not the fast-path alias.
        general = _dominated_hypervolume_grid(front, reference)
        assert float(general) == pytest.approx(float(exact), abs=1e-6)
        assert float(general) == pytest.approx(6.0, abs=1e-6)

    def test_general_grid_path_matches_2d_with_candidate(self) -> None:
        """Agreement holds after merging a candidate into the 2-D front."""
        front = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [1.5, 2.5]])
        reference = jnp.array([4.0, 4.0])
        exact = _pareto_volume_2d(front, reference)
        general = _dominated_hypervolume_grid(front, reference)
        assert float(general) == pytest.approx(float(exact), abs=1e-6)
        assert float(general) == pytest.approx(6.25, abs=1e-6)

    def test_a_dominated_point_adds_no_volume_in_2d(self) -> None:
        """A redundant front point must not change the 2-D hypervolume.

        The public entry point documents that the front "need not be screened of
        dominated points" and that such points "contribute no extra volume", so
        passing one must leave the result at the non-dominated value.
        """
        front = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        with_redundant = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0], [3.0, 3.0]])
        reference = jnp.array([4.0, 4.0])
        assert float(_pareto_volume_2d(front, reference)) == pytest.approx(6.0, abs=1e-6)
        assert float(_pareto_volume_2d(with_redundant, reference)) == pytest.approx(6.0, abs=1e-6)
        # The general path is the independent oracle: it already handles this input.
        assert float(_dominated_hypervolume_grid(with_redundant, reference)) == pytest.approx(
            6.0, abs=1e-6
        )

    def test_public_dominated_hypervolume_routes_2d_to_staircase(self) -> None:
        """The public entry point matches the exact 2-D result."""
        front = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        reference = jnp.array([4.0, 4.0])
        assert float(dominated_hypervolume(front, reference)) == pytest.approx(6.0, abs=1e-6)

    def test_general_hypervolume_3d_hand_computed(self) -> None:
        """Hand-constructed M=3 front matches the hand-computed HV.

        Front {(1,1,3),(3,3,1)} with reference (4,4,4):
        box A from (1,1,3) has volume 3*3*1 = 9, box B from (3,3,1)
        has volume 1*1*3 = 3, their overlap (the corner z>=3 in every
        objective) has volume 1, so the union HV = 9 + 3 - 1 = 11.
        """
        front = jnp.array([[1.0, 1.0, 3.0], [3.0, 3.0, 1.0]])
        reference = jnp.array([4.0, 4.0, 4.0])
        assert float(dominated_hypervolume(front, reference)) == pytest.approx(11.0, abs=1e-6)

    def test_general_hypervolume_single_point_box(self) -> None:
        """A single M=3 point yields the full dominated box volume."""
        front = jnp.array([[1.0, 1.0, 1.0]])
        reference = jnp.array([4.0, 4.0, 4.0])
        assert float(dominated_hypervolume(front, reference)) == pytest.approx(27.0, abs=1e-6)

    def test_general_hypervolume_ignores_dominated_points(self) -> None:
        """Dominated front points contribute no extra dominated volume."""
        front = jnp.array([[1.0, 1.0, 3.0], [3.0, 3.0, 1.0], [3.5, 3.5, 3.5]])
        reference = jnp.array([4.0, 4.0, 4.0])
        assert float(dominated_hypervolume(front, reference)) == pytest.approx(11.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Accuracy: thresholds, published cases and the properties of the indicator
# ---------------------------------------------------------------------------

#: BoTorch compares hypervolumes **exactly** wherever the inputs are exactly
#: representable -- ``torch.equal`` and ``assertEqual`` throughout
#: ``test/utils/multi_objective/box_decompositions/test_dominated.py``, which contains
#: no ``allclose`` and no ``atol`` -- and falls back to ``EPS = 1e-4`` only for the
#: fronts whose coordinates are not, in ``test/utils/multi_objective/test_hypervolume.py``.
#: The cases below follow that same rule, so this tolerance is reached by exactly one of
#: them: the ``M = 3`` case whose reference point is ``(-2.1, -2.5, -2.3)``.
PUBLISHED_HYPERVOLUME_EPS = 1e-4

#: Published hypervolumes, taken from BoTorch's own test suite rather than hand-computed
#: here, as ``(label, front, reference_point, expected, exactly_representable)``. BoTorch
#: maximises and this module minimises, and the indicator is invariant under reflecting
#: both the front and the reference point through the origin, so :func:`_as_minimisation`
#: negates each pair. Every front below is one BoTorch asserts a value for.
PUBLISHED_CASES: tuple[tuple[str, list[list[float]], list[float], float, bool], ...] = (
    # test_dominated.py, m=2: the staircase 24+7+6+5+4+3.
    (
        "m=2 staircase",
        [[3.0, 8.0], [4.0, 7.0], [5.0, 6.0], [6.0, 5.0], [7.0, 4.0], [8.0, 3.0]],
        [0.0, 0.0],
        49.0,
        True,
    ),
    # test_dominated.py, m=2: the same front plus the two points that file calls
    # dominated and removes. The value must not move -- the property this PR restores.
    (
        "m=2 staircase with BoTorch's two dominated extras",
        [
            [3.0, 8.0],
            [4.0, 7.0],
            [5.0, 6.0],
            [6.0, 5.0],
            [7.0, 4.0],
            [8.0, 3.0],
            [8.0, 2.0],
            [7.0, 1.0],
        ],
        [0.0, 0.0],
        49.0,
        True,
    ),
    # test_dominated.py, m=2: reference at ``max + 1``, so no point beats it anywhere.
    (
        "m=2 whole front worse than the reference",
        [[3.0, 8.0], [4.0, 7.0], [5.0, 6.0], [6.0, 5.0], [7.0, 4.0], [8.0, 3.0]],
        [9.0, 9.0],
        0.0,
        True,
    ),
    # test_dominated.py, m=3: inclusion-exclusion 176+234+280-132-140-180+120.
    (
        "m=3 non-dominated front",
        [[1.0, 6.0, 8.0], [2.0, 4.0, 10.0], [3.0, 5.0, 7.0]],
        [-1.0, -2.0, -3.0],
        358.0,
        True,
    ),
    (
        "m=3 whole front worse than the reference",
        [[1.0, 6.0, 8.0], [2.0, 4.0, 10.0], [3.0, 5.0, 7.0]],
        [4.0, 7.0, 11.0],
        0.0,
        True,
    ),
    # test_hypervolume.py, m=2: ``[8.5, 3.0]`` is dominated by ``[8.5, 3.5]``. Every
    # coordinate is a multiple of a half, so this is exact here even though BoTorch's
    # own recursive algorithm needs ``EPS`` for it.
    (
        "m=2 with a dominated point",
        [[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]],
        [0.0, 0.0],
        37.75,
        True,
    ),
    (
        "m=2 with a dominated point, tighter reference",
        [[8.5, 3.0], [8.5, 3.5], [5.0, 5.0], [9.0, 1.0], [4.0, 5.0]],
        [1.0, 0.5],
        28.75,
        True,
    ),
    # test_hypervolume.py, m=3: the one case whose reference point is not a dyadic
    # rational, and so the one that needs BoTorch's own ``EPS``.
    (
        "m=3 unit vectors",
        [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        [-2.1, -2.5, -2.3],
        11.075,
        False,
    ),
    (
        "m=2 single point worse than the reference",
        [[-1.0, -1.0]],
        [0.0, 0.0],
        0.0,
        True,
    ),
)

#: The randomised differential fronts draw every coordinate as an integer multiple of
#: ``1 / DIFFERENTIAL_DENOMINATOR`` no larger than
#: ``DIFFERENTIAL_MAX_NUMERATOR / DIFFERENTIAL_DENOMINATOR``. That choice is what lets the
#: two paths be compared with **no tolerance at all**: every width and height is then a
#: multiple of ``1 / 8``, so every box volume is a multiple of ``1 / 64``, and no partial
#: sum of either path can exceed ``(48 / 8) ** 2 = 36``. Since ``36 * 64 = 2304`` is far
#: below ``2 ** 24 = 16777216`` -- the last integer float32 represents exactly -- every
#: partial sum of either path is exact, whatever order XLA associates it in.
DIFFERENTIAL_DENOMINATOR = 8
DIFFERENTIAL_MAX_NUMERATOR = 48
DIFFERENTIAL_MAX_FRONT_SIZE = 6
DIFFERENTIAL_CASES = 300
DIFFERENTIAL_SEED = 20261110

#: Floor on how much of the corpus must hold a point past the reference on objective 0.
#: Derived: with both drawn uniformly from the same ``49`` values, a front of ``P`` points
#: lies wholly inside the reference on objective 0 with probability
#: ``sum_{k=1..49} k ** P / 49 ** (P + 1)``, which is ``1 / (P + 1)`` to within a percent.
#: Averaged over ``P`` drawn from 1 to 6 that is ``(1/2 + 1/3 + ... + 1/7) / 6 ~ 0.27``, so
#: close to three quarters of the corpus reaches the input class and requiring half leaves
#: a wide margin. It is a corpus control, not a property of the code: without it a change
#: to the generator could quietly make every case a screened front while the differential
#: test kept passing.
DIFFERENTIAL_MIN_PAST_REFERENCE = DIFFERENTIAL_CASES // 2

#: Reassociation budget for a float32 reduction, after NVIDIA torch-harmonics PR #339's
#: ``compare_tensors``: ``atol = ulps * eps * max|reference|`` with ``rtol = 0`` and
#: ``ulps = max(64, sqrt(terms))``, on the grounds that reassociating a sum carries an
#: absolute error set by the size of the summed terms rather than by the entry it lands on.
MINIMUM_REDUCTION_ULPS = 64.0

#: A draw from a candidate's predictive sits within this many standard deviations of its
#: mean. Beyond five the two-sided normal tail is below ``6e-7`` per coordinate, which no
#: sample count in this file comes close to exhausting.
CANDIDATE_DRAW_BOUND = 5.0

#: Step for the central difference that checks the q-EHVI gradient. With the sample draws
#: held fixed, the estimator is piecewise-multilinear in the candidate's mean, so a step
#: taken inside one linear piece carries **no truncation error** and the difference
#: quotient is limited by cancellation alone. That is why the step is this large rather
#: than small: measured in float64, the disagreement with autodiff *grows* as the step
#: falls below it (2e-16 at 1e-1, 1e-14 at 1e-2, 1e-13 at 1e-3), which is cancellation,
#: not truncation. An order of magnitude inside the nearest partition boundary.
GRADIENT_STEP = 1e-2


def _as_minimisation(
    front: list[list[float]], reference: list[float]
) -> tuple[jax.Array, jax.Array]:
    """Reflect a maximisation front and reference point into this module's convention.

    The dominated hypervolume is unchanged by negating both, since negation maps the
    region dominated by ``P`` below ``r`` onto the region dominating ``-P`` above ``-r``.
    """
    return -jnp.array(front), -jnp.array(reference)


def _reduction_budget(*, terms: int, scale: float, dtype: DTypeLike = jnp.float32) -> float:
    """Absolute tolerance for two orderings of the same ``terms``-addend float32 sum."""
    epsilon = float(jnp.finfo(dtype).eps)
    ulps = max(MINIMUM_REDUCTION_ULPS, float(np.sqrt(terms)))
    return ulps * epsilon * abs(scale)


def _hypervolume_increment(
    front: jax.Array, candidates: jax.Array, reference_point: jax.Array
) -> float:
    """Exact volume the candidates add to the front, from the indicator itself.

    The deterministic limit of q-EHVI. Taking it from :func:`dominated_hypervolume`
    rather than restating a hand-computed number means the expected value is derived
    from the same definition the acquisition integrates, and the cases above have
    already pinned that function against BoTorch's published values.
    """
    merged = jnp.concatenate([front, candidates], axis=0)
    return float(dominated_hypervolume(merged, reference_point)) - float(
        dominated_hypervolume(front, reference_point)
    )


def _deterministic_limit_budget(
    *,
    front: jax.Array,
    candidates: jax.Array,
    reference_point: jax.Array,
    standard_deviation: float,
) -> float:
    """How far a q-EHVI estimate may sit from the increment evaluated at the mean.

    The dominated hypervolume is piecewise-multilinear in a candidate's coordinates, so
    displacing one coordinate by ``d`` changes it by at most ``d`` times the area of the
    opposite face, which the box between the reference point and the best value present
    bounds by ``span ** (M - 1)``. Summing over the ``M`` objectives and taking the worst
    draw within :data:`CANDIDATE_DRAW_BOUND` standard deviations gives the bound here.

    This is a Lipschitz bound on the integrand, read off the geometry of the reference
    box. It is not a measured spread of the estimator, and it does not shrink with the
    sample count: what it bounds is the bias of standing in a near-zero but non-zero
    predictive for a point mass, which is why these tests pass ``std = 1e-6`` rather
    than zero.
    """
    merged = jnp.concatenate([front, candidates], axis=0)
    num_objectives = int(reference_point.shape[0])
    span = float(jnp.max(reference_point - jnp.min(merged, axis=0)))
    face_area = span ** (num_objectives - 1)
    return num_objectives * face_area * CANDIDATE_DRAW_BOUND * standard_deviation


class TestPublishedHypervolumeCases:
    """Values asserted by BoTorch's own hypervolume tests, not computed here.

    BoTorch is the reference implementation of the box decomposition this module cites
    (Daulton, Balandat & Bakshy 2020, arXiv:2006.05078 -- the same authors), so its test
    suite is the external oracle. Its two "worse than the reference point" cases are the
    reason this class exists: opifex had no test of that input class at any ``M``.
    """

    @pytest.mark.parametrize(
        ("label", "front", "reference", "expected", "exactly_representable"),
        PUBLISHED_CASES,
        ids=[case[0] for case in PUBLISHED_CASES],
    )
    def test_the_public_entry_point_matches_botorch(
        self,
        label: str,
        front: list[list[float]],
        reference: list[float],
        expected: float,
        exactly_representable: bool,
    ) -> None:
        """The exported hypervolume reproduces each published value."""
        points, reference_point = _as_minimisation(front, reference)
        measured = float(dominated_hypervolume(points, reference_point))
        if exactly_representable:
            assert measured == expected, f"{label}: {measured} != {expected}"
        else:
            assert measured == pytest.approx(expected, abs=PUBLISHED_HYPERVOLUME_EPS), label

    @pytest.mark.parametrize(
        ("label", "front", "reference", "expected", "exactly_representable"),
        [case for case in PUBLISHED_CASES if len(case[2]) == 2],
        ids=[case[0] for case in PUBLISHED_CASES if len(case[2]) == 2],
    )
    def test_both_paths_match_botorch_at_two_objectives(
        self,
        label: str,
        front: list[list[float]],
        reference: list[float],
        expected: float,
        exactly_representable: bool,
    ) -> None:
        """The staircase and the grid both reproduce the published two-objective values.

        The public entry point routes ``M = 2`` to the staircase, so without this the
        grid path would be unmeasured on every two-objective published case.
        """
        points, reference_point = _as_minimisation(front, reference)
        staircase = float(_pareto_volume_2d(points, reference_point))
        grid = float(_dominated_hypervolume_grid(points, reference_point))
        assert staircase == grid, f"{label}: staircase {staircase} != grid {grid}"
        if exactly_representable:
            assert staircase == expected, f"{label}: {staircase} != {expected}"
        else:
            assert staircase == pytest.approx(expected, abs=PUBLISHED_HYPERVOLUME_EPS), label


class TestTheTwoPathsAgreeOnAnyFront:
    """The ``M = 2`` fast path must equal the dimension-agnostic grid on *any* front.

    The module docstring promises the staircase "is unit-tested to agree with it at
    ``M = 2``", and two hand-picked screened fronts were what stood behind that. A front
    the reference point cuts through satisfies neither path's hand-checked cases and is
    where they diverged.
    """

    @staticmethod
    def _randomised_cases() -> list[tuple[jax.Array, jax.Array]]:
        """Fronts and reference points on the dyadic grid :data:`DIFFERENTIAL_DENOMINATOR`."""
        generator = np.random.default_rng(DIFFERENTIAL_SEED)
        cases: list[tuple[jax.Array, jax.Array]] = []
        for _ in range(DIFFERENTIAL_CASES):
            size = int(generator.integers(1, DIFFERENTIAL_MAX_FRONT_SIZE + 1))
            front = generator.integers(0, DIFFERENTIAL_MAX_NUMERATOR + 1, size=(size, 2))
            reference = generator.integers(0, DIFFERENTIAL_MAX_NUMERATOR + 1, size=(2,))
            cases.append(
                (
                    jnp.asarray(front.astype(np.float32) / DIFFERENTIAL_DENOMINATOR),
                    jnp.asarray(reference.astype(np.float32) / DIFFERENTIAL_DENOMINATOR),
                )
            )
        return cases

    def test_the_corpus_actually_holds_fronts_past_the_reference(self) -> None:
        """Corpus control: the generated cases must exercise the input class.

        Without this the differential test below passes just as happily over a corpus of
        screened fronts, which is what it is there to go beyond.

        The count is of points past the reference **on objective 0** specifically, not on
        either objective. A point past it on objective 1 only was always handled correctly,
        because a height is clamped at zero where a width is not, so counting those would
        overstate how much of the corpus reaches the defect.
        """
        cases = self._randomised_cases()
        assert len(cases) == DIFFERENTIAL_CASES
        past_reference = sum(
            1 for front, reference in cases if bool(jnp.any(front[:, 0] > reference[0]))
        )
        assert past_reference >= DIFFERENTIAL_MIN_PAST_REFERENCE, (
            f"only {past_reference} of {DIFFERENTIAL_CASES} cases hold a point past the "
            "reference point on objective 0, so the corpus no longer exercises the cap"
        )

    def test_the_staircase_equals_the_grid_exactly(self) -> None:
        """Bit-exact agreement, with no tolerance, over the dyadic corpus.

        The coordinates are chosen so that neither path can round (see
        :data:`DIFFERENTIAL_DENOMINATOR`), which turns the usual approximate cross-check
        into an exact one: any disagreement at all is a defect in one of the two paths.
        """
        disagreements: list[str] = []
        for front, reference in self._randomised_cases():
            staircase = float(_pareto_volume_2d(front, reference))
            grid = float(_dominated_hypervolume_grid(front, reference))
            if staircase != grid:
                disagreements.append(
                    f"front={front.tolist()} reference={reference.tolist()} "
                    f"staircase={staircase} grid={grid}"
                )
        assert not disagreements, (
            f"{len(disagreements)} of {DIFFERENTIAL_CASES} fronts disagree; first three:\n"
            + "\n".join(disagreements[:3])
        )

    @pytest.mark.parametrize(
        ("front", "reference", "expected"),
        [
            # The trailing point is past the reference on objective 0, so it can dominate
            # nothing below the reference: the volume is (4-1)*(4-1) from (1, 1) alone.
            ([[1.0, 1.0], [5.0, 1.0]], [4.0, 4.0], 9.0),
            # Past the reference on both objectives.
            ([[1.0, 1.0], [5.0, 5.0]], [4.0, 4.0], 9.0),
            # Past the reference on objective 0 and sorted into the middle of the front:
            # 2*1 from (1, 3) plus 1*3 from (3, 1).
            ([[1.0, 3.0], [6.0, 2.0], [3.0, 1.0]], [4.0, 4.0], 5.0),
            # Past the reference on objective 1 only -- already correct before this change,
            # because a height is clamped at zero where a width was not.
            ([[1.0, 5.0], [2.0, 1.0]], [4.0, 4.0], 6.0),
        ],
    )
    def test_a_point_past_the_reference_contributes_nothing(
        self, front: list[list[float]], reference: list[float], expected: float
    ) -> None:
        """A point worse than the reference on any objective adds no volume.

        Nothing can be both dominated by such a point and below the reference, so it is
        exactly BoTorch's ``better_than_ref`` filter expressed as an expected value. The
        four fronts are hand-computable and dyadic, so the comparison is exact.
        """
        points = jnp.array(front)
        reference_point = jnp.array(reference)
        assert float(_pareto_volume_2d(points, reference_point)) == expected
        assert float(_dominated_hypervolume_grid(points, reference_point)) == expected

    def test_two_objectives_are_routed_to_the_staircase(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``M = 2`` reaches the fast path, not merely a value the grid also returns.

        The neighbouring value assertion cannot show this: now that the two paths agree
        everywhere, routing ``M = 2`` to the grid would leave every other test green.
        """
        calls: list[str] = []
        real = batch_active._pareto_volume_2d

        def spy(points: jax.Array, reference_point: jax.Array) -> jax.Array:
            calls.append("_pareto_volume_2d")
            return real(points, reference_point)

        monkeypatch.setattr(batch_active, "_pareto_volume_2d", spy)
        front = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        assert float(batch_active.dominated_hypervolume(front, jnp.array([4.0, 4.0]))) == 6.0
        assert calls == ["_pareto_volume_2d"]


class TestThePublishedIndicatorProperties:
    """Properties the hypervolume indicator is published as having.

    Strict monotonicity with respect to Pareto dominance and invariance of the induced
    order under scaling are the indicator's defining properties (Zitzler, Thiele,
    Laumanns, Fonseca & Grunert da Fonseca, "Performance assessment of multiobjective
    optimizers", IEEE Trans. Evol. Comput. 7(2):117-132, 2003); the hypervolume is the
    only known unary indicator that is strictly monotonic. Insensitivity to dominated
    points is stated by moocore, whose 100-point example gives the same
    ``93.55331425585321`` before and after ``filter_dominated``.

    Every multiplier and offset below is a power of two or a dyadic rational, so each
    identity holds bit-exactly in float32 and none of these tests carries a tolerance.
    """

    FRONT = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    REFERENCE = jnp.array([4.0, 4.0])
    FRONT_3D = jnp.array([[1.0, 1.0, 3.0], [3.0, 3.0, 1.0]])
    REFERENCE_3D = jnp.array([4.0, 4.0, 4.0])

    @pytest.mark.parametrize("scale", [[2.0, 0.5], [0.25, 4.0], [8.0, 8.0]])
    def test_scaling_each_objective_scales_the_volume_by_the_product(
        self, scale: list[float]
    ) -> None:
        """Scaling objective ``m`` by ``s_m > 0`` multiplies the volume by ``prod(s)``."""
        factors = jnp.array(scale)
        base = float(dominated_hypervolume(self.FRONT, self.REFERENCE))
        scaled = float(
            dominated_hypervolume(self.FRONT * factors[None, :], self.REFERENCE * factors)
        )
        assert scaled == base * float(jnp.prod(factors))

    @pytest.mark.parametrize("offset", [[0.25, -0.5], [-2.0, 2.0], [1.5, 1.5]])
    def test_translating_front_and_reference_together_changes_nothing(
        self, offset: list[float]
    ) -> None:
        """The volume depends on the front only through its offsets from the reference."""
        shift = jnp.array(offset)
        base = float(dominated_hypervolume(self.FRONT, self.REFERENCE))
        moved = float(dominated_hypervolume(self.FRONT + shift[None, :], self.REFERENCE + shift))
        assert moved == base

    def test_a_dominated_point_changes_nothing_at_two_objectives(self) -> None:
        """Adding a redundant point leaves the two-objective volume alone."""
        base = float(dominated_hypervolume(self.FRONT, self.REFERENCE))
        for redundant in ([3.0, 3.0], [2.5, 3.5], [3.5, 1.5], [2.0, 2.0]):
            padded = jnp.concatenate([self.FRONT, jnp.array([redundant])], axis=0)
            assert float(dominated_hypervolume(padded, self.REFERENCE)) == base

    def test_a_dominated_point_changes_nothing_at_three_objectives(self) -> None:
        """The same at ``M = 3``, where the grid path computes it."""
        base = float(dominated_hypervolume(self.FRONT_3D, self.REFERENCE_3D))
        padded = jnp.concatenate([self.FRONT_3D, jnp.array([[3.5, 3.5, 3.5]])], axis=0)
        assert float(dominated_hypervolume(padded, self.REFERENCE_3D)) == base

    def test_a_dominating_front_scores_strictly_higher(self) -> None:
        """Strict monotonicity: if every point improves, the volume must grow.

        The front is shifted towards the ideal corner by a dyadic step, which dominates
        the original front point-for-point, so a merely non-decreasing result is a defect.
        """
        step = jnp.array([0.5, 0.5])
        better = self.FRONT - step[None, :]
        base = float(dominated_hypervolume(self.FRONT, self.REFERENCE))
        improved = float(dominated_hypervolume(better, self.REFERENCE))
        assert improved > base
        # And the dominating front's own volume is hand-computable: the staircase
        # (0.5, 2.5), (1.5, 1.5), (2.5, 0.5) under (4, 4) spans 1*1.5 + 1*2.5 + 1.5*3.5.
        assert improved == 1.0 * 1.5 + 1.0 * 2.5 + 1.5 * 3.5


class TestQEHVITransformsAgree:
    """What ``jit``, ``vmap`` and ``grad`` compute, not merely that they trace.

    ``test_qehvi_is_jit_and_grad_differentiable`` asserts a finite value and a shape
    under each transform, which any transform returning the wrong number also satisfies.
    These compare each transform against the untransformed computation.
    """

    FRONT = jnp.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    REFERENCE = jnp.array([4.0, 4.0])
    STD = jnp.array([[0.1, 0.1]])
    NUM_SAMPLES = 64
    MEAN = jnp.array([[1.5, 2.5]])

    @classmethod
    def _acquisition(cls, candidate_mean: jax.Array) -> jax.Array:
        """q-EHVI at a fixed key, so eager and transformed runs share their draws."""
        return q_expected_hypervolume_improvement(
            candidate_mean=candidate_mean,
            candidate_std=cls.STD,
            pareto_front=cls.FRONT,
            reference_point=cls.REFERENCE,
            num_samples=cls.NUM_SAMPLES,
            rngs=jax.random.PRNGKey(0),
        )

    @classmethod
    def _terms(cls) -> int:
        """Addends in the longest reduction the estimator performs.

        The non-dominated partition prepends the anti-ideal edge, so each axis carries
        ``P + 2`` edges and the grid holds ``(P + 2) ** M`` cells; the outer mean runs
        over the samples. Their product bounds the reduction length, which is all
        :func:`_reduction_budget` needs.
        """
        num_points, num_objectives = cls.FRONT.shape
        return cls.NUM_SAMPLES * (num_points + 2) ** num_objectives

    def test_jit_matches_eager(self) -> None:
        """Compiling must not move the value beyond a reassociation of its own sum."""
        eager = float(self._acquisition(self.MEAN))
        compiled = float(jax.jit(self._acquisition)(self.MEAN))
        assert compiled == pytest.approx(
            eager, abs=_reduction_budget(terms=self._terms(), scale=eager)
        )

    def test_vmap_matches_the_loop(self) -> None:
        """A batched map must equal the same candidates evaluated one at a time."""
        means = jnp.array([[[1.5, 2.5]], [[2.5, 1.5]], [[1.25, 1.25]]])
        batched = jax.vmap(self._acquisition)(means)
        looped = jnp.array([self._acquisition(mean) for mean in means])
        assert batched.shape == looped.shape
        budget = _reduction_budget(terms=self._terms(), scale=float(jnp.max(jnp.abs(looped))))
        assert float(jnp.max(jnp.abs(batched - looped))) <= budget
        # The three candidates must not all coincide, or agreement says nothing.
        assert float(jnp.max(looped) - jnp.min(looped)) > budget

    def test_the_gradient_points_towards_the_ideal_corner(self) -> None:
        """Under minimisation, improving any objective can only raise the volume.

        Tolerance-free, and the one gradient claim that does not need a reference: a
        non-negative partial derivative in either objective would be a sign error.
        """
        gradient = jax.grad(lambda mean: self._acquisition(mean).sum())(self.MEAN)
        assert float(jnp.max(gradient)) < 0.0
        for delta in (0.05, 0.1, 0.2):
            moved = float(self._acquisition(self.MEAN - delta))
            assert moved > float(self._acquisition(self.MEAN))

    @pytest.mark.usefixtures("float64")
    def test_the_gradient_matches_a_central_difference(self) -> None:
        """Autodiff must equal a central difference of the same function.

        Run in float64 through the repository's own ``float64`` fixture: in float32 the
        quotient's cancellation error swamps the comparison, since it divides a
        difference of two values near ``0.24`` by ``2 * 1e-2``.

        Because the estimator is piecewise-multilinear in the mean at a fixed draw, a
        step inside one linear piece has no truncation error, so the budget is purely
        the reduction error of the two function evaluations divided by the step --
        see :data:`GRADIENT_STEP`.
        """
        mean = jnp.asarray(self.MEAN, dtype=jnp.float64)
        value = float(self._acquisition(mean))
        gradient = jax.grad(lambda candidate: self._acquisition(candidate).sum())(mean)
        budget = (
            _reduction_budget(terms=self._terms(), scale=value, dtype=jnp.float64) / GRADIENT_STEP
        )
        for axis in range(mean.shape[-1]):
            step = jnp.zeros_like(mean).at[0, axis].set(GRADIENT_STEP)
            central = (
                float(self._acquisition(mean + step)) - float(self._acquisition(mean - step))
            ) / (2.0 * GRADIENT_STEP)
            assert central == pytest.approx(float(gradient[0, axis]), abs=budget), (
                f"objective {axis}: autodiff {float(gradient[0, axis])!r} against central "
                f"difference {central!r}, budget {budget!r}"
            )
