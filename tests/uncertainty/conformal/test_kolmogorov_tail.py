"""The Kolmogorov tail, and the two ways a single series gets it wrong.

``Q(lambda)`` has two series expressions, each accurate where the other is not. A single
alternating series -- the form usually written down -- fails for small ``lambda``, and the
failure is total rather than gradual: its terms do not decay until ``k > 4.29 / lambda``,
so a truncation returns nothing like the answer below that. The complementary theta form
fails at the other end, where the tail is small and forming it as ``1 - body`` loses every
significant digit.

The split is at the median of the distribution (van Mulbregt 2018, sec. 4.1), which is
what makes both safe: whichever branch runs computes the smaller of the tail and the body,
so the complement is never taken from a number near one.

Accuracy is checked against ``scipy.special.kolmogorov``, which is cephes and switches at
the same point, so the two agree by construction rather than by coincidence.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import kolmogorov

from opifex.uncertainty.conformal.exchangeability import _kolmogorov_tail, ks_two_sample_pvalue


FLOAT32_EPS = float(np.finfo(np.float32).eps)


def _tail(lam: float) -> float:
    return float(_kolmogorov_tail(jnp.asarray(lam, jnp.float32)))


class TestItMatchesTheReference:
    """Agreement with cephes across the whole usable range."""

    @pytest.mark.parametrize(
        "lam", [0.05, 0.1, 0.3, 0.5, 0.7, 0.81, 0.82, 0.83, 1.0, 1.5, 2.0, 3.0, 4.0]
    )
    def test_the_relative_error_is_at_the_float32_floor(self, lam: float) -> None:
        # Relative, not absolute: the tail spans twenty orders of magnitude over this
        # range, so an absolute bound would be satisfied by returning zero.
        reference = kolmogorov(lam)

        assert abs(_tail(lam) - reference) / reference < 2.0 * FLOAT32_EPS

    def test_the_branches_agree_where_they_meet(self) -> None:
        # A step at the cutover would be a real defect: the same argument would give two
        # answers depending on which side of a constant it fell. Comparing the change
        # across the cutover with the change over an equal interval wholly inside one
        # branch separates a genuine jump from the function's own slope, which is about
        # -1.59 here and would otherwise be mistaken for one.
        step = 2e-4
        across = abs(_tail(0.82 + step) - _tail(0.82 - step))
        within = abs(_tail(0.82 - step) - _tail(0.82 - 3 * step))

        assert across == pytest.approx(within, rel=0.2)


class TestTheSmallArgumentRegime:
    """Where a lone alternating series fails."""

    def test_identical_samples_are_perfectly_compatible(self) -> None:
        # D = 0, so lambda = 0 and the answer is one. An alternating series truncated at
        # an even count returns zero here, which reads as "certainly different" for data
        # compared against itself.
        assert _tail(0.0) == pytest.approx(1.0)

    @pytest.mark.parametrize("lam", [1e-8, 1e-4, 1e-3, 0.01, 0.03])
    def test_a_tiny_statistic_reports_near_certainty(self, lam: float) -> None:
        assert _tail(lam) == pytest.approx(1.0, abs=1e-6)

    def test_the_whole_small_regime_is_covered_not_just_the_origin(self) -> None:
        # Guarding lambda == 0 alone would leave this interval wrong, since the failure is
        # of the series' convergence rather than of one special value.
        for lam in np.geomspace(1e-9, 0.05, 40):
            assert _tail(float(lam)) == pytest.approx(kolmogorov(lam), rel=2.0 * FLOAT32_EPS)


class TestTheLargeArgumentRegime:
    """Where the complementary form loses the answer to cancellation."""

    @pytest.mark.parametrize(("lam", "expected"), [(3.0, 3.0460e-08), (4.0, 2.5328e-14)])
    def test_a_small_tail_keeps_its_significant_digits(self, lam: float, expected: float) -> None:
        assert _tail(lam) == pytest.approx(expected, rel=1e-4)

    def test_the_tail_decreases_monotonically(self) -> None:
        values = [_tail(lam) for lam in np.linspace(0.05, 4.0, 60)]

        assert all(later <= earlier for earlier, later in itertools.pairwise(values))


class TestItSurvivesATransform:
    """The reciprocal in the theta branch is evaluated on every lane, taken or not."""

    def test_the_gradient_is_finite_at_the_origin(self) -> None:
        # Both branches of a `where` are evaluated, so an unguarded 1 / lambda would make
        # inf * 0 here and return NaN from the branch that was not selected.
        gradient = jax.grad(_kolmogorov_tail)(jnp.asarray(0.0, jnp.float32))

        assert bool(jnp.isfinite(gradient))

    def test_the_gradient_is_finite_across_both_branches(self) -> None:
        gradients = jax.vmap(jax.grad(_kolmogorov_tail))(
            jnp.asarray([0.0, 0.1, 0.5, 0.82, 1.0, 3.0], jnp.float32)
        )

        assert bool(jnp.all(jnp.isfinite(gradients)))

    def test_it_batches(self) -> None:
        batched = jax.vmap(_kolmogorov_tail)(jnp.asarray([0.0, 1.0, 3.0], jnp.float32))

        assert batched.shape == (3,)
        assert float(batched[0]) == pytest.approx(1.0)

    def test_the_p_value_traces_once_under_jit(self) -> None:
        traces = {"count": 0}

        @jax.jit
        def probability(first: jax.Array, second: jax.Array) -> jax.Array:
            traces["count"] += 1
            return ks_two_sample_pvalue(calibration_scores=first, evaluation_scores=second)

        key = jax.random.key(0)
        for seed in range(3):
            sample = jax.random.normal(jax.random.key(seed), (64,))
            probability(jax.random.normal(key, (64,)), sample)

        assert traces["count"] == 1


class TestTheReportedOutcome:
    """The pass flag is traced data, so it batches and costs no recompilation."""

    def test_identical_samples_pass(self) -> None:
        from opifex.uncertainty.conformal.exchangeability import check_exchangeability

        sample = jax.random.normal(jax.random.key(0), (64,))
        report = check_exchangeability(calibration_scores=sample, evaluation_scores=sample)

        assert bool(report.passes)
        assert float(report.p_value) == pytest.approx(1.0)

    def test_a_shifted_sample_fails(self) -> None:
        from opifex.uncertainty.conformal.exchangeability import check_exchangeability

        base = jax.random.normal(jax.random.key(0), (64,))
        shifted = jax.random.normal(jax.random.key(1), (64,)) + 5.0
        report = check_exchangeability(calibration_scores=base, evaluation_scores=shifted)

        assert not bool(report.passes)

    def test_the_outcome_is_not_part_of_the_pytree_metadata(self) -> None:
        # As a static field each outcome would be a separate treedef, so passing a report
        # into a compiled function would recompile per result.
        from opifex.uncertainty.conformal.exchangeability import check_exchangeability

        base = jax.random.normal(jax.random.key(0), (64,))
        shifted = jax.random.normal(jax.random.key(1), (64,)) + 5.0
        passing = check_exchangeability(calibration_scores=base, evaluation_scores=base)
        failing = check_exchangeability(calibration_scores=base, evaluation_scores=shifted)

        assert jax.tree.structure(passing) == jax.tree.structure(failing)
