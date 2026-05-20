"""Contract tests for :func:`aggregate_solver_solutions`.

Replaces the four solver-side ``*Wrapper`` shim classes that previously
lived under ``opifex.solvers.wrappers``. The aggregation function is a
single utility: callers run the solver(s) themselves and pass the list
of resulting :class:`Solution` objects in. The function packages
per-field means + sample-stack + ddof=1 variance + optional quantile
bands into a fresh :class:`Solution` whose ``auxiliary_data["uq"]``
carries the :class:`SolutionDistribution`-shaped payload.

Covers:

* Empty input rejection.
* Mismatched field-key rejection across sub-solutions.
* Out-of-range quantile rejection.
* Per-field mean and ddof=1 variance correctness.
* Quantile placement under ``auxiliary_data["uq"]["quantiles"]``.
* ``ensemble_size`` metadata bookkeeping.
* ``converged`` AND-aggregation across sub-solutions.
* No-in-place-mutation contract on the input list.
"""

from __future__ import annotations

import dataclasses as dc

import jax.numpy as jnp
import pytest

from opifex.core.solver.interface import Solution
from opifex.uncertainty.scientific.solutions import aggregate_solver_solutions


def _make_solution(value: float, *, key: str = "u", converged: bool = True) -> Solution:
    return Solution(
        fields={key: jnp.full((4,), value)},
        metrics={"loss": value},
        execution_time=0.5,
        converged=converged,
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_aggregate_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="at least one solution"):
        aggregate_solver_solutions([])


def test_aggregate_rejects_mismatched_field_keys() -> None:
    a = _make_solution(1.0, key="u")
    b = Solution(
        fields={"p": jnp.zeros((4,))},
        metrics={},
        execution_time=0.0,
        converged=True,
    )
    with pytest.raises(ValueError, match="same field keys"):
        aggregate_solver_solutions([a, b])


def test_aggregate_rejects_out_of_range_quantile() -> None:
    solutions = [_make_solution(v) for v in (1.0, 2.0, 3.0)]
    with pytest.raises(ValueError, match="quantile"):
        aggregate_solver_solutions(solutions, quantiles=(1.5,))


# ---------------------------------------------------------------------------
# Mean + variance contract
# ---------------------------------------------------------------------------


def test_aggregate_returns_mean_per_field() -> None:
    solutions = [_make_solution(v) for v in (1.0, 3.0, 5.0)]
    out = aggregate_solver_solutions(solutions)
    assert jnp.allclose(out.fields["u"], 3.0)  # mean of 1, 3, 5


def test_aggregate_uses_ddof_one_variance() -> None:
    solutions = [_make_solution(v) for v in (1.0, 3.0, 5.0)]
    out = aggregate_solver_solutions(solutions)
    # variance with ddof=1 of (1, 3, 5) is 4.0
    assert jnp.allclose(out.auxiliary_data["uq"]["variance"]["u"], 4.0)


def test_aggregate_singleton_input_returns_zero_variance() -> None:
    out = aggregate_solver_solutions([_make_solution(2.5)])
    assert jnp.allclose(out.auxiliary_data["uq"]["variance"]["u"], 0.0)
    assert jnp.allclose(out.fields["u"], 2.5)


# ---------------------------------------------------------------------------
# Samples preserved
# ---------------------------------------------------------------------------


def test_aggregate_preserves_raw_sample_stack_under_aux_data() -> None:
    solutions = [_make_solution(v) for v in (1.0, 3.0, 5.0)]
    out = aggregate_solver_solutions(solutions)
    samples = out.auxiliary_data["uq"]["samples"]["u"]
    assert samples.shape == (3, 4)
    assert jnp.allclose(samples[:, 0], jnp.array([1.0, 3.0, 5.0]))


# ---------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------


def test_aggregate_stores_quantiles_when_requested() -> None:
    solutions = [_make_solution(v) for v in (1.0, 2.0, 3.0, 4.0, 5.0)]
    out = aggregate_solver_solutions(solutions, quantiles=(0.05, 0.95))
    quantiles = out.auxiliary_data["uq"]["quantiles"]
    assert set(quantiles.keys()) == {0.05, 0.95}
    assert jnp.allclose(quantiles[0.05]["u"], 1.2)
    assert jnp.allclose(quantiles[0.95]["u"], 4.8)


def test_aggregate_omits_quantiles_block_when_none_requested() -> None:
    out = aggregate_solver_solutions([_make_solution(1.0), _make_solution(2.0)])
    assert out.auxiliary_data["uq"]["quantiles"] == {}


# ---------------------------------------------------------------------------
# Metadata + bookkeeping
# ---------------------------------------------------------------------------


def test_aggregate_metadata_carries_ensemble_size_and_extra_entries() -> None:
    solutions = [_make_solution(v) for v in (1.0, 2.0, 3.0)]
    out = aggregate_solver_solutions(
        solutions, metadata=(("method", "deep_ensemble"),)
    )
    metadata = dict(out.auxiliary_data["uq"]["metadata"])
    assert metadata["uncertainty_sources"] == ("ensemble",)
    assert metadata["ensemble_size"] == 3
    assert metadata["method"] == "deep_ensemble"
    assert out.metrics["ensemble_size"] == 3


def test_aggregate_converged_is_and_aggregation() -> None:
    solutions = [
        _make_solution(1.0, converged=True),
        _make_solution(2.0, converged=False),
    ]
    out = aggregate_solver_solutions(solutions)
    assert out.converged is False


def test_aggregate_execution_time_is_sum_across_subsolutions() -> None:
    solutions = [_make_solution(1.0), _make_solution(2.0), _make_solution(3.0)]
    out = aggregate_solver_solutions(solutions)
    assert out.execution_time == pytest.approx(1.5)  # 3 * 0.5


# ---------------------------------------------------------------------------
# Immutability — source solutions must not be mutated
# ---------------------------------------------------------------------------


def test_aggregate_does_not_mutate_input_solutions() -> None:
    solutions = [_make_solution(v) for v in (1.0, 3.0)]
    original_fields_u_0 = solutions[0].fields["u"]
    out = aggregate_solver_solutions(solutions)
    # Mutating the aggregate's mean array does not propagate back.
    out.fields["u"] = jnp.full((4,), 99.0)
    assert jnp.allclose(solutions[0].fields["u"], original_fields_u_0)


def test_aggregate_returns_frozen_dataclass() -> None:
    """``Solution`` itself is ``frozen=True``; the aggregated output
    inherits that contract."""
    out = aggregate_solver_solutions([_make_solution(1.0)])
    with pytest.raises(dc.FrozenInstanceError):
        out.execution_time = 99.0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# summarize_stacked_sample_solution — generative case
# ---------------------------------------------------------------------------


def _make_stacked_solution(values: list[float]) -> Solution:
    """Single Solution whose 'u' field is a (num_samples, 4) sample stack."""
    return Solution(
        fields={"u": jnp.stack([jnp.full((4,), v) for v in values], axis=0)},
        metrics={"loss": 0.1, "log_likelihood": -1.23},
        execution_time=2.5,
        converged=True,
    )


def test_summarize_stacked_returns_per_field_mean() -> None:
    from opifex.uncertainty.scientific.solutions import summarize_stacked_sample_solution

    out = summarize_stacked_sample_solution(_make_stacked_solution([1.0, 3.0, 5.0]))
    assert jnp.allclose(out.fields["u"], 3.0)


def test_summarize_stacked_uses_ddof_one_variance() -> None:
    from opifex.uncertainty.scientific.solutions import summarize_stacked_sample_solution

    out = summarize_stacked_sample_solution(_make_stacked_solution([1.0, 3.0, 5.0]))
    assert jnp.allclose(out.auxiliary_data["uq"]["variance"]["u"], 4.0)


def test_summarize_stacked_records_log_likelihood_alias() -> None:
    from opifex.uncertainty.scientific.solutions import summarize_stacked_sample_solution

    out = summarize_stacked_sample_solution(_make_stacked_solution([1.0, 2.0]))
    assert out.metrics["uq_method"] == "generative_sampling"
    assert out.metrics["mean_log_likelihood"] == pytest.approx(-1.23)


def test_summarize_stacked_passes_through_scalar_fields() -> None:
    from opifex.uncertainty.scientific.solutions import summarize_stacked_sample_solution

    raw = Solution(
        fields={
            "u": jnp.stack([jnp.zeros((4,)), jnp.ones((4,))], axis=0),
            "scalar_field": jnp.asarray(42.0),
        },
        metrics={},
        execution_time=0.0,
        converged=True,
    )
    out = summarize_stacked_sample_solution(raw)
    # Scalar field passes through unchanged in fields; stacked field is meaned.
    assert jnp.allclose(out.fields["scalar_field"], 42.0)
    assert jnp.allclose(out.fields["u"], 0.5)


def test_summarize_stacked_rejects_out_of_range_quantile() -> None:
    from opifex.uncertainty.scientific.solutions import summarize_stacked_sample_solution

    with pytest.raises(ValueError, match="quantile"):
        summarize_stacked_sample_solution(
            _make_stacked_solution([1.0, 2.0]), quantiles=(2.0,)
        )


def test_summarize_stacked_rejects_mismatched_sample_lengths() -> None:
    from opifex.uncertainty.scientific.solutions import summarize_stacked_sample_solution

    raw = Solution(
        fields={
            "u": jnp.stack([jnp.zeros((4,)), jnp.ones((4,))], axis=0),
            "p": jnp.stack([jnp.zeros((4,))], axis=0),  # length 1 vs 2 → mismatch
        },
        metrics={},
        execution_time=0.0,
        converged=True,
    )
    with pytest.raises(ValueError, match="sample-axis length"):
        summarize_stacked_sample_solution(raw)
