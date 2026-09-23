"""The residual-shift diagnostic, and why its outcome is data rather than metadata.

The report carries a p-value and a pass flag, both arrays. Holding either as a Python
value -- or holding a status string derived from one -- would put the outcome in the
pytree's static metadata, so a passing and a failing report would have different
structures and a compiled consumer would recompile per result. The text a person reads
comes from :func:`shift_status` instead, on the host.
"""

from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.uncertainty.ood import residual_shift_diagnostic, shift_status, ShiftReport


def _residuals(seed: int, size: int, shift: float = 0.0) -> jax.Array:
    return jnp.asarray(shift + np.random.default_rng(seed).standard_normal(size))


def _diagnose(reference: jax.Array, observed: jax.Array) -> ShiftReport:
    return residual_shift_diagnostic(
        reference_residuals=reference, observed_residuals=observed, alpha=0.05
    )


class TestTheReport:
    """What a completed diagnostic holds."""

    def test_it_is_frozen(self) -> None:
        report = ShiftReport(p_value=jnp.asarray(0.5), passes=jnp.asarray(True))

        with pytest.raises(dc.FrozenInstanceError):
            report.passes = jnp.asarray(False)  # type: ignore[misc]

    def test_every_leaf_is_an_array(self) -> None:
        # A Python bool among the leaves would be traced into a weakly typed scalar;
        # one in the metadata would be part of the cache key.
        leaves = jax.tree.leaves(ShiftReport(p_value=jnp.asarray(0.5), passes=jnp.asarray(True)))

        assert leaves and all(isinstance(leaf, jax.Array) for leaf in leaves)

    def test_it_names_its_method(self) -> None:
        report = _diagnose(_residuals(0, 64), _residuals(1, 64))

        assert report.method == "ks_two_sample_residual"


class TestTheOutcome:
    """The diagnostic's verdict on two residual streams."""

    def test_exchangeable_residuals_pass(self) -> None:
        report = _diagnose(_residuals(0, 512), _residuals(1, 512))

        assert bool(report.passes)
        assert float(report.p_value) > 0.05

    def test_a_location_shift_is_detected(self) -> None:
        report = _diagnose(_residuals(2, 512), _residuals(3, 512, shift=2.0))

        assert not bool(report.passes)
        assert float(report.p_value) < 0.05

    def test_a_pass_is_reported_as_exchangeable(self) -> None:
        report = _diagnose(_residuals(0, 512), _residuals(1, 512))

        assert shift_status(report) == ("no_shift", "exchangeable")

    def test_a_failure_does_not_claim_conformal_validity(self) -> None:
        # Distribution-free coverage rests on exchangeability. Once the diagnostic
        # rejects it, the assumption must be reported as broken rather than omitted.
        report = _diagnose(_residuals(4, 256), _residuals(5, 256, shift=3.0))

        assert shift_status(report) == ("shift_detected", "shift_detected")


class TestTheMetadata:
    """Static fields record only what the outcome cannot change."""

    def test_it_records_the_level_and_both_sample_sizes(self) -> None:
        report = residual_shift_diagnostic(
            reference_residuals=_residuals(6, 256),
            observed_residuals=_residuals(7, 128),
            alpha=0.05,
        )
        metadata = dict(report.metadata)

        assert metadata["alpha"] == pytest.approx(0.05)
        assert int(metadata["reference_size"]) == 256
        assert int(metadata["observed_size"]) == 128

    def test_both_outcomes_share_one_structure(self) -> None:
        # The property the whole arrangement exists for: a status string stored per
        # outcome would give these two reports different treedefs, and a compiled
        # consumer a compilation apiece.
        reference = _residuals(8, 256)
        passing = _diagnose(reference, _residuals(9, 256))
        failing = _diagnose(reference, _residuals(10, 256, shift=3.0))

        assert jax.tree.structure(passing) == jax.tree.structure(failing)


class TestItSurvivesATransform:
    """The diagnostic runs where the residuals are produced."""

    def test_it_traces_once_for_either_outcome(self) -> None:
        traces = {"count": 0}

        @jax.jit
        def diagnose(reference: jax.Array, observed: jax.Array) -> ShiftReport:
            traces["count"] += 1
            return _diagnose(reference, observed)

        reference = _residuals(11, 128)
        diagnose(reference, _residuals(12, 128))
        diagnose(reference, _residuals(13, 128, shift=4.0))

        assert traces["count"] == 1

    def test_it_batches_with_one_verdict_per_element(self) -> None:
        reference = _residuals(14, 128)
        observed = jnp.stack(
            [_residuals(15, 128), _residuals(16, 128, shift=4.0), _residuals(17, 128)]
        )

        passes = jax.vmap(lambda row: _diagnose(reference, row).passes)(observed)

        np.testing.assert_array_equal(np.asarray(passes), [True, False, True])
