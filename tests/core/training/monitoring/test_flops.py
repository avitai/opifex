"""Tests for the FLOPs counter monitoring component."""

from opifex.core.training.monitoring.flops import FlopsCounter


def test_get_summary_on_fresh_instance_returns_dict() -> None:
    """get_summary must work on a freshly-constructed counter.

    Regression: a fresh FlopsCounter (without a prior reset_counters call)
    used to raise AttributeError because self.profile_data was only assigned
    inside reset_counters. A new instance must already expose an empty
    profile_data so the summary can be produced.
    """
    counter = FlopsCounter()

    summary = counter.get_summary()

    assert isinstance(summary, dict)
    assert summary["total_flops"] == 0
    assert summary["operation_counts"] == {}
    assert summary["profile_data"] == {}
    assert summary["enable_profiling"] is True
