"""Tests for profiling event coordinator."""

import json

import pytest

from opifex.benchmarking.profiling.event_coordinator import (
    create_shared_coordinator,
    EventCoordinator,
    ProfilingEvent,
    ProfilingTimeline,
)


class TestProfilingEvent:
    """Tests for ProfilingEvent dataclass."""

    def test_create_minimal_event(self):
        """Create event with required fields only."""
        event = ProfilingEvent(timestamp=0.0, event_type="test", profiler_id="p1")
        assert event.event_type == "test"
        assert event.profiler_id == "p1"
        assert event.duration_ms is None
        assert event.data == {}

    def test_frozen_event(self):
        """Event is immutable."""
        event = ProfilingEvent(timestamp=0.0, event_type="test", profiler_id="p1")
        with pytest.raises(AttributeError):
            event.timestamp = 1.0  # type: ignore[misc]


class TestProfilingTimeline:
    """Tests for ProfilingTimeline."""

    def test_empty_timeline_duration(self):
        """Empty timeline has zero duration."""
        tl = ProfilingTimeline()
        assert tl.get_timeline_duration() == 0.0

    def test_add_and_retrieve_events(self):
        """Events can be added and retrieved."""
        tl = ProfilingTimeline()
        tl.start_timeline()
        tl.add_event("start", "profiler_a")
        tl.add_event("stop", "profiler_a", duration_ms=10.0)

        events = tl.get_events()
        assert len(events) == 2
        assert events[0].event_type == "start"
        assert events[1].duration_ms == 10.0

    def test_filter_events_by_profiler(self):
        """Events can be filtered by profiler ID."""
        tl = ProfilingTimeline()
        tl.start_timeline()
        tl.add_event("e1", "profiler_a")
        tl.add_event("e2", "profiler_b")
        tl.add_event("e3", "profiler_a")

        a_events = tl.get_events(profiler_id="profiler_a")
        assert len(a_events) == 2

    def test_start_clears_events(self):
        """Starting timeline clears previous events."""
        tl = ProfilingTimeline()
        tl.start_timeline()
        tl.add_event("old", "p1")
        tl.start_timeline()
        assert len(tl.get_events()) == 0


class TestEventCoordinator:
    """Tests for EventCoordinator."""

    def test_register_unregister_profiler(self):
        """Profilers can be registered and unregistered."""
        coord = EventCoordinator()
        coord.register_profiler("p1")
        coord.register_profiler("p2")
        assert "p1" in coord.active_profilers
        assert "p2" in coord.active_profilers

        coord.unregister_profiler("p1")
        assert "p1" not in coord.active_profilers

    def test_events_ignored_outside_session(self):
        """Events added outside a session are silently ignored."""
        coord = EventCoordinator()
        coord.add_event("test", "p1")
        assert len(coord.timeline.get_events()) == 0

    def test_profiling_session_records_events(self):
        """Events within a session are recorded."""
        coord = EventCoordinator()
        coord.register_profiler("p1")

        with coord.profiling_session(enable_jax_profiler=False):
            coord.add_event("compute", "p1", duration_ms=5.0)

        events = coord.timeline.get_events()
        # Should have session start, compute, session end events
        assert any(e.event_type == "compute" for e in events)

    def test_time_function(self):
        """time_function measures execution and records events."""
        coord = EventCoordinator()

        with coord.profiling_session(enable_jax_profiler=False):
            result, duration = coord.time_function(
                lambda: sum(range(1000)),
                profiler_id="test",
                operation_name="sum_range",
            )

        assert result == 499500
        assert duration > 0

    def test_profiling_summary(self):
        """Summary contains expected fields."""
        coord = EventCoordinator()
        coord.register_profiler("p1")

        with coord.profiling_session(enable_jax_profiler=False):
            coord.add_event("op", "p1", duration_ms=10.0)

        summary = coord.get_profiling_summary()
        assert "session_duration_s" in summary
        assert "total_events" in summary
        assert "profiler_statistics" in summary
        assert summary["total_events"] > 0

    def test_export_json(self):
        """Export produces valid JSON."""
        coord = EventCoordinator()
        with coord.profiling_session(enable_jax_profiler=False):
            coord.add_event("test", "p1")

        output = coord.export_timeline("json")
        data = json.loads(output)
        assert "events" in data
        assert len(data["events"]) > 0

    def test_export_csv(self):
        """Export produces CSV with header."""
        coord = EventCoordinator()
        with coord.profiling_session(enable_jax_profiler=False):
            coord.add_event("test", "p1")

        output = coord.export_timeline("csv")
        lines = output.strip().split("\n")
        assert lines[0].startswith("timestamp")
        assert len(lines) > 1

    def test_nested_session_reuses_outer(self):
        """Nested profiling session reuses the outer session."""
        coord = EventCoordinator()
        with coord.profiling_session(enable_jax_profiler=False):
            coord.add_event("outer", "p1")
            with coord.profiling_session(enable_jax_profiler=False):
                coord.add_event("inner", "p1")

        events = coord.timeline.get_events()
        assert any(e.event_type == "outer" for e in events)
        assert any(e.event_type == "inner" for e in events)


class TestCreateSharedCoordinator:
    """Tests for factory function."""

    def test_creates_coordinator(self):
        """Factory returns a fresh EventCoordinator."""
        coord = create_shared_coordinator()
        assert isinstance(coord, EventCoordinator)
        assert len(coord.active_profilers) == 0
