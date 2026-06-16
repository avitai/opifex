"""Smoke tests enforcing the example contract across ``examples/``.

Every example must expose a callable ``main() -> dict`` and guard execution
under ``if __name__ == "__main__":`` so it can be imported (and, for the
lightweight ones, run) without side effects. See
``docs/development/example-documentation-design.md``.

- ``test_example_exposes_main`` runs for every example (cheap: imports only).
- ``test_example_main_runs`` is marked ``slow``; it actually calls ``main()``
  and asserts the returned metrics are finite. Deselect with ``-m "not slow"``.
"""

from __future__ import annotations

import math

import pytest
from tests.examples._harness import discover_examples, load_example


_EXAMPLES = discover_examples()
_IDS = [str(p.relative_to(p.parents[1])) for p in _EXAMPLES]


@pytest.mark.parametrize("path", _EXAMPLES, ids=_IDS)
def test_example_exposes_main(path) -> None:
    """Each example imports cleanly and exposes a callable ``main``."""
    module = load_example(path)
    assert callable(getattr(module, "main", None)), (
        f"{path} must expose a callable main(); flat scripts that run on import "
        "must be refactored per docs/development/example-documentation-design.md"
    )


@pytest.mark.slow
@pytest.mark.parametrize("path", _EXAMPLES, ids=_IDS)
def test_example_main_runs(path) -> None:
    """Running ``main()`` returns a dict of finite scalar metrics."""
    module = load_example(path)
    summary = module.main()
    assert isinstance(summary, dict) and summary, f"{path}: main() must return a non-empty dict"
    for key, value in summary.items():
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue  # non-scalar entries (shapes, labels) are allowed
        assert math.isfinite(scalar), f"{path}: metric {key!r} is not finite ({value})"
