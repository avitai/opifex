"""Shaping of experiment records into what a run logger accepts.

Both functions are pure: the backend calls them and forwards the result to its
``RunLogger``.
"""

from __future__ import annotations

import dataclasses
import statistics
from collections import deque
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from opifex.mlops.experiment import PhysicsMetadata, PhysicsMetrics


type Param = str | int | float | bool

_PREFIX = "physics"


def physics_metadata_params(metadata: PhysicsMetadata) -> dict[str, Param]:
    """Render physics metadata as run parameters under the ``physics.`` prefix.

    Scalars are kept as they are, string lists are comma-joined, tuples are rendered
    with ``str``, and mappings contribute one parameter per key
    (``physics.<field>.<key>``). Unset fields contribute nothing.

    Args:
        metadata: The metadata to render.

    Returns:
        Parameter name to value, every value one MLflow accepts.
    """
    params: dict[str, Param] = {}
    for field in dataclasses.fields(metadata):
        value = getattr(metadata, field.name)
        if value is None:
            continue
        key = f"{_PREFIX}.{field.name}"
        if isinstance(value, Mapping):
            params.update({f"{key}.{name}": _param(item) for name, item in value.items()})
        elif isinstance(value, list):
            params[key] = ",".join(str(item) for item in value)
        else:
            params[key] = _param(value)
    return params


def _param(value: object) -> Param:
    """Keep the values MLflow accepts; render every other one with ``str``."""
    return value if isinstance(value, str | int | float | bool) else str(value)


def flatten_physics_metrics(metrics: PhysicsMetrics) -> dict[str, float]:
    """Flatten a metrics record into one float per dotted key.

    Mappings are flattened recursively (``field.key``), number sequences contribute
    their ``mean``, ``min`` and ``max``, and unset fields and empty sequences
    contribute nothing.

    Args:
        metrics: One of the physics metrics records.

    Returns:
        Metric name to value.

    Raises:
        TypeError: If a value is neither a number, a mapping nor a sequence of numbers.
    """
    flat: dict[str, float] = {}
    pending = deque(
        (field.name, getattr(metrics, field.name)) for field in dataclasses.fields(metrics)
    )
    while pending:
        key, value = pending.popleft()
        if value is None:
            continue
        if isinstance(value, bool | int | float):
            flat[key] = float(value)
        elif isinstance(value, Mapping):
            pending.extend((f"{key}.{name}", item) for name, item in value.items())
        elif isinstance(value, Sequence) and not isinstance(value, str):
            if not value:
                continue
            if not all(isinstance(item, bool | int | float) for item in value):
                raise TypeError(f"{key}: metric sequences must hold numbers, got {value!r}")
            numbers = [float(item) for item in value]
            flat[f"{key}.mean"] = statistics.fmean(numbers)
            flat[f"{key}.min"] = min(numbers)
            flat[f"{key}.max"] = max(numbers)
        else:
            raise TypeError(f"{key}: a {type(value).__name__} cannot be logged as a metric")
    return flat
