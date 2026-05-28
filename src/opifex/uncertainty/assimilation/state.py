"""Digital-twin / data-assimilation state container (Task 6.7).

``AssimilationState`` is a Pattern-B PyTree carrying:

* ``mean`` — physical-state estimate, shape ``(state_dim,)``.
* ``covariance`` — state covariance, shape ``(state_dim, state_dim)``.
* ``time`` — scalar timestamp the estimate corresponds to.
* ``metadata`` — static, hashable digital-twin annotations
  (``tuple[tuple[str, Any], ...]``).

Per GUIDE_ALIGNMENT §5a pattern (B): ``flax.struct.dataclass`` with
``slots=True`` + ``kw_only=True``; ``metadata`` is marked
``pytree_node=False`` so it stays static under JAX transforms.
``validate()`` is a public method, **not** wired into
``__post_init__`` or ``tree_unflatten`` (GUIDE_ALIGNMENT item 7).
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from flax import struct


if TYPE_CHECKING:
    import jax


_REQUIRED_METADATA_KEYS = (
    "physical_state",
    "observation_uncertainty",
    "model_discrepancy",
    "numerical_uncertainty",
    "calibration_uncertainty",
)


@struct.dataclass(frozen=True)
class AssimilationState:
    """Digital-twin state estimate + uncertainty bookkeeping.

    Attributes:
        mean: Physical-state mean, shape ``(state_dim,)``.
        covariance: State covariance, shape ``(state_dim, state_dim)``.
        time: Scalar timestamp.
        metadata: Static, hashable annotations distinguishing physical
            state, observation uncertainty, model discrepancy,
            numerical uncertainty, and calibration uncertainty. Stored
            as ``tuple[tuple[str, Any], ...]`` so it survives
            ``jax.tree.flatten`` as static aux data.
    """

    mean: jax.Array
    covariance: jax.Array
    time: jax.Array
    metadata: tuple[tuple[str, Any], ...] = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return ``metadata`` as a regular dict for ergonomic read access."""
        return dict(self.metadata)

    def validate(self) -> None:
        """Public validation — must be called by callers, not the tree machinery.

        Raises:
            ValueError: If ``mean`` is not 1-D, ``covariance`` is not
                square / matching ``mean`` size, or required metadata
                fields are missing.
        """
        if self.mean.ndim != 1:
            raise ValueError(f"mean must be 1-D; got shape {self.mean.shape}.")
        if self.covariance.shape != (self.mean.shape[0], self.mean.shape[0]):
            raise ValueError(
                f"covariance shape {self.covariance.shape} does not match "
                f"mean dim {self.mean.shape[0]}."
            )
        meta_dict = self.metadata_dict()
        missing = [k for k in _REQUIRED_METADATA_KEYS if k not in meta_dict]
        if missing:
            raise ValueError(
                f"AssimilationState metadata is missing required keys: {missing}. "
                f"Required keys are {_REQUIRED_METADATA_KEYS}."
            )


def build_default_metadata(
    *,
    physical_state: str,
    observation_uncertainty: float,
    model_discrepancy: float,
    numerical_uncertainty: float,
    calibration_uncertainty: float,
    extra: dict[str, Any] | None = None,
) -> tuple[tuple[str, Any], ...]:
    """Helper that builds a metadata tuple satisfying ``validate()``.

    Designed so the canonical five uncertainty-source slots are named
    consistently across callers; ``extra`` allows experiment-specific
    annotations without touching the required slots.
    """
    items: list[tuple[str, Any]] = [
        ("physical_state", physical_state),
        ("observation_uncertainty", observation_uncertainty),
        ("model_discrepancy", model_discrepancy),
        ("numerical_uncertainty", numerical_uncertainty),
        ("calibration_uncertainty", calibration_uncertainty),
    ]
    if extra:
        items.extend(extra.items())
    return tuple(items)


__all__ = ["AssimilationState", "build_default_metadata"]
