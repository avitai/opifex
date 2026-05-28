"""Typed PAC-Bayes certificate container and driver.

:class:`PACBayesCertificate` is the canonical record produced after evaluating
a PAC-Bayes bound on a fitted posterior. It is a Pattern-B container
(``flax.struct.dataclass``) so array fields (``bound_value``, ``kl``,
``dataset_size``) flow transparently through ``jax.jit`` / ``jax.grad`` while
the bound name and metadata stay as hashable static aux_data.

The driver :func:`pac_bayes_certificate` is intentionally generic: ``model``,
``posterior``, and ``data`` are protocol-light — the function only consults

* ``posterior.kl_divergence`` (scalar JAX array) or ``posterior.kl`` (alias),
* ``data["empirical_risk"]`` (scalar) and ``data["dataset_size"]`` (positive
  scalar / int),

so any backend (Bayesian NNX module, variational state container, deep-ensemble
adapter) that exposes those two scalars can be certified.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

from opifex.uncertainty.pac_bayes.bounds import (
    catoni_bound,
    mcallester_bound,
    quadratic_bound,
)
from opifex.uncertainty.types import metadata_to_dict, MetadataItems


# The three canonical PAC-Bayes bound family names. ``_BOUND_REGISTRY`` is the
# only place the family name strings live; consumers must not duplicate them.
#
# A posterior accepted by :func:`pac_bayes_certificate` exposes ``kl_divergence``
# (attribute or zero-arg method) or ``kl`` (attribute alias). We keep that
# contract documented in the driver docstring rather than as a Protocol, since
# Pyright's runtime-checkable Protocol with a callable-or-attribute member
# yields false-positive type errors at every call site.
_BOUND_REGISTRY: dict[str, Any] = {
    "mcallester": mcallester_bound,
    "catoni": catoni_bound,
    "quadratic": quadratic_bound,
}


def _extract_kl(posterior: Any) -> jax.Array:
    """Resolve a scalar KL from ``posterior.kl_divergence`` or ``posterior.kl``.

    Both attributes are accepted because Opifex Bayesian layers expose
    ``kl_divergence()`` as a method while variational-inference results record
    it under the ``kl`` field. Callable attributes are evaluated with no
    arguments.

    Raises:
        TypeError: If neither attribute exists.

    """
    for name in ("kl_divergence", "kl"):
        if hasattr(posterior, name):
            value = getattr(posterior, name)
            if callable(value):
                value = value()
            return jnp.asarray(value, dtype=jnp.float32)
    raise TypeError(
        "posterior must expose a 'kl_divergence' or 'kl' attribute (scalar or callable)."
    )


def _extract_scalar(data: dict[str, Any], key: str) -> jax.Array:
    """Pull a required scalar from the ``data`` mapping."""
    if key not in data:
        raise KeyError(f"data is missing required key {key!r} for PAC-Bayes certification.")
    return jnp.asarray(data[key], dtype=jnp.float32)


@struct.dataclass
class PACBayesCertificate:
    """Typed PAC-Bayes certificate record.

    Array fields (``bound_value``, ``kl``, ``dataset_size``) are pytree leaves
    so the container flows transparently through ``jax.jit`` / ``jax.grad``;
    ``bound_name``, ``delta``, and ``metadata`` are static aux_data.

    Fields:

    * ``bound_value`` — scalar JAX array; the population-risk upper bound.
    * ``kl`` — scalar JAX array; KL divergence between posterior and prior.
    * ``dataset_size`` — scalar JAX array; ``n`` used to evaluate the bound.
    * ``bound_name`` — one of ``"mcallester"``, ``"catoni"``, ``"quadratic"``.
    * ``delta`` — confidence parameter in ``(0, 1)``.
    * ``metadata`` — immutable ``tuple[tuple[str, Any], ...]`` aux_data; at
      minimum carries a ``"tightness"`` entry (``bound_value - empirical_risk``).

    ``validate()`` is public and not wired into ``__post_init__`` /
    ``tree_unflatten`` so JIT-traced reconstructions never trigger spurious
    failures.
    """

    bound_value: jax.Array
    kl: jax.Array
    dataset_size: jax.Array
    bound_name: str = struct.field(pytree_node=False)
    delta: float = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata tuple."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate the certificate; never called from the pytree path.

        Raises:
            ValueError: If ``bound_name`` is not a known family, ``delta`` is
                outside ``(0, 1)``, or any scalar is non-finite.

        """
        if self.bound_name not in _BOUND_REGISTRY:
            raise ValueError(
                f"unknown bound_name {self.bound_name!r}; "
                f"expected one of {sorted(_BOUND_REGISTRY)}."
            )
        if not 0.0 < float(self.delta) < 1.0:
            raise ValueError(f"delta must be in (0, 1); got {self.delta!r}.")
        for name, value in (
            ("bound_value", self.bound_value),
            ("kl", self.kl),
            ("dataset_size", self.dataset_size),
        ):
            if not bool(jnp.all(jnp.isfinite(value))):
                raise ValueError(f"PACBayesCertificate.{name} must be finite.")


def pac_bayes_certificate(
    model: Any,
    posterior: Any,
    data: dict[str, Any],
    *,
    delta: float,
    bound_name: str = "mcallester",
    beta: float = 1.0,
) -> PACBayesCertificate:
    """Evaluate a PAC-Bayes bound and return a typed certificate.

    Args:
        model: Reference to the model whose posterior is being certified. The
            driver does not introspect the model; it is recorded in metadata
            for traceability.
        posterior: An object exposing ``kl_divergence`` (attribute or method)
            returning a scalar JAX array. Bayesian-NNX layers and
            ``VariationalResult`` containers both satisfy this protocol.
        data: Mapping with required keys ``"empirical_risk"`` (scalar) and
            ``"dataset_size"`` (positive scalar / int).
        delta: Confidence parameter in ``(0, 1)``.
        bound_name: Which bound family to evaluate; one of
            ``"mcallester"``, ``"catoni"``, ``"quadratic"``.
        beta: Catoni temperature; ignored for other bound families.

    Returns:
        A :class:`PACBayesCertificate` whose ``bound_value`` is a scalar JAX
        array.

    Raises:
        ValueError: If ``delta`` is outside ``(0, 1)`` or ``bound_name`` is
            not a known family.
        KeyError: If ``data`` lacks ``"empirical_risk"`` or ``"dataset_size"``.

    """
    if bound_name not in _BOUND_REGISTRY:
        raise ValueError(
            f"unknown bound_name {bound_name!r}; expected one of {sorted(_BOUND_REGISTRY)}."
        )
    if not 0.0 < float(delta) < 1.0:
        raise ValueError(f"delta must be in (0, 1); got {delta!r}.")

    empirical_risk = _extract_scalar(data, "empirical_risk")
    dataset_size = _extract_scalar(data, "dataset_size")
    kl_value = _extract_kl(posterior)

    if bound_name == "catoni":
        bound_value = catoni_bound(empirical_risk, kl_value, dataset_size, delta, beta=beta)
    else:
        bound_fn = _BOUND_REGISTRY[bound_name]
        bound_value = bound_fn(empirical_risk, kl_value, dataset_size, delta)

    tightness = bound_value - empirical_risk
    metadata: MetadataItems = (
        ("tightness", float(tightness)),
        ("empirical_risk", float(empirical_risk)),
        ("model_repr", type(model).__name__),
    )
    return PACBayesCertificate(
        bound_value=bound_value,
        kl=kl_value,
        dataset_size=dataset_size,
        bound_name=bound_name,
        delta=float(delta),
        metadata=metadata,
    )


__all__ = ["PACBayesCertificate", "pac_bayes_certificate"]
