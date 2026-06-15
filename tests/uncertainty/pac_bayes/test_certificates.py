"""Tests for the PAC-Bayes certificate container and driver.

These tests pin the contract from Task 8.1 TDD steps #4 and #6:

* :func:`pac_bayes_certificate` returns a :class:`PACBayesCertificate`
  whose ``bound_value`` is a scalar JAX array.
* The driver records ``bound_name``, ``delta``, ``dataset_size``, ``kl`` and
  a ``tightness`` entry in metadata.
* ``delta`` outside ``(0, 1)`` raises ``ValueError`` — both at the driver
  level and via ``PACBayesCertificate.validate()``.
* The container is a valid JAX pytree (round-trips through
  ``jax.tree_util.tree_flatten`` / ``tree_unflatten``).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.pac_bayes.bounds import (
    catoni_bound,
    mcallester_bound,
    quadratic_bound,
)
from opifex.uncertainty.pac_bayes.certificates import (
    pac_bayes_certificate,
    PACBayesCertificate,
)


@dataclass
class _FakePosterior:
    """Synthetic posterior carrying a scalar KL — the only attribute the driver consults."""

    kl_divergence: jax.Array


@dataclass
class _FakeModel:
    """Synthetic model recorded only by ``type().__name__`` in metadata."""

    name: str = "fake"


def _data(empirical_risk: float = 0.05, dataset_size: int = 256) -> dict[str, jax.Array]:
    return {
        "empirical_risk": jnp.asarray(empirical_risk),
        "dataset_size": jnp.asarray(dataset_size),
    }


def _posterior(kl: float = 2.0) -> _FakePosterior:
    return _FakePosterior(kl_divergence=jnp.asarray(kl))


# ---- driver returns a typed scalar bound -----------------------------------


def test_pac_bayes_certificate_returns_typed_certificate_with_scalar_bound() -> None:
    cert = pac_bayes_certificate(_FakeModel(), _posterior(), _data(), delta=0.05)
    assert isinstance(cert, PACBayesCertificate)
    assert isinstance(cert.bound_value, jax.Array)
    assert cert.bound_value.shape == ()
    assert cert.bound_name == "mcallester"
    assert cert.delta == 0.05


def test_pac_bayes_certificate_matches_mcallester_formula() -> None:
    cert = pac_bayes_certificate(
        _FakeModel(), _posterior(kl=2.0), _data(empirical_risk=0.1, dataset_size=512), delta=0.05
    )
    expected = mcallester_bound(jnp.asarray(0.1), jnp.asarray(2.0), 512, 0.05)
    assert float(cert.bound_value) == pytest.approx(float(expected), rel=1e-5)


@pytest.mark.parametrize(
    ("bound_name", "expected_fn"),
    [
        ("catoni", lambda: catoni_bound(jnp.asarray(0.1), jnp.asarray(2.0), 512, 0.05)),
        ("quadratic", lambda: quadratic_bound(jnp.asarray(0.1), jnp.asarray(2.0), 512, 0.05)),
    ],
)
def test_pac_bayes_certificate_dispatches_to_named_bound(bound_name: str, expected_fn) -> None:
    cert = pac_bayes_certificate(
        _FakeModel(),
        _posterior(kl=2.0),
        _data(empirical_risk=0.1, dataset_size=512),
        delta=0.05,
        bound_name=bound_name,
    )
    assert cert.bound_name == bound_name
    assert float(cert.bound_value) == pytest.approx(float(expected_fn()), rel=1e-5)


def test_pac_bayes_certificate_records_required_metadata() -> None:
    cert = pac_bayes_certificate(_FakeModel(), _posterior(), _data(), delta=0.05)
    md = cert.metadata_dict()
    assert "tightness" in md
    assert "empirical_risk" in md
    assert "model_repr" in md
    # Tightness equals ``bound - empirical_risk`` and must be non-negative for a sensible bound.
    assert md["tightness"] == pytest.approx(
        float(cert.bound_value) - md["empirical_risk"], rel=1e-5
    )
    assert md["tightness"] >= 0.0


# ---- delta gating (TDD requirement #6) -------------------------------------


@pytest.mark.parametrize("bad_delta", [0.0, -0.1, 1.0, 1.5])
def test_pac_bayes_certificate_driver_rejects_delta_outside_open_unit_interval(
    bad_delta: float,
) -> None:
    with pytest.raises(ValueError, match=r"delta"):
        pac_bayes_certificate(_FakeModel(), _posterior(), _data(), delta=bad_delta)


def test_certificate_validate_rejects_invalid_delta() -> None:
    cert = PACBayesCertificate(
        bound_value=jnp.asarray(0.1),
        kl=jnp.asarray(1.0),
        dataset_size=jnp.asarray(100.0),
        bound_name="mcallester",
        delta=1.5,
    )
    with pytest.raises(ValueError, match=r"delta"):
        cert.validate()


def test_certificate_validate_rejects_unknown_bound_name() -> None:
    cert = PACBayesCertificate(
        bound_value=jnp.asarray(0.1),
        kl=jnp.asarray(1.0),
        dataset_size=jnp.asarray(100.0),
        bound_name="bogus",
        delta=0.05,
    )
    with pytest.raises(ValueError, match=r"bound_name"):
        cert.validate()


def test_certificate_validate_rejects_non_finite_bound() -> None:
    cert = PACBayesCertificate(
        bound_value=jnp.asarray(jnp.inf),
        kl=jnp.asarray(1.0),
        dataset_size=jnp.asarray(100.0),
        bound_name="mcallester",
        delta=0.05,
    )
    with pytest.raises(ValueError, match=r"bound_value"):
        cert.validate()


# ---- container is a valid JAX pytree ---------------------------------------


def test_pac_bayes_certificate_is_valid_pytree() -> None:
    """Array fields are leaves; ``bound_name`` / ``delta`` / ``metadata`` are static aux_data."""
    cert = pac_bayes_certificate(_FakeModel(), _posterior(), _data(), delta=0.05)
    leaves, treedef = jax.tree_util.tree_flatten(cert)
    # Three array leaves: bound_value, kl, dataset_size.
    assert len(leaves) == 3
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, PACBayesCertificate)
    assert rebuilt.bound_name == "mcallester"
    assert float(rebuilt.bound_value) == pytest.approx(float(cert.bound_value), rel=1e-6)


def test_pac_bayes_certificate_driver_is_jit_compatible() -> None:
    """The driver is not itself jit'd, but the bound it calls is — checked via the leaf."""

    def f(empirical_risk: jax.Array, kl: jax.Array) -> jax.Array:
        cert = pac_bayes_certificate(
            _FakeModel(),
            _FakePosterior(kl_divergence=kl),
            {"empirical_risk": empirical_risk, "dataset_size": jnp.asarray(256.0)},
            delta=0.05,
        )
        return cert.bound_value

    value = f(jnp.asarray(0.1), jnp.asarray(2.0))
    assert bool(jnp.isfinite(value))


def test_pac_bayes_certificate_supports_kl_as_callable_method() -> None:
    """Posteriors exposing ``kl_divergence`` as a callable method are accepted."""

    class _CallablePosterior:
        def kl_divergence(self) -> jax.Array:
            return jnp.asarray(1.5)

    cert = pac_bayes_certificate(_FakeModel(), _CallablePosterior(), _data(), delta=0.05)
    assert float(cert.kl) == pytest.approx(1.5, rel=1e-6)


def test_pac_bayes_certificate_rejects_posterior_without_kl_attribute() -> None:
    class _NoKL:
        pass

    with pytest.raises(TypeError, match=r"kl_divergence"):
        pac_bayes_certificate(_FakeModel(), _NoKL(), _data(), delta=0.05)


def test_pac_bayes_certificate_rejects_missing_data_keys() -> None:
    with pytest.raises(KeyError, match=r"empirical_risk"):
        pac_bayes_certificate(
            _FakeModel(), _posterior(), {"dataset_size": jnp.asarray(256.0)}, delta=0.05
        )


def test_pac_bayes_certificate_rejects_unknown_bound_name_at_driver() -> None:
    with pytest.raises(ValueError, match=r"bound_name"):
        pac_bayes_certificate(_FakeModel(), _posterior(), _data(), delta=0.05, bound_name="bogus")
