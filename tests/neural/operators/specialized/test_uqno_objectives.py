"""Marker test file — UQNO Bayesian-objective surface intentionally absent.

After Task 3.8's rewrite, ``UncertaintyQuantificationNeuralOperator`` is a
conformal operator (not a Bayesian model). It does not implement
:class:`VariationalModule` and intentionally exposes none of
``kl_divergence``, ``negative_elbo``, ``loss_components``, or
``predict_distribution``.

This file pins that contract so a future regression cannot silently
re-add a Bayesian objective surface to UQNO; the audit-mandated
"objectives" test file is therefore present, even though every check is
a non-existence assertion.

The actual UQNO contract — the three-stage conformal pipeline
(``predict_base`` → ``calibrate`` → ``predict_with_bands``) — is tested
in ``tests/neural/operators/specialized/test_uqno.py`` and the
integration suite ``tests/uncertainty/integration/test_uqno_conformal_pipeline.py``.
"""

from __future__ import annotations

from opifex.neural.operators.specialized.uqno import (
    UncertaintyQuantificationNeuralOperator,
)


def test_uqno_does_not_expose_kl_divergence() -> None:
    assert not hasattr(UncertaintyQuantificationNeuralOperator, "kl_divergence")


def test_uqno_does_not_expose_negative_elbo() -> None:
    assert not hasattr(UncertaintyQuantificationNeuralOperator, "negative_elbo")


def test_uqno_does_not_expose_loss_components() -> None:
    assert not hasattr(UncertaintyQuantificationNeuralOperator, "loss_components")


def test_uqno_does_not_expose_predict_distribution() -> None:
    assert not hasattr(UncertaintyQuantificationNeuralOperator, "predict_distribution")
