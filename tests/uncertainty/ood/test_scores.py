"""OOD score contracts.

References (canonical):
* Hendrycks & Gimpel 2017 ("A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples", arXiv:1610.02136) — Maximum Softmax
  Probability (MSP) score; lower → more OOD.
* Gal & Ghahramani 2016 — predictive entropy / mutual information from
  ensembles (already in :mod:`opifex.uncertainty.metrics`).

AUROC / AUPRC use the canonical CalibraX kernels directly
(``calibrax.metrics.functional.classification.{roc_auc, average_precision}``)
rather than re-exporting from this module (no forward shims). FPR95 has
no CalibraX equivalent and lives locally.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _import_ood():
    from opifex.uncertainty import ood

    return ood


# ---------------------------------------------------------------------------
# Maximum softmax probability
# ---------------------------------------------------------------------------


def test_max_softmax_probability_picks_per_sample_max() -> None:
    ood = _import_ood()
    probabilities = jnp.array(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.6, 0.3],
            [0.5, 0.5, 0.0],
        ]
    )
    out = ood.max_softmax_probability(probabilities=probabilities)
    assert bool(jnp.allclose(out, jnp.array([0.7, 0.6, 0.5]), atol=1e-6))


def test_max_softmax_probability_is_jit_compatible() -> None:
    ood = _import_ood()
    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.standard_normal((8, 5)))
    probs = jax.nn.softmax(logits, axis=-1)
    jitted = jax.jit(lambda p: ood.max_softmax_probability(probabilities=p))
    out = jitted(probs)
    assert out.shape == (8,)
    assert bool(jnp.all(out >= 1.0 / 5))  # min possible MSP is uniform = 1/K


# ---------------------------------------------------------------------------
# FPR95: false-positive rate when TPR ≥ 0.95
# ---------------------------------------------------------------------------


def test_fpr95_on_perfectly_separable_scores_is_zero() -> None:
    ood = _import_ood()
    # In-distribution scores: high (negative class label 0).
    # OOD scores: low (positive class label 1, lower = more OOD).
    # Convention: higher OOD score → more likely OOD; here we use entropy-like
    # scores so OOD samples have higher score.
    scores = jnp.array([0.1, 0.2, 0.1, 0.3, 0.9, 0.95, 0.99])
    labels = jnp.array([0, 0, 0, 0, 1, 1, 1])  # last 3 are OOD
    out = float(ood.fpr95(scores=scores, labels=labels))
    assert out == pytest.approx(0.0, abs=1e-6)


def test_fpr95_on_random_scores_is_near_chance() -> None:
    ood = _import_ood()
    rng = np.random.default_rng(0)
    n = 2048
    scores = jnp.asarray(rng.uniform(size=(n,)))
    labels = jnp.asarray(rng.integers(0, 2, size=(n,)))
    out = float(ood.fpr95(scores=scores, labels=labels))
    # Pure noise: FPR at 95% TPR threshold should be near 0.95 (lots of false positives).
    assert 0.7 <= out <= 1.0


# ---------------------------------------------------------------------------
# AUROC / AUPRC via direct CalibraX import (no shim)
# ---------------------------------------------------------------------------


def test_auroc_via_calibrax_separates_known_labels() -> None:
    """Tests reach into CalibraX directly rather than going through an
    `ood.auroc` shim — the OOD module does not re-export AUROC."""
    from calibrax.metrics.functional.classification import roc_auc

    scores = jnp.array([0.1, 0.2, 0.3, 0.9, 0.85, 0.95])
    labels = jnp.array([0, 0, 0, 1, 1, 1])
    out = float(roc_auc(scores, labels))
    assert out == pytest.approx(1.0, abs=1e-6)


def test_auprc_via_calibrax_separates_known_labels() -> None:
    from calibrax.metrics.functional.classification import average_precision

    scores = jnp.array([0.1, 0.2, 0.3, 0.9, 0.85, 0.95])
    labels = jnp.array([0, 0, 0, 1, 1, 1])
    out = float(average_precision(scores, labels))
    assert out > 0.9


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_ood_surface_includes_local_kernels_only() -> None:
    """OOD module exposes only the locally-implemented kernels — not
    re-exports of CalibraX / `opifex.uncertainty.metrics` symbols."""
    ood = _import_ood()
    expected = {
        "max_softmax_probability",
        "fpr95",
    }
    missing = expected - set(dir(ood))
    assert not missing, f"missing public OOD symbols: {sorted(missing)}"
    # Negative contract: predictive entropy / mutual information are NOT
    # re-exported from `ood` — callers import them from
    # `opifex.uncertainty.metrics` directly.
    for forbidden in ("predictive_entropy", "mutual_information", "roc_auc"):
        assert forbidden not in dir(ood), (
            f"`{forbidden}` should not be re-exported from `opifex.uncertainty.ood`"
        )
