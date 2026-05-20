"""Classification conformal prediction-set contracts.

References:

* Sadinle, Lei, Wasserman 2019, "Least Ambiguous Set-Valued Classifiers
  With Bounded Error Levels", JASA — LAC score ``1 - p_y``.
* Romano, Sesia, Candes 2020, "Classification with Valid and Adaptive
  Coverage", NeurIPS — APS cumulative-sorted-probability score.
* Angelopoulos, Bates, Jordan, Malik 2021, "Uncertainty Sets for Image
  Classifiers Using Conformal Prediction", ICLR (arXiv:2009.14193) — RAPS
  regularised APS.

Cross-checked against Fortuna's
``fortuna.conformal.classification.simple_prediction`` and
``adaptive_prediction`` for LAC and APS respectively.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _import_classification():
    from opifex.uncertainty.conformal import classification

    return classification


def _import_conformal():
    from opifex.uncertainty import conformal

    return conformal


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


def test_lac_score_is_one_minus_true_class_probability() -> None:
    classification = _import_classification()
    probabilities = jnp.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]])
    targets = jnp.array([0, 1])
    out = classification.lac_score(probabilities=probabilities, targets=targets)
    assert bool(jnp.allclose(out, jnp.array([0.3, 0.2])))


def test_aps_score_sums_sorted_descending_until_true_class() -> None:
    """APS score is the cumulative sum of sorted-descending probabilities up
    to and including the true class index in the sort order."""
    classification = _import_classification()
    probabilities = jnp.array([[0.5, 0.3, 0.2]])
    # Sort descending: [0.5, 0.3, 0.2]. True class 0 has p=0.5 → cumsum to it = 0.5.
    targets = jnp.array([0])
    out_first = classification.aps_score(probabilities=probabilities, targets=targets)
    assert float(out_first[0]) == pytest.approx(0.5, abs=1e-6)

    # True class 2 has p=0.2 → it sorts to position 2 → cumsum = 0.5+0.3+0.2 = 1.0
    targets = jnp.array([2])
    out_last = classification.aps_score(probabilities=probabilities, targets=targets)
    assert float(out_last[0]) == pytest.approx(1.0, abs=1e-6)


def test_aps_score_produces_nested_sets_from_sorted_probabilities() -> None:
    """For fixed probabilities, increasing the threshold q produces a nested
    sequence of prediction sets."""
    classification = _import_classification()
    probabilities = jnp.array([[0.5, 0.3, 0.15, 0.05]])
    set_at_06 = classification.aps_prediction_set(
        probabilities=probabilities, threshold=jnp.asarray(0.6)
    )
    set_at_08 = classification.aps_prediction_set(
        probabilities=probabilities, threshold=jnp.asarray(0.8)
    )
    set_at_10 = classification.aps_prediction_set(
        probabilities=probabilities, threshold=jnp.asarray(1.0)
    )
    # 0.6 → top class (cumsum 0.5+0.3=0.8 first hits ≥0.6 at 2 classes).
    # Nesting: set_at_06 ⊆ set_at_08 ⊆ set_at_10.
    assert bool(jnp.all(set_at_06[0] <= set_at_08[0]))
    assert bool(jnp.all(set_at_08[0] <= set_at_10[0]))


def test_raps_score_preserves_aps_ordering_without_regularisation() -> None:
    """With ``k_reg=num_classes`` and ``lambda_reg=0`` RAPS == APS."""
    classification = _import_classification()
    probabilities = jnp.array([[0.5, 0.3, 0.15, 0.05]])
    targets = jnp.array([2])
    aps = classification.aps_score(probabilities=probabilities, targets=targets)
    raps = classification.raps_score(
        probabilities=probabilities, targets=targets, k_reg=4, lambda_reg=0.0
    )
    assert bool(jnp.allclose(aps, raps))


def test_raps_score_increases_with_regularisation_lambda() -> None:
    """Larger ``lambda_reg`` strictly increases the score for low-ranked targets."""
    classification = _import_classification()
    probabilities = jnp.array([[0.5, 0.3, 0.15, 0.05]])
    targets = jnp.array([3])  # lowest-prob class
    raps_no_reg = float(
        classification.raps_score(
            probabilities=probabilities, targets=targets, k_reg=1, lambda_reg=0.0
        )[0]
    )
    raps_reg = float(
        classification.raps_score(
            probabilities=probabilities, targets=targets, k_reg=1, lambda_reg=0.5
        )[0]
    )
    assert raps_reg > raps_no_reg


# ---------------------------------------------------------------------------
# Probability validation
# ---------------------------------------------------------------------------


def test_classification_score_rejects_negative_probabilities() -> None:
    classification = _import_classification()
    probabilities = jnp.array([[0.7, -0.1, 0.4]])
    targets = jnp.array([0])
    with pytest.raises(ValueError, match=r"(?i)(probabilities|negative)"):
        classification.lac_score(probabilities=probabilities, targets=targets, validate=True)


def test_classification_score_rejects_rows_not_summing_to_one() -> None:
    classification = _import_classification()
    probabilities = jnp.array([[0.7, 0.1, 0.0]])  # sums to 0.8
    targets = jnp.array([0])
    with pytest.raises(ValueError, match=r"(?i)(probabilities|sum)"):
        classification.lac_score(probabilities=probabilities, targets=targets, validate=True)


# ---------------------------------------------------------------------------
# Calibrator (split conformal LAC)
# ---------------------------------------------------------------------------


def _synthetic_classification(
    n_calib: int = 1024,
    n_test: int = 512,
    num_classes: int = 4,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    def _draw(n: int, salt: int) -> tuple[jax.Array, jax.Array]:
        local = np.random.default_rng(seed + salt)
        labels = local.integers(low=0, high=num_classes, size=(n,))
        logits = local.normal(size=(n, num_classes))
        for idx in range(n):
            logits[idx, labels[idx]] += 1.5
        probs = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return jnp.asarray(probs), jnp.asarray(labels.astype(np.int32))

    cal_probs, cal_labels = _draw(n_calib, salt=0)
    test_probs, test_labels = _draw(n_test, salt=1)
    return cal_probs, cal_labels, test_probs, test_labels


def test_lac_conformal_classifier_hits_target_coverage() -> None:
    classification = _import_classification()
    cal_probs, cal_labels, test_probs, test_labels = _synthetic_classification(seed=1)
    clf = classification.LACConformalClassifier(alpha=0.1)
    state = clf.fit(probabilities=cal_probs, targets=cal_labels)
    prediction_set = clf.with_state(state).predict(probabilities=test_probs)
    # Set must include the true class on ~ 1 - alpha fraction of points.
    n_test, num_classes = test_probs.shape
    sample_idx = jnp.arange(n_test)
    covered = prediction_set.values[sample_idx, test_labels]
    empirical = float(jnp.mean(covered.astype(jnp.float32)))
    assert empirical == pytest.approx(0.9, abs=0.05)
    assert prediction_set.method == "lac"
    assert prediction_set.values.shape == (n_test, num_classes)


def test_lac_conformal_classifier_predict_before_fit_raises() -> None:
    classification = _import_classification()
    clf = classification.LACConformalClassifier(alpha=0.1)
    with pytest.raises(RuntimeError, match=r"(?i)(fit|calibrate)"):
        clf.predict(probabilities=jnp.array([[0.5, 0.5]]))


# ---------------------------------------------------------------------------
# Transform compatibility
# ---------------------------------------------------------------------------


def test_lac_score_is_vmap_and_jit_compatible() -> None:
    classification = _import_classification()
    rng = np.random.default_rng(0)
    probs = np.exp(rng.standard_normal((4, 32, 5)))
    probs = jnp.asarray(probs / probs.sum(axis=-1, keepdims=True))
    targets = jnp.asarray(rng.integers(0, 5, size=(4, 32)).astype(np.int32))
    scores = jax.vmap(lambda p, t: classification.lac_score(probabilities=p, targets=t))(
        probs, targets
    )
    assert scores.shape == (4, 32)
    jitted = jax.jit(lambda p, t: classification.lac_score(probabilities=p, targets=t))
    out = jitted(probs[0], targets[0])
    assert out.shape == (32,)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_public_conformal_surface_includes_classification_components() -> None:
    conformal = _import_conformal()
    expected = {
        "lac_score",
        "aps_score",
        "raps_score",
        "aps_prediction_set",
        "LACConformalClassifier",
        "LACConformalState",
    }
    missing = expected - set(dir(conformal))
    assert not missing, f"missing public conformal symbols: {sorted(missing)}"
