"""Classification conformal prediction sets.

Three score functions and an LAC split-conformal classifier:

* :func:`lac_score` — Sadinle, Lei, Wasserman 2019 ("Least Ambiguous
  Set-Valued Classifiers With Bounded Error Levels", JASA). ``score = 1 - p_y``.
* :func:`aps_score` — Romano, Sesia, Candes 2020 ("Classification with
  Valid and Adaptive Coverage", NeurIPS). Cumulative sum of
  sorted-descending probabilities up to and including the true class.
* :func:`raps_score` — Angelopoulos, Bates, Jordan, Malik 2021 ("RAPS",
  arXiv:2009.14193). APS with an additive penalty
  ``lambda_reg * max(0, rank - k_reg)`` discouraging large sets.

The :class:`LACConformalClassifier` follows the standard
``with_state``/``fit``/``predict`` ergonomics shared with the regression
calibrators and returns :class:`opifex.uncertainty.types.PredictionSet` so
downstream code consumes a single typed boolean-set value object.

Reference: ``fortuna.conformal.classification.simple_prediction`` and
``adaptive_prediction`` are the canonical JAX-native implementations the
scoring kernels here mirror.
"""

from __future__ import annotations

import dataclasses as dc
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct

from opifex.uncertainty.conformal.scores import conformal_quantile
from opifex.uncertainty.types import PredictionSet


if TYPE_CHECKING:
    from opifex.uncertainty.types import MetadataItems


_PROBABILITY_SUM_TOL: float = 1e-3


def _validate_probabilities(probabilities: jax.Array) -> None:
    """Eagerly check that probabilities are non-negative and rows sum to 1."""
    if bool(jnp.any(probabilities < 0.0)):
        raise ValueError("Probabilities contain negative entries; expected non-negative.")
    row_sums = jnp.sum(probabilities, axis=-1)
    if bool(jnp.any(jnp.abs(row_sums - 1.0) > _PROBABILITY_SUM_TOL)):
        raise ValueError(
            "Probabilities do not sum to 1 along the last axis within tolerance "
            f"{_PROBABILITY_SUM_TOL}."
        )


# ---------------------------------------------------------------------------
# Score functions
# ---------------------------------------------------------------------------


def lac_score(
    *,
    probabilities: jax.Array,
    targets: jax.Array,
    validate: bool = False,
) -> jax.Array:
    """LAC score ``1 - p_y`` per Sadinle, Lei, Wasserman 2019.

    Args:
        probabilities: ``(batch, num_classes)`` softmax-normalised probabilities.
        targets: ``(batch,)`` integer class labels.
        validate: When ``True``, raise on negative probabilities or rows that
            do not sum to 1.

    Returns:
        ``(batch,)`` per-sample LAC scores.
    """
    if validate:
        _validate_probabilities(probabilities)
    chosen = jnp.take_along_axis(probabilities, targets[:, None], axis=-1).squeeze(-1)
    return 1.0 - chosen


def aps_score(
    *,
    probabilities: jax.Array,
    targets: jax.Array,
    validate: bool = False,
) -> jax.Array:
    """APS cumulative-sorted-probability score per Romano et al. 2020.

    For each sample, sort probabilities descending and return the cumulative
    sum up to and including the position of the true class.
    """
    if validate:
        _validate_probabilities(probabilities)
    sorted_descending = jnp.sort(probabilities, axis=-1)[..., ::-1]
    argsorted = jnp.argsort(-probabilities, axis=-1)
    target_rank = jnp.argmax((argsorted == targets[..., None]).astype(jnp.int32), axis=-1)
    cumulative = jnp.cumsum(sorted_descending, axis=-1)
    return jnp.take_along_axis(cumulative, target_rank[..., None], axis=-1).squeeze(-1)


def aps_prediction_set(*, probabilities: jax.Array, threshold: jax.Array) -> jax.Array:
    """Boolean APS prediction set at the given cumulative-probability threshold.

    Matches the canonical Angelopoulos reference
    (``aangelopoulos/conformal-prediction/notebooks/imagenet-aps.ipynb``)::

        val_pi = val_smx.argsort(1)[:, ::-1]
        val_srt = np.take_along_axis(val_smx, val_pi, axis=1).cumsum(axis=1)
        prediction_sets = np.take_along_axis(
            val_srt <= qhat, val_pi.argsort(axis=1), axis=1
        )

    A class is included when its cumulative sorted-descending probability
    (up to and including itself) is ``<= threshold``. Returns
    ``(batch, num_classes)`` boolean array in the original class order.
    """
    argsorted = jnp.argsort(-probabilities, axis=-1)
    sorted_descending = jnp.take_along_axis(probabilities, argsorted, axis=-1)
    cumulative = jnp.cumsum(sorted_descending, axis=-1)
    included_in_sorted = cumulative <= threshold
    inverse_perm = jnp.argsort(argsorted, axis=-1)
    return jnp.take_along_axis(included_in_sorted, inverse_perm, axis=-1)


def raps_score(
    *,
    probabilities: jax.Array,
    targets: jax.Array,
    k_reg: int,
    lambda_reg: float,
    validate: bool = False,
) -> jax.Array:
    """RAPS score per Angelopoulos et al. 2021.

    Adds ``lambda_reg * max(0, rank_in_sorted - k_reg + 1)`` to the APS score,
    discouraging large prediction sets.
    """
    if validate:
        _validate_probabilities(probabilities)
    base = aps_score(probabilities=probabilities, targets=targets, validate=False)
    argsorted = jnp.argsort(-probabilities, axis=-1)
    target_rank = jnp.argmax((argsorted == targets[..., None]).astype(jnp.int32), axis=-1)
    penalty = lambda_reg * jnp.maximum(0.0, (target_rank + 1) - k_reg)
    return base + penalty


# ---------------------------------------------------------------------------
# LAC split-conformal classifier
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class LACConformalState:
    """Fitted threshold for an LAC split-conformal classifier."""

    threshold: jax.Array
    alpha: float = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class LACConformalClassifier:
    """Split-conformal classifier using the LAC score.

    Usage::

        clf = LACConformalClassifier(alpha=0.1)
        state = clf.fit(probabilities=val_probs, targets=val_labels)
        prediction_set = clf.with_state(state).predict(probabilities=test_probs)
    """

    alpha: float
    _state: LACConformalState | None = dc.field(default=None)

    def with_state(self, state: LACConformalState) -> LACConformalClassifier:
        """Return a fresh classifier carrying ``state`` (immutable update)."""
        return dc.replace(self, _state=state)

    def fit(self, *, probabilities: jax.Array, targets: jax.Array) -> LACConformalState:
        """Fit the LAC threshold from calibration ``(probabilities, targets)``."""
        scores = lac_score(probabilities=probabilities, targets=targets)
        threshold = conformal_quantile(scores=scores, alpha=self.alpha)
        metadata: MetadataItems = (
            ("method", "lac"),
            ("score_type", "lac"),
            ("alpha", float(self.alpha)),
            ("calibration_size", int(probabilities.shape[0])),
            ("num_classes", int(probabilities.shape[-1])),
        )
        return LACConformalState(threshold=threshold, alpha=self.alpha, metadata=metadata)

    def predict(self, *, probabilities: jax.Array) -> PredictionSet:
        """Return a :class:`PredictionSet` of classes with ``1 - p >= threshold``."""
        state = self._state
        if state is None:
            raise RuntimeError(
                "LACConformalClassifier.predict called before fit; "
                "call fit(...) first or .with_state(state)."
            )
        per_class_scores = 1.0 - probabilities
        values = per_class_scores <= state.threshold
        return PredictionSet(
            values=values,
            scores=per_class_scores,
            threshold=float(state.threshold),
            method="lac",
            metadata=state.metadata,
        )
