"""Out-of-distribution detection scores.

Locally-implemented OOD scores that have no canonical CalibraX
equivalent. AUROC / AUPRC live in
``calibrax.metrics.functional.classification`` and are imported directly
by callers (no forward shims here).

Reference: Hendrycks & Gimpel 2017, "A Baseline for Detecting
Misclassified and Out-of-Distribution Examples" (arXiv:1610.02136) —
Maximum Softmax Probability (MSP) as a baseline OOD score.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def max_softmax_probability(*, probabilities: jax.Array) -> jax.Array:
    """Maximum softmax probability per sample.

    Convention: HIGHER MSP indicates more in-distribution. Pair with
    AUROC / AUPRC where label 1 = OOD and the score = ``1 - MSP`` (or
    use ``negative MSP``) to express "higher score = more OOD".

    Args:
        probabilities: ``(batch, num_classes)`` softmax probabilities.

    Returns:
        Per-sample max probability of shape ``(batch,)``.
    """
    return jnp.max(probabilities, axis=-1)


def fpr95(*, scores: jax.Array, labels: jax.Array) -> jax.Array:
    """False-positive rate when the true-positive rate is at least 95%.

    Convention: ``scores`` larger → more OOD; ``labels == 1`` indicates
    the OOD class. We sweep the threshold from highest score down,
    keeping every sample whose score is at or above it as a predicted
    positive. The FPR at the first threshold where ``TPR >= 0.95`` is
    returned.

    Args:
        scores: 1-D array of per-sample OOD scores.
        labels: 1-D binary array; 1 = OOD positive, 0 = in-distribution.

    Returns:
        Scalar FPR in ``[0, 1]``. Lower is better.
    """
    labels_f = labels.astype(jnp.float32)
    # Sort by descending score.
    order = jnp.argsort(-scores)
    sorted_labels = labels_f[order]
    cumulative_positive = jnp.cumsum(sorted_labels)
    cumulative_negative = jnp.cumsum(1.0 - sorted_labels)
    total_positive = jnp.sum(labels_f)
    total_negative = scores.shape[0] - total_positive
    tpr = cumulative_positive / jnp.maximum(total_positive, 1.0)
    fpr = cumulative_negative / jnp.maximum(total_negative, 1.0)
    # First index where TPR >= 0.95.
    above_target = (tpr >= 0.95).astype(jnp.int32)
    any_above = jnp.any(above_target == 1)
    chosen_idx = jnp.where(any_above, jnp.argmax(above_target), scores.shape[0] - 1)
    return fpr[chosen_idx]
