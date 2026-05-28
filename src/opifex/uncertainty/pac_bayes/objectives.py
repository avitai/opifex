"""PAC-Bayes objective wrappers for variational training.

The single user-facing entry point is :func:`pac_bayes_kl_objective`, which
produces the McAllester-style training objective

    ``R_hat + sqrt((KL + log(2 * sqrt(n) / delta)) / (2 * n))``

used by Dziugaite & Roy (2017) and Pérez-Ortiz et al. (JMLR v22). The helper
delegates to :func:`opifex.uncertainty.pac_bayes.bounds.mcallester_bound`; no
formula is duplicated here.

Edge cases:

* When ``delta == 1`` the McAllester confidence-correction ``log(2 sqrt(n)/delta)``
  collapses to ``log(2 sqrt(n))``, which is still positive — so the bound does
  *not* reduce to the per-example ELBO. The TDD requirement asks for an
  ELBO-style scaling ``R_hat + KL / n`` in the ``delta == 1`` limit, which we
  achieve by switching dispatch: ``delta == 1`` returns the Phase-1 ELBO term
  exactly (``empirical_risk + kl / dataset_size``). For ``delta in (0, 1)`` the
  full McAllester bound is used.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from opifex.uncertainty.pac_bayes.bounds import mcallester_bound


def _validate_delta_for_objective(delta: float) -> None:
    """Reject ``delta`` outside the closed-open interval ``(0, 1]``.

    The objective accepts the boundary ``delta == 1`` (ELBO limit); the bound
    functions in :mod:`opifex.uncertainty.pac_bayes.bounds` reject it because
    a generalization bound at confidence ``0`` is meaningless.
    """
    if not 0.0 < float(delta) <= 1.0:
        raise ValueError(f"delta must be in (0, 1] for the PAC-Bayes objective; got {delta!r}.")


def pac_bayes_kl_objective(
    empirical_risk: jax.Array,
    kl: jax.Array,
    dataset_size: jax.Array | int,
    *,
    delta: float = 0.05,
) -> jax.Array:
    """Differentiable PAC-Bayes training objective.

    Returns the McAllester bound

        ``R_hat + sqrt((KL + log(2 * sqrt(n) / delta)) / (2 * n))``

    for ``delta in (0, 1)``. When ``delta == 1`` the helper reduces to the
    Phase-1 ELBO scaling ``R_hat + KL / n`` so PAC-Bayes consumers can fall
    back to ELBO training by setting ``delta=1``.

    Args:
        empirical_risk: Scalar empirical risk (per-example loss after
            reduction).
        kl: Scalar KL divergence between posterior and prior.
        dataset_size: Sample size ``n``; must be positive.
        delta: Confidence parameter in ``(0, 1]``. ``delta=1`` selects the
            ELBO limit.

    Returns:
        Scalar differentiable objective.

    Raises:
        ValueError: If ``delta`` is outside ``(0, 1]``.

    """
    _validate_delta_for_objective(delta)
    risk = jnp.asarray(empirical_risk, dtype=jnp.float32)
    kl_arr = jnp.asarray(kl, dtype=jnp.float32)
    n = jnp.asarray(dataset_size, dtype=jnp.float32)
    if float(delta) >= 1.0:
        return risk + kl_arr / n
    return mcallester_bound(risk, kl_arr, n, delta)


__all__ = ["pac_bayes_kl_objective"]
