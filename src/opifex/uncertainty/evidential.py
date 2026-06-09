r"""Normal-Inverse-Gamma (NIG) evidential-regression primitive.

Deep Evidential Regression (Amini, Schwarting, Soleimany, Rus, *NeurIPS* 2020)
places a Gaussian prior on the unknown mean :math:`\mu` and an Inverse-Gamma
prior on the unknown variance :math:`\sigma^2`. The conjugate prior is the
Normal-Inverse-Gamma distribution with evidential parameters
:math:`m = (\gamma, \nu, \alpha, \beta)`. A single network output head emits
``m`` per example, giving aleatoric **and** epistemic uncertainty in one
deterministic forward pass (no sampling, no ensemble).

This module provides the three pure-JAX surfaces used by the atomistic
evidential energy head:

* :class:`NIGParams` — a frozen, JAX-pytree NIG-parameter container.
* :func:`positive_evidential_params` — the softplus parameterisation that maps
  4 raw network logits ``(gamma, nu, alpha, beta)`` to valid NIG parameters
  (``nu, beta > 0``; ``alpha > 1`` via ``softplus + 1``).
* :func:`evidential_nll` — the Amini 2020 loss
  :math:`\mathcal{L} = \mathcal{L}^{\mathrm{NLL}}_{\mathrm{NIG}}
  + \lambda\,|y-\gamma|\,(2\nu + \alpha)`.
* :func:`nig_to_predictive_distribution` — maps a :class:`NIGParams` to an
  Opifex :class:`~opifex.uncertainty.types.PredictiveDistribution` using the
  closed-form NIG moments.

All functions are ``jit``/``grad``/``vmap`` safe (pure ``jax.numpy``; no Python
control flow on array values, no in-place mutation).

References
----------
* Amini, A.; Schwarting, W.; Soleimany, A.; Rus, D. "Deep Evidential
  Regression." *NeurIPS* 2020.
  https://proceedings.neurips.cc/paper_files/paper/2020/file/aab085461de182608ee9f607f3f7d18f-Paper.pdf
* Tan, A. R.; et al. "Evidential deep learning for interatomic potentials" (eIP).
  *Nat. Commun.* 2025 (arXiv:2407.13994) — the per-atom NIG prediction /
  aleatoric / epistemic decomposition reused below.
* Code reference: ``../chemprop/chemprop/nn/predictors.py`` (``EvidentialFFN``
  softplus parameterisation, ``+1`` on ``alpha``) and
  ``../chemprop/chemprop/nn/metrics.py`` (``EvidentialLoss`` closed form).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import struct
from jaxtyping import Array, Float  # noqa: TC002

from opifex.uncertainty.types import PredictiveDistribution


# Default error-evidence regularisation weight ``lambda`` (Amini 2020 Eq. 10;
# chemprop ``EvidentialLoss.v_kl`` default). Penalises evidence placed on
# wrong predictions, scaling total evidence ``2 nu + alpha`` by the residual.
_DEFAULT_REGULARIZER_COEFFICIENT: float = 0.2

# Numerical floor matching the chemprop ``EvidentialLoss.eps`` default; the
# regulariser term is offset by this so a perfect fit contributes exactly zero.
_REGULARIZER_EPSILON: float = 1e-8

# Strictly-positive floor for ``nu``, ``beta`` and the ``alpha - 1`` evidence.
# ``softplus`` underflows to exactly ``0`` for very negative float32 logits
# (e.g. ``softplus(-50) == 0``), which would make ``alpha == 1`` and divide the
# closed-form moments ``beta/(alpha-1)`` by zero. Clamping keeps the NIG moments
# finite and the ``alpha > 1`` contract strict without changing the chemprop
# softplus parameterisation in the well-conditioned regime.
_POSITIVITY_FLOOR: float = 1e-6


@struct.dataclass(slots=True, kw_only=True)
class NIGParams:
    r"""Normal-Inverse-Gamma evidential parameters :math:`(\gamma, \nu, \alpha, \beta)`.

    A JAX pytree (via :func:`flax.struct.dataclass`): all four fields are
    array leaves, so an instance flows cleanly through ``jit``/``grad``/``vmap``
    and ``.replace(...)`` immutable updates.

    Field contract (Amini 2020): :math:`\gamma` is the predictive mean
    (unconstrained), while :math:`\nu > 0`, :math:`\alpha > 1`, :math:`\beta > 0`
    parameterise the conjugate prior. Shapes broadcast together; scalars and
    batched ``(batch,)`` / ``(n_atoms,)`` arrays are both valid.
    """

    gamma: Float[Array, "*batch"]
    nu: Float[Array, "*batch"]
    alpha: Float[Array, "*batch"]
    beta: Float[Array, "*batch"]


def positive_evidential_params(raw: Float[Array, "*batch 4"]) -> NIGParams:
    r"""Map 4 raw logits to valid NIG parameters via the softplus parameterisation.

    Transcribes the chemprop ``EvidentialFFN.forward`` reference
    (``../chemprop/chemprop/nn/predictors.py:197-200``):

    .. math::
       \gamma = z_0,\quad
       \nu = \mathrm{softplus}(z_1),\quad
       \alpha = \mathrm{softplus}(z_2) + 1,\quad
       \beta = \mathrm{softplus}(z_3).

    The ``+1`` offset enforces :math:`\alpha > 1` so the closed-form aleatoric
    variance :math:`\beta/(\alpha-1)` is finite and positive.

    Args:
        raw: Network logits with a trailing axis of size 4 ordered
            ``(gamma, nu, alpha, beta)``; leading axes are the batch / per-atom
            dimensions.

    Returns:
        A :class:`NIGParams` whose ``nu, beta`` are strictly positive and whose
        ``alpha`` is strictly greater than one.
    """
    gamma = raw[..., 0]
    nu = jax.nn.softplus(raw[..., 1]) + _POSITIVITY_FLOOR
    alpha = jax.nn.softplus(raw[..., 2]) + 1.0 + _POSITIVITY_FLOOR
    beta = jax.nn.softplus(raw[..., 3]) + _POSITIVITY_FLOOR
    return NIGParams(gamma=gamma, nu=nu, alpha=alpha, beta=beta)


def aleatoric_variance(params: NIGParams) -> Array:
    r"""Return the aleatoric variance :math:`\mathbb{E}[\sigma^2] = \beta/(\alpha-1)`.

    Closed-form NIG moment (Amini 2020; eIP arXiv:2407.13994). Requires
    ``alpha > 1`` (guaranteed by :func:`positive_evidential_params`).
    """
    return params.beta / (params.alpha - 1.0)


def epistemic_variance(params: NIGParams) -> Array:
    r"""Return the model (epistemic) variance :math:`\mathrm{Var}[\mu] = \beta/(\nu(\alpha-1))`.

    Closed-form NIG moment (Amini 2020; eIP arXiv:2407.13994). Requires
    ``nu > 0`` and ``alpha > 1`` (guaranteed by
    :func:`positive_evidential_params`).
    """
    return params.beta / (params.nu * (params.alpha - 1.0))


def evidential_nll(
    params: NIGParams,
    target: Float[Array, "*batch"],
    *,
    coefficient: float = _DEFAULT_REGULARIZER_COEFFICIENT,
    epsilon: float = _REGULARIZER_EPSILON,
) -> Array:
    r"""Deep-Evidential-Regression loss for one (or a batch of) NIG prediction(s).

    Transcribes the chemprop ``EvidentialLoss._calc_unreduced_loss`` reference
    (``../chemprop/chemprop/nn/metrics.py:241-257``), itself Eqs. 8-10 of
    Amini 2020. With :math:`r = y - \gamma` and
    :math:`\Omega = 2\beta(1+\nu)`:

    .. math::
       \mathcal{L}^{\mathrm{NLL}} =
       \tfrac12 \log\!\frac{\pi}{\nu}
       - \alpha \log \Omega
       + (\alpha + \tfrac12)\log(\nu r^2 + \Omega)
       + \log\Gamma(\alpha) - \log\Gamma(\alpha + \tfrac12),

    and the total loss adds the error-evidence regulariser

    .. math::
       \mathcal{L} = \mathcal{L}^{\mathrm{NLL}}
       + \lambda\big(|r|\,(2\nu + \alpha) - \epsilon\big).

    The loss is returned **unreduced** (per element); callers reduce over the
    batch.

    Args:
        params: The predicted NIG parameters.
        target: Ground-truth value(s) broadcasting against ``params.gamma``.
        coefficient: Regulariser weight :math:`\lambda` (Amini 2020 Eq. 10).
            ``0.0`` disables the regulariser.
        epsilon: Numerical offset matching the chemprop ``eps`` default so a
            zero-residual fit contributes exactly zero regulariser.

    Returns:
        The per-element evidential loss, the same shape as the broadcast of
        ``params.gamma`` and ``target``.
    """
    residual = target - params.gamma
    two_b_lambda = 2.0 * params.beta * (1.0 + params.nu)

    nll = (
        0.5 * jnp.log(jnp.pi / params.nu)
        - params.alpha * jnp.log(two_b_lambda)
        + (params.alpha + 0.5) * jnp.log(params.nu * residual**2 + two_b_lambda)
        + jax.lax.lgamma(params.alpha)
        - jax.lax.lgamma(params.alpha + 0.5)
    )

    regularizer = jnp.abs(residual) * (2.0 * params.nu + params.alpha)
    return nll + coefficient * (regularizer - epsilon)


def nig_to_predictive_distribution(
    params: NIGParams,
    *,
    metadata: tuple[tuple[str, object], ...] = (),
) -> PredictiveDistribution:
    r"""Map NIG evidential parameters to an Opifex :class:`PredictiveDistribution`.

    Uses the closed-form NIG moments (Amini 2020; eIP arXiv:2407.13994):

    * ``mean`` :math:`= \gamma`,
    * ``aleatoric`` :math:`= \mathbb{E}[\sigma^2] = \beta/(\alpha-1)`,
    * ``epistemic`` :math:`= \mathrm{Var}[\mu] = \beta/(\nu(\alpha-1))`,
    * ``variance`` :math:`=` ``total_uncertainty`` :math:`=` aleatoric + epistemic.

    The total/epistemic/aleatoric fields are *variances* (the
    :class:`PredictiveDistribution` contract), and the additivity
    ``total == epistemic + aleatoric`` holds by construction, so
    :meth:`PredictiveDistribution.validate` passes.

    Args:
        params: The predicted NIG parameters.
        metadata: Extra immutable, hashable metadata pairs to attach (the
            ``"method"`` key is prepended automatically).

    Returns:
        A :class:`PredictiveDistribution` carrying the evidential mean and the
        epistemic/aleatoric/total variance decomposition.
    """
    aleatoric = aleatoric_variance(params)
    epistemic = epistemic_variance(params)
    total = aleatoric + epistemic
    composed_metadata = (("method", "deep_evidential_regression"), *metadata)
    return PredictiveDistribution(
        mean=params.gamma,
        variance=total,
        aleatoric=aleatoric,
        epistemic=epistemic,
        total_uncertainty=total,
        metadata=composed_metadata,
    )


__all__ = [
    "NIGParams",
    "aleatoric_variance",
    "epistemic_variance",
    "evidential_nll",
    "nig_to_predictive_distribution",
    "positive_evidential_params",
]
