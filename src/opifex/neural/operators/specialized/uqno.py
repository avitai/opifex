"""JAX-native port of the conformal Uncertainty Quantification Neural Operator (UQNO).

Mirrors the three-stage conformal pipeline from Ma, Pitt,
Azizzadenesheli, Anandkumar (TMLR 2024 —
`arXiv:2402.01960 <https://arxiv.org/abs/2402.01960>`_). The canonical
PyTorch reference lives at
``../neuraloperator/neuralop/models/uqno.py`` +
``../neuraloperator/scripts/train_uqno_darcy.py``; the numerical core
(``PointwiseQuantileLoss``, ``get_coeff_quantile_idx``, the scaling-factor
derivation) is cross-checked test-by-test against that reference.

**Differences from the canonical PyTorch implementation:**

* The canonical ``UQNO.__init__(base_model, residual_model=None)``
  defaults the residual to ``deepcopy(base_model)``. Here both
  operators are required keyword-only (``base=``, ``residual=``) so
  the call site is explicit about which model is which.
* The canonical performs calibration externally in the training
  script; this port packages :meth:`calibrate` on the class for
  ergonomics, returning a typed :class:`UQNOConformalCalibrator`.
* :meth:`predict_with_bands` returns a typed
  :class:`PredictiveDistribution` with a populated
  :class:`PredictionInterval`; the canonical returns untyped tensors.
* JAX/NNX semantics: ``jax.lax.stop_gradient`` replaces
  ``torch.no_grad()`` + ``model.eval()`` for the base operator inside
  the residual-stage forward pass.

**Algorithmic core mirrored faithfully:**

1. **Base solution operator** ``G_hat(a, x)`` — a standard deterministic
   :class:`opifex.neural.operators.fno.base.FourierNeuralOperator`
   (wrapped here as :class:`UQNOBaseSolutionOperator` for tagging).
2. **Residual operator** ``E(a, x)`` — a separately-trained
   :class:`FourierNeuralOperator` (wrapped as
   :class:`UQNOResidualOperator`) producing per-grid-point quantile
   widths via the canonical pointwise pinball loss
   :class:`opifex.uncertainty.losses.PointwiseQuantileLoss`.
3. **Scalar conformal calibration** — on a held-out calibration set,
   :meth:`UncertaintyQuantificationNeuralOperator.calibrate` derives a
   single ``uncertainty_scaling_factor`` from per-grid ratios
   ``|y - G_hat(x)| / E(x)`` via :func:`get_coeff_quantile_idx`. The
   fitted factor lives in a :class:`UQNOConformalCalibrator` (a
   ``flax.struct``-decorated pytree).

At test time,
:meth:`UncertaintyQuantificationNeuralOperator.predict_with_bands`
returns a :class:`PredictiveDistribution` whose
:attr:`PredictiveDistribution.interval` is populated with
``G_hat(x) ± E(x) * scaling_factor``; ``epistemic`` and ``samples``
stay ``None`` (conformal is a distribution-free calibration of the
deterministic predictor, not a Bayesian posterior).

Canonical reference (cross-checked numerically in
``tests/neural/operators/specialized/test_uqno.py``):
``../neuraloperator/neuralop/models/uqno.py``;
``../neuraloperator/scripts/train_uqno_darcy.py`` for the calibration
recipe.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from flax import nnx, struct

from opifex.neural.operators.fno.base import FourierNeuralOperator  # noqa: TC001
from opifex.uncertainty.layers.bayesian import (
    BayesianLinear,
    BayesianSpectralConvolution,
)
from opifex.uncertainty.types import (
    MetadataItems,
    PredictionInterval,
    PredictiveDistribution,
)


# ---------------------------------------------------------------------------
# Tagged wrappers around the deterministic FNO base + residual
# ---------------------------------------------------------------------------


class UQNOBaseSolutionOperator(nnx.Module):
    """Thin wrapper tagging an FNO as the UQNO's base solution operator.

    Trained with a standard regression objective (MSE / H1) against
    ``y_true``. The orchestrator routes its forward through
    ``jax.lax.stop_gradient`` inside
    :meth:`UncertaintyQuantificationNeuralOperator.__call__` so the
    residual-stage gradients never reach this operator's parameters.
    """

    def __init__(self, fno: FourierNeuralOperator) -> None:
        self.fno = fno

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the base solution operator to ``x``."""
        return self.fno(x)


class UQNOResidualOperator(nnx.Module):
    """Thin wrapper tagging an FNO as the UQNO's residual quantile operator.

    Trained with :class:`opifex.uncertainty.losses.PointwiseQuantileLoss`
    against the residuals of the (frozen) base operator. Conventionally
    the raw output is passed through ``softplus`` / ``jnp.abs`` so
    quantile widths are non-negative; the orchestrator's
    :meth:`UncertaintyQuantificationNeuralOperator.predict_residual`
    helper applies ``jnp.abs`` as a safe default.
    """

    def __init__(self, fno: FourierNeuralOperator) -> None:
        self.fno = fno

    def __call__(self, x: jax.Array) -> jax.Array:
        """Apply the residual quantile operator to ``x``."""
        return self.fno(x)


# ---------------------------------------------------------------------------
# Fitted conformal calibrator (flax.struct pytree)
# ---------------------------------------------------------------------------


@struct.dataclass(slots=True, kw_only=True)
class UQNOConformalCalibrator:
    """Fitted scalar conformal scaling factor + the alpha/delta used to derive it.

    The scaling factor is a scalar :class:`jax.Array`; the integer
    ``alpha`` / ``delta`` configuration is stored as
    ``struct.field(pytree_node=False)`` so it travels as static
    aux_data and never enters jit traces as a leaf.
    """

    scaling_factor: jax.Array
    alpha: float = struct.field(pytree_node=False)
    delta: float = struct.field(pytree_node=False)
    domain_idx: int = struct.field(pytree_node=False)
    function_idx: int = struct.field(pytree_node=False)
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def validate(self) -> None:
        """Public validation hook; not called from ``__post_init__``."""
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError(f"alpha must lie in (0, 1); got {self.alpha!r}.")
        if not 0.0 < float(self.delta) < 1.0:
            raise ValueError(f"delta must lie in (0, 1); got {self.delta!r}.")


# ---------------------------------------------------------------------------
# Conformal-index helper
# ---------------------------------------------------------------------------


def get_coeff_quantile_idx(
    *, alpha: float, delta: float, n_samples: int, n_gridpts: int
) -> tuple[int, int]:
    """Domain + function quantile indices for UQNO conformal calibration.

    Direct JAX-free Python port of the canonical
    ``get_coeff_quantile_idx`` in
    ``../neuraloperator/scripts/train_uqno_darcy.py``. Returns the
    ``(domain_idx, function_idx)`` pair: take the ``domain_idx``-th
    largest pointwise ratio per function, then the ``function_idx``-th
    largest of those per-function values across the calibration set.

    Args:
        alpha: Desired pointwise miscoverage rate in ``(0, 1)``.
        delta: Desired function-level miscoverage rate in ``(0, 1)``.
        n_samples: Number of calibration samples.
        n_gridpts: Number of grid points per sample.
    """
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError(f"alpha must lie in (0, 1); got {alpha!r}.")
    if not 0.0 < float(delta) < 1.0:
        raise ValueError(f"delta must lie in (0, 1); got {delta!r}.")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be positive; got {n_samples!r}.")
    if n_gridpts <= 0:
        raise ValueError(f"n_gridpts must be positive; got {n_gridpts!r}.")

    lb = math.sqrt(-math.log(delta) / (2.0 * n_gridpts))
    t = (alpha - lb) / 3.0 + lb
    percentile = alpha - t
    domain_idx = math.ceil(percentile * n_gridpts)
    function_percentile = (
        math.ceil((n_samples + 1) * (delta - math.exp(-2.0 * n_gridpts * t * t))) / n_samples
    )
    function_idx = math.ceil(function_percentile * n_samples)
    return domain_idx, function_idx


# ---------------------------------------------------------------------------
# UQNO orchestrator
# ---------------------------------------------------------------------------


class UncertaintyQuantificationNeuralOperator(nnx.Module):
    """Three-stage conformal UQNO orchestrator.

    Holds a base solution operator, a residual quantile operator, and
    an optional fitted :class:`UQNOConformalCalibrator`. Use:

    1. Train ``self.base`` to convergence on the regression task with
       any standard FNO training loop.
    2. Train ``self.residual`` against
       :class:`opifex.uncertainty.losses.PointwiseQuantileLoss` on
       ``base(x) - y_true`` residuals (gradients through
       :meth:`__call__` are stopped at the base via
       ``jax.lax.stop_gradient`` so residual-stage updates do not
       contaminate the base).
    3. Call :meth:`calibrate` on a held-out calibration set to obtain
       a :class:`UQNOConformalCalibrator`; attach it via
       :meth:`with_calibrator`.
    4. Call :meth:`predict_with_bands` at test time.

    The class never claims native Bayesian or distributional support;
    the matching capability declaration is
    :class:`opifex.uncertainty.adapters.operators.FNOConformalAdapterSpec`.
    """

    calibrator: nnx.Data[UQNOConformalCalibrator | None]

    def __init__(
        self,
        *,
        base: UQNOBaseSolutionOperator,
        residual: UQNOResidualOperator,
        calibrator: UQNOConformalCalibrator | None = None,
    ) -> None:
        self.base = base
        self.residual = residual
        self.calibrator = calibrator

    # ------------------------------------------------------------------
    # Forward + per-stage helpers
    # ------------------------------------------------------------------

    def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return ``(solution, quantile_width)`` for ``x``.

        Gradients through the base are stopped via
        ``jax.lax.stop_gradient`` — residual-stage training that calls
        this method only updates the residual operator and the
        calibrator.
        """
        solution = jax.lax.stop_gradient(self.base(x))
        quantile = jnp.abs(self.residual(x))
        return solution, quantile

    def predict_base(self, x: jax.Array) -> jax.Array:
        """Apply the base solution operator only."""
        return self.base(x)

    def predict_residual(self, x: jax.Array) -> jax.Array:
        """Apply the residual quantile operator (non-negative)."""
        return jnp.abs(self.residual(x))

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(
        self,
        x_calib: jax.Array,
        y_calib: jax.Array,
        *,
        alpha: float,
        delta: float,
        eps: float = 1e-12,
    ) -> UQNOConformalCalibrator:
        """Derive a scalar uncertainty scaling factor on a calibration set.

        Mirrors ``../neuraloperator/scripts/train_uqno_darcy.py``: for
        every calibration sample, compute per-grid ratios
        ``|y - base(x)| / (residual(x) + eps)``; take the
        ``domain_idx``-th largest ratio per function (per-batch);
        then the ``function_idx``-th largest of those across the
        batch is the scalar scaling factor.

        Args:
            x_calib: Calibration inputs, shape ``(n_samples, ...)``.
            y_calib: Calibration targets, same shape as the base
                model output.
            alpha: Target pointwise miscoverage in ``(0, 1)``.
            delta: Target function-level miscoverage in ``(0, 1)``.
            eps: Floor added to ``residual(x)`` before division to
                avoid divide-by-zero on near-zero predicted widths.
        """
        base_pred = self.predict_base(x_calib)
        residual_pred = self.predict_residual(x_calib) + eps
        ratios = jnp.abs(y_calib - base_pred) / residual_pred  # (n_samples, ...)
        n_samples = int(ratios.shape[0])
        n_gridpts = int(jnp.prod(jnp.array(ratios.shape[1:])))
        flat = ratios.reshape(n_samples, n_gridpts)

        domain_idx, function_idx = get_coeff_quantile_idx(
            alpha=alpha, delta=delta, n_samples=n_samples, n_gridpts=n_gridpts
        )
        # Per-sample: the (domain_idx)-th largest pointwise ratio.
        domain_k = min(max(domain_idx + 1, 1), n_gridpts)
        per_sample_topk = jnp.sort(flat, axis=1)[:, -domain_k:]
        per_sample_value = per_sample_topk[:, 0]  # smallest of top-k == k-th largest

        # Across samples: the (function_idx)-th largest of those.
        function_k = min(max(function_idx + 1, 1), n_samples)
        across_topk = jnp.sort(per_sample_value)[-function_k:]
        scaling_factor = jnp.abs(across_topk[0])

        return UQNOConformalCalibrator(
            scaling_factor=scaling_factor,
            alpha=float(alpha),
            delta=float(delta),
            domain_idx=int(domain_idx),
            function_idx=int(function_idx),
            metadata=(
                ("source", "uqno_conformal_calibration"),
                ("n_samples", n_samples),
                ("n_gridpts", n_gridpts),
            ),
        )

    def with_calibrator(
        self, calibrator: UQNOConformalCalibrator
    ) -> UncertaintyQuantificationNeuralOperator:
        """Attach ``calibrator`` to this operator and return ``self``.

        NNX modules support in-place mutation; ``with_*`` is the
        fluent-attach name (matches the canonical neuraloperator
        ``uqno_data_proc.set_scale_factor`` pattern in spirit).
        """
        self.calibrator = calibrator
        return self

    # ------------------------------------------------------------------
    # Test-time prediction with calibrated bands
    # ------------------------------------------------------------------

    def predict_with_bands(self, x: jax.Array) -> PredictiveDistribution:
        """Return ``PredictiveDistribution`` with bands ``base ± E * scaling_factor``.

        Requires a fitted :class:`UQNOConformalCalibrator` (attach via
        :meth:`with_calibrator` or by passing ``calibrator=`` at
        construction). The metadata records
        ``("method", "conformal"), ("alpha", alpha), ("delta", delta)``;
        ``epistemic`` and ``samples`` stay ``None`` (conformal is not
        Bayesian).
        """
        if self.calibrator is None:
            raise RuntimeError(
                "UQNO predict_with_bands requires a fitted calibrator. "
                "Call .calibrate(x_calib, y_calib, alpha=..., delta=...) "
                "and attach via .with_calibrator(...)."
            )
        solution = self.predict_base(x)
        widths = self.predict_residual(x) * self.calibrator.scaling_factor
        # ``scaling_factor`` is a traced jax.Array under jit and cannot be
        # cast to float here — keep it on the calibrator object; the static
        # alpha/delta are sufficient identifiers for the band semantics.
        interval = PredictionInterval(
            lower=solution - widths,
            upper=solution + widths,
            coverage=1.0 - self.calibrator.alpha,
            method="conformal",
            metadata=(
                ("alpha", self.calibrator.alpha),
                ("delta", self.calibrator.delta),
            ),
        )
        return PredictiveDistribution(
            mean=solution,
            interval=interval,
            metadata=(
                ("method", "conformal"),
                ("alpha", self.calibrator.alpha),
                ("delta", self.calibrator.delta),
                ("source", "uqno"),
            ),
        )


__all__ = [
    "BayesianLinear",
    "BayesianSpectralConvolution",
    "UQNOBaseSolutionOperator",
    "UQNOConformalCalibrator",
    "UQNOResidualOperator",
    "UncertaintyQuantificationNeuralOperator",
    "get_coeff_quantile_idx",
]
