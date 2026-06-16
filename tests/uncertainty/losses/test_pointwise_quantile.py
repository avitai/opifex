"""`PointwiseQuantileLoss` numerical contract.

The opifex implementation must match the canonical PyTorch reference
at ``../neuraloperator/neuralop/losses/data_losses.py::PointwiseQuantileLoss``
numerically (within float tolerance) on identical inputs. The reference
formula:

    quantile = 1 - alpha
    y_abs    = abs(y)
    diff     = y_abs - y_pred
    yscale   = max(y_abs, dim=0) + eps        # per-batch element max
    ptwise   = max(quantile * diff, -(1 - quantile) * diff)
    scaled   = ptwise / (2 * quantile * (1 - quantile) * yscale)
    ptavg    = mean over spatial axes (keeps batch + channel)
    loss     = sum or mean over batch + channel

Tests cover the formula, the reduction modes, edge cases, and JAX/NNX
transform compatibility (jax.jit / jax.grad / jax.vmap) per the hard
task exit criterion.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.losses import PointwiseQuantileLoss


def _reference_loss_numpy(y_pred: jax.Array, y: jax.Array, alpha: float, reduction: str) -> float:
    """Re-implement the PyTorch reference formula in numpy for cross-check."""
    import numpy as np

    yp = np.asarray(y_pred)
    yt = np.asarray(y)
    eps = 1e-7
    quantile = 1.0 - alpha
    y_abs = np.abs(yt)
    diff = y_abs - yp
    yscale = np.max(y_abs, axis=0) + eps
    ptwise = np.maximum(quantile * diff, -(1.0 - quantile) * diff)
    scaled = ptwise / (2.0 * quantile * (1.0 - quantile) * yscale)
    batch = scaled.shape[0]
    ptavg = scaled.reshape(batch, -1).mean(axis=1, keepdims=True)
    if reduction == "sum":
        return float(ptavg.sum())
    return float(ptavg.mean())


def _toy_inputs(
    seed: int = 0, shape: tuple[int, ...] = (4, 1, 8, 8)
) -> tuple[jax.Array, jax.Array]:
    key = jax.random.PRNGKey(seed)
    k_pred, k_true = jax.random.split(key)
    y_pred = jax.nn.softplus(jax.random.normal(k_pred, shape))  # quantile widths ≥ 0
    y = jax.random.normal(k_true, shape) * 0.5
    return y_pred, y


# ---------------------------------------------------------------------------
# Construction + parameter validation
# ---------------------------------------------------------------------------


def test_pointwise_quantile_loss_construction_records_alpha_and_reduction() -> None:
    loss = PointwiseQuantileLoss(alpha=0.1, reduction="sum")
    assert loss.alpha == 0.1
    assert loss.reduction == "sum"


def test_pointwise_quantile_loss_rejects_invalid_reduction() -> None:
    with pytest.raises(ValueError, match="reduction"):
        PointwiseQuantileLoss(alpha=0.1, reduction="median")  # type: ignore[arg-type]


@pytest.mark.parametrize("alpha", [-0.1, 0.0, 1.0, 1.5])
def test_pointwise_quantile_loss_rejects_out_of_range_alpha(alpha: float) -> None:
    with pytest.raises(ValueError, match="alpha"):
        PointwiseQuantileLoss(alpha=alpha)


# ---------------------------------------------------------------------------
# Numerical match with PyTorch reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("alpha", [0.02, 0.05, 0.1, 0.5, 0.9])
@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_matches_pytorch_reference_formula(alpha: float, reduction: str) -> None:
    y_pred, y = _toy_inputs(seed=int(alpha * 1000) + (1 if reduction == "sum" else 0))
    loss_fn = PointwiseQuantileLoss(alpha=alpha, reduction=reduction)  # type: ignore[arg-type]
    out = float(loss_fn(y_pred=y_pred, y=y))
    expected = _reference_loss_numpy(y_pred, y, alpha, reduction)
    assert out == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_zero_predicted_widths_produce_positive_loss() -> None:
    """Predicting zero quantile widths over non-zero residuals is penalised."""
    y = jnp.array([[1.0, -0.5, 0.25, -2.0]]).reshape((4, 1, 1, 1))
    y_pred = jnp.zeros_like(y)
    loss_fn = PointwiseQuantileLoss(alpha=0.1, reduction="mean")
    out = float(loss_fn(y_pred=y_pred, y=y))
    assert out > 0.0


def test_perfect_prediction_loss_is_smaller_than_zero_prediction() -> None:
    """Predicting widths == |y| should beat predicting zero widths."""
    y = jnp.array([[1.0, -0.5, 0.25, -2.0]]).reshape((4, 1, 1, 1))
    perfect_widths = jnp.abs(y)
    zero_widths = jnp.zeros_like(y)
    loss_fn = PointwiseQuantileLoss(alpha=0.1, reduction="mean")
    out_perfect = float(loss_fn(y_pred=perfect_widths, y=y))
    out_zero = float(loss_fn(y_pred=zero_widths, y=y))
    assert out_perfect < out_zero


# ---------------------------------------------------------------------------
# JAX / NNX transform compatibility (hard exit criterion)
# ---------------------------------------------------------------------------


def test_pointwise_quantile_loss_is_jit_compatible() -> None:
    """`PointwiseQuantileLoss` traces under ``jax.jit`` when closed over."""
    y_pred, y = _toy_inputs()
    loss_fn = PointwiseQuantileLoss(alpha=0.1, reduction="sum")

    @jax.jit
    def jitted_loss(yp: jax.Array, yt: jax.Array) -> jax.Array:
        return loss_fn(y_pred=yp, y=yt)

    out = jitted_loss(y_pred, y)
    eager_out = loss_fn(y_pred=y_pred, y=y)
    assert bool(jnp.allclose(out, eager_out, rtol=1e-5, atol=1e-6))


def test_pointwise_quantile_loss_passes_directly_as_jit_static_arg() -> None:
    """The loss can be passed DIRECTLY as a ``jax.jit`` argument.

    Because the loss is a hashable, array-free config object registered as a
    static pytree (``jax.tree_util.register_static`` — the same idiom flax NNX
    uses for its metadata classes), transforms treat the instance as static.
    No ``static_argnames`` and no closure wrapper are required, so it composes
    cleanly inside jitted train steps that receive it as a parameter.
    """
    y_pred, y = _toy_inputs()

    @jax.jit
    def jitted(loss: PointwiseQuantileLoss, yp: jax.Array, yt: jax.Array) -> jax.Array:
        return loss(y_pred=yp, y=yt)

    loss = PointwiseQuantileLoss(alpha=0.1, reduction="mean")
    out = jitted(loss, y_pred, y)
    assert bool(jnp.isfinite(out))
    assert bool(jnp.allclose(out, loss(y_pred=y_pred, y=y), rtol=1e-5, atol=1e-6))


def test_pointwise_quantile_loss_passes_directly_through_nnx_jit() -> None:
    """The loss flows through ``nnx.jit`` as a direct parameter.

    Mirrors the UQNO residual training step, which receives the loss as an
    argument alongside its NNX modules. A static pytree config object is the
    principled way to pass non-array hyperparameters through ``nnx.jit``.
    """
    from flax import nnx

    y_pred, y = _toy_inputs()

    @nnx.jit
    def jitted(loss: PointwiseQuantileLoss, yp: jax.Array, yt: jax.Array) -> jax.Array:
        return loss(y_pred=yp, y=yt)

    loss = PointwiseQuantileLoss(alpha=0.1, reduction="mean")
    out = jitted(loss, y_pred, y)
    assert bool(jnp.isfinite(out))


def test_pointwise_quantile_loss_is_grad_compatible() -> None:
    """Gradients flow back to ``y_pred``; widths can be tuned via SGD."""
    y_pred, y = _toy_inputs()
    loss_fn = PointwiseQuantileLoss(alpha=0.1, reduction="sum")

    def loss_of_pred(yp: jax.Array) -> jax.Array:
        return loss_fn(y_pred=yp, y=y)

    grad = jax.grad(loss_of_pred)(y_pred)
    assert grad.shape == y_pred.shape
    assert bool(jnp.all(jnp.isfinite(grad)))
    # Gradients should not be identically zero — the loss depends on y_pred.
    assert bool(jnp.any(jnp.abs(grad) > 0))


def test_pointwise_quantile_loss_is_vmap_compatible() -> None:
    """`vmap` over a leading "trial" dimension produces the right shape."""
    loss_fn = PointwiseQuantileLoss(alpha=0.1, reduction="mean")
    base_shape = (4, 1, 8, 8)
    y_pred = jax.nn.softplus(jax.random.normal(jax.random.PRNGKey(0), (3, *base_shape)))
    y = jax.random.normal(jax.random.PRNGKey(1), (3, *base_shape))
    out = jax.vmap(lambda p, t: loss_fn(y_pred=p, y=t))(y_pred, y)
    assert out.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(out)))


# ---------------------------------------------------------------------------
# Hashability (frozen dataclass → safe as static arg / closed over by jit)
# ---------------------------------------------------------------------------


def test_pointwise_quantile_loss_is_hashable_and_frozen() -> None:
    import dataclasses as dc

    loss_a = PointwiseQuantileLoss(alpha=0.1)
    loss_b = PointwiseQuantileLoss(alpha=0.1)
    loss_c = PointwiseQuantileLoss(alpha=0.5)
    # Equal-by-field dataclasses are equal + share a hash.
    assert loss_a == loss_b
    assert hash(loss_a) == hash(loss_b)
    assert loss_a != loss_c
    # Frozen — cannot mutate fields post-construction.
    with pytest.raises(dc.FrozenInstanceError):
        loss_a.alpha = 0.5  # type: ignore[misc]
