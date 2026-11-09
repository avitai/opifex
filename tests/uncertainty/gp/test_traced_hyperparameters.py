"""Hyperparameters can be traced, which is what optimising them requires.

The leave-one-out objective this package ships is Rasmussen & Williams 2006 §5.4.2, from the
*Model Selection and Adaptation of Hyperparameters* chapter. Its purpose is to be
differentiated with respect to the hyperparameters, and doing that under ``jax.jit`` means
tracing them.

**The existing jit tests cannot catch this.** Every one of them traces the training *data*
and passes the hyperparameters as Python literals inside the jitted function -- see
``test_loocv_log_predictive_is_jit_compatible``, whose docstring says "with traced training
data". A Python ``if value <= 0.0`` never sees a tracer that way, so the contract was
asserted in the one case that cannot fail.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp import (
    exact_gp_loocv_log_predictive,
    fit_exact_gp,
    matern52_kernel,
    multi_output_icm_kernel,
    rbf_kernel,
)


def _toy() -> tuple[jax.Array, jax.Array]:
    """A small regression problem.

    Returns:
        Inputs of shape ``(8, 1)`` and targets of shape ``(8,)``.
    """
    x = jnp.linspace(-1.0, 1.0, 8).reshape(-1, 1)
    return x, jnp.sin(3.0 * x[:, 0])


def test_rbf_kernel_accepts_a_traced_lengthscale() -> None:
    """The guard must not forbid compiling a kernel whose scale is traced."""
    x, _ = _toy()

    gram = jax.jit(lambda ls: rbf_kernel(x, x, lengthscale=ls, output_scale=1.0))(jnp.asarray(0.4))

    assert gram.shape == (8, 8)
    assert jnp.all(jnp.isfinite(gram))


def test_matern52_kernel_accepts_a_traced_lengthscale() -> None:
    """The shared kernel guard covers more than one kernel, so check a second."""
    x, _ = _toy()

    gram = jax.jit(lambda ls: matern52_kernel(x, x, lengthscale=ls, output_scale=1.0))(
        jnp.asarray(0.4)
    )

    assert jnp.all(jnp.isfinite(gram))


def test_the_loocv_objective_is_differentiable_under_jit() -> None:
    """The defect this module exists for: optimising what the objective selects.

    Before the guards became trace-aware this raised ``TracerBoolConversionError`` from
    ``fit_exact_gp``, so the objective could not be differentiated with respect to the
    hyperparameters it exists to choose.
    """
    x, y = _toy()

    def objective(params: dict[str, jax.Array]) -> jax.Array:
        state = fit_exact_gp(
            x_train=x,
            y_train=y,
            lengthscale=jnp.exp(params["log_lengthscale"]),
            output_scale=jnp.exp(params["log_output_scale"]),
            noise_std=jnp.exp(params["log_noise"]),
        )
        return -jnp.sum(exact_gp_loocv_log_predictive(state=state))

    start = {
        "log_lengthscale": jnp.asarray(0.0),
        "log_output_scale": jnp.asarray(0.0),
        "log_noise": jnp.asarray(-2.0),
    }
    value, grads = jax.jit(jax.value_and_grad(objective))(start)

    assert jnp.isfinite(value)
    assert all(jnp.isfinite(g) for g in grads.values())
    assert any(abs(float(g)) > 0.0 for g in grads.values()), (
        "a zero gradient everywhere would mean the hyperparameters are not reaching the "
        "objective, so this would pass without testing anything"
    )


def test_the_icm_coregionalisation_is_differentiable_under_jit() -> None:
    """Multi-output: the coregionalisation matrix is the thing a caller fits."""
    feats = jnp.linspace(-1.0, 1.0, 6).reshape(-1, 1)
    tasks = 2
    x = jnp.concatenate(
        [
            jnp.tile(feats, (tasks, 1)),
            jnp.repeat(jnp.arange(tasks), feats.shape[0])[:, None].astype(feats.dtype),
        ],
        axis=1,
    )
    y = jnp.sin(jnp.tile(feats, (tasks, 1))[:, 0])

    def objective(w: jax.Array) -> jax.Array:
        state = fit_exact_gp(
            x_train=x,
            y_train=y,
            lengthscale=0.5,
            output_scale=1.0,
            noise_std=0.1,
            kernel_fn=multi_output_icm_kernel(
                base_kernel_fn=rbf_kernel,
                coregionalisation=w @ w.T + jnp.eye(tasks) * 0.1,
            ),
        )
        return -jnp.sum(exact_gp_loocv_log_predictive(state=state))

    grad = jax.jit(jax.grad(objective))(jnp.asarray([[0.9, 0.1], [0.2, 0.8]]))

    assert grad.shape == (tasks, tasks)
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.max(jnp.abs(grad))) > 0.0


def test_a_concrete_nonpositive_value_still_raises() -> None:
    """The check is kept where it can be decided, which is every eager call."""
    x, y = _toy()

    with pytest.raises(ValueError, match="noise_std"):
        fit_exact_gp(x_train=x, y_train=y, lengthscale=0.4, output_scale=1.0, noise_std=0.0)
    with pytest.raises(ValueError, match="lengthscale"):
        rbf_kernel(x, x, lengthscale=-1.0, output_scale=1.0)
    with pytest.raises(ValueError, match="output_scale"):
        matern52_kernel(x, x, lengthscale=0.4, output_scale=0.0)


def test_a_concrete_nonpositive_value_raises_under_grad_too() -> None:
    """A ``grad`` tracer carries a concrete value, so the check still fires there."""
    x, _ = _toy()

    with pytest.raises(ValueError, match="lengthscale"):
        jax.grad(lambda ls: jnp.sum(rbf_kernel(x, x, lengthscale=ls, output_scale=1.0)))(
            jnp.asarray(-1.0)
        )


def test_the_kernel_is_vmappable_over_its_lengthscale() -> None:
    """``vmap`` raises the same error as ``jit``, so it is covered by the same fix."""
    x, _ = _toy()

    grams = jax.vmap(lambda ls: rbf_kernel(x, x, lengthscale=ls, output_scale=1.0))(
        jnp.asarray([0.3, 0.6])
    )

    assert grams.shape == (2, 8, 8)
    assert jnp.all(jnp.isfinite(grams))
