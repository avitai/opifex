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

import math
import typing as tp

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.uncertainty.gp import (
    exact_gp_loocv_log_predictive,
    fit_exact_gp,
    matern52_kernel,
    multi_output_icm_kernel,
    predict_exact_gp,
    rbf_kernel,
)
from tests.uncertainty.gp._accuracy import (
    assert_reduction_agreement,
    assert_reference_agreement,
    assert_within_budget,
    exponential_kernel_budget,
    TAYLOR_RATE_FLOOR,
    taylor_remainder_rates,
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


#: The problem every accuracy check below is posed on. Small on purpose: these assert
#: floating-point agreement, and a larger fit would measure the fit instead.
TRAIN_SIZE = 8
LENGTHSCALE = 0.4
OUTPUT_SCALE = 1.0
NOISE_STD = 0.1

#: Roundings on the path to each kernel's exponent, and after it, read off the source.
#:
#: ``rbf_kernel`` (``exact.py``): ``x1 - x2``, ``** 2``, ``lengthscale ** 2`` and the divide
#: reach the exponent --- the sum over a one-dimensional input adds none --- then ``exp``,
#: ``output_scale ** 2`` and the final multiply follow it.
RBF_ROUNDINGS = (4, 3)

#: ``matern52_kernel`` (``kernels.py``): ``x1 - x2``, ``** 2``, ``sqrt``, ``sqrt(5)``, the
#: multiply and the divide reach the exponent; the polynomial's six, ``exp``,
#: ``output_scale ** 2`` and two multiplies follow it.
MATERN52_ROUNDINGS = (6, 10)


def _squared_distances(x: jax.Array) -> np.ndarray:
    """Pairwise squared distances, as float64 numpy.

    Args:
        x: Inputs of shape ``(n, d)``.

    Returns:
        The ``(n, n)`` matrix of squared distances.
    """
    return np.asarray(jnp.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1), dtype=np.float64)


class TestValueAccuracy:
    """Tracing a hyperparameter must not change the numbers, only the guard's decidability.

    The tests above this class assert shape and ``isfinite``. A kernel that silently dropped
    its ``-1/2``, or divided by ``lengthscale`` instead of its square, passes every one of
    them -- so the contract that tracing is value-preserving was never actually checked.
    """

    def test_the_rbf_gram_matrix_is_unchanged_by_tracing(self) -> None:
        """Budget is the operation count weighted by the exponent; see ``_accuracy``."""
        x, _ = _toy()

        traced = jax.jit(lambda ls: rbf_kernel(x, x, lengthscale=ls, output_scale=OUTPUT_SCALE))(
            jnp.asarray(LENGTHSCALE)
        )
        concrete = rbf_kernel(x, x, lengthscale=LENGTHSCALE, output_scale=OUTPUT_SCALE)

        amplified, flat = RBF_ROUNDINGS
        assert_within_budget(
            traced,
            concrete,
            budget=exponential_kernel_budget(
                concrete,
                exp_argument=_squared_distances(x) / (2.0 * LENGTHSCALE**2),
                amplified_roundings=amplified,
                flat_roundings=flat,
            ),
            what="rbf_kernel traced versus concrete lengthscale",
        )

    def test_the_matern52_gram_matrix_is_unchanged_by_tracing(self) -> None:
        """A second kernel, with a different exponent and a polynomial prefactor.

        Matern-5/2 is the case that makes reading the exponent back out of the Gram matrix
        wrong: ``-log(k)`` is short of the true exponent by ``log(poly)``, which is 11.18
        against 7.19 here, and the resulting budget is violated on CUDA float32.
        """
        x, _ = _toy()

        traced = jax.jit(
            lambda ls: matern52_kernel(x, x, lengthscale=ls, output_scale=OUTPUT_SCALE)
        )(jnp.asarray(LENGTHSCALE))
        concrete = matern52_kernel(x, x, lengthscale=LENGTHSCALE, output_scale=OUTPUT_SCALE)

        amplified, flat = MATERN52_ROUNDINGS
        assert_within_budget(
            traced,
            concrete,
            budget=exponential_kernel_budget(
                concrete,
                exp_argument=np.sqrt(5.0) * np.sqrt(_squared_distances(x)) / LENGTHSCALE,
                amplified_roundings=amplified,
                flat_roundings=flat,
            ),
            what="matern52_kernel traced versus concrete lengthscale",
        )

    def test_an_exactly_representable_lengthscale_gives_bit_identical_values(self) -> None:
        """Why the budget above is not zero, demonstrated rather than asserted in prose.

        A Python ``float`` is weakly typed, so ``lengthscale ** 2`` is evaluated by Python in
        double precision and only then bound to float32; a traced float32 scalar is squared in
        float32. For ``0.4`` those differ by one rounding of the square, which the exponent
        amplifies. For ``0.5`` the square is exact in both, and nothing is left to differ ---
        both calls are eager here, so no fusion difference enters either.
        """
        x, _ = _toy()

        from_array = rbf_kernel(x, x, lengthscale=jnp.asarray(0.5), output_scale=OUTPUT_SCALE)
        from_float = rbf_kernel(x, x, lengthscale=0.5, output_scale=OUTPUT_SCALE)

        np.testing.assert_array_equal(np.asarray(from_array), np.asarray(from_float))


class TestPredictAgainstAnIndependentReference:
    """The one check here that an agreement test cannot make.

    Every other accuracy test in this module compares two evaluations of the same code, so a
    wrong formula changes both sides equally and passes. Measured: scaling the predictive
    variance's subtrahend by ``1.000002`` leaves all 62 tests in this directory green,
    because the pre-existing formula check in ``test_exact.py`` allows ``atol=1e-5`` on a
    quantity whose value is 9.6e-03 --- a thousandfold more room than the quantity's own
    precision. This compares against Rasmussen & Williams Algorithm 2.1 written out
    independently in float64 NumPy, using an LU solve rather than a Cholesky one.
    """

    @staticmethod
    def _reference(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Algorithm 2.1 in float64 NumPy, independent of the code under test.

        Args:
            x: Training inputs, used as test inputs too.

        Returns:
            The predictive mean and the marginal predictive variance.
        """
        targets = np.sin(3.0 * x[:, 0])
        squared = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
        cross = OUTPUT_SCALE**2 * np.exp(-0.5 * squared / LENGTHSCALE**2)
        gram = cross + NOISE_STD**2 * np.eye(x.shape[0])
        mean = cross.T @ np.linalg.solve(gram, targets)
        variance = OUTPUT_SCALE**2 - np.einsum("ij,ij->j", cross, np.linalg.solve(gram, cross))
        return mean, variance

    def test_the_predictive_mean_matches_the_reference(self) -> None:
        """Budget is the forward error of the ``n``-term inner product."""
        x, y = _toy()
        state = fit_exact_gp(
            x_train=x,
            y_train=y,
            lengthscale=LENGTHSCALE,
            output_scale=OUTPUT_SCALE,
            noise_std=NOISE_STD,
        )
        mean_reference, _ = self._reference(np.asarray(x, dtype=np.float64))

        assert_reference_agreement(
            jnp.asarray(predict_exact_gp(state=state, x_test=x).mean),
            mean_reference,
            terms=TRAIN_SIZE,
            scale=float(np.max(np.abs(mean_reference))),
            what="predictive mean against a float64 reference",
        )

    def test_the_predictive_variance_matches_the_reference(self) -> None:
        """The cancelling branch, where the scale is the operands' and not the result's.

        ``k_diag - sum(v * v)`` is about 9.6e-03 while both operands are about 1, so the
        budget is set by ``output_scale ** 2``. Measured, float32 uses 1.33 ULP of it on CPU
        and 1.83 on CUDA against this reference, out of the 8 the ``n``-term sum allows.
        """
        x, y = _toy()
        state = fit_exact_gp(
            x_train=x,
            y_train=y,
            lengthscale=LENGTHSCALE,
            output_scale=OUTPUT_SCALE,
            noise_std=NOISE_STD,
        )
        _, variance_reference = self._reference(np.asarray(x, dtype=np.float64))

        assert_reference_agreement(
            jnp.asarray(predict_exact_gp(state=state, x_test=x).variance),
            variance_reference,
            terms=TRAIN_SIZE,
            scale=OUTPUT_SCALE**2,
            what="predictive variance against a float64 reference",
        )


class TestGradientAccuracy:
    """The capability this module exists for, checked against something other than itself.

    ``test_the_loocv_objective_is_differentiable_under_jit`` asserts the gradient is finite
    and somewhere non-zero. A ``stop_gradient`` anywhere in the fit passes that, because the
    remaining paths still carry signal. These compare the gradient against the objective.
    """

    @staticmethod
    def _objective(x: jax.Array, y: jax.Array) -> tp.Callable[[jax.Array], jax.Array]:
        """Negative LOOCV log-predictive as a function of log-hyperparameters.

        Args:
            x: Training inputs.
            y: Training targets.

        Returns:
            A scalar objective over ``[log lengthscale, log output scale, log noise]``.
        """

        def objective(params: jax.Array) -> jax.Array:
            state = fit_exact_gp(
                x_train=x,
                y_train=y,
                lengthscale=jnp.exp(params[0]),
                output_scale=jnp.exp(params[1]),
                noise_std=jnp.exp(params[2]),
            )
            return -exact_gp_loocv_log_predictive(state=state)

        return objective

    def test_the_taylor_remainder_converges_at_second_order(self) -> None:
        """pyadjoint's ``taylor_test``: a correct gradient gives rate 2, a wrong one gives 1.

        Run in float64 because the verdict is a convergence rate. In float32 the remainder at
        these step sizes is dominated by roundoff rather than by the truncation term, and the
        measured rates are 1.84, -0.26 and 0.57 for a gradient that is in fact correct -- so
        the instrument, not the gradient, is what float32 would be testing.
        """
        with jax.enable_x64(True):
            x = jnp.linspace(-1.0, 1.0, TRAIN_SIZE, dtype=jnp.float64).reshape(-1, 1)
            y = jnp.sin(3.0 * x[:, 0])
            objective = self._objective(x, y)
            start = jnp.asarray([0.0, 0.0, -2.0], dtype=jnp.float64)

            residuals, rates = taylor_remainder_rates(objective, start, jax.grad(objective)(start))

        assert len(rates) >= 2, f"too few usable residuals to read a rate: {residuals}"
        assert min(rates) >= TAYLOR_RATE_FLOOR, (
            f"Taylor remainder rates {rates} fall below {TAYLOR_RATE_FLOOR}; a first-order "
            f"remainder means the gradient disagrees with the objective. residuals={residuals}"
        )

    def test_the_central_difference_error_falls_as_the_square_of_the_step(self) -> None:
        """A second tolerance-free check, on a different mechanism from the Taylor test.

        A central difference of a correct gradient has error ``O(h^2)``, so halving the step
        quarters it. Asserting the *rate* rather than a level means there is no tolerance to
        choose. Steps stay where truncation dominates: in float64 the error is 2.99e-07 at
        1e-03 and 2.97e-09 at 1e-04, a ratio of 100 for a tenfold step, and only below 1e-05
        does roundoff take over and the ratio invert.
        """
        with jax.enable_x64(True):
            x = jnp.linspace(-1.0, 1.0, TRAIN_SIZE, dtype=jnp.float64).reshape(-1, 1)
            y = jnp.sin(3.0 * x[:, 0])
            objective = self._objective(x, y)
            start = jnp.asarray([0.0, 0.0, -2.0], dtype=jnp.float64)
            analytic = jax.grad(objective)(start)

            errors = []
            for step in (2e-3, 1e-3):
                differenced = jnp.stack(
                    [
                        (
                            objective(start + jnp.zeros_like(start).at[axis].set(step))
                            - objective(start - jnp.zeros_like(start).at[axis].set(step))
                        )
                        / (2.0 * step)
                        for axis in range(start.shape[0])
                    ]
                )
                errors.append(float(jnp.max(jnp.abs(differenced - analytic) / jnp.abs(analytic))))

        coarse, fine = errors
        assert fine < coarse, f"halving the step did not reduce the error: {errors}"
        order = math.log(coarse / fine) / math.log(2.0)
        assert order >= 1.8, (
            f"central-difference error fell at order {order:.3f}, not the 2 a correct "
            f"gradient gives; errors={errors}"
        )

    def test_the_gradient_is_unchanged_by_jit_and_by_vmap(self) -> None:
        """Compiling or batching the gradient must not move it.

        Budget is torch-harmonics' reduction tolerance over the ``n x n`` solve the LOOCV
        objective performs. Worst observed across CPU and CUDA in both precisions is 0.40 of
        budget, for the batched comparison on CUDA float32.
        """
        x, y = _toy()
        objective = self._objective(x, y)
        start = jnp.asarray([0.0, 0.0, -2.0])
        eager = jax.grad(objective)(start)

        assert_reduction_agreement(
            jax.jit(jax.grad(objective))(start),
            eager,
            terms=TRAIN_SIZE**2,
            what="LOOCV gradient, compiled versus eager",
        )

        shifted = start + 0.1
        assert_reduction_agreement(
            jax.vmap(jax.grad(objective))(jnp.stack([start, shifted])),
            jnp.stack([eager, jax.grad(objective)(shifted)]),
            terms=TRAIN_SIZE**2,
            what="LOOCV gradient, batched versus looped",
        )


class TestTheGuardsRefuseNonPositiveScales:
    """Both branches of both guards, which the tests above leave half-covered.

    Coverage of this pull request reports one missing line and one partial branch in each of
    ``exact.py`` and ``kernels.py``: ``rbf_kernel``'s ``output_scale`` refusal and the shared
    kernel guard's ``lengthscale`` refusal. A guard whose refusing branch never runs is a
    guard whose message, and whose decision to refuse at all, is untested.
    """

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_the_rbf_output_scale_is_refused(self, value: float) -> None:
        """``exact.py``: the branch no test reached."""
        x, _ = _toy()

        with pytest.raises(ValueError, match="output_scale must be strictly positive"):
            rbf_kernel(x, x, lengthscale=LENGTHSCALE, output_scale=value)

    @pytest.mark.parametrize("value", [0.0, -1.0])
    def test_the_shared_kernel_guard_refuses_a_lengthscale(self, value: float) -> None:
        """``kernels.py``: the other branch no test reached.

        ``matern52_kernel`` routes through ``_require_positive_kernel_hparams``, which is a
        different guard from ``rbf_kernel``'s own pair in ``exact.py``.
        """
        x, _ = _toy()

        with pytest.raises(ValueError, match="lengthscale must be strictly positive"):
            matern52_kernel(x, x, lengthscale=value, output_scale=OUTPUT_SCALE)

    def test_the_message_names_the_offending_value(self) -> None:
        """The message is the only thing a caller gets, since the value cannot be inspected."""
        x, _ = _toy()

        with pytest.raises(ValueError, match=r"got -2\.5"):
            matern52_kernel(x, x, lengthscale=-2.5, output_scale=OUTPUT_SCALE)


class TestTheHyperparameterIsTracedAndNotBakedIn:
    """The contract, stated as the property that distinguishes traced from static.

    Every other test here shows a traced hyperparameter is *accepted*. None shows it is
    *traced*: a value frozen into the compiled graph as a constant would satisfy them all,
    one recompilation at a time, and an optimiser would then pay a full compile per step
    while still getting right answers. ``chex.assert_max_traces`` is the repo's existing
    instrument for this -- see ``tests/core/spectral/test_spectral_utils_jit.py``.
    """

    def test_changing_the_lengthscale_does_not_recompile(self) -> None:
        """One trace across four different length-scales."""
        x, _ = _toy()
        chex.clear_trace_counter()

        @jax.jit
        @chex.assert_max_traces(n=1)
        def gram(lengthscale: jax.Array) -> jax.Array:
            return rbf_kernel(x, x, lengthscale=lengthscale, output_scale=OUTPUT_SCALE)

        produced = [gram(jnp.asarray(value)) for value in (0.3, 0.4, 0.5, 0.6)]

        assert len({float(jnp.sum(value)) for value in produced}) == len(produced), (
            "the four length-scales gave repeated sums, so the value is not reaching the "
            "kernel and the trace count would be uninformative"
        )

    def test_optimising_the_objective_compiles_once_for_every_step(self) -> None:
        """The defect's real cost: a static hyperparameter recompiles once per step."""
        x, y = _toy()
        chex.clear_trace_counter()

        @jax.jit
        @chex.assert_max_traces(n=1)
        def step(params: jax.Array) -> jax.Array:
            state = fit_exact_gp(
                x_train=x,
                y_train=y,
                lengthscale=jnp.exp(params[0]),
                output_scale=jnp.exp(params[1]),
                noise_std=jnp.exp(params[2]),
            )
            return jax.grad(lambda p: -exact_gp_loocv_log_predictive(state=state) * p[0])(params)

        current = jnp.asarray([0.0, 0.0, -2.0])
        for _ in range(5):
            current = current - 1e-4 * step(current)

        assert jnp.all(jnp.isfinite(current))

    def test_grad_of_jit_and_jit_of_grad_agree(self) -> None:
        """Transform composition in both orders, which an optimiser picks between freely."""
        x, y = _toy()
        objective = TestGradientAccuracy._objective(x, y)
        start = jnp.asarray([0.0, 0.0, -2.0])

        assert_reduction_agreement(
            jax.grad(jax.jit(objective))(start),
            jax.jit(jax.grad(objective))(start),
            terms=TRAIN_SIZE**2,
            what="grad(jit(f)) versus jit(grad(f))",
        )
