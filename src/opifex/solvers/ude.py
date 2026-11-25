"""Universal Differential Equation (UDE) and pure-JAX adaptive Tsit5 ODE solver.

Implements the UDE grey-box modelling framework of Rackauckas et al. (2020,
``arXiv:2001.04385``) together with a pure-JAX adaptive 7-stage FSAL Tsitouras
5(4) Runge-Kutta integrator (:func:`tsit5_solve`) with in-step cursor-driven
4th-degree continuous-extension dense output (:math:`O(h^5)` global dense accuracy),
Hairer-Nørsett-Wanner (§II.4) deadbeat I-controller step-size adaptation, and
logarithmic checkpointed discretize-then-optimize reverse-mode automatic
differentiation.

References:
    * Rackauckas, C., et al. (2020). "Universal Differential Equations for
      Scientific Machine Learning." ``arXiv:2001.04385``.
    * Tsitouras, Ch. (2011). "Runge-Kutta pairs of order 5(4) satisfying only
      the first column simplifying assumption." *Comput. Math. Appl.* 62(2),
      770-775.
    * Hairer, E., Nørsett, S. P., & Wanner, G. (2008). *Solving Ordinary
      Differential Equations I: Nonstiff Problems* (2nd rev. ed.), §II.4, Eq. (4.13).
    * Chen, R. T. Q., et al. (2018). "Neural Ordinary Differential Equations."
      *NeurIPS*.
"""

# ruff: noqa: F821  # jaxtyping dimension names (dim, n_times, etc.)

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final

import jax
import jax.numpy as jnp
from flax import nnx, struct
from jaxtyping import Array, Float

from opifex.core.solver.implicit import infer_inexact_dtype
from opifex.core.solver.loop import bounded_while_loop
from opifex.core.solver.solution import Solution
from opifex.core.solver.status import is_successful, Status


# Tsitouras (2011, Table 1) 7-stage FSAL 5(4) explicit Runge-Kutta tableau.
_TSIT5_C: Final[tuple[float, ...]] = (
    0.161,
    0.327,
    0.9,
    0.9800255409045097,
    1.0,
    1.0,
)

_TSIT5_A2: Final[tuple[float, ...]] = (0.161,)
_TSIT5_A3: Final[tuple[float, ...]] = (
    -0.008480655492356989,
    0.335480655492357,
)
_TSIT5_A4: Final[tuple[float, ...]] = (
    2.8971530571054935,
    -6.359448489975075,
    4.3622954328695815,
)
_TSIT5_A5: Final[tuple[float, ...]] = (
    5.325864828439257,
    -11.748883564062828,
    7.4955393428898365,
    -0.09249506636175525,
)
_TSIT5_A6: Final[tuple[float, ...]] = (
    5.86145544294642,
    -12.92096931784711,
    8.159367898576159,
    -0.071584973281401,
    -0.028269050394068383,
)
_TSIT5_B_SOL: Final[tuple[float, ...]] = (
    0.09646076681806523,
    0.01,
    0.4798896504144996,
    1.3790085741037418,
    -3.2900695154360807,
    2.324710524099774,
)
_TSIT5_B_ERR: Final[tuple[float, ...]] = (
    0.0017800110522257771,
    0.0008164344596567469,
    -0.007880878010261996,
    0.1447110071732629,
    -0.5823571654525552,
    0.45808210592918697,
    -1.0 / 66.0,
)


@dataclass(frozen=True)
class UDEConfig:
    """Configuration for Tsit5 UDE / NeuralODE integration.

    Attributes:
        dt0: Initial step size (or constant step size when ``adaptive=False``).
        rtol: Relative tolerance for local truncation error control.
        atol: Absolute tolerance for local truncation error control.
        max_steps: Maximum integration step attempts.
        dtmin: Optional minimum step size ``> 0`` for adaptive control.
        force_dtmin: When ``True`` (default), clamp ``|dt|`` to ``dtmin`` and accept
            steps taken at ``dtmin``; when ``False``, halt with
            :attr:`~opifex.core.solver.status.Status.STEP_TOO_SMALL` if ``|dt| < dtmin``.
        adaptive: When ``True`` (default), adapt step sizes via the
            Hairer-Nørsett-Wanner (§II.4) deadbeat I-controller; when ``False``,
            advance with constant step size ``dt0`` (clamping the final step to ``t1``).
    """

    dt0: float = 0.01
    rtol: float = 1e-5
    atol: float = 1e-5
    max_steps: int = 4096
    dtmin: float | None = None
    force_dtmin: bool = True
    adaptive: bool = True


class _Tsit5Carry(struct.PyTreeNode):
    """Internal carry threaded through the checkpointed bounded Tsit5 loop."""

    t: Array
    y: Array
    f: Array
    dt: Array
    at_dtmin: Array
    step_attempts: Array
    accepted_steps: Array
    save_idx: Array
    done: Array
    has_nonfinite: Array
    step_too_small: Array
    ys: Array


def _validate_ude_config(config: UDEConfig) -> None:
    """Validate static adaptive-integration parameters before tracing.

    Args:
        config: Solver configuration.

    Raises:
        TypeError: When ``force_dtmin`` is not a ``bool``.
        ValueError: When ``max_steps < 1`` or ``dt0`` / ``rtol`` / ``atol`` / ``dtmin``
            are non-positive or non-finite.
    """
    if config.max_steps < 1:
        raise ValueError(f"max_steps must be at least 1; got {config.max_steps}.")
    if not all(x > 0.0 and math.isfinite(x) for x in (config.dt0, config.rtol, config.atol)):
        raise ValueError(
            "dt0, rtol, and atol must be finite and strictly positive; got "
            f"dt0={config.dt0}, rtol={config.rtol}, atol={config.atol}."
        )
    if config.dtmin is not None and (config.dtmin <= 0.0 or not math.isfinite(config.dtmin)):
        raise ValueError(f"dtmin must be finite and strictly positive; got dtmin={config.dtmin}.")
    if not isinstance(config.force_dtmin, bool):
        raise TypeError(f"force_dtmin must be a bool; got {type(config.force_dtmin).__name__}.")
    if not isinstance(config.adaptive, bool):
        raise TypeError(f"adaptive must be a bool; got {type(config.adaptive).__name__}.")


def _tsit5_single_step(
    rhs_fn: Callable[[Array, Array], Array],
    t_c: Array,
    y_c: Array,
    f1: Array,
    dt: Array,
) -> tuple[Array, Array, Array, Array]:
    """Advance one 7-stage FSAL Tsit5 step (6 RHS evaluations)."""
    dtype = y_c.dtype
    c = tuple(jnp.asarray(val, dtype=dtype) for val in _TSIT5_C)

    y2 = y_c + dt * (_TSIT5_A2[0] * f1)
    f2 = rhs_fn(t_c + c[0] * dt, y2)

    y3 = y_c + dt * (_TSIT5_A3[0] * f1 + _TSIT5_A3[1] * f2)
    f3 = rhs_fn(t_c + c[1] * dt, y3)

    y4 = y_c + dt * (_TSIT5_A4[0] * f1 + _TSIT5_A4[1] * f2 + _TSIT5_A4[2] * f3)
    f4 = rhs_fn(t_c + c[2] * dt, y4)

    y5 = y_c + dt * (_TSIT5_A5[0] * f1 + _TSIT5_A5[1] * f2 + _TSIT5_A5[2] * f3 + _TSIT5_A5[3] * f4)
    f5 = rhs_fn(t_c + c[3] * dt, y5)

    y6 = y_c + dt * (
        _TSIT5_A6[0] * f1
        + _TSIT5_A6[1] * f2
        + _TSIT5_A6[2] * f3
        + _TSIT5_A6[3] * f4
        + _TSIT5_A6[4] * f5
    )
    f6 = rhs_fn(t_c + c[4] * dt, y6)

    y_cand = y_c + dt * (
        _TSIT5_B_SOL[0] * f1
        + _TSIT5_B_SOL[1] * f2
        + _TSIT5_B_SOL[2] * f3
        + _TSIT5_B_SOL[3] * f4
        + _TSIT5_B_SOL[4] * f5
        + _TSIT5_B_SOL[5] * f6
    )
    f7 = rhs_fn(t_c + c[5] * dt, y_cand)

    fs = jnp.stack([f1, f2, f3, f4, f5, f6, f7], axis=0)
    b_err = jnp.asarray(_TSIT5_B_ERR, dtype=dtype)
    y_err = dt * jnp.einsum("s,s...->...", b_err, fs)
    ks = dt * fs
    return y_cand, f7, y_err, ks


def _tsit5_poly_eval_one(
    seg_t0: Array,
    seg_t1: Array,
    seg_y0: Array,
    seg_ks: Array,
    t_query: Array,
) -> Array:
    """Evaluate Tsitouras (2011) 4th-degree continuous extension at scalar ``t_query``."""
    dtype = seg_y0.dtype
    span = seg_t1 - seg_t0
    has_span = jnp.abs(span) > 0.0
    safe_span = jnp.where(has_span, span, jnp.ones_like(span))
    theta = jnp.where(has_span, (t_query - seg_t0) / safe_span, jnp.zeros_like(t_query))
    theta2 = theta * theta

    b1 = (
        -1.0530884977290216
        * theta
        * (theta - 1.3299890189751412)
        * (theta2 - 1.4364028541716351 * theta + 0.7139816917074209)
    )
    b2 = 0.1017 * theta2 * (theta2 - 2.1966568338249754 * theta + 1.2949852507374631)
    b3 = 2.4906272856512528 * theta2 * (theta2 - 2.3853564547206166 * theta + 1.5780346820809249)
    b4 = -16.548102889244903 * (theta - 1.2171292729553324) * (theta - 0.6162040603780009) * theta2
    b5 = 47.37952196281928 * (theta - 1.2030712083723626) * (theta - 0.6580472926535474) * theta2
    b6 = -34.87065786149661 * (theta - 1.2) * (theta - 2.0 / 3.0) * theta2
    b7 = 2.5 * (theta - 1.0) * (theta - 0.6) * theta2

    weights = jnp.stack([b1, b2, b3, b4, b5, b6, b7], axis=0).astype(dtype)
    return seg_y0 + jnp.matmul(weights, seg_ks)


def _spacing_min_step(t_val: Array, direction: Array) -> Array:
    """Floating-point number-spacing floor ``10 * |nextafter(t, dir * inf) - t|``."""
    t_clean = jax.lax.stop_gradient(t_val)
    target = direction * jnp.asarray(jnp.inf, dtype=t_val.dtype)
    return 10.0 * jnp.abs(jnp.nextafter(t_clean, target) - t_clean)


def _i_controller_adapt_step(
    y_c: Array,
    y_cand: Array,
    y_err: Array,
    at_dtmin: Array,
    *,
    rtol: float,
    atol: float,
) -> tuple[Array, Array]:
    """Compute step acceptance and Hairer-Nørsett-Wanner (§II.4, Eq. 4.13) step factor."""
    dtype = y_c.dtype
    nan_cand = ~jnp.all(jnp.isfinite(y_cand)) | ~jnp.all(jnp.isfinite(y_err))
    safe_cand = jnp.where(nan_cand, y_c, y_cand)
    scale = jnp.asarray(atol, dtype=dtype) + jnp.asarray(rtol, dtype=dtype) * jnp.maximum(
        jnp.abs(y_c), jnp.abs(safe_cand)
    )
    ratio = jnp.where(nan_cand, jnp.inf, y_err / scale)
    scaled_err = jnp.sqrt(jnp.mean(ratio * ratio))
    accept = (~nan_cand) & ((scaled_err <= 1.0) | at_dtmin)

    inv_err = jax.lax.stop_gradient(
        jnp.clip(1.0 / jnp.maximum(scaled_err, jnp.asarray(1e-10, dtype=dtype)), 1e-4, 1e4)
    )
    # Hairer, Nørsett & Wanner (2008, §II.4, Eq. 4.13) deadbeat I-controller:
    # safety = 0.9, exponent = 1 / 5 for the 5(4) pair, factor floor 1.0 on accepted
    # steps (never shrinking an accepted step) and ceiling 0.9 on rejected steps.
    raw_factor = 0.9 * (inv_err ** (1.0 / 5.0))
    f_min = jnp.where(accept, jnp.asarray(1.0, dtype=dtype), jnp.asarray(0.2, dtype=dtype))
    f_max = jnp.where(accept, jnp.asarray(10.0, dtype=dtype), jnp.asarray(0.9, dtype=dtype))
    factor = jax.lax.stop_gradient(jnp.clip(raw_factor, f_min, f_max))
    return accept, factor


def _update_step_size(
    carry: _Tsit5Carry,
    step_dt: Array,
    factor: Array,
    accept: Array,
    next_done: Array,
    next_t: Array,
    direction: Array,
    cfg: UDEConfig,
) -> tuple[Array, Array, Array]:
    """Update ``(next_dt, next_at_dtmin, step_too_small)`` with ``dtmin`` and spacing guards."""
    dtype = step_dt.dtype
    dt_next_abs = jnp.abs(jax.lax.stop_gradient(step_dt)) * factor
    hit_dtmin_err = jnp.asarray(False)
    next_at_dtmin = jnp.asarray(False)

    if cfg.dtmin is not None:
        dtmin_val = jnp.asarray(cfg.dtmin, dtype=dtype)
        if not cfg.force_dtmin:
            hit_dtmin_err = (~next_done) & (dt_next_abs < dtmin_val)
        dt_next_abs = jnp.where(carry.at_dtmin & (factor == 1.0), dtmin_val, dt_next_abs)
        dt_next_abs = jnp.maximum(dt_next_abs, dtmin_val)

    min_step_curr = _spacing_min_step(carry.t, direction)
    min_step_next = _spacing_min_step(next_t, direction)
    hit_spacing = (~next_done) & (~accept) & (dt_next_abs < min_step_curr)
    dt_next_abs = jnp.where(accept, jnp.maximum(dt_next_abs, min_step_next), dt_next_abs)
    if cfg.dtmin is not None:
        next_at_dtmin = jnp.asarray(cfg.force_dtmin) & (
            dt_next_abs <= jnp.asarray(cfg.dtmin, dtype=dtype)
        )
    return direction * dt_next_abs, next_at_dtmin, hit_dtmin_err | hit_spacing


def _lane_is_active(carry: _Tsit5Carry, max_steps: int) -> Array:
    """Return ``True`` while a trajectory lane still has integration steps to take."""
    return (
        (~carry.done)
        & (~carry.has_nonfinite)
        & (~carry.step_too_small)
        & (carry.step_attempts < max_steps)
    )


def _advance_saved_points_in_step(
    c: _Tsit5Carry,
    t_cand: Array,
    ks: Array,
    ts_f: Array,
    direction: Array,
    step_accepted: Array,
) -> tuple[Array, Array]:
    """Interpolate only the ``ts_f`` entries falling inside the accepted step ``(c.t, t_cand]``."""
    n_times = ts_f.shape[0]
    dir_t_cand = jax.lax.stop_gradient(direction * t_cand)

    def _in_step_at_cursor(idx: Array) -> Array:
        safe_idx = jnp.minimum(idx, n_times - 1)
        t_target = ts_f[safe_idx]
        return (
            step_accepted
            & (idx < n_times)
            & (jax.lax.stop_gradient(direction * t_target) <= dir_t_cand)
        )

    def save_cond(save_state: tuple[Array, Array]) -> Array:
        idx, _ys_curr = save_state
        return _in_step_at_cursor(idx)

    def save_body(save_state: tuple[Array, Array]) -> tuple[Array, Array]:
        idx, ys_curr = save_state
        in_range = _in_step_at_cursor(idx)
        safe_idx = jnp.minimum(idx, n_times - 1)
        y_interp = _tsit5_poly_eval_one(c.t, t_cand, c.y, ks, ts_f[safe_idx])
        updated_row = jnp.where(in_range, y_interp, ys_curr[safe_idx])
        return idx + in_range.astype(jnp.int32), ys_curr.at[safe_idx].set(updated_row)

    return bounded_while_loop(
        save_cond,
        save_body,
        (c.save_idx, c.ys),
        max_steps=n_times,
    )


def _make_tsit5_step_body(
    rhs_fn: Callable[[Array, Array], Array],
    ts_f: Array,
    t1: Array,
    direction: Array,
    cfg: UDEConfig,
) -> Callable[[_Tsit5Carry], _Tsit5Carry]:
    """Build the per-lane-freezing adaptive Tsit5 step body with cursor ``ts`` saving."""
    dtype = ts_f.dtype

    def body_fn(c: _Tsit5Carry) -> _Tsit5Carry:
        """Advance or reject one Tsit5 step on active lanes and freeze finished lanes."""
        active = _lane_is_active(c, cfg.max_steps)
        rem = t1 - c.t
        rem_abs = jax.lax.stop_gradient(jnp.abs(rem))
        dt_abs = jax.lax.stop_gradient(jnp.abs(c.dt))
        hitting_end = dt_abs >= rem_abs * (1.0 - 10.0 * float(jnp.finfo(dtype).eps))
        step_dt = jnp.where(hitting_end, rem, c.dt)
        t_cand = jnp.where(hitting_end, t1, c.t + step_dt)

        y_cand, f7, y_err, ks = _tsit5_single_step(rhs_fn, c.t, c.y, c.f, step_dt)
        nonfinite_step = ~jnp.all(jnp.isfinite(y_cand)) | ~jnp.all(jnp.isfinite(f7))
        if cfg.adaptive:
            accept, factor = _i_controller_adapt_step(
                c.y,
                y_cand,
                y_err,
                c.at_dtmin,
                rtol=cfg.rtol,
                atol=cfg.atol,
            )
            next_t = jnp.where(accept, t_cand, c.t)
            next_y = jnp.where(accept, y_cand, c.y)
            next_f = jnp.where(accept, f7, c.f)
            next_done = accept & hitting_end
            next_dt, next_at_dtmin, hit_small = _update_step_size(
                c,
                step_dt,
                factor,
                accept,
                next_done,
                next_t,
                direction,
                cfg,
            )
            terminal_nonfinite = nonfinite_step & c.at_dtmin
        else:
            accept = ~nonfinite_step
            next_t = jnp.where(accept, t_cand, c.t)
            next_y = jnp.where(accept, y_cand, c.y)
            next_f = jnp.where(accept, f7, c.f)
            next_done = accept & hitting_end
            next_dt = c.dt
            next_at_dtmin = jnp.asarray(False)
            hit_small = jnp.asarray(False)
            terminal_nonfinite = nonfinite_step

        next_save_idx, cand_ys = _advance_saved_points_in_step(
            c, t_cand, ks, ts_f, direction, active & accept
        )

        return _Tsit5Carry(
            t=jnp.where(active, next_t, c.t),
            y=jnp.where(active, next_y, c.y),
            f=jnp.where(active, next_f, c.f),
            dt=jnp.where(active, next_dt, c.dt),
            at_dtmin=jnp.where(active, next_at_dtmin, c.at_dtmin),
            step_attempts=c.step_attempts + active.astype(jnp.int32),
            accepted_steps=c.accepted_steps + (active & accept).astype(jnp.int32),
            save_idx=next_save_idx,
            done=c.done | (active & next_done),
            has_nonfinite=c.has_nonfinite | (active & terminal_nonfinite),
            step_too_small=c.step_too_small | (active & hit_small),
            ys=jnp.where(active, cand_ys, c.ys),
        )

    return body_fn


def _validate_tsit5_inputs(y0: Array, ts: Array) -> None:
    """Validate 1-D initial state and concrete monotone 1-D save-time grid."""
    if y0.ndim != 1 or y0.shape[0] < 1:
        raise ValueError(
            "y0 must be a non-empty 1-D state vector (use jax.vmap for batches); "
            f"got shape {y0.shape}."
        )
    if ts.ndim != 1 or ts.shape[0] < 1:
        raise ValueError(f"ts must be a non-empty 1-D array; got shape {ts.shape}.")
    if ts.shape[0] >= 2:
        diffs = jnp.diff(ts)
        try:
            is_monotone = bool(jnp.all(diffs >= 0) or jnp.all(diffs <= 0))
        except jax.errors.TracerBoolConversionError:
            is_monotone = True
        if not is_monotone:
            raise ValueError("ts must be monotonically non-decreasing or non-increasing.")


def tsit5_solve(
    vector_field: Callable[[Array, Array, Any], Array],
    y0: Array,
    ts: Array,
    *,
    config: UDEConfig | None = None,
    args: Any = None,
) -> Solution:
    """Integrate ``dy/dt = vector_field(t, y, args)`` with pure-JAX FSAL Tsit5.

    Args:
        vector_field: Callable ``(t, y, args) -> dy/dt``.
        y0: Initial state vector of shape ``(state_dim,)``.
        ts: Monotone 1-D save times ``(n_times,)`` spanning ``[ts[0], ts[-1]]``.
        config: Optional :class:`UDEConfig` (uses defaults when ``None``).
        args: Optional pytree forwarded as the third argument to ``vector_field``.

    Returns:
        :class:`~opifex.core.solver.solution.Solution` with ``fields={"y": ..., "t": ...}``,
        ``status`` (``SUCCESS``, ``INITIAL_FAILURE``, ``STEP_TOO_SMALL``, ``MAX_ITERS``,
        or ``NONFINITE``), and ``stats={"num_steps": ...}``.
    """
    cfg = config or UDEConfig()
    _validate_ude_config(cfg)
    _validate_tsit5_inputs(y0, ts)

    dtype = infer_inexact_dtype(y0, ts, args)
    y0_f = jnp.asarray(y0, dtype=dtype)
    ts_f = jnp.asarray(ts, dtype=dtype)
    t0 = ts_f[0]
    t1 = ts_f[-1]
    total_span = t1 - t0
    direction = jnp.where(
        jax.lax.stop_gradient(total_span) >= 0.0,
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(-1.0, dtype=dtype),
    )
    if ts_f.shape[0] >= 2:
        diffs = jnp.diff(jax.lax.stop_gradient(ts_f))
        ts_valid = jnp.all(diffs >= 0.0) | jnp.all(diffs <= 0.0)
    else:
        ts_valid = jnp.asarray(True)

    span_abs = jax.lax.stop_gradient(jnp.abs(total_span))
    trivial_span = span_abs == 0.0
    tiny = jnp.asarray(float(jnp.finfo(dtype).tiny), dtype=dtype)
    raw_dt0 = jnp.asarray(cfg.dt0, dtype=dtype)
    if cfg.adaptive and cfg.dtmin is not None:
        raw_dt0 = jnp.maximum(raw_dt0, jnp.asarray(cfg.dtmin, dtype=dtype))
    raw_dt0 = jnp.maximum(raw_dt0, _spacing_min_step(t0, direction))
    dt0_abs = jnp.minimum(raw_dt0, jnp.maximum(span_abs, tiny))
    dt_init = direction * dt0_abs

    def rhs_fn(t_val: Array, y_val: Array) -> Array:
        """Evaluate the vector field in the promoted working floating dtype."""
        return jnp.asarray(vector_field(t_val, y_val, args), dtype=dtype)

    f0 = rhs_fn(t0, y0_f)
    init_ys = jnp.where(
        trivial_span,
        y0_f[None, :] + (ts_f - t0)[:, None] * f0[None, :],
        jnp.repeat(y0_f[None, :], ts_f.shape[0], axis=0),
    )
    init_save_idx = jnp.sum(
        (jax.lax.stop_gradient(ts_f) == jax.lax.stop_gradient(t0)).astype(jnp.int32)
    )
    step_body = _make_tsit5_step_body(rhs_fn, ts_f, t1, direction, cfg)

    init_carry = _Tsit5Carry(
        t=t0,
        y=y0_f,
        f=f0,
        dt=dt_init,
        at_dtmin=jnp.asarray(False),
        step_attempts=jnp.asarray(0, dtype=jnp.int32),
        accepted_steps=jnp.asarray(0, dtype=jnp.int32),
        save_idx=init_save_idx,
        done=trivial_span | (~ts_valid),
        has_nonfinite=~jnp.all(jnp.isfinite(y0_f)) | ~jnp.all(jnp.isfinite(f0)),
        step_too_small=jnp.asarray(False),
        ys=init_ys,
    )
    final_carry = bounded_while_loop(
        lambda c: _lane_is_active(c, cfg.max_steps),
        step_body,
        init_carry,
        max_steps=cfg.max_steps,
    )
    raw_ys = final_carry.ys
    has_bad_output = final_carry.has_nonfinite | ~jnp.all(jnp.isfinite(raw_ys))
    valid_ts_status = jnp.where(
        has_bad_output,
        jnp.asarray(int(Status.NONFINITE), dtype=jnp.int32),
        jnp.where(
            final_carry.done,
            jnp.asarray(int(Status.SUCCESS), dtype=jnp.int32),
            jnp.where(
                final_carry.step_too_small,
                jnp.asarray(int(Status.STEP_TOO_SMALL), dtype=jnp.int32),
                jnp.asarray(int(Status.MAX_ITERS), dtype=jnp.int32),
            ),
        ),
    )
    status = jnp.where(
        ts_valid,
        valid_ts_status,
        jnp.asarray(int(Status.INITIAL_FAILURE), dtype=jnp.int32),
    )
    ys = jnp.where(ts_valid, raw_ys, jnp.full_like(raw_ys, jnp.nan))
    return Solution(
        fields={"y": ys, "t": ts_f},
        metrics={},
        status=status,
        stats={"num_steps": final_carry.accepted_steps},
    )


def _validate_ode_module_mode(module: nnx.Module) -> None:
    """Reject active stochastic Dropout or batch-statistics BatchNorm in ODE vector fields."""
    for _path, sub in nnx.iter_graph(module):
        if not isinstance(sub, nnx.Module):
            continue
        det = getattr(sub, "deterministic", None)
        if det is False and not (isinstance(sub, nnx.Dropout) and sub.rate == 0.0):
            raise ValueError(
                "Adaptive ODE integration requires a deterministic vector field; found "
                f"{type(sub).__name__} with deterministic=False (call .eval() first)."
            )
        if getattr(sub, "use_running_average", None) is False:
            raise ValueError(
                "Adaptive ODE integration evaluates single state vectors (batch=1) and "
                f"cannot compute batch statistics; found {type(sub).__name__} with "
                "use_running_average=False (call .eval() first)."
            )


class NeuralODE(nnx.Module):
    """Neural ODE wrapping an ``nnx.Module`` vector field integrated via pure-JAX Tsit5."""

    def __init__(
        self,
        net: nnx.Module,
        state_dim: int,
        *,
        config: UDEConfig | None = None,
        rngs: nnx.Rngs,  # noqa: ARG002 - nnx.Module constructor contract
    ) -> None:
        """Initialize NeuralODE.

        Args:
            net: Neural network mapping ``(batch, state_dim) -> (batch, state_dim)``.
            state_dim: Dimensionality of the state vector.
            config: Solver configuration.
            rngs: Random number generators.
        """
        self.net = net
        self.state_dim = state_dim
        self.config = config or UDEConfig()

    def vector_field(
        self,
        t: float | Array,  # noqa: ARG002 - autonomous vector-field signature (t, y, args)
        y: Array,
        args: object = None,  # noqa: ARG002 - ODE vector-field signature (t, y, args)
    ) -> Array:
        """Evaluate ``dy/dt = net(y)``.

        Args:
            t: Current time (unused for autonomous systems, kept for interface parity).
            y: Current state vector ``(state_dim,)``.
            args: Optional auxiliary arguments (unused).

        Returns:
            State derivative ``(state_dim,)``.
        """
        return self.net(y[None, :])[0]  # type: ignore[reportCallIssue]

    def solve(
        self,
        y0: Float[Array, "dim"],
        ts: Float[Array, "n_times"],
        config: UDEConfig | None = None,
    ) -> Solution:
        """Integrate the Neural ODE from ``y0`` over ``ts`` and return a :class:`Solution`.

        Args:
            y0: Initial state ``(state_dim,)``.
            ts: Monotone save time points ``(n_times,)``.
            config: Optional override configuration.

        Returns:
            :class:`Solution` carrying ``fields["y"]``, ``fields["t"]``, ``status``,
            and ``stats["num_steps"]``.
        """
        return _solve_nnx_ode(self, y0, ts, config)


class UDESolver(nnx.Module):
    """Universal Differential Equation solver combining known physics and a neural residual."""

    def __init__(
        self,
        known_dynamics_fn: Callable[[Any, Array], Array],
        neural_residual: nnx.Module,
        state_dim: int,
        *,
        config: UDEConfig | None = None,
        rngs: nnx.Rngs,  # noqa: ARG002 - nnx.Module constructor contract
    ) -> None:
        """Initialize UDE Solver.

        Args:
            known_dynamics_fn: Pure physics function ``(t, y) -> dy/dt``.
            neural_residual: ``nnx.Module`` mapping ``(batch, state_dim) -> (batch, state_dim)``.
            state_dim: Dimensionality of the state vector.
            config: Solver configuration.
            rngs: Random number generators.
        """
        self.known_dynamics_fn = known_dynamics_fn
        self.neural_residual = neural_residual
        self.state_dim = state_dim
        self.config = config or UDEConfig()

    def vector_field(
        self,
        t: float | Array,
        y: Array,
        args: object = None,  # noqa: ARG002 - ODE vector-field signature (t, y, args)
    ) -> Array:
        """Evaluate ``du/dt = known(t, y) + neural(y)``.

        Args:
            t: Current time.
            y: Current state ``(state_dim,)``.
            args: Optional auxiliary arguments (unused).

        Returns:
            Combined dynamics ``(state_dim,)``.
        """
        known = self.known_dynamics_fn(t, y)
        neural = self.neural_residual(y[None, :])[0]  # type: ignore[reportCallIssue]
        return known + neural

    def solve(
        self,
        y0: Float[Array, "dim"],
        ts: Float[Array, "n_times"],
        config: UDEConfig | None = None,
    ) -> Solution:
        """Integrate the UDE from ``y0`` over ``ts`` and return a :class:`Solution`.

        Args:
            y0: Initial state ``(state_dim,)``.
            ts: Monotone save time points ``(n_times,)``.
            config: Optional override configuration.

        Returns:
            :class:`Solution` carrying ``fields["y"]``, ``fields["t"]``, ``status``,
            and ``stats["num_steps"]``.
        """
        return _solve_nnx_ode(self, y0, ts, config)

    def trajectory_loss(
        self,
        y0: Float[Array, "dim"],
        t_obs: Float[Array, "n_obs"],
        y_obs: Float[Array, "n_obs dim"],
    ) -> Float[Array, ""]:
        """Compute MSE trajectory loss, returning ``NaN`` if integration fails.

        Args:
            y0: Initial state ``(state_dim,)``.
            t_obs: Observation times ``(n_obs,)``.
            y_obs: Observed states ``(n_obs, state_dim)``.

        Returns:
            Scalar MSE loss when ``is_successful(sol.status)``, else ``NaN``.
        """
        sol = self.solve(y0, t_obs)
        y_pred = sol.fields["y"]
        mse = jnp.mean((y_pred - jnp.asarray(y_obs, dtype=y_pred.dtype)) ** 2)
        return jnp.where(
            is_successful(sol.status),
            mse,
            jnp.asarray(jnp.nan, dtype=y_pred.dtype),
        )


def _solve_nnx_ode(
    module: NeuralODE | UDESolver,
    y0: Array,
    ts: Array,
    config: UDEConfig | None,
) -> Solution:
    """Validate NNX mode, split trainable parameters, and integrate ``module.vector_field``."""
    cfg = config or module.config
    _validate_ode_module_mode(module)
    graphdef, params, rest_state = nnx.split(module, nnx.Param, ...)

    def rhs(t_val: Array, y_val: Array, mod_params: nnx.State) -> Array:
        """Reconstruct ``module`` with traced parameters and dispatch ``vector_field``."""
        merged = nnx.merge(graphdef, mod_params, rest_state)
        return merged.vector_field(t_val, y_val)

    return tsit5_solve(rhs, y0, ts, config=cfg, args=params)


class _ResidualMLP(nnx.Module):
    """Multi-layer perceptron for the neural residual term in :func:`create_ude`."""

    def __init__(
        self,
        layers: list[nnx.Linear],
        activation: Callable[[Array], Array],
    ) -> None:
        """Store residual linear layers inside an ``nnx.List`` container."""
        self.layers = nnx.List(layers)
        self.activation = activation

    def __call__(self, x: Array) -> Array:
        """Evaluate the residual MLP on batched states ``(batch, state_dim)``."""
        h = x
        for layer in list(self.layers)[:-1]:
            h = self.activation(layer(h))
        return list(self.layers)[-1](h)


def create_ude(
    known_dynamics_fn: Callable[[Any, Array], Array],
    state_dim: int,
    hidden_dims: list[int],
    *,
    config: UDEConfig | None = None,
    activation: Callable[[Array], Array] = jnp.tanh,
    rngs: nnx.Rngs,
) -> UDESolver:
    """Create a UDE solver with an MLP residual network.

    Args:
        known_dynamics_fn: Known physics function ``(t, y) -> dy/dt``.
        state_dim: State vector dimensionality.
        hidden_dims: List of hidden layer sizes for the neural residual.
        config: Solver configuration.
        activation: Activation function for hidden layers.
        rngs: Random number generators.

    Returns:
        Configured :class:`UDESolver` instance.
    """
    layers: list[nnx.Linear] = []
    in_features = state_dim
    for h in hidden_dims:
        layers.append(nnx.Linear(in_features, h, rngs=rngs))
        in_features = h
    layers.append(nnx.Linear(in_features, state_dim, rngs=rngs))
    net = _ResidualMLP(layers, activation)
    return UDESolver(
        known_dynamics_fn=known_dynamics_fn,
        neural_residual=net,
        state_dim=state_dim,
        config=config,
        rngs=rngs,
    )


__all__ = [
    "NeuralODE",
    "UDEConfig",
    "UDESolver",
    "create_ude",
    "tsit5_solve",
]
