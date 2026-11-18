r"""Pure-JAX stiff Rosenbrock-Wanner and ESDIRK ordinary differential equation solvers.

Integrates stiff initial-value problems

.. math::

    \frac{\mathrm{d}y}{\mathrm{d}t} = f(t, y, \theta), \quad y(t_0) = y_0, \quad t \in [t_0, t_1]

entirely in JAX on top of :func:`opifex.core.solver.integrate` (:func:`jax.lax.scan` +
per-step :func:`jax.checkpoint` discretize-then-optimize reverse-mode AD, avoiding
continuous-adjoint trajectory drift and requiring zero ``equinox`` or ``diffrax``
dependencies).

Three canonical L-stable (:math:`R(\infty) = 0`) implicit methods are provided directly
from their published tableaux and permissive-licensed reference implementations:

* :class:`Rodas4` — Hairer & Wanner's 6-stage, **4th-order** L-stable stiffly accurate
  Rosenbrock scheme (:math:`\gamma = 0.25`; Hairer & Wanner 1996, §IV.7), ported from
  ``SciML/OrdinaryDiffEq.jl`` (``14f664a``, MIT, ``rosenbrock_tableaus.jl:77-111`` &
  ``rosenbrock_perform_step.jl:431-504``). Uses Hairer & Wanner's transformed matrix
  :math:`W = (h\gamma)^{-1} I - J` so each step performs **one**
  :func:`jax.scipy.linalg.lu_factor`, **zero** :math:`J v` matrix-vector products, and
  **one** :func:`jax.linearize` primal evaluation at stage 1 for :math:`(f_0, \partial_t f, J)`.
* :class:`Ros34PW2` — Rang & Angermann's 4-stage, **3rd-order** L-stable Rosenbrock-Wanner
  W-method (:math:`\gamma \approx 0.4358665215`; Rang & Angermann 2005), ported from
  ``SciML/OrdinaryDiffEq.jl`` (``14f664a``, MIT, ``rosenbrock_tableaus.jl:814-836``).
  Satisfies W-method stiff order conditions on singular-perturbation and non-autonomous
  Prothero-Robinson systems.
* :class:`Kvaerno3` — Kværnø's 4-stage, **3rd-order** (stage order 2) A-L stable stiffly
  accurate ESDIRK3(2)4L[2]SA method with an explicit first stage (Kværnø 2004, *BIT*
  44(3), 489-502), ported from ``patrick-kidger/diffrax`` (``e950aa2``, Apache-2.0,
  ``diffrax/_solver/kvaerno3.py:15-42``) including its stage-derivative predictor tableau
  and dimensionless :math:`f`-space simplified-Newton iteration.

All three schemes preserve linear invariants :math:`c^\top y_n = c^\top y_0` (such as
chemical mass conservation) to floating-point rounding precision and support both uniform
and multi-decade geometric step schedules.

References:
    * Hairer, E., & Wanner, G. (1996). *Solving Ordinary Differential Equations II: Stiff
      and Differential-Algebraic Problems* (2nd ed.). Springer, §IV.6-IV.8.
    * Rang, J., & Angermann, L. (2005). "New Rosenbrock W-methods of order 3 for partial
      differential algebraic equations of index 1." *BIT Numer. Math.* 45(4), 761-787.
    * Kværnø, A. (2004). "Singly diagonally implicit Runge-Kutta methods with an explicit
      first stage." *BIT Numer. Math.* 44(3), 489-502.
    * Rackauckas, C., & Nie, Q. (2017). "DifferentialEquations.jl — A Performant and
      Feature-Rich Ecosystem for Solving Differential Equations in Julia"
      (``SciML/OrdinaryDiffEq.jl`` ``14f664a``, MIT).
    * Kidger, P. (2022). *On Neural Differential Equations* (``patrick-kidger/diffrax``
      ``e950aa2``, Apache-2.0).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg
import optax
from flax import struct
from jax.flatten_util import ravel_pytree

from opifex.core.solver.integrate import FINAL_STATE, integrate, Snapshots
from opifex.core.solver.problem import Method, Problem, solve
from opifex.core.solver.solution import Solution
from opifex.core.solver.status import Status


GridKind = Literal["uniform", "geometric"]


@dataclass(frozen=True, slots=True, kw_only=True)
class _RosenbrockTableau:
    """Immutable Hairer-Wanner transformed Rosenbrock tableau."""

    gamma: float
    a_rows: tuple[tuple[float, ...], ...]
    c_rows: tuple[tuple[float, ...], ...]
    nodes: tuple[float, ...]
    d_time: tuple[float, ...]
    b_weights: tuple[float, ...]


# Hairer & Wanner (1996) 6-stage, 4th-order L-stable stiffly accurate Rodas4 tableau
# (SciML/OrdinaryDiffEq.jl 14f664a, MIT, rosenbrock_tableaus.jl:77-111).
_RODAS4_TABLEAU: Final[_RosenbrockTableau] = _RosenbrockTableau(
    gamma=0.25,
    a_rows=(
        (),
        (1.544,),
        (0.9466785280815826, 0.2557011698983284),
        (3.314825187068521, 2.896124015972201, 0.9986419139977817),
        (1.221224509226641, 6.019134481288629, 12.53708332932087, -0.687886036105895),
        (1.221224509226641, 6.019134481288629, 12.53708332932087, -0.687886036105895, 1.0),
    ),
    c_rows=(
        (),
        (-5.6688,),
        (-2.430093356833875, -0.2063599157091915),
        (-0.1073529058151375, -9.594562251023355, -20.47028614809616),
        (7.496443313967647, -10.24680431464352, -33.99990352819905, 11.7089089320616),
        (
            8.083246795921522,
            -7.981132988064893,
            -31.52159432874371,
            16.31930543123136,
            -6.058818238834054,
        ),
    ),
    nodes=(0.0, 0.386, 0.21, 0.63, 1.0, 1.0),
    d_time=(0.25, -0.1043, 0.1035, -0.0362, 0.0, 0.0),
    b_weights=(
        1.221224509226641,
        6.019134481288629,
        12.53708332932087,
        -0.687886036105895,
        1.0,
        1.0,
    ),
)

# Rang & Angermann (2005) 4-stage, 3rd-order L-stable W-method Ros34PW2 tableau
# (SciML/OrdinaryDiffEq.jl 14f664a, MIT, rosenbrock_tableaus.jl:814-836).
_ROS34PW2_TABLEAU: Final[_RosenbrockTableau] = _RosenbrockTableau(
    gamma=0.435866521508459,
    a_rows=(
        (),
        (2.0,),
        (1.4192173174557647, -0.2592322116729697),
        (4.18476048231916, -0.28519201735549593, 2.294280360279042),
    ),
    c_rows=(
        (),
        (-4.588560720558084,),
        (-4.18476048231916, 0.28519201735549593),
        (-6.368179200128359, -6.795620944466837, 2.8700986043310563),
    ),
    nodes=(0.0, 0.871733043016918, 0.7315799577888524, 1.0),
    d_time=(
        0.435866521508459,
        -0.435866521508459,
        -0.4133333762338865,
        -5.551115123125783e-17,
    ),
    b_weights=(4.1847604823191595, -0.28519201735549565, 2.2942803602790414, 1.0),
)

# Kværnø (2004) 4-stage, 3rd-order (stage order 2) A-L stable ESDIRK3(2)4L[2]SA tableau
# with Diffrax stage-derivative predictors (patrick-kidger/diffrax e950aa2, Apache-2.0).
_KV_GAMMA: Final[float] = 0.43586652150845899942
_KV_A21: Final[float] = _KV_GAMMA
_KV_A31: Final[float] = (-4.0 * _KV_GAMMA * _KV_GAMMA + 6.0 * _KV_GAMMA - 1.0) / (4.0 * _KV_GAMMA)
_KV_A32: Final[float] = (-2.0 * _KV_GAMMA + 1.0) / (4.0 * _KV_GAMMA)
_KV_A41: Final[float] = (6.0 * _KV_GAMMA - 1.0) / (12.0 * _KV_GAMMA)
_KV_A42: Final[float] = -1.0 / ((24.0 * _KV_GAMMA - 12.0) * _KV_GAMMA)
_KV_A43: Final[float] = (-6.0 * _KV_GAMMA * _KV_GAMMA + 6.0 * _KV_GAMMA - 1.0) / (
    6.0 * _KV_GAMMA - 3.0
)
_KV_THETA: Final[float] = 1.0 / (2.0 * _KV_GAMMA)
_KV_LOWER: Final[tuple[tuple[float, ...], ...]] = (
    (_KV_A21,),
    (_KV_A31, _KV_A32),
    (_KV_A41, _KV_A42, _KV_A43),
)
_KV_PREDICTOR: Final[tuple[tuple[float, ...], ...]] = (
    (1.0,),
    (1.0 - _KV_THETA, _KV_THETA),
    (_KV_A31, _KV_A32, _KV_GAMMA),
)
_KV_IMPLICIT_NODES: Final[tuple[float, float, float]] = (2.0 * _KV_GAMMA, 1.0, 1.0)
_KV_WEIGHTS: Final[tuple[float, float, float, float]] = (
    _KV_A41,
    _KV_A42,
    _KV_A43,
    _KV_GAMMA,
)


class InitialValueProblem(Problem):
    r"""Ordinary differential equation initial-value problem on :math:`[t_0, t_1]`.

    State ``y0``, interval endpoints ``t0`` and ``t1``, and dynamic parameters ``args``
    are pytree leaves so ensembles of trajectories batch under :func:`jax.vmap` and
    differentiate with respect to ``y0``, ``t0``, ``t1``, and ``args``.

    Attributes:
        vector_field: Callable ``f(t, y, args) -> dy_dt`` returning a pytree matching ``y0``.
        y0: Initial state pytree at ``t0``.
        t0: Scalar initial time.
        t1: Scalar final time.
        args: Optional parameter pytree passed as the third argument of ``vector_field``.
    """

    vector_field: Callable[[jax.Array, Any, Any], Any] = struct.field(pytree_node=False)
    y0: Any
    t0: jax.Array
    t1: jax.Array
    args: Any = None


def _validate_stiff_schedule(
    num_steps: int,
    grid: str,
    geometric_ratio: float,
) -> None:
    """Validate static time-grid parameters before tracing."""
    if num_steps < 1:
        msg = f"num_steps must be at least 1, got {num_steps}"
        raise ValueError(msg)
    if grid not in ("uniform", "geometric"):
        msg = f"grid must be 'uniform' or 'geometric', got {grid!r}"
        raise ValueError(msg)
    if grid == "geometric" and geometric_ratio <= 0.0:
        msg = f"geometric_ratio must be strictly positive, got {geometric_ratio}"
        raise ValueError(msg)


def _grid_time_at_index(
    index: jax.Array,
    t0: jax.Array,
    t1: jax.Array,
    *,
    num_steps: int,
    grid: GridKind,
    geometric_ratio: float,
) -> jax.Array:
    r"""Evaluate grid node :math:`t_k` with cancellation-free endpoints."""
    dtype = t0.dtype
    span = t1 - t0
    if grid == "uniform":
        frac = index.astype(dtype) / num_steps
    else:
        ratio = jnp.asarray(geometric_ratio, dtype=dtype)
        log_r = jnp.log1p(ratio)
        frac = jnp.expm1((index.astype(dtype) / num_steps) * log_r) / ratio
    t_val = t0 + span * frac
    t_val = jnp.where(index == 0, t0, t_val)
    return jnp.where(index == num_steps, t1, t_val)


def _rosenbrock_single_step(
    rhs_vec: Callable[[jax.Array, jax.Array], jax.Array],
    t_curr: jax.Array,
    t_next: jax.Array,
    y_k: jax.Array,
    tableau: _RosenbrockTableau,
) -> jax.Array:
    """Advance one Hairer-Wanner transformed Rosenbrock step with zero ``J @ v`` products."""
    dt = t_next - t_curr
    dim = y_k.shape[0]
    f0, lin = jax.linearize(rhs_vec, t_curr, y_k)
    zero_t = jnp.zeros_like(t_curr)
    df_dt = lin(jnp.ones_like(t_curr), jnp.zeros_like(y_k))
    eye = jnp.eye(dim, dtype=y_k.dtype)
    jac = jax.vmap(lambda e: lin(zero_t, e), out_axes=-1)(eye)

    w_hw = eye / (dt * tableau.gamma) - jac
    lu_piv = jsp_linalg.lu_factor(w_hw)

    ks: list[jax.Array] = [jsp_linalg.lu_solve(lu_piv, f0 + (dt * tableau.d_time[0]) * df_dt)]
    for stage_idx in range(1, len(tableau.b_weights)):
        u_stage = y_k
        c_accum = jnp.zeros_like(y_k)
        for prev_idx in range(stage_idx):
            u_stage = u_stage + tableau.a_rows[stage_idx][prev_idx] * ks[prev_idx]
            c_accum = c_accum + (tableau.c_rows[stage_idx][prev_idx] / dt) * ks[prev_idx]
        node = tableau.nodes[stage_idx]
        t_stage = t_next if node == 1.0 else t_curr + node * dt
        f_stage = rhs_vec(t_stage, u_stage)
        rhs_stage = f_stage + (dt * tableau.d_time[stage_idx]) * df_dt + c_accum
        ks.append(jsp_linalg.lu_solve(lu_piv, rhs_stage))

    y_next = y_k
    for stage_idx, weight in enumerate(tableau.b_weights):
        y_next = y_next + weight * ks[stage_idx]
    return y_next


def _kvaerno3_single_step(
    rhs_vec: Callable[[jax.Array, jax.Array], jax.Array],
    t_curr: jax.Array,
    t_next: jax.Array,
    y_k: jax.Array,
    newton_iterations: int,
) -> jax.Array:
    """Advance one Kværnø (2004) ESDIRK3(2)4L[2]SA step in dimensionless ``f``-space."""
    dt = t_next - t_curr
    dim = y_k.shape[0]
    f1, lin_y = jax.linearize(lambda u: rhs_vec(t_curr, u), y_k)
    eye = jnp.eye(dim, dtype=y_k.dtype)
    jac = jax.vmap(lin_y, out_axes=-1)(eye)
    lu_piv = jsp_linalg.lu_factor(eye - (dt * _KV_GAMMA) * jac)
    dt_gamma = dt * _KV_GAMMA

    stage_fs: list[jax.Array] = [f1]
    for imp_idx in range(3):
        y_partial = y_k
        f_stage = jnp.zeros_like(y_k)
        for prev_idx in range(imp_idx + 1):
            y_partial = y_partial + (dt * _KV_LOWER[imp_idx][prev_idx]) * stage_fs[prev_idx]
            f_stage = f_stage + _KV_PREDICTOR[imp_idx][prev_idx] * stage_fs[prev_idx]
        node = _KV_IMPLICIT_NODES[imp_idx]
        t_stage = t_next if node == 1.0 else t_curr + node * dt
        for _ in range(newton_iterations):
            residual = rhs_vec(t_stage, y_partial + dt_gamma * f_stage) - f_stage
            f_stage = f_stage + jsp_linalg.lu_solve(lu_piv, residual)
        stage_fs.append(f_stage)

    y_next = y_k
    for stage_idx, weight in enumerate(_KV_WEIGHTS):
        y_next = y_next + (dt * weight) * stage_fs[stage_idx]
    return y_next


def _common_floating_dtype(*trees: Any) -> jnp.dtype:
    """Promote all inexact array leaves across ``trees`` with a ``float32`` floor."""
    dtypes = [jnp.float32]
    for tree in trees:
        for leaf in jax.tree_util.tree_leaves(tree):
            arr_dtype = getattr(leaf, "dtype", None)
            if arr_dtype is not None and jnp.issubdtype(arr_dtype, jnp.inexact):
                dtypes.append(arr_dtype)
    return jnp.result_type(*dtypes)


def _execute_stiff_integration(
    problem: InitialValueProblem,
    step_fn: Callable[
        [Callable[[jax.Array, jax.Array], jax.Array], jax.Array, jax.Array, jax.Array],
        jax.Array,
    ],
    *,
    num_steps: int,
    grid: GridKind,
    geometric_ratio: float,
    save: Snapshots,
    checkpoint: bool,
) -> Solution:
    """Shared driver bridging flat vector-field steps to :func:`opifex.core.solver.integrate`."""
    dtype = _common_floating_dtype(problem.y0, problem.t0, problem.t1, problem.args)
    y0_typed = jax.tree.map(
        lambda x: (
            jnp.asarray(x, dtype=dtype) if jnp.issubdtype(jnp.asarray(x).dtype, jnp.inexact) else x
        ),
        problem.y0,
    )
    y0_vec, unravel = ravel_pytree(y0_typed)
    t0 = jnp.asarray(problem.t0, dtype=dtype)
    t1 = jnp.asarray(problem.t1, dtype=dtype)

    def rhs_vec(tau: jax.Array, u_vec: jax.Array) -> jax.Array:
        """Evaluate flattened ODE vector field at ``(tau, u_vec)``."""
        dy_tree = problem.vector_field(tau, unravel(u_vec), problem.args)
        dy_flat, _ = ravel_pytree(dy_tree)
        return dy_flat.astype(dtype)

    def advance(state: dict[str, jax.Array], index: jax.Array) -> dict[str, jax.Array]:
        """Advance ``(y, t)`` across global grid step ``index``."""
        t_curr = _grid_time_at_index(
            index,
            t0,
            t1,
            num_steps=num_steps,
            grid=grid,
            geometric_ratio=geometric_ratio,
        )
        t_next = _grid_time_at_index(
            index + 1,
            t0,
            t1,
            num_steps=num_steps,
            grid=grid,
            geometric_ratio=geometric_ratio,
        )
        y_next = step_fn(rhs_vec, t_curr, t_next, state["y"])
        return {"y": y_next, "t": t_next}

    saved = integrate(
        advance,
        {"y": y0_vec, "t": t0},
        num_steps=num_steps,
        save=save,
        checkpoint=checkpoint,
    )
    saved_y_tree = jax.vmap(unravel)(saved["y"])
    finite_ok = jnp.all(jnp.isfinite(saved["y"])) & jnp.all(jnp.isfinite(saved["t"]))
    status = jnp.where(
        finite_ok,
        jnp.asarray(Status.SUCCESS, dtype=jnp.int32),
        jnp.asarray(Status.NONFINITE, dtype=jnp.int32),
    )
    return Solution(
        fields={"y": saved_y_tree, "t": saved["t"]},
        metrics={"final_norm": optax.safe_norm(saved["y"][-1], 0.0)},
        status=status,
        stats={"num_steps": jnp.asarray(num_steps, dtype=jnp.int32)},
    )


class _RosenbrockMethodBase(Method[InitialValueProblem]):
    """Shared base configuration and execution path for Rosenbrock-Wanner methods."""

    num_steps: int = struct.field(pytree_node=False, default=128)
    grid: GridKind = struct.field(pytree_node=False, default="uniform")
    geometric_ratio: float = struct.field(pytree_node=False, default=1e6)
    save: Snapshots = struct.field(pytree_node=False, default=FINAL_STATE)
    checkpoint: bool = struct.field(pytree_node=False, default=True)

    def _solve_with_tableau(
        self,
        problem: InitialValueProblem,
        tableau: _RosenbrockTableau,
    ) -> Solution:
        """Validate static grid parameters and integrate ``problem`` with ``tableau``."""
        _validate_stiff_schedule(self.num_steps, self.grid, self.geometric_ratio)

        def step_fn(
            rhs_vec: Callable[[jax.Array, jax.Array], jax.Array],
            t_curr: jax.Array,
            t_next: jax.Array,
            y_k: jax.Array,
        ) -> jax.Array:
            """Advance one Rosenbrock step using ``tableau``."""
            return _rosenbrock_single_step(rhs_vec, t_curr, t_next, y_k, tableau)

        return _execute_stiff_integration(
            problem,
            step_fn,
            num_steps=self.num_steps,
            grid=self.grid,
            geometric_ratio=self.geometric_ratio,
            save=self.save,
            checkpoint=self.checkpoint,
        )


class Rodas4(_RosenbrockMethodBase):
    r"""Hairer & Wanner (1996) 6-stage, 4th-order L-stable stiffly accurate Rosenbrock solver.

    Attributes:
        num_steps: Static number of time steps across :math:`[t_0, t_1]`.
        grid: Step spacing schedule, ``"uniform"`` or ``"geometric"``.
        geometric_ratio: Ratio parameter :math:`R > 0` for ``grid="geometric"``.
        save: Trajectory snapshot schedule passed to :func:`~opifex.core.solver.integrate`.
        checkpoint: Rematerialize each step in reverse-mode AD.
    """

    def solve(self, problem: InitialValueProblem) -> Solution:
        """Integrate ``problem`` with the 6-stage, 4th-order ``Rodas4`` method.

        Args:
            problem: Stiff initial-value problem instance.

        Returns:
            Solution with ``fields={"y": ..., "t": ...}``, ``metrics={"final_norm": ...}``,
            traced ``status``, and ``stats={"num_steps": ...}``.
        """
        return self._solve_with_tableau(problem, _RODAS4_TABLEAU)


class Ros34PW2(_RosenbrockMethodBase):
    r"""Rang & Angermann (2005) 4-stage, 3rd-order L-stable Rosenbrock-Wanner W-method.

    Attributes:
        num_steps: Static number of time steps across :math:`[t_0, t_1]`.
        grid: Step spacing schedule, ``"uniform"`` or ``"geometric"``.
        geometric_ratio: Ratio parameter :math:`R > 0` for ``grid="geometric"``.
        save: Trajectory snapshot schedule passed to :func:`~opifex.core.solver.integrate`.
        checkpoint: Rematerialize each step in reverse-mode AD.
    """

    def solve(self, problem: InitialValueProblem) -> Solution:
        """Integrate ``problem`` with the 4-stage, 3rd-order ``Ros34PW2`` W-method.

        Args:
            problem: Stiff initial-value problem instance.

        Returns:
            Solution with ``fields={"y": ..., "t": ...}``, ``metrics={"final_norm": ...}``,
            traced ``status``, and ``stats={"num_steps": ...}``.
        """
        return self._solve_with_tableau(problem, _ROS34PW2_TABLEAU)


class Kvaerno3(Method[InitialValueProblem]):
    r"""Kværnø (2004) 4-stage, 3rd-order A-L stable stiffly accurate ESDIRK solver.

    Attributes:
        num_steps: Static number of time steps across :math:`[t_0, t_1]`.
        newton_iterations: Static number of dimensionless ``f``-space chord-Newton
            iterations per implicit stage reusing the step's LU factorization.
        grid: Step spacing schedule, ``"uniform"`` or ``"geometric"``.
        geometric_ratio: Ratio parameter :math:`R > 0` for ``grid="geometric"``.
        save: Trajectory snapshot schedule passed to :func:`~opifex.core.solver.integrate`.
        checkpoint: Rematerialize each step in reverse-mode AD.
    """

    num_steps: int = struct.field(pytree_node=False, default=128)
    newton_iterations: int = struct.field(pytree_node=False, default=4)
    grid: GridKind = struct.field(pytree_node=False, default="uniform")
    geometric_ratio: float = struct.field(pytree_node=False, default=1e6)
    save: Snapshots = struct.field(pytree_node=False, default=FINAL_STATE)
    checkpoint: bool = struct.field(pytree_node=False, default=True)

    def solve(self, problem: InitialValueProblem) -> Solution:
        """Integrate ``problem`` with Kværnø's 4-stage ESDIRK3(2)4L[2]SA method.

        Args:
            problem: Stiff initial-value problem instance.

        Returns:
            Solution with ``fields={"y": ..., "t": ...}``, ``metrics={"final_norm": ...}``,
            traced ``status``, and ``stats={"num_steps": ...}``.

        Raises:
            ValueError: If ``newton_iterations`` is less than 1.
        """
        if self.newton_iterations < 1:
            msg = f"newton_iterations must be at least 1, got {self.newton_iterations}"
            raise ValueError(msg)
        _validate_stiff_schedule(self.num_steps, self.grid, self.geometric_ratio)
        n_iters = self.newton_iterations

        def step_fn(
            rhs_vec: Callable[[jax.Array, jax.Array], jax.Array],
            t_curr: jax.Array,
            t_next: jax.Array,
            y_k: jax.Array,
        ) -> jax.Array:
            """Advance one Kvaerno3 step with ``n_iters`` chord-Newton iterations."""
            return _kvaerno3_single_step(rhs_vec, t_curr, t_next, y_k, n_iters)

        return _execute_stiff_integration(
            problem,
            step_fn,
            num_steps=self.num_steps,
            grid=self.grid,
            geometric_ratio=self.geometric_ratio,
            save=self.save,
            checkpoint=self.checkpoint,
        )


def solve_stiff_ode(
    vector_field: Callable[[jax.Array, Any, Any], Any],
    y0: Any,
    *,
    t0: jax.Array | float,
    t1: jax.Array | float,
    args: Any = None,
    method: Method[InitialValueProblem] = Rodas4(),
) -> Solution:
    """Functional convenience wrapper integrating a stiff initial-value problem.

    Args:
        vector_field: Right-hand side ``f(t, y, args) -> dy_dt``.
        y0: Initial state pytree at ``t0``.
        t0: Initial time.
        t1: Final time.
        args: Optional dynamic parameters passed to ``vector_field``.
        method: Stiff ODE method (:class:`Rodas4`, :class:`Ros34PW2`, or :class:`Kvaerno3`).

    Returns:
        Traced :class:`~opifex.core.solver.solution.Solution`.
    """
    problem = InitialValueProblem(
        vector_field=vector_field,
        y0=y0,
        t0=jnp.asarray(t0),
        t1=jnp.asarray(t1),
        args=args,
    )
    return solve(problem, method)


__all__ = [
    "GridKind",
    "InitialValueProblem",
    "Kvaerno3",
    "Rodas4",
    "Ros34PW2",
    "solve_stiff_ode",
]
