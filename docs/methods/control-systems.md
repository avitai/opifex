# Control Systems in Opifex

## Overview

The `opifex.optimization.control` module provides differentiable predictive control components for scientific machine learning. It is organized into two layers:

- **Model Predictive Control (MPC)** — a differentiable, JAX/NNX-based receding-horizon controller with neural or user-supplied dynamics, constraint projection, real-time optimization, and a safety-critical variant with control barrier functions, emergency control, and backup policies.
- **System identification** — neural networks that learn system dynamics from input/output data, with optional physics constraints, online adaptation, and joint control-policy learning.

All components are built on `flax.nnx`, so they compose with `jax.jit`, `jax.grad`/`nnx.grad`, and `jax.vmap`.

```python
from opifex.optimization.control import (
    # MPC
    MPCConfig,
    MPCObjective,
    MPCResult,
    DifferentiableMPC,
    RecedingHorizonController,
    SafetyCriticalMPC,
    ControlBarrier,
    ConstraintProjector,
    PredictiveModel,
    RealTimeOptimizer,
    # System identification
    SystemIdentifier,
    PhysicsConstrainedSystemID,
    PhysicsConstraint,
    OnlineSystemLearner,
    ControlIntegratedSystemID,
    SystemDynamicsModel,
    BenchmarkValidationResult,
)
```

## Theoretical Foundation

### System Identification

System identification learns a mathematical model of a dynamical system from input/output data. In opifex, a neural network learns the discrete-time one-step map:

$$x_{k+1} = f_{\theta}(x_k, u_k)$$

where $x_k$ is the system state, $u_k$ is the input/control vector, and $f_{\theta}$ is a neural network parameterized by $\theta$.

### Model Predictive Control

MPC is an optimization-based control strategy that solves a finite-horizon optimal control problem at each time step:

$$\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \ell(x_k, u_k) + \ell_f(x_N)$$

subject to:

- $x_{k+1} = f(x_k, u_k)$ (system dynamics)
- $x_k \in \mathcal{X}$ (state constraints)
- $u_k \in \mathcal{U}$ (input constraints)

Only the first control action of the optimized sequence is applied; the problem is re-solved at the next step (receding horizon).

## Model Predictive Control

### Configuration: `MPCConfig`

`MPCConfig` is a frozen dataclass that captures the horizon, dimensions, objective weights, and optimizer settings. The `objective_weights` dictionary uses the keys `"state"`, `"control"`, and (optionally) `"terminal"`.

```python
from opifex.optimization.control import MPCConfig

config = MPCConfig(
    horizon=15,
    control_dim=1,
    state_dim=2,
    objective_weights={"state": 1.0, "control": 0.1, "terminal": 10.0},
)
```

Available fields (with defaults): `horizon=10`, `control_dim=2`, `state_dim=4`, `prediction_steps=None` (defaults to `horizon`), `objective_weights=None` (defaults to `{"state": 1.0, "control": 0.1, "terminal": 10.0}`), `max_iterations=50`, `tolerance=1e-4`, `time_limit=0.01`, `learning_rate=0.01`.

### `DifferentiableMPC`

`DifferentiableMPC` solves the receding-horizon problem. `compute_control` is `@nnx.jit`-compiled and returns an `MPCResult` whose first control action is applied to the plant. You can supply your own dynamics with `set_dynamics(fn)` (`fn(state, control) -> state_derivative`, integrated with a simple Euler step) or let it build a default neural `PredictiveModel`.

```python
import jax.numpy as jnp
from opifex.optimization.control import MPCConfig, DifferentiableMPC

config = MPCConfig(
    horizon=15,
    control_dim=1,
    state_dim=2,
    objective_weights={"state": 1.0, "control": 0.1},
)
mpc = DifferentiableMPC(config=config)

# Damped oscillator: x_dot = A x + B u
A = jnp.array([[0.0, 1.0], [-1.0, -0.5]])
B = jnp.array([[0.0], [1.0]])
mpc.set_dynamics(lambda x, u: A @ x + B @ u)

state = jnp.array([1.0, 0.0])
reference = jnp.zeros((15, 2))           # horizon x state_dim
result = mpc.compute_control(state, reference)

print(result.control_action.shape)       # (1,) — first action only
print(result.predicted_trajectory.shape) # (15, 2)
print(float(result.objective_value), bool(result.converged))
```

`MPCResult` is a `NamedTuple` with fields `control_action`, `predicted_trajectory`, `objective_value`, `converged`, `iterations`, `computation_time`, `emergency_activated`, `backup_used`, and `timeout_occurred`.

Because `compute_control` is differentiable, you can take gradients of the objective with respect to the state or the controller parameters:

```python
from flax import nnx

def loss_fn(mpc, state, reference):
    return mpc.compute_control(state, reference).objective_value

grad_fn = nnx.grad(loss_fn, argnums=1)   # gradient w.r.t. the state
gradients = grad_fn(mpc, state, reference)
```

A batch of independent MPC problems can be solved at once with `compute_control_batch`, which returns a `BatchMPCResult` (`control_actions`, `predicted_trajectories`, `objective_values`).

### Objective: `MPCObjective`

`MPCObjective` wraps the weighted quadratic cost. It is callable as `objective(states, controls, reference)`:

```python
import jax.numpy as jnp
from opifex.optimization.control import MPCObjective

objective = MPCObjective({"state": 1.0, "control": 0.1, "terminal": 10.0})
cost = objective(
    states=jnp.ones((10, 4)),
    controls=jnp.ones((10, 2)),
    reference=jnp.zeros((10, 4)),
)
```

The objective is `state_cost + control_cost + terminal_cost`, where the terminal term is applied only to the last state when a `"terminal"` weight is present.

### Neural dynamics: `PredictiveModel`

When no custom dynamics are supplied, `DifferentiableMPC` constructs a `PredictiveModel`: an NNX network that predicts the next state via a residual connection (`x + Δx`). You can also build and pass one explicitly.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.optimization.control import MPCConfig, DifferentiableMPC, PredictiveModel

model = PredictiveModel(
    state_dim=4,
    control_dim=2,
    hidden_dims=[64, 32],
    prediction_horizon=10,
    rngs=nnx.Rngs(0),
)

# Single-step and multi-step rollouts
next_state = model.predict_step(jnp.zeros(4), jnp.zeros(2))      # (4,)
trajectory = model.predict_trajectory(jnp.zeros(4), jnp.ones((10, 2)))  # (11, 4)

mpc = DifferentiableMPC(
    config=MPCConfig(horizon=10, control_dim=2, state_dim=4),
    dynamics_model=model,
)
```

`PredictiveModel` also supports physics-informed conservation terms via `physics_informed=True` and `conservation_laws=["energy", "momentum"]`.

### Constraint projection: `ConstraintProjector`

`ConstraintProjector` enforces box bounds on states and controls (and optional learned safety projections). It is an NNX module that can be passed to `DifferentiableMPC` to clip control sequences during optimization.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.optimization.control import ConstraintProjector

projector = ConstraintProjector(
    state_dim=4,
    control_dim=2,
    state_bounds={"lower": [-2, -1, -jnp.pi, -5], "upper": [2, 1, jnp.pi, 5]},
    control_bounds={"lower": [-1, -1], "upper": [1, 1]},
    rngs=nnx.Rngs(0),
)

clipped = projector.project_control(jnp.array([2.0, -2.0]))   # -> [1., -1.]
projected_state = projector.project_state(jnp.array([3.0, 2.0, 0.0, 0.0]))
```

Custom nonlinear constraints can be added with `add_custom_constraint(fn)`, where `fn(state)` returns a scalar that should be `<= 0` when feasible. The projector applies a gradient step toward feasibility for any violated custom constraint.

### Real-time optimization: `RealTimeOptimizer`

`RealTimeOptimizer` is the gradient-descent solver `DifferentiableMPC` uses internally. The JIT-compatible `optimize` runs a fixed number of iterations; `optimize_with_time_limit` adds wall-clock enforcement (and is therefore not JIT-compatible).

```python
import jax.numpy as jnp
from opifex.optimization.control import RealTimeOptimizer

optimizer = RealTimeOptimizer(max_iterations=50, tolerance=1e-4, learning_rate=0.1)
result = optimizer.optimize(lambda x: jnp.sum((x - 1.0) ** 2), None, jnp.zeros(2))

print(result.solution, bool(result.converged))   # -> ~[1., 1.] True
```

`optimize` returns an `OptimizationResult` (`solution`, `converged`, `iterations`, `timeout_occurred`). It accepts a `warm_start_solution` argument to seed from a previous solve.

## Receding Horizon Control

`RecedingHorizonController` wraps an MPC controller and drives it through a closed-loop simulation. It pads short reference windows automatically and exposes `compute_control` and `simulate_tracking`.

```python
import jax.numpy as jnp
from opifex.optimization.control import RecedingHorizonController

controller = RecedingHorizonController(
    mpc_horizon=10,
    control_horizon=5,
    state_dim=4,
    control_dim=2,
    sampling_time=0.1,
)

# One control step
result = controller.compute_control(
    jnp.array([1.0, 0.0, 0.0, 0.0]),
    jnp.zeros((10, 4)),
)
print(result.control_action.shape)   # (2,)

# Track a reference trajectory end-to-end
t = jnp.linspace(0, 2 * jnp.pi, 50)
reference = jnp.column_stack([jnp.sin(t), jnp.cos(t), jnp.zeros_like(t), jnp.zeros_like(t)])
final_state = controller.simulate_tracking(reference[0], reference)
```

Passing `safety_critical=True` makes the controller use a `SafetyCriticalMPC` internally.

## Safety and Constraints

### `SafetyCriticalMPC` and `ControlBarrier`

`SafetyCriticalMPC` extends `DifferentiableMPC` with three safety layers: control barrier functions, an emergency controller (engaged when the state leaves a safe region), and a backup policy (engaged when the MPC problem is judged infeasible). Use `compute_safe_control` instead of `compute_control` to activate these checks.

```python
import jax.numpy as jnp
from opifex.optimization.control import SafetyCriticalMPC, ControlBarrier

safe_mpc = SafetyCriticalMPC(
    horizon=8,
    control_dim=2,
    state_dim=4,
    safety_barriers=True,
    emergency_control=True,
    backup_policy=True,
)

# A control barrier function returns a value that is >= 0 in the safe set.
barrier = ControlBarrier(constraint=lambda state: 1.0 - jnp.sum(state**2))
safe_mpc.add_barrier(barrier)

# A dangerous state triggers the emergency controller.
dangerous_state = jnp.array([1.5, 1.5, 0.0, 0.0])
result = safe_mpc.compute_safe_control(dangerous_state, jnp.zeros((8, 4)))
print(bool(result.emergency_activated))   # True

# Barriers can also be queried directly.
is_safe = barrier.is_safe_control(jnp.array([0.5, 0.5, 0.0, 0.0]), jnp.array([0.1, 0.1]))
print(bool(is_safe))                       # True
```

The returned `MPCResult` exposes `emergency_activated` and `backup_used` flags so the caller can tell which path produced the action.

## System Identification

### `SystemIdentifier`

`SystemIdentifier` is the base neural one-step predictor. It is called as `model(state, input_val)` and returns the predicted next state.

```python
import jax
import jax.numpy as jnp
from flax import nnx
from opifex.optimization.control import SystemIdentifier

system_id = SystemIdentifier(
    state_dim=4,
    input_dim=2,
    hidden_dim=64,
    num_layers=3,
    rngs=nnx.Rngs(0),
)

next_state = system_id(jnp.zeros(4), jnp.zeros(2))   # (4,)
```

Train it with standard NNX gradients on one-step prediction error:

```python
def loss_fn(model, states, inputs, targets):
    predictions = jax.vmap(model, in_axes=(0, 0))(states, inputs)
    return jnp.mean((predictions - targets) ** 2)

# states[:-1], inputs[:-1] -> states[1:]
loss, grads = nnx.value_and_grad(loss_fn)(
    system_id, states[:-1], inputs[:-1], states[1:]
)
```

`validate_on_benchmark(name, test_data)` evaluates one-step prediction error and returns a `BenchmarkValidationResult` (`benchmark_name`, `metrics`, `validation_passed`, `details`):

```python
result = system_id.validate_on_benchmark(
    "linear_system",
    {"states": states[:-1], "inputs": inputs[:-1], "targets": states[1:]},
)
print(result.metrics["prediction_error"])
```

### Parameterized dynamics: `SystemDynamicsModel`

`SystemDynamicsModel` provides an explicit linear (`x_{k+1} = A x_k + B u_k`, with learnable `A`/`B`) or nonlinear (MLP) state-transition model.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.optimization.control import SystemDynamicsModel

linear_model = SystemDynamicsModel(
    model_type="linear", state_dim=3, input_dim=2, rngs=nnx.Rngs(0)
)
nonlinear_model = SystemDynamicsModel(
    model_type="nonlinear", state_dim=2, input_dim=1, hidden_dims=[32, 32], rngs=nnx.Rngs(0)
)

next_state = linear_model(jnp.zeros(3), jnp.zeros(2))   # (3,)
```

### Physics-constrained identification: `PhysicsConstrainedSystemID`

`PhysicsConstrainedSystemID` extends `SystemIdentifier` with a learned energy function and a list of `PhysicsConstraint`s. Each constraint declares a `name`, a `constraint_type` (`"conservation"`, `"stability"`, or `"symmetry"`), a `tolerance`, and a `weight`.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.optimization.control import PhysicsConstrainedSystemID, PhysicsConstraint

energy_conservation = PhysicsConstraint(
    name="energy_conservation",
    constraint_type="conservation",
    tolerance=1e-3,
    weight=1.0,
)

model = PhysicsConstrainedSystemID(
    state_dim=2,
    input_dim=1,
    constraints=[energy_conservation],
    rngs=nnx.Rngs(0),
)

out = model.predict_with_constraints(jnp.array([1.0, -0.5]), jnp.array([0.1]))
print(out.keys())                       # dict_keys(['prediction', 'constraint_violations'])
energy = model.compute_energy(jnp.zeros(2))
```

`predict_with_constraints` returns the prediction together with per-constraint violation diagnostics (`constraint`, `violation`, `satisfied`).

### Online adaptation: `OnlineSystemLearner`

`OnlineSystemLearner` adapts the model in real time from streaming observations via `update_online(state, input_val, target)`.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.optimization.control import OnlineSystemLearner

learner = OnlineSystemLearner(
    state_dim=2,
    input_dim=1,
    learning_rate=1e-3,
    adaptation_rate=0.95,
    buffer_size=100,
    rngs=nnx.Rngs(0),
)

update = learner.update_online(
    jnp.array([1.0, -0.5]), jnp.array([0.1]), jnp.array([0.9, -0.4])
)
print(update.keys())                    # loss, adaptation_strength, effective_lr
print(learner.get_memory_info())        # buffer_size, current_size, total_updates
```

Set `adaptive_lr=True` to shrink the effective learning rate when recent prediction error spikes.

### Joint control learning: `ControlIntegratedSystemID`

`ControlIntegratedSystemID` learns the system model and a control policy together. It exposes `compute_control_action`, `joint_optimization`, `simulate_closed_loop`, and `validate_control_benchmark`.

```python
import jax.numpy as jnp
from flax import nnx
from opifex.optimization.control import ControlIntegratedSystemID

model = ControlIntegratedSystemID(
    state_dim=2, input_dim=1, control_dim=1, rngs=nnx.Rngs(0)
)

# Policy action toward a target state
action = model.compute_control_action(
    current_state=jnp.array([1.0, -0.5]), target_state=jnp.zeros(2)
)
print(action.shape)                     # (1,)

# Roll out the closed loop with the learned model + policy
sim = model.simulate_closed_loop(
    initial_state=jnp.array([2.0, -1.0]), target_state=jnp.zeros(2), steps=10
)
print(sim["states"].shape, sim["actions"].shape)   # (11, 2) (10, 1)

# Joint identification + control loss for training
losses = model.joint_optimization(jnp.zeros((20, 2)), jnp.zeros((20, 2)))
print(losses.keys())  # system_id_loss, control_loss, total_loss
```

## Best Practices

### System Identification

- **Data quality**: ensure rich, informative excitation in the training trajectories.
- **Physics constraints**: encode known conservation/stability laws with `PhysicsConstrainedSystemID` when available.
- **Validation**: always evaluate one-step (and multi-step) prediction error on held-out data via `validate_on_benchmark`.

### MPC Design

- **Horizon selection**: balance tracking performance against per-step compute (`compute_control` is `@nnx.jit`-compiled).
- **Objective weights**: tune the `"state"`, `"control"`, and `"terminal"` weights in `MPCConfig.objective_weights` to trade off tracking vs. control effort.
- **Warm starting**: `DifferentiableMPC` caches the previous control sequence; `RealTimeOptimizer.optimize` accepts an explicit `warm_start_solution`.

### Safety-Critical Applications

- **Barriers**: register `ControlBarrier`s and call `compute_safe_control` so emergency/backup paths can engage.
- **Graceful degradation**: inspect `result.emergency_activated` and `result.backup_used` to log and react to fallback activations.
- **Constraints**: clip controls and states with `ConstraintProjector`, including custom nonlinear constraints.

## See Also

- [Optimization User Guide](../user-guide/optimization.md) — general optimization concepts
- [Meta-Optimization](meta-optimization.md) — meta-learning for optimization
- [Neural Networks](../user-guide/neural-networks.md) — neural network integration
- [API Reference](../api/optimization.md) — complete API documentation
