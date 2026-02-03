"""
Model Predictive Control (MPC) Framework for Opifex.

Provides differentiable MPC implementation with neural network-based predictive models,
constraint handling and projection, real-time control policy optimization, and
safety-critical system support.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from flax import nnx


@dataclass
class MPCConfig:
    """Configuration for MPC controller."""

    horizon: int = 10
    control_dim: int = 2
    state_dim: int = 4
    prediction_steps: int | None = None
    objective_weights: dict[str, float] | None = None
    max_iterations: int = 50
    tolerance: float = 1e-4
    time_limit: float = 0.01  # Real-time constraint (10ms)
    learning_rate: float = 0.01

    def __post_init__(self):
        if self.prediction_steps is None:
            self.prediction_steps = self.horizon
        if self.objective_weights is None:
            self.objective_weights = {"state": 1.0, "control": 0.1, "terminal": 10.0}


class MPCResult(NamedTuple):
    """Result from MPC computation."""

    control_action: jnp.ndarray
    predicted_trajectory: jnp.ndarray
    objective_value: float | jax.Array
    converged: bool = True
    iterations: int = 0
    computation_time: float = 0.0
    emergency_activated: bool = False
    backup_used: bool = False
    timeout_occurred: bool = False


class OptimizationResult(NamedTuple):
    """Result from optimization."""

    solution: jnp.ndarray
    converged: bool
    iterations: int
    timeout_occurred: bool = False


class BatchMPCResult(NamedTuple):
    """Result from batch MPC computation."""

    control_actions: jnp.ndarray
    predicted_trajectories: jnp.ndarray
    objective_values: jnp.ndarray


class PredictiveModel(nnx.Module):
    """Neural network-based predictive model for system dynamics."""

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        hidden_dims: list[int] | None = None,
        prediction_horizon: int = 10,
        model_type: str = "neural",
        physics_informed: bool = False,
        conservation_laws: list[str] | None = None,
        *,
        rngs: nnx.Rngs,
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.hidden_dims = hidden_dims or [64, 32]
        self.prediction_horizon = prediction_horizon
        self.model_type = model_type
        self.physics_informed = physics_informed
        self.conservation_laws = conservation_laws or []

        # Build dynamics network
        layers = []
        input_dim = state_dim + control_dim

        for hidden_dim in self.hidden_dims:
            layers.append(nnx.Linear(input_dim, hidden_dim, rngs=rngs))
            layers.append(nnx.tanh)
            input_dim = hidden_dim

        layers.append(nnx.Linear(input_dim, state_dim, rngs=rngs))
        self.dynamics_network = nnx.Sequential(*layers)

        # Physics loss components
        if physics_informed:
            self.physics_loss = self._setup_physics_loss()

    def _setup_physics_loss(self):
        """Setup physics-informed loss components."""
        physics_loss = {}

        for law in self.conservation_laws:
            if law == "energy":
                physics_loss["energy"] = self._energy_conservation_loss
            elif law == "momentum":
                physics_loss["momentum"] = self._momentum_conservation_loss

        return physics_loss

    def _energy_conservation_loss(self, state_prev, state_next, control):
        """Energy conservation constraint."""
        # Simple kinetic energy conservation (example)
        energy_prev = 0.5 * jnp.sum(state_prev[:2] ** 2)  # First 2 states as velocities
        energy_next = 0.5 * jnp.sum(state_next[:2] ** 2)
        return (energy_next - energy_prev) ** 2

    def _momentum_conservation_loss(self, state_prev, state_next, control):
        """Momentum conservation constraint."""
        # Simple momentum conservation (example)
        momentum_prev = jnp.sum(state_prev[:2])
        momentum_next = jnp.sum(state_next[:2])
        return (momentum_next - momentum_prev) ** 2

    def predict_step(self, state: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
        """Predict next state given current state and control."""
        inputs = jnp.concatenate([state, control])
        delta_state = self.dynamics_network(inputs)
        return state + delta_state  # Residual connection

    def predict_trajectory(
        self, initial_state: jnp.ndarray, control_sequence: jnp.ndarray
    ) -> jnp.ndarray:
        """Predict state trajectory given control sequence."""

        def scan_fn(current_state, control):
            next_state = self.predict_step(current_state, control)
            return next_state, next_state

        _, trajectory = jax.lax.scan(scan_fn, initial_state, control_sequence)

        return jnp.concatenate([initial_state[None, :], trajectory], axis=0)


class ConstraintProjector(nnx.Module):
    """Neural network-based constraint projection."""

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        state_bounds: dict[str, list[float]] | None = None,
        control_bounds: dict[str, list[float]] | None = None,
        safety_constraints: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.state_bounds = state_bounds
        self.control_bounds = control_bounds
        self.safety_constraints = safety_constraints
        self.custom_constraints = []

        # Neural projection networks
        if safety_constraints:
            self.state_projector = nnx.Sequential(
                nnx.Linear(state_dim, 32, rngs=rngs),
                nnx.tanh,
                nnx.Linear(32, state_dim, rngs=rngs),
            )

            self.control_projector = nnx.Sequential(
                nnx.Linear(control_dim, 16, rngs=rngs),
                nnx.tanh,
                nnx.Linear(16, control_dim, rngs=rngs),
            )

    def add_custom_constraint(self, constraint_fn: Callable):
        """Add custom constraint function."""
        self.custom_constraints.append(constraint_fn)

    def project_state(self, state: jnp.ndarray) -> jnp.ndarray:
        """Project state to satisfy constraints."""
        projected_state = state

        # Box constraints
        if self.state_bounds:
            lower = jnp.array(self.state_bounds["lower"])
            upper = jnp.array(self.state_bounds["upper"])
            projected_state = jnp.clip(projected_state, lower, upper)

        # Neural projection for safety constraints
        if self.safety_constraints:
            projected_state = self.state_projector(projected_state)

        # Custom constraints
        for constraint_fn in self.custom_constraints:
            violation = constraint_fn(projected_state)
            if violation > 0:
                # Simple projection: move towards feasible region
                gradient = jax.grad(constraint_fn)(projected_state)
                projected_state = projected_state - 0.1 * violation * gradient

        return projected_state

    def project_control(self, control: jnp.ndarray) -> jnp.ndarray:
        """Project control to satisfy constraints."""
        projected_control = control

        # Box constraints
        if self.control_bounds:
            lower = jnp.array(self.control_bounds["lower"])
            upper = jnp.array(self.control_bounds["upper"])
            projected_control = jnp.clip(projected_control, lower, upper)

        # Neural projection for safety constraints
        if self.safety_constraints:
            projected_control = self.control_projector(projected_control)

        return projected_control


class ControlBarrier:
    """Control barrier function for safety."""

    def __init__(self, constraint: Callable, alpha: float = 1.0):
        self.constraint = constraint
        self.alpha = alpha

    def is_safe_control(self, state: jnp.ndarray, control: jnp.ndarray) -> bool:
        """Check if control is safe given current state."""
        # Simple barrier function check
        barrier_value = self.constraint(state)
        return barrier_value >= 0  # Positive means safe


class RealTimeOptimizer(nnx.Module):
    """Real-time optimizer for MPC problems."""

    def __init__(
        self,
        max_iterations: int = 50,
        tolerance: float = 1e-4,
        learning_rate: float = 0.01,
        warm_start: bool = True,
        time_limit: float = 0.01,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.learning_rate = learning_rate
        self.warm_start = warm_start
        self.time_limit = time_limit

    def optimize(
        self,
        objective: Callable,
        constraints: Callable | None,
        initial_guess: jnp.ndarray,
        warm_start_solution: jnp.ndarray | None = None,
    ) -> OptimizationResult:
        """Optimize objective subject to constraints.

        Note: This is not JIT-compatible due to time limits.

        Args:
            objective: Objective function to optimize.
            constraints: Constraints to enforce.
            initial_guess: Initial guess for the solution.
            warm_start_solution: Solution from previous iteration for warm start.

        Returns:
            OptimizationResult: Result of the optimization.
        """
        # Use warm start if available
        if self.warm_start and warm_start_solution is not None:
            x = warm_start_solution
        else:
            x = initial_guess

        # Pre-compute gradient function
        grad_fn = jax.grad(objective)

        if constraints is not None:
            constraint_grad_fn = jax.grad(lambda x: jnp.sum(constraints(x) ** 2))

        # Simple fixed-iteration optimization for JIT compatibility
        for _ in range(self.max_iterations):
            # Compute gradient
            grad = grad_fn(x)

            # Apply constraints if any
            if constraints is not None:
                constraint_grad = constraint_grad_fn(x)
                grad = grad + 0.1 * constraint_grad

            # Update solution
            x = x - self.learning_rate * grad

        # Compute final convergence check
        grad = jax.grad(objective)(x)
        if constraints is not None:
            constraint_grad = jax.grad(lambda x: jnp.sum(constraints(x) ** 2))(x)
            grad = grad + 0.1 * constraint_grad

        # Simple convergence check based on gradient norm
        converged = jnp.linalg.norm(grad) < self.tolerance

        return OptimizationResult(
            solution=x, converged=converged, iterations=self.max_iterations
        )

    def optimize_with_time_limit(
        self,
        objective: Callable,
        constraints: Callable | None,
        initial_guess: jnp.ndarray,
        warm_start_solution: jnp.ndarray | None = None,
    ) -> OptimizationResult:
        """
        Optimize with real-time constraints (not JIT-compatible due to time limits).

        This method includes time limit enforcement and therefore cannot be
        JIT-compiled.
        Use optimize() for JIT-compatible optimization without time limits.
        """

        start_time = time.time()

        # Use warm start if available
        if self.warm_start and warm_start_solution is not None:
            x = warm_start_solution
        else:
            x = initial_guess

        # Simple gradient descent optimization with time limit checking
        converged = False
        final_iteration = 0

        # Pre-compute gradient function
        grad_fn = jax.grad(objective)

        if constraints is not None:
            constraint_grad_fn = jax.grad(lambda x: jnp.sum(constraints(x) ** 2))

        for iteration in range(self.max_iterations):
            # Check time limit
            if time.time() - start_time > self.time_limit:
                return OptimizationResult(
                    solution=x,
                    converged=False,
                    iterations=iteration,
                    timeout_occurred=True,
                )

            # Compute gradient
            grad = grad_fn(x)

            # Apply constraints if any
            if constraints is not None:
                constraint_grad = constraint_grad_fn(x)
                grad = grad + 0.1 * constraint_grad

            # Update solution
            x_new = x - self.learning_rate * grad

            # Check convergence
            convergence_norm = jnp.linalg.norm(x_new - x)
            x = x_new
            final_iteration = iteration + 1

            # Check convergence
            if convergence_norm < self.tolerance:
                converged = True
                break

        return OptimizationResult(
            solution=x, converged=converged, iterations=final_iteration
        )


class MPCObjective:
    """MPC objective function."""

    def __init__(self, weights: dict[str, float]):
        self.weights = weights

    def __call__(
        self, states: jnp.ndarray, controls: jnp.ndarray, reference: jnp.ndarray
    ) -> jax.Array:
        """Compute MPC objective."""
        state_cost = self.weights["state"] * jnp.sum((states - reference) ** 2)
        control_cost = self.weights["control"] * jnp.sum(controls**2)
        terminal_cost = self.weights.get("terminal", 0.0) * jnp.sum(
            (states[-1] - reference[-1]) ** 2
        )

        return state_cost + control_cost + terminal_cost


class DifferentiableMPC(nnx.Module):
    """Differentiable Model Predictive Control implementation."""

    def __init__(
        self,
        config: MPCConfig,
        dynamics_model: PredictiveModel | None = None,
        constraint_projector: ConstraintProjector | None = None,
    ):
        self.config = config
        self.horizon = config.horizon
        self.control_dim = config.control_dim
        self.state_dim = config.state_dim

        # Create default predictive model if not provided
        if dynamics_model is None:
            key = jax.random.PRNGKey(42)
            self.predictive_model = PredictiveModel(
                state_dim=config.state_dim,
                control_dim=config.control_dim,
                rngs=nnx.Rngs(key),
            )
        else:
            self.predictive_model = dynamics_model

        # Create constraint projector if provided
        self.constraint_projector = constraint_projector

        # Create objective function
        weights = (
            config.objective_weights
            if config.objective_weights is not None
            else {"state": 1.0, "control": 0.1, "terminal": 10.0}
        )
        self.objective = MPCObjective(weights)

        # Create optimizer
        self.optimizer = RealTimeOptimizer(
            max_iterations=config.max_iterations,
            tolerance=config.tolerance,
            time_limit=config.time_limit,
            learning_rate=config.learning_rate,
        )

        # Warm start cache (flattened control sequence)
        self.warm_start_cache = nnx.Variable(jnp.zeros(self.horizon * self.control_dim))

        # Custom dynamics function
        self._custom_dynamics = None

    def set_dynamics(self, dynamics_fn: Callable):
        """Set custom dynamics function."""
        self._custom_dynamics = dynamics_fn

    def _predict_trajectory(
        self, initial_state: jnp.ndarray, control_sequence: jnp.ndarray
    ) -> jnp.ndarray:
        """Predict trajectory using either neural model or custom dynamics."""
        if self._custom_dynamics is not None:
            # Use custom dynamics
            trajectory = [initial_state]
            current_state = initial_state

            for control in control_sequence:
                next_state = current_state + 0.1 * self._custom_dynamics(
                    current_state, control
                )  # Euler integration
                trajectory.append(next_state)
                current_state = next_state

            return jnp.stack(trajectory)
        # Use neural predictive model
        return self.predictive_model.predict_trajectory(initial_state, control_sequence)

    def compute_objective(
        self, states: jnp.ndarray, controls: jnp.ndarray, reference: jnp.ndarray
    ) -> jax.Array:
        """Compute MPC objective function."""
        return self.objective(states, controls, reference)

    @nnx.jit
    def compute_control(
        self, current_state: jnp.ndarray, reference_trajectory: jnp.ndarray
    ) -> MPCResult:
        """Compute optimal control action."""
        start_time = time.time()

        # Initialize control sequence
        initial_controls = jnp.zeros((self.horizon, self.control_dim))

        # Define optimization objective
        def mpc_objective(control_sequence_flat):
            control_sequence = control_sequence_flat.reshape(
                (self.horizon, self.control_dim)
            )

            # Project controls if projector available
            if self.constraint_projector is not None:
                control_sequence = jax.vmap(self.constraint_projector.project_control)(
                    control_sequence
                )

            # Predict trajectory
            predicted_trajectory = self._predict_trajectory(
                current_state, control_sequence
            )
            states = predicted_trajectory[1:]  # Exclude initial state

            # Compute objective
            return self.compute_objective(
                states, control_sequence, reference_trajectory
            )

        # Optimize control sequence
        result = self.optimizer.optimize(
            mpc_objective,
            None,  # No explicit constraints (handled by projector)
            initial_controls.flatten(),
            warm_start_solution=self.warm_start_cache[...],
        )

        # Update warm start cache
        self.warm_start_cache[...] = result.solution

        # Extract optimal control
        optimal_controls = result.solution.reshape((self.horizon, self.control_dim))

        # Apply constraint projection
        if self.constraint_projector is not None:
            optimal_controls = jax.vmap(self.constraint_projector.project_control)(
                optimal_controls
            )

        # Get predicted trajectory
        predicted_trajectory = self._predict_trajectory(current_state, optimal_controls)

        # Return only the first control action (receding horizon)
        control_action = optimal_controls[0]

        computation_time = time.time() - start_time

        return MPCResult(
            control_action=control_action,
            predicted_trajectory=predicted_trajectory[1:],  # Exclude initial state
            objective_value=mpc_objective(result.solution),
            converged=result.converged,
            iterations=result.iterations,
            computation_time=computation_time,
            timeout_occurred=result.timeout_occurred,
        )

    def compute_control_batch(
        self, batch_states: jnp.ndarray, batch_references: jnp.ndarray
    ) -> BatchMPCResult:
        """Compute control for batch of states."""

        # Use functional API to avoid mutating self with batched state
        # (which would change warm_start_cache shape)
        graphdef, state = nnx.split(self)

        # Use jax.vmap instead of nnx.vmap to avoid state handling issues
        # since we don't need to propagate state updates in batch mode
        def batch_step(graphdef, state, s, r):
            model = nnx.merge(graphdef, state)
            return model.compute_control(s, r)

        batch_step_vmap = jax.vmap(batch_step, in_axes=(None, None, 0, 0))

        # Execute batch computation
        # This returns a batched MPCResult
        batched_result = batch_step_vmap(
            graphdef, state, batch_states, batch_references
        )

        return BatchMPCResult(
            control_actions=batched_result.control_action,
            predicted_trajectories=batched_result.predicted_trajectory,
            objective_values=jnp.asarray(batched_result.objective_value),
        )


class SafetyCriticalMPC(DifferentiableMPC):
    """Safety-critical MPC with emergency control and backup policies."""

    def __init__(
        self,
        horizon: int = 10,
        control_dim: int = 2,
        state_dim: int = 4,
        safety_barriers: bool = True,
        emergency_control: bool = True,
        backup_policy: bool = True,
        **kwargs,
    ):
        config = MPCConfig(
            horizon=horizon, control_dim=control_dim, state_dim=state_dim, **kwargs
        )
        super().__init__(config)

        self.safety_barriers = safety_barriers
        self.emergency_control = emergency_control
        self.backup_policy = backup_policy

        # Safety components
        self.control_barriers = []
        self.emergency_controller = self._create_emergency_controller()
        self.backup_policy_fn = self._create_backup_policy()

    def _create_emergency_controller(self):
        """Create emergency controller."""

        def emergency_control(state):
            # Simple emergency controller: drive to origin
            return -0.5 * state[: self.control_dim]

        return emergency_control

    def _create_backup_policy(self):
        """Create backup policy."""

        def backup_policy(state, reference):
            # Simple backup: proportional control towards reference
            target = reference[0] if reference.ndim > 1 else reference
            return 0.1 * (target[: self.control_dim] - state[: self.control_dim])

        return backup_policy

    def add_barrier(self, barrier: ControlBarrier):
        """Add control barrier function."""
        self.control_barriers.append(barrier)

    def _is_safe_state(self, state: jnp.ndarray) -> bool:
        """Check if state is in safe region."""
        # For dangerous state test, check if norm is too large
        return jnp.linalg.norm(state) < 2.0  # Simple safety threshold

    def _is_mpc_feasible(self, state: jnp.ndarray, reference: jnp.ndarray) -> bool:
        """Check if MPC problem is feasible."""
        # Simple heuristic: check if state is not too far from reference
        target = reference[0] if reference.ndim > 1 else reference
        distance = jnp.linalg.norm(state - target[: self.state_dim])
        return distance < 5.0  # Threshold for feasibility

    def compute_safe_control(
        self, current_state: jnp.ndarray, reference_trajectory: jnp.ndarray
    ) -> MPCResult:
        """Compute safe control action with emergency and backup policies."""
        emergency_activated = False
        backup_used = False

        # Check if emergency control is needed
        if self.emergency_control and not self._is_safe_state(current_state):
            emergency_activated = True
            control_action = self.emergency_controller(current_state)

            return MPCResult(
                control_action=control_action,
                predicted_trajectory=jnp.zeros((self.horizon, self.state_dim)),
                objective_value=jnp.inf,
                converged=False,
                emergency_activated=emergency_activated,
                backup_used=backup_used,
            )

        # Check if MPC is feasible
        if self.backup_policy and not self._is_mpc_feasible(
            current_state, reference_trajectory
        ):
            backup_used = True
            control_action = self.backup_policy_fn(current_state, reference_trajectory)

            return MPCResult(
                control_action=control_action,
                predicted_trajectory=jnp.zeros((self.horizon, self.state_dim)),
                objective_value=jnp.inf,
                converged=False,
                emergency_activated=emergency_activated,
                backup_used=backup_used,
            )

        # Normal MPC computation
        return super().compute_control(current_state, reference_trajectory)  # type: ignore[reportCallIssue]


class RecedingHorizonController(nnx.Module):
    """Receding horizon controller implementation."""

    def __init__(
        self,
        mpc_horizon: int = 10,
        control_horizon: int = 5,
        state_dim: int = 4,
        control_dim: int = 2,
        sampling_time: float = 0.1,
        safety_critical: bool = False,
    ):
        self.mpc_horizon = mpc_horizon
        self.control_horizon = control_horizon
        self.state_dim = state_dim
        self.control_dim = control_dim
        self.sampling_time = sampling_time

        # Create MPC controller
        config = MPCConfig(
            horizon=mpc_horizon, control_dim=control_dim, state_dim=state_dim
        )

        if safety_critical:
            self.mpc = SafetyCriticalMPC(
                horizon=mpc_horizon, control_dim=control_dim, state_dim=state_dim
            )
        else:
            self.mpc = DifferentiableMPC(config)

    def compute_control(
        self, current_state: jnp.ndarray, reference_trajectory: jnp.ndarray
    ) -> MPCResult:
        """Compute control using receding horizon."""
        # Pad reference if needed
        if reference_trajectory.shape[0] < self.mpc_horizon:
            last_ref = reference_trajectory[-1]
            padding_needed = self.mpc_horizon - reference_trajectory.shape[0]
            padding = jnp.tile(last_ref, (padding_needed, 1))
            reference_trajectory = jnp.vstack([reference_trajectory, padding])

        return self.mpc.compute_control(
            current_state, reference_trajectory[: self.mpc_horizon]
        )  # type: ignore[reportCallIssue]

    def simulate_tracking(
        self,
        initial_state: jnp.ndarray,
        reference_trajectory: jnp.ndarray,
        simulation_steps: int | None = None,
    ) -> jnp.ndarray:
        """Simulate reference tracking."""
        if simulation_steps is None:
            simulation_steps = reference_trajectory.shape[0] - 1

        current_state = initial_state

        for step in range(simulation_steps):
            # Get reference window
            ref_start = min(step, reference_trajectory.shape[0] - self.mpc_horizon)
            ref_end = min(ref_start + self.mpc_horizon, reference_trajectory.shape[0])
            ref_window = reference_trajectory[ref_start:ref_end]

            # Compute control
            result = self.compute_control(current_state, ref_window)

            # Simple integration - handle shape compatibility properly
            control_action = result.control_action

            # Ensure control has same dimension as state for integration
            if len(control_action) < self.state_dim:
                # Pad control with zeros for remaining state dimensions
                control_input = jnp.concatenate(
                    [control_action, jnp.zeros(self.state_dim - len(control_action))]
                )
            elif len(control_action) > self.state_dim:
                # Truncate control to state dimension
                control_input = control_action[: self.state_dim]
            else:
                control_input = control_action

            current_state = current_state + self.sampling_time * control_input

        return current_state
