# Optimization Examples

## Overview

This document provides practical examples of using the Opifex optimization module for various scientific computing applications. The examples demonstrate meta-optimization, production deployment, control systems, and integration with other Opifex components.

## Meta-Optimization Examples

### Example 1: Basic Learn-to-Optimize

Learn an optimizer for a family of quadratic functions:

```python
import jax
import jax.numpy as jnp
import flax.nnx as nnx
from opifex.optimization.meta_optimizers import LearnToOptimize, MetaOptimizerConfig

# Define a family of quadratic optimization problems
def create_quadratic_problem(key, dim=10):
    """Create a random quadratic optimization problem."""
    A = jax.random.normal(key, (dim, dim))
    A = A @ A.T + 0.1 * jnp.eye(dim)  # Ensure positive definite
    b = jax.random.normal(key, (dim,))
    c = jax.random.normal(key, ())

    def objective(x):
        return 0.5 * x.T @ A @ x + b.T @ x + c

    return objective, A, b, c

# Create training problems
key = jax.random.PRNGKey(42)
training_problems = []
for i in range(100):
    key, subkey = jax.random.split(key)
    obj_fn, A, b, c = create_quadratic_problem(subkey)
    training_problems.append(obj_fn)

# Configure L2O
config = MetaOptimizerConfig(
    meta_learning_rate=1e-3,
    num_unroll_steps=20,
    num_meta_epochs=100,
    adaptation_strategy="cosine_annealing"
)

# Initialize L2O optimizer
l2o = LearnToOptimize(config=config, rngs=nnx.Rngs(42))

# Meta-train the optimizer
print("Meta-training L2O optimizer...")
l2o.meta_train(training_problems, num_meta_epochs=100)

# Test on a new problem
key, test_key = jax.random.split(key)
test_objective, _, _, _ = create_quadratic_problem(test_key)

# Initialize parameters
initial_params = jax.random.normal(test_key, (10,))

# Optimize with learned optimizer
optimized_params = l2o.optimize(
    initial_params=initial_params,
    objective_fn=test_objective,
    num_steps=50
)

print(f"Initial loss: {test_objective(initial_params):.6f}")
print(f"Final loss: {test_objective(optimized_params):.6f}")
```

### Example 2: Physics-Informed Meta-Optimization

Meta-optimization for physics-informed neural networks:

```python
from opifex.optimization.meta_optimizers import MetaOptimizer
from opifex.core.physics.losses import PhysicsInformedLoss
from opifex.neural import MLP

# Define a simple PDE: u_t + u * u_x = 0 (Burgers' equation)
def burgers_pde_residual(u, x, t):
    """Compute PDE residual for Burgers' equation."""
    u_t = jax.grad(u, argnums=1)(x, t)
    u_x = jax.grad(u, argnums=0)(x, t)
    return u_t + u(x, t) * u_x

# Create physics-informed loss
physics_loss = PhysicsInformedLoss(
    pde_loss_weight=1.0,
    boundary_loss_weight=10.0,
    initial_loss_weight=10.0
)

# Configure physics-aware meta-optimizer
config = MetaOptimizerConfig(
    physics_aware=True,
    conservation_weighting=True,
    pde_constraint_strength=1.0,
    meta_learning_rate=1e-3
)

meta_optimizer = MetaOptimizer(config=config, rngs=nnx.Rngs(42))

# Create neural network
model = MLP(
    features=[50, 50, 50, 1],
    activation="tanh",
    rngs=nnx.Rngs(42)
)

# Training data
x_train = jnp.linspace(0, 1, 100)
t_train = jnp.linspace(0, 1, 100)
X, T = jnp.meshgrid(x_train, t_train)
training_points = jnp.stack([X.flatten(), T.flatten()], axis=1)

# Boundary and initial conditions
x_bc = jnp.array([0.0, 1.0])
t_bc = jnp.linspace(0, 1, 50)
boundary_points = jnp.stack([
    jnp.concatenate([x_bc, x_bc]),
    jnp.concatenate([t_bc, t_bc])
], axis=1)

x_ic = jnp.linspace(0, 1, 100)
t_ic = jnp.zeros_like(x_ic)
initial_points = jnp.stack([x_ic, t_ic], axis=1)
initial_values = jnp.sin(jnp.pi * x_ic)  # u(x, 0) = sin(πx)

# Optimize with physics-informed meta-optimizer
optimized_model = meta_optimizer.optimize_physics_informed(
    model=model,
    pde_residual_fn=burgers_pde_residual,
    training_points=training_points,
    boundary_points=boundary_points,
    initial_points=initial_points,
    initial_values=initial_values,
    num_steps=1000
)

print("Physics-informed optimization completed!")
```

### Example 3: Multi-Objective Optimization

Optimize a neural network for both accuracy and efficiency:

```python
from opifex.optimization.l2o import MultiObjectiveL2OEngine, MultiObjectiveConfig
from opifex.neural import CNN

# Define multiple objectives
def accuracy_loss(params, model, data):
    """Compute accuracy loss."""
    x, y = data
    predictions = model.apply(params, x)
    return jnp.mean((predictions - y) ** 2)

def efficiency_loss(params, model, data):
    """Compute efficiency loss (model complexity)."""
    param_count = sum(p.size for p in jax.tree.leaves(params))
    return param_count / 1e6  # Normalize by 1M parameters

def inference_time_loss(params, model, data):
    """Compute inference time loss."""
    x, _ = data

    # Time a forward pass
    start_time = time.time()
    _ = model.apply(params, x)
    end_time = time.time()

    return (end_time - start_time) * 1000  # Convert to milliseconds

# Configure multi-objective optimization
config = MultiObjectiveConfig(
    num_objectives=3,
    pareto_frontier_approximation=True,
    scalarization_method="weighted_sum",
    diversity_preservation=True
)

# Create multi-objective optimizer
mo_optimizer = MultiObjectiveL2OEngine(config=config, rngs=nnx.Rngs(42))

# Create model
model = CNN(
    features=[32, 64, 128],
    kernel_sizes=[3, 3, 3],
    strides=[1, 2, 2],
    num_classes=10,
    rngs=nnx.Rngs(42)
)

# Training data (example)
key = jax.random.PRNGKey(42)
x_train = jax.random.normal(key, (1000, 32, 32, 3))
y_train = jax.random.randint(key, (1000,), 0, 10)
data = (x_train, y_train)

# Define objectives
objectives = [
    lambda params: accuracy_loss(params, model, data),
    lambda params: efficiency_loss(params, model, data),
    lambda params: inference_time_loss(params, model, data)
]

# Optimize with multiple objectives
pareto_solutions = mo_optimizer.optimize(
    initial_params=model.parameters,
    objectives=objectives,
    num_solutions=50,
    num_steps=500
)

print(f"Found {len(pareto_solutions)} Pareto-optimal solutions")
for i, solution in enumerate(pareto_solutions[:5]):
    acc_loss = accuracy_loss(solution.params, model, data)
    eff_loss = efficiency_loss(solution.params, model, data)
    time_loss = inference_time_loss(solution.params, model, data)
    print(f"Solution {i}: Accuracy={acc_loss:.4f}, Efficiency={eff_loss:.4f}, Time={time_loss:.2f}ms")
```

## Production Optimization Examples

### Example 4: Model Deployment with Adaptive Optimization

Deploy a model with production optimization:

```python
from opifex.optimization.production import HybridPerformancePlatform, OptimizationStrategy
from opifex.optimization.adaptive_deployment import AdaptiveDeploymentSystem, DeploymentConfig
from opifex.neural import FNO

# Create a Fourier Neural Operator model
fno_model = FNO(
    modes=32,
    width=64,
    input_dim=2,
    output_dim=1,
    rngs=nnx.Rngs(42)
)

# Configure production optimization platform
platform = HybridPerformancePlatform(
    gpu_memory_optimization=True,
    adaptive_jit=True,
    performance_monitoring=True,
    workload_profiling=True
)

# Optimize model for production
optimized_model = platform.optimize_model(
    model=fno_model,
    optimization_strategy=OptimizationStrategy.BALANCED,
    target_latency_ms=50.0,
    target_throughput_rps=1000
)

# Configure adaptive deployment
deployment_config = DeploymentConfig(
    canary_percentage=10,
    rollback_threshold=0.95,
    monitoring_window_minutes=30,
    success_criteria=["latency", "accuracy", "error_rate"]
)

deployment_system = AdaptiveDeploymentSystem(
    config=deployment_config,
    ai_driven_strategies=True,
    automatic_rollback=True
)

# Deploy with monitoring
deployment_result = deployment_system.deploy(
    model=optimized_model,
    strategy="canary",
    target_environment="production"
)

print(f"Deployment status: {deployment_result.status}")
print(f"Performance improvement: {deployment_result.performance_improvement}%")
print(f"Memory usage reduction: {deployment_result.memory_reduction}%")
```

### Example 5: Global Resource Management

Optimize resource allocation across multiple cloud providers:

```python
from opifex.optimization.resource_management import (
    GlobalResourceManager,
    CloudProvider,
    OptimizationObjective
)

# Configure global resource manager
resource_manager = GlobalResourceManager(
    cloud_providers=[CloudProvider.AWS, CloudProvider.GCP, CloudProvider.AZURE],
    optimization_objective=OptimizationObjective.COST_PERFORMANCE,
    sustainability_tracking=True
)

# Define workload requirements
workload_requirements = {
    "compute_units": 2000,
    "memory_gb": 1000,
    "gpu_count": 16,
    "storage_tb": 50,
    "network_bandwidth_gbps": 10
}

# Define constraints
constraints = {
    "max_latency_ms": 100,
    "availability_requirement": 0.999,
    "budget_limit_usd": 20000,
    "carbon_footprint_limit_kg": 1000
}

# Optimize resource allocation
allocation = resource_manager.optimize_allocation(
    workload_requirements=workload_requirements,
    constraints=constraints,
    time_horizon_hours=24
)

print("Optimal Resource Allocation:")
for provider, resources in allocation.provider_allocation.items():
    print(f"{provider}: {resources}")

print(f"Total cost: ${allocation.total_cost:.2f}")
print(f"Carbon footprint: {allocation.carbon_footprint:.2f} kg CO2")
print(f"Expected latency: {allocation.expected_latency:.1f} ms")
```

### Example 6: Edge Network Optimization

Deploy models to edge locations with latency optimization:

```python
from opifex.optimization.edge_network import (
    IntelligentEdgeNetwork,
    EdgeRegion,
    LatencyOptimizer
)

# Configure edge network
edge_network = IntelligentEdgeNetwork(
    regions=[
        EdgeRegion.US_EAST,
        EdgeRegion.US_WEST,
        EdgeRegion.EU_WEST,
        EdgeRegion.ASIA_PACIFIC
    ],
    latency_target_ms=5.0,
    failover_enabled=True
)

# Define traffic pattern
traffic_pattern = {
    EdgeRegion.US_EAST: 0.4,    # 40% of traffic
    EdgeRegion.US_WEST: 0.2,    # 20% of traffic
    EdgeRegion.EU_WEST: 0.25,   # 25% of traffic
    EdgeRegion.ASIA_PACIFIC: 0.15  # 15% of traffic
}

# Latency requirements
latency_requirements = {
    "p50": 2.0,  # 50th percentile: 2ms
    "p95": 5.0,  # 95th percentile: 5ms
    "p99": 10.0  # 99th percentile: 10ms
}

# Deploy to edge
edge_deployment = edge_network.deploy_to_edge(
    model=optimized_model,
    traffic_pattern=traffic_pattern,
    latency_requirements=latency_requirements
)

print("Edge Deployment Results:")
for region, deployment in edge_deployment.regional_deployments.items():
    print(f"{region}: {deployment.status} (latency: {deployment.latency:.1f}ms)")

print(f"Global P95 latency: {edge_deployment.global_p95_latency:.1f}ms")
print(f"Failover regions configured: {len(edge_deployment.failover_regions)}")
```

## Control Systems Examples

### Example 7: System Identification

Learn system dynamics from data:

```python
from opifex.optimization.control import SystemIdentifier, SystemDynamicsModel
import matplotlib.pyplot as plt

# Generate synthetic system data (nonlinear pendulum)
def true_pendulum_dynamics(state, control, dt=0.01):
    """True pendulum dynamics: [theta, theta_dot]"""
    theta, theta_dot = state
    torque = control[0]

    # Pendulum parameters
    g, l, m, b = 9.81, 1.0, 1.0, 0.1

    # Dynamics
    theta_ddot = -(g/l) * jnp.sin(theta) - (b/(m*l**2)) * theta_dot + torque/(m*l**2)

    # Euler integration
    theta_new = theta + theta_dot * dt
    theta_dot_new = theta_dot + theta_ddot * dt

    return jnp.array([theta_new, theta_dot_new])

# Generate training data
key = jax.random.PRNGKey(42)
num_trajectories = 50
trajectory_length = 100
dt = 0.01

state_data = []
input_data = []

for _ in range(num_trajectories):
    # Random initial condition
    key, subkey = jax.random.split(key)
    state = jax.random.uniform(subkey, (2,), minval=-1.0, maxval=1.0)

    trajectory_states = [state]
    trajectory_inputs = []

    for t in range(trajectory_length):
        # Random control input
        key, subkey = jax.random.split(key)
        control = jax.random.uniform(subkey, (1,), minval=-2.0, maxval=2.0)

        # Apply dynamics
        state = true_pendulum_dynamics(state, control, dt)

        trajectory_states.append(state)
        trajectory_inputs.append(control)

    state_data.append(jnp.array(trajectory_states))
    input_data.append(jnp.array(trajectory_inputs))

# Concatenate all data
all_states = jnp.concatenate(state_data, axis=0)
all_inputs = jnp.concatenate(input_data, axis=0)

# Define neural dynamics model
dynamics_model = SystemDynamicsModel(
    state_dim=2,
    input_dim=1,
    hidden_dims=[64, 64, 32],
    activation="tanh",
    physics_informed=True
)

# Create system identifier
system_id = SystemIdentifier(
    model=dynamics_model,
    learning_rate=1e-3,
    regularization_strength=1e-4
)

# Train the model
print("Training system identification model...")
trained_model = system_id.fit(
    state_data=all_states[:-1],  # Current states
    next_state_data=all_states[1:],  # Next states
    input_data=all_inputs,
    num_epochs=1000
)

# Test the learned model
test_initial_state = jnp.array([0.5, 0.0])
test_control_sequence = jnp.sin(jnp.linspace(0, 4*jnp.pi, 200)).reshape(-1, 1)

# Simulate with true dynamics
true_trajectory = [test_initial_state]
state = test_initial_state
for control in test_control_sequence:
    state = true_pendulum_dynamics(state, control, dt)
    true_trajectory.append(state)

# Simulate with learned dynamics
learned_trajectory = [test_initial_state]
state = test_initial_state
for control in test_control_sequence:
    state = trained_model.predict_next_state(state, control)
    learned_trajectory.append(state)

true_trajectory = jnp.array(true_trajectory)
learned_trajectory = jnp.array(learned_trajectory)

# Plot comparison
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(true_trajectory[:, 0], label='True')
plt.plot(learned_trajectory[:, 0], '--', label='Learned')
plt.xlabel('Time Step')
plt.ylabel('Angle (rad)')
plt.legend()
plt.title('Angle Trajectory')

plt.subplot(1, 2, 2)
plt.plot(true_trajectory[:, 1], label='True')
plt.plot(learned_trajectory[:, 1], '--', label='Learned')
plt.xlabel('Time Step')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.title('Angular Velocity Trajectory')

plt.tight_layout()
plt.show()

# Compute prediction error
prediction_error = jnp.mean((true_trajectory - learned_trajectory) ** 2)
print(f"Mean squared prediction error: {prediction_error:.6f}")
```

### Example 8: Model Predictive Control

Implement MPC for trajectory tracking:

```python
from opifex.optimization.control import DifferentiableMPC, MPCConfig, MPCObjective

# Use the trained dynamics model from previous example
# Define MPC objective
objective = MPCObjective(
    state_cost_weight=1.0,
    input_cost_weight=0.1,
    terminal_cost_weight=10.0
)

# Configure MPC
mpc_config = MPCConfig(
    prediction_horizon=20,
    control_horizon=10,
    state_constraints={
        "lower": jnp.array([-jnp.pi, -5.0]),
        "upper": jnp.array([jnp.pi, 5.0])
    },
    input_constraints={
        "lower": jnp.array([-3.0]),
        "upper": jnp.array([3.0])
    }
)

# Create MPC controller
mpc_controller = DifferentiableMPC(
    system_model=trained_model,
    objective=objective,
    config=mpc_config
)

# Define reference trajectory (swing up and stabilize)
time_steps = 300
reference_trajectory = jnp.zeros((time_steps, 2))
# First half: swing up to inverted position
reference_trajectory = reference_trajectory.at[:150, 0].set(jnp.pi)
# Second half: stabilize at inverted position
reference_trajectory = reference_trajectory.at[150:, 0].set(jnp.pi)

# Simulate MPC control
current_state = jnp.array([0.0, 0.0])  # Start at bottom
state_trajectory = [current_state]
control_trajectory = []

print("Running MPC simulation...")
for t in range(time_steps - mpc_config.prediction_horizon):
    # Get reference for prediction horizon
    ref_horizon = reference_trajectory[t:t+mpc_config.prediction_horizon]

    # Solve MPC problem
    mpc_result = mpc_controller.solve(
        current_state=current_state,
        reference_trajectory=ref_horizon
    )

    # Apply first control action
    control_action = mpc_result.optimal_control[0]
    control_trajectory.append(control_action)

    # Simulate system (using true dynamics for realistic simulation)
    current_state = true_pendulum_dynamics(current_state, control_action, dt)
    state_trajectory.append(current_state)

    if t % 50 == 0:
        print(f"Step {t}: State = [{current_state[0]:.3f}, {current_state[1]:.3f}], "
              f"Control = {control_action[0]:.3f}")

state_trajectory = jnp.array(state_trajectory)
control_trajectory = jnp.array(control_trajectory)

# Plot MPC results
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(state_trajectory[:, 0], label='Actual')
plt.plot(reference_trajectory[:len(state_trajectory), 0], '--', label='Reference')
plt.xlabel('Time Step')
plt.ylabel('Angle (rad)')
plt.legend()
plt.title('Angle Tracking')

plt.subplot(1, 3, 2)
plt.plot(state_trajectory[:, 1])
plt.xlabel('Time Step')
plt.ylabel('Angular Velocity (rad/s)')
plt.title('Angular Velocity')

plt.subplot(1, 3, 3)
plt.plot(control_trajectory)
plt.xlabel('Time Step')
plt.ylabel('Control Torque (N⋅m)')
plt.title('Control Input')

plt.tight_layout()
plt.show()

# Compute tracking performance
tracking_error = jnp.mean((state_trajectory[:, 0] - reference_trajectory[:len(state_trajectory), 0]) ** 2)
print(f"Mean squared tracking error: {tracking_error:.6f}")
```

## Integration Examples

### Example 9: L2O with Neural Operators

Use L2O to optimize Fourier Neural Operator training:

```python
from opifex.neural import FNO
from opifex.optimization.l2o import L2OEngine, L2OEngineConfig
from opifex.data import PDEDataGenerator

# Generate PDE training data
data_generator = PDEDataGenerator(
    pde_type="heat_equation",
    domain_size=(64, 64),
    time_steps=100
)

train_data = data_generator.generate_dataset(num_samples=1000)
val_data = data_generator.generate_dataset(num_samples=200)

# Create FNO model
fno_model = FNO(
    modes=16,
    width=32,
    input_dim=1,
    output_dim=1,
    rngs=nnx.Rngs(42)
)

# Configure L2O for FNO optimization
l2o_config = L2OEngineConfig(
    meta_learning_rate=1e-3,
    num_unroll_steps=10,
    problem_encoding_dim=64,
    optimizer_network_hidden_dims=[128, 64, 32]
)

# Create L2O engine
l2o_engine = L2OEngine(config=l2o_config, rngs=nnx.Rngs(42))

# Define FNO training objective
def fno_objective(params):
    def loss_fn(batch):
        x, y = batch
        pred = fno_model.apply(params, x)
        return jnp.mean((pred - y) ** 2)

    # Compute loss over training data
    total_loss = 0.0
    for batch in train_data:
        total_loss += loss_fn(batch)
    return total_loss / len(train_data)

# Meta-train L2O on FNO problems
print("Meta-training L2O for FNO optimization...")
l2o_engine.meta_train(
    problem_family=[fno_objective],
    num_meta_epochs=50
)

# Use L2O to optimize FNO
optimized_fno_params = l2o_engine.optimize(
    initial_params=fno_model.parameters,
    objective_fn=fno_objective,
    num_steps=200
)

# Evaluate performance
initial_loss = fno_objective(fno_model.parameters)
final_loss = fno_objective(optimized_fno_params)

print(f"Initial FNO loss: {initial_loss:.6f}")
print(f"L2O optimized loss: {final_loss:.6f}")
print(f"Improvement: {(initial_loss - final_loss) / initial_loss * 100:.2f}%")
```

### Example 10: End-to-End Scientific Computing Pipeline

Complete pipeline with optimization, training, and deployment:

```python
from opifex.core import create_pde_problem
from opifex.neural import PINN
from opifex.core.training.trainer import Trainer
from opifex.optimization.meta_optimizers import MetaOptimizer
from opifex.optimization.production import HybridPerformancePlatform
from opifex.optimization.adaptive_deployment import AdaptiveDeploymentSystem

# 1. Define scientific problem (2D heat equation)
pde_problem = create_pde_problem(
    pde_type="heat_equation_2d",
    domain={"x": [0, 1], "y": [0, 1], "t": [0, 1]},
    boundary_conditions="dirichlet",
    initial_condition=lambda x, y: jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
)

# 2. Create physics-informed neural network
pinn_model = PINN(
    features=[50, 50, 50, 1],
    activation="tanh",
    rngs=nnx.Rngs(42)
)

# 3. Configure meta-optimizer for physics-informed training
meta_config = MetaOptimizerConfig(
    physics_aware=True,
    conservation_weighting=True,
    meta_learning_rate=1e-3
)

meta_optimizer = MetaOptimizer(config=meta_config, rngs=nnx.Rngs(42))

# 4. Train with meta-optimization
trainer = Trainer(
    model=pinn_model,
    config=training_config
)

trained_model, training_metrics = trainer.train(
    problem=pde_problem,
    meta_optimizer=meta_optimizer,
    num_epochs=1000
)

print(f"Training completed. Final loss: {training_metrics['final_loss']:.6f}")

# 5. Production optimization
platform = HybridPerformancePlatform(
    gpu_memory_optimization=True,
    adaptive_jit=True,
    performance_monitoring=True
)

production_model = platform.optimize_model(
    model=trained_model,
    optimization_strategy="aggressive",
    target_latency_ms=10.0
)

# 6. Deploy with adaptive deployment system
deployment_config = DeploymentConfig(
    canary_percentage=5,
    rollback_threshold=0.98,
    monitoring_window_minutes=15
)

deployment_system = AdaptiveDeploymentSystem(
    config=deployment_config,
    ai_driven_strategies=True
)

deployment_result = deployment_system.deploy(
    model=production_model,
    strategy="canary",
    target_environment="production"
)

print("End-to-End Pipeline Results:")
print(f"- Training loss reduction: {training_metrics['loss_reduction']:.2f}%")
print(f"- Production optimization speedup: {platform.speedup_factor:.1f}x")
print(f"- Deployment status: {deployment_result.status}")
print(f"- Production latency: {deployment_result.latency:.1f}ms")
```

## Performance Benchmarking

### Example 11: Optimization Performance Comparison

Compare different optimization approaches:

```python
from opifex.optimization.benchmarking import OptimizationBenchmark
import time

# Define benchmark problem (Rosenbrock function)
def rosenbrock(x):
    """N-dimensional Rosenbrock function."""
    return jnp.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)

# Benchmark configuration
benchmark = OptimizationBenchmark(
    problem_fn=rosenbrock,
    problem_dim=10,
    num_trials=50,
    max_iterations=1000
)

# Test different optimizers
optimizers = {
    "Adam": optax.adam(1e-3),
    "L2O": l2o_engine,
    "Meta-Optimizer": meta_optimizer
}

results = {}

for name, optimizer in optimizers.items():
    print(f"Benchmarking {name}...")

    trial_results = []
    for trial in range(benchmark.num_trials):
        # Random initialization
        key = jax.random.PRNGKey(trial)
        initial_params = jax.random.normal(key, (benchmark.problem_dim,))

        # Time optimization
        start_time = time.time()

        if name == "Adam":
            # Standard optimization
            opt_state = optimizer.init(initial_params)
            params = initial_params

            for _ in range(benchmark.max_iterations):
                grad = jax.grad(rosenbrock)(params)
                updates, opt_state = optimizer.update(grad, opt_state)
                params = optax.apply_updates(params, updates)

        else:
            # L2O or Meta-optimizer
            params = optimizer.optimize(
                initial_params=initial_params,
                objective_fn=rosenbrock,
                num_steps=benchmark.max_iterations
            )

        end_time = time.time()

        # Record results
        final_loss = rosenbrock(params)
        optimization_time = end_time - start_time

        trial_results.append({
            "final_loss": final_loss,
            "time": optimization_time,
            "converged": final_loss < 1e-6
        })

    # Aggregate results
    results[name] = {
        "mean_loss": jnp.mean([r["final_loss"] for r in trial_results]),
        "std_loss": jnp.std([r["final_loss"] for r in trial_results]),
        "mean_time": jnp.mean([r["time"] for r in trial_results]),
        "convergence_rate": jnp.mean([r["converged"] for r in trial_results])
    }

# Print benchmark results
print("\nBenchmark Results:")
print("-" * 60)
for name, result in results.items():
    print(f"{name:15} | Loss: {result['mean_loss']:.2e} ± {result['std_loss']:.2e} | "
          f"Time: {result['mean_time']:.3f}s | Convergence: {result['convergence_rate']:.1%}")
```

## Troubleshooting Examples

### Example 12: Debugging Optimization Issues

Common optimization problems and solutions:

```python
from opifex.optimization.debugging import OptimizationDebugger

# Create debugger
debugger = OptimizationDebugger(
    verbose=True,
    save_trajectory=True,
    check_gradients=True
)

# Debug optimization problem
def problematic_objective(x):
    """Objective with potential issues."""
    # Potential issues: NaN gradients, exploding values, etc.
    return jnp.sum(x**4) + jnp.exp(jnp.sum(x**2))

# Debug with L2O
debug_result = debugger.debug_optimization(
    optimizer=l2o_engine,
    objective_fn=problematic_objective,
    initial_params=jnp.ones(5),
    num_steps=100
)

print("Debug Report:")
print(f"Issues found: {debug_result.issues}")
print(f"Recommendations: {debug_result.recommendations}")

# Common fixes
if "gradient_explosion" in debug_result.issues:
    print("Applying gradient clipping...")
    # Apply gradient clipping or reduce learning rate

if "nan_gradients" in debug_result.issues:
    print("Checking for numerical instabilities...")
    # Add numerical stability checks

if "slow_convergence" in debug_result.issues:
    print("Trying adaptive learning rate...")
    # Use adaptive learning rate scheduling
```

## See Also

- [Optimization User Guide](../user-guide/optimization.md) - Comprehensive usage guide
- [Meta-Optimization Methods](../methods/meta-optimization.md) - Theoretical background
- [Production Optimization](../methods/production-optimization.md) - Enterprise features
- [Control Systems](../methods/control-systems.md) - Control theory applications
- [API Reference](../api/optimization.md) - Complete API documentation
