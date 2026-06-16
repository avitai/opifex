# Production Optimization

## Overview

Production optimization in Opifex provides the **Hybrid Performance Platform**: a set of
components for taking a trained neural operator and preparing it for production
deployment. It covers adaptive JIT compilation, intelligent GPU memory management,
AI-powered performance monitoring and prediction, adaptive deployment with predictive
scaling, edge-network routing, global resource management, and physics-aware scientific
validation.

Every public class below lives in a single module, `opifex.optimization.production`,
which re-exports the building blocks from the supporting submodules. You can therefore
import the entire production surface from one place:

```python
from opifex.optimization.production import (
    AIAnomalyDetector,
    AdaptiveDeploymentSystem,
    AdaptiveJAXOptimizer,
    GlobalResourceManager,
    HybridPerformancePlatform,
    IntelligentEdgeNetwork,
    IntelligentGPUMemoryManager,
    OptimizationStrategy,
    OptimizedModel,
    PerformanceMetrics,
    PerformanceMonitor,
    PerformancePredictor,
    PhysicsDomain,
    PredictiveScaler,
    ScientificComputingIntegrator,
    WorkloadProfile,
)
```

The examples assume a simple Flax NNX model with a `linear` attribute. This is the same
fixture used by the production test suite:

```python
import jax.numpy as jnp
from flax import nnx


class SimpleModel(nnx.Module):
    """Minimal neural operator stand-in for the optimization examples."""

    def __init__(self, features: int = 64, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.linear = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


model = SimpleModel(features=64, rngs=nnx.Rngs(0))
```

## Core Components

### 1. Workload Profiles and Optimization Strategies

Optimization in Opifex is driven by a `WorkloadProfile`: an immutable description of the
production workload. The optimizer inspects the profile to pick one of four
`OptimizationStrategy` values.

```python
from opifex.optimization.production import OptimizationStrategy, WorkloadProfile

workload = WorkloadProfile(
    batch_size=32,
    sequence_length=128,
    memory_footprint=2.0,          # GB
    compute_intensity=8.0,         # FLOPS / byte
    latency_requirement=10.0,      # milliseconds
    throughput_requirement=100.0,  # requests / second
    model_complexity="medium",     # "simple", "medium", or "complex"
)

# Available strategies
list(OptimizationStrategy)
# [AGGRESSIVE_FUSION, MEMORY_EFFICIENT, LATENCY_OPTIMIZED, BALANCED]
```

Strategy selection is rule-based:

| Condition on the workload                | Selected strategy                       |
| ---------------------------------------- | --------------------------------------- |
| `compute_intensity > 10.0`               | `OptimizationStrategy.AGGRESSIVE_FUSION` |
| `memory_footprint > 8.0`                 | `OptimizationStrategy.MEMORY_EFFICIENT`  |
| `latency_requirement < 5.0`              | `OptimizationStrategy.LATENCY_OPTIMIZED` |
| otherwise                                | `OptimizationStrategy.BALANCED`          |

### 2. Adaptive JIT Optimization

`AdaptiveJAXOptimizer` analyses a workload, applies the matching JIT strategy to the
model, and benchmarks the result against the un-optimized baseline. The measured speedup
is recorded as `improvement_factor` (baseline latency / optimized latency).

```python
from opifex.optimization.production import AdaptiveJAXOptimizer

optimizer = AdaptiveJAXOptimizer(
    performance_threshold=1.1,       # min speedup to cache the result
    memory_efficiency_target=0.85,
    cache_size=100,
)

# Inspect which strategy the workload selects
strategy = optimizer.analyze_workload_patterns(workload)  # OptimizationStrategy.BALANCED

# Run the full optimization
optimized = optimizer.optimize_neural_operator(model, workload)

print(optimized.optimization_type)                       # OptimizationStrategy.BALANCED
print(optimized.performance_metrics.improvement_factor)  # measured speedup ratio

# The optimized model is a drop-in callable
output = optimized.model(jnp.ones((workload.batch_size, 64)))
print(output.shape)  # (32, 64)
```

`optimize_neural_operator` returns an `OptimizedModel` container with the optimized
`model`, its `optimization_type`, a `performance_metrics` (`PerformanceMetrics`) record,
and an `optimization_metadata` dictionary holding the baseline latency, workload profile,
timestamp, and JAX backend.

You can also apply a single strategy directly:

```python
fused_model = optimizer.apply_aggressive_kernel_fusion(model)
memory_model = optimizer.apply_memory_optimization(model)       # adds jax.checkpoint
latency_model = optimizer.apply_latency_optimization(model)     # warms up JIT
balanced_model = optimizer.apply_balanced_optimization(model)
```

### 3. Performance Metrics

`PerformanceMetrics` holds the measured performance of an optimized model. It reports only
directly measured quantities; GPU utilization and energy efficiency are intentionally
absent because they require device/power telemetry (e.g. NVML) that is not a dependency of
the framework.

```python
from opifex.optimization.production import PerformanceMetrics

metrics = PerformanceMetrics(
    latency_ms=5.2,
    throughput_rps=192.3,
    memory_usage_gb=1.8,
    improvement_factor=1.35,
)
```

Throughput is derived from the measured latency (`throughput = 1000 / latency`), so the
two fields are always consistent.

### 4. Intelligent GPU Memory Management

`IntelligentGPUMemoryManager` plans memory pools and estimates per-model memory usage so
that multiple models can be co-located efficiently.

```python
from opifex.optimization.production import IntelligentGPUMemoryManager

memory_manager = IntelligentGPUMemoryManager(
    fragmentation_threshold=0.15,
    gc_trigger_threshold=0.85,
    # pool_sizes defaults to small/medium/large/xlarge (MB ranges)
)

# Pick the right pool for an allocation size (in MB)
memory_manager.select_memory_pool(128)   # "medium"
memory_manager.select_memory_pool(2048)  # "large"

# Estimate the memory a model needs at a given batch size (MB)
estimate_mb = memory_manager.estimate_model_memory_usage(model, batch_size=32)

# Plan allocations for several concurrent models
allocation_plan = memory_manager.optimize_multi_model_allocation(
    [(model, 16)]  # list of (model, batch_size) tuples
)
# Keys: model_allocations, shared_regions, total_memory_mb, efficiency_score
```

Custom pools can be supplied as `{pool_name: (min_size_mb, max_size_mb)}`:

```python
manager = IntelligentGPUMemoryManager(
    fragmentation_threshold=0.2,
    gc_trigger_threshold=0.9,
    pool_sizes={"tiny": (1, 16), "big": (16, 1024)},
)
```

### 5. Performance Monitoring and Prediction

The monitoring stack combines three classes:

- `AIAnomalyDetector` — an autoencoder that flags anomalous metric vectors.
- `PerformancePredictor` — a small network that forecasts latency, throughput, and memory.
- `PerformanceMonitor` — a real-time monitor that collects metrics and drives the two
  models above.

```python
from flax import nnx
from opifex.optimization.production import (
    AIAnomalyDetector,
    PerformanceMonitor,
    PerformancePredictor,
)

rngs = nnx.Rngs(0)

anomaly_detector = AIAnomalyDetector(rngs=rngs)
performance_predictor = PerformancePredictor(rngs=rngs)

monitor = PerformanceMonitor(
    anomaly_detector=anomaly_detector,
    performance_predictor=performance_predictor,
    collection_interval=1.0,  # seconds
)
```

`PerformanceMonitor` exposes async methods (`start_monitoring`, `stop_monitoring`,
`collect_current_metrics`, `predict_future_performance`) and keeps a `metrics_history`
list. Collecting a single snapshot:

```python
import asyncio

metrics = asyncio.run(monitor.collect_current_metrics())
print(metrics.latency_ms, metrics.throughput_rps)
```

The anomaly detector returns a boolean mask and per-sample reconstruction error:

```python
import jax.numpy as jnp

is_anomaly, reconstruction_error = anomaly_detector.detect_anomalies(jnp.ones((1, 16)))
```

### 6. Predictive Scaling

`PredictiveScaler` wraps a `PerformanceMonitor` and turns its forecasts into scale-up /
scale-down / maintain decisions, bounded by replica limits.

```python
from opifex.optimization.production import PredictiveScaler

scaler = PredictiveScaler(
    performance_monitor=monitor,
    scale_up_threshold=1.2,
    scale_down_threshold=0.8,
    min_replicas=1,
    max_replicas=10,
)

print(scaler.current_replicas)  # 1

# evaluate_scaling_decision() is async and needs >= 10 metrics in history;
# it returns a dict with "action", "target_replicas", "reason", "confidence".
```

### 7. Adaptive Deployment

`AdaptiveDeploymentSystem` orchestrates AI-driven deployments (canary, blue-green,
rolling) with automatic rollback. It is composed from four collaborators, all of which
share a single `DeploymentAI`.

```python
from flax import nnx
from opifex.optimization.adaptive_deployment import (
    AdaptiveDeploymentSystem,
    CanaryController,
    DeploymentAI,
    RollbackEngine,
    TrafficShaper,
)

rngs = nnx.Rngs(0)
deployment_ai = DeploymentAI(rngs=rngs)

deployment_system = AdaptiveDeploymentSystem(
    deployment_ai=deployment_ai,
    canary_controller=CanaryController(deployment_ai=deployment_ai),
    traffic_shaper=TrafficShaper(deployment_ai=deployment_ai),
    rollback_engine=RollbackEngine(deployment_ai=deployment_ai),
)

stats = deployment_system.get_system_statistics()
# Keys include: total_deployments, active_deployments, successful_deployments,
# rolled_back_deployments, success_rate, rollback_rate, ...
```

`deploy_model(deployment_id, config, system_features)` (async) drives a deployment using
a `DeploymentConfig` and a system-state feature vector; `get_deployment_status` and
`get_system_statistics` report progress.

#### Deployment Strategies

The available strategies are defined by `DeploymentStrategy` in
`opifex.optimization.adaptive_deployment`:

1. **Canary** (`DeploymentStrategy.CANARY`): gradual traffic rollout with health checks.
2. **Blue-Green** (`DeploymentStrategy.BLUE_GREEN`): zero-downtime swap with instant rollback.
3. **Rolling** (`DeploymentStrategy.ROLLING`): sequential instance updates.
4. **A/B Test** (`DeploymentStrategy.A_B_TEST`), **Shadow**, **Feature Flag**: additional
   traffic-management modes.

### 8. Intelligent Edge Network

`IntelligentEdgeNetwork` routes inference requests to the lowest-latency edge region with
caching and regional failover. It is built from an `EdgeGateway`, a `LatencyOptimizer`, an
`EdgeCache`, and a `RegionalFailover`.

```python
from flax import nnx
from opifex.optimization.edge_network import (
    EdgeCache,
    EdgeGateway,
    EdgeRegion,
    IntelligentEdgeNetwork,
    LatencyOptimizer,
    RegionalFailover,
)

gateway = EdgeGateway(primary_regions=[EdgeRegion.US_EAST, EdgeRegion.EU_WEST])
latency_optimizer = LatencyOptimizer(rngs=nnx.Rngs(0))
edge_cache = EdgeCache()
regional_failover = RegionalFailover(edge_gateway=gateway)

edge_network = IntelligentEdgeNetwork(
    edge_gateway=gateway,
    latency_optimizer=latency_optimizer,
    edge_cache=edge_cache,
    regional_failover=regional_failover,
    target_latency_ms=0.5,
)
```

`process_inference_request(...)` (async) is the main entry point; it consults the cache,
selects an optimal region via the gateway, and falls back through `RegionalFailover` on
health-check failure. Available regions are enumerated by `EdgeRegion` (e.g. `US_EAST`,
`US_WEST`, `EU_WEST`, `EU_CENTRAL`, `ASIA_PACIFIC`, `ASIA_NORTHEAST`).

### 9. Global Resource Management

`GlobalResourceManager` coordinates multi-cloud allocation, GPU pooling, cost control, and
sustainability tracking. It lives in `opifex.deployment.resource_management` and is
re-exported through `opifex.optimization.production`. It is assembled from four
sub-managers.

```python
from flax import nnx
from opifex.deployment.resource_management.global_manager import (
    CostController,
    GPUPoolManager,
    GlobalResourceManager,
    ResourceOrchestrator,
    SustainabilityTracker,
)

orchestrator = ResourceOrchestrator(rngs=nnx.Rngs(0))

resource_manager = GlobalResourceManager(
    resource_orchestrator=orchestrator,
    gpu_pool_manager=GPUPoolManager(resource_orchestrator=orchestrator),
    cost_controller=CostController(budget_limit_usd_per_day=10000.0),
    sustainability_tracker=SustainabilityTracker(carbon_reduction_target_percentage=30.0),
)
```

`allocate_resources_with_intelligence(resource_requirements, constraints,
sustainability_priority)` (async) returns an allocation result with GPU allocations, a
cost estimate, a carbon footprint, and a performance estimate.

### 10. Scientific Computing Integration

`ScientificComputingIntegrator` adds physics-aware validation to the optimization
pipeline. It checks conservation laws, numerical precision, and domain benchmarks against
reference data, and produces an overall scientific score.

```python
import jax.numpy as jnp
from opifex.optimization.production import (
    PhysicsDomain,
    ScientificComputingIntegrator,
)

integrator = ScientificComputingIntegrator(domain=PhysicsDomain.FLUID_DYNAMICS)

model_output = jnp.ones((8, 4))
reference_data = {"numerical_reference": model_output}

results = integrator.comprehensive_scientific_validation(model_output, reference_data)
print(results["overall_scientific_score"])

# Turn the validation results into actionable recommendations
recommendations = integrator.optimize_for_scientific_accuracy(model_output, results)
```

The supported domains are enumerated by `PhysicsDomain`: `QUANTUM_CHEMISTRY`,
`FLUID_DYNAMICS`, `MATERIALS_SCIENCE`, `PLASMA_PHYSICS`, `MOLECULAR_DYNAMICS`,
`SOLID_STATE`, and `GENERAL`.

## The Hybrid Performance Platform

`HybridPerformancePlatform` is the top-level orchestrator. It wires together the JIT
optimizer, memory manager, performance monitor, scientific integrator, and predictive
scaler, and exposes a single `optimize_for_production` entry point. It is a Flax NNX module
and therefore requires an `rngs` argument so it can build its internal AI components.

```python
from flax import nnx
from opifex.optimization.production import HybridPerformancePlatform

platform = HybridPerformancePlatform(rngs=nnx.Rngs(0))

optimized = platform.optimize_for_production(model, workload)

# optimization_metadata is enriched by every stage of the pipeline
print(optimized.optimization_metadata["production_ready"])        # True / False
print(optimized.optimization_metadata["platform_optimization"])   # True
print(optimized.optimization_metadata["memory_plan"])             # memory allocation plan

# The optimized model is callable
output = optimized.model(jnp.ones((workload.batch_size, 64)))
```

`optimize_for_production` runs a multi-stage pipeline: JIT optimization, memory planning,
performance-monitoring setup, scientific validation, predictive-scaling recommendations,
and a final production-readiness check (latency, throughput, and memory against the
workload requirements).

You can inject custom components and tune the latency target and physics domain:

```python
from opifex.optimization.production import (
    AdaptiveJAXOptimizer,
    HybridPerformancePlatform,
    IntelligentGPUMemoryManager,
    PhysicsDomain,
)

platform = HybridPerformancePlatform(
    jit_optimizer=AdaptiveJAXOptimizer(performance_threshold=1.5),
    memory_manager=IntelligentGPUMemoryManager(fragmentation_threshold=0.2),
    physics_domain=PhysicsDomain.FLUID_DYNAMICS,
    target_latency_ms=1.0,
    rngs=nnx.Rngs(0),
)
```

### Continuous Monitoring and Status

The platform can run continuous monitoring (async) and report a consolidated status:

```python
import asyncio

# Start / stop continuous monitoring (async)
# await platform.start_continuous_monitoring()
# await platform.stop_continuous_monitoring()

status = platform.get_comprehensive_status()
# Keys: platform_type, physics_domain, monitoring_active,
#       metrics_history_length, current_replicas, latest_metrics (if available)
```

## End-to-End Example

The following ties the pieces together: profile a workload, optimize the model for
production, and inspect the result.

```python
import jax.numpy as jnp
from flax import nnx

from opifex.optimization.production import (
    HybridPerformancePlatform,
    WorkloadProfile,
)


class SimpleModel(nnx.Module):
    def __init__(self, features: int = 64, *, rngs: nnx.Rngs) -> None:
        super().__init__()
        self.linear = nnx.Linear(features, features, rngs=rngs)

    def __call__(self, x):
        return self.linear(x)


model = SimpleModel(features=64, rngs=nnx.Rngs(0))

workload = WorkloadProfile(
    batch_size=32,
    sequence_length=128,
    memory_footprint=2.0,
    compute_intensity=8.0,
    latency_requirement=10.0,
    throughput_requirement=100.0,
    model_complexity="medium",
)

platform = HybridPerformancePlatform(rngs=nnx.Rngs(0))
optimized = platform.optimize_for_production(model, workload)

print("strategy:", optimized.optimization_type)
print("latency (ms):", optimized.performance_metrics.latency_ms)
print("throughput (rps):", optimized.performance_metrics.throughput_rps)
print("production ready:", optimized.optimization_metadata["production_ready"])

output = optimized.model(jnp.ones((workload.batch_size, 64)))
print("output shape:", output.shape)  # (32, 64)
```

## Best Practices

### 1. Deployment Strategy Selection

- **Low-Risk Changes**: prefer `DeploymentStrategy.ROLLING`.
- **High-Risk Changes**: use `DeploymentStrategy.CANARY` with the `RollbackEngine` enabled.
- **Critical Systems**: use `DeploymentStrategy.BLUE_GREEN` for instant rollback.
- **Comparisons**: use `DeploymentStrategy.A_B_TEST`.

### 2. Resource Optimization

- Let `IntelligentGPUMemoryManager.optimize_multi_model_allocation` plan co-located models
  rather than sizing pools by hand.
- Use `GlobalResourceManager` with a `SustainabilityTracker` when carbon footprint matters.
- Drive scaling from `PredictiveScaler` so replica changes follow forecasts, not raw load.

### 3. Monitoring and Alerting

- Keep a `PerformanceMonitor` running and feed its `metrics_history` to `PredictiveScaler`.
- Use `AIAnomalyDetector` for unsupervised anomaly flagging on metric vectors.
- Surface platform health with `HybridPerformancePlatform.get_comprehensive_status`.

### 4. Performance Optimization

- Profile first: build an accurate `WorkloadProfile` before optimizing.
- Measure impact: rely on the measured `improvement_factor`, not assumed speedups.
- For physics workloads, gate releases on the scientific score from
  `ScientificComputingIntegrator`.

## See Also

- [Optimization User Guide](../user-guide/optimization.md) - Practical usage guide
- [Meta-Optimization](meta-optimization.md) - Meta-learning approaches
- [API Reference](../api/optimization.md) - Complete API documentation
- [Deployment Guide](../deployment/local-development.md) - Deployment best practices
