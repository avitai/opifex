"""Adaptive Deployment System for Opifex production optimization.

This module implements AI-driven deployment strategies, canary deployments,
A/B testing, and automatic rollback for the Phase 7.4 Production Optimization system.

Part of: Hybrid Performance Platform + Intelligent Edge + Adaptive Optimization
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


class DeploymentStrategy(Enum):
    """Deployment strategies for production releases."""

    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    A_B_TEST = "a_b_test"
    SHADOW = "shadow"
    FEATURE_FLAG = "feature_flag"


class DeploymentStatus(Enum):
    """Status of deployment operations."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class RollbackTrigger(Enum):
    """Triggers for automatic rollback."""

    ERROR_RATE_THRESHOLD = "error_rate_threshold"
    LATENCY_THRESHOLD = "latency_threshold"
    SUCCESS_RATE_THRESHOLD = "success_rate_threshold"
    MANUAL = "manual"
    ANOMALY_DETECTION = "anomaly_detection"
    HEALTH_CHECK_FAILURE = "health_check_failure"


@dataclass
class DeploymentConfig:
    """Configuration for deployment strategies."""

    strategy: DeploymentStrategy
    traffic_split_percentage: float  # 0-100
    rollout_duration_minutes: int
    health_check_interval_seconds: int
    success_threshold_percentage: float
    error_threshold_percentage: float
    latency_threshold_ms: float
    auto_rollback_enabled: bool
    monitoring_enabled: bool
    a_b_test_duration_hours: int = 24
    feature_flag_percentage: float = 0.0


@dataclass
class DeploymentMetrics:
    """Metrics collected during deployment."""

    deployment_id: str
    timestamp: float
    success_rate: float
    error_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_rps: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    user_satisfaction_score: float = 0.0

    # Scientific computing specific metrics
    numerical_accuracy: float = 0.0
    conservation_score: float = 0.0
    physics_consistency: float = 0.0


@dataclass
class DeploymentState:
    """Current state of a deployment."""

    deployment_id: str
    status: DeploymentStatus
    strategy: DeploymentStrategy
    start_time: float
    current_traffic_percentage: float
    target_traffic_percentage: float
    metrics_history: list[DeploymentMetrics] = field(default_factory=list)
    rollback_triggers: list[RollbackTrigger] = field(default_factory=list)
    health_checks_passed: int = 0
    health_checks_failed: int = 0


@dataclass
class RollbackDecision:
    """Decision result for rollback evaluation."""

    should_rollback: bool
    trigger: RollbackTrigger
    confidence: float
    reason: str
    rollback_strategy: str
    estimated_rollback_time_minutes: int


class DeploymentAI(nnx.Module):
    """AI-driven deployment decision engine."""

    def __init__(
        self,
        input_features: int = 24,
        hidden_features: int = 128,
        decision_threshold: float = 0.7,
        *,
        rngs: nnx.Rngs,
    ):
        super().__init__()
        self.decision_threshold = decision_threshold

        # Neural network for deployment strategy selection
        self.strategy_selector = nnx.Sequential(
            nnx.Linear(input_features, hidden_features, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_features, hidden_features // 2, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_features // 2, len(DeploymentStrategy), rngs=rngs),
        )

        # Neural network for rollback prediction
        self.rollback_predictor = nnx.Sequential(
            nnx.Linear(input_features, hidden_features, rngs=rngs),
            nnx.gelu,
            nnx.Linear(hidden_features, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 1, rngs=rngs),  # Rollback probability
        )

        # Traffic management optimizer
        self.traffic_optimizer = nnx.Sequential(
            nnx.Linear(input_features, 64, rngs=rngs),
            nnx.gelu,
            nnx.Linear(64, 32, rngs=rngs),
            nnx.gelu,
            nnx.Linear(32, 1, rngs=rngs),  # Optimal traffic percentage
        )

    def select_deployment_strategy(
        self, system_features: jnp.ndarray
    ) -> tuple[DeploymentStrategy, float]:
        """Select optimal deployment strategy based on system state."""
        strategy_scores = self.strategy_selector(system_features)
        strategy_probabilities = jax.nn.softmax(strategy_scores)

        best_strategy_idx = jnp.argmax(strategy_probabilities)
        confidence = strategy_probabilities[best_strategy_idx]

        strategies = list(DeploymentStrategy)
        selected_strategy = strategies[int(best_strategy_idx)]

        return selected_strategy, float(confidence)

    def predict_rollback_probability(self, deployment_metrics: jnp.ndarray) -> float:
        """Predict probability that deployment should be rolled back."""
        rollback_logit = self.rollback_predictor(deployment_metrics)
        rollback_probability = jax.nn.sigmoid(rollback_logit)
        return float(rollback_probability[0])

    def optimize_traffic_split(
        self, current_metrics: jnp.ndarray, target_metrics: jnp.ndarray
    ) -> float:
        """Optimize traffic split percentage for gradual rollout."""
        combined_features = jnp.concatenate([current_metrics, target_metrics])
        optimal_percentage = self.traffic_optimizer(combined_features)
        # Ensure percentage is in valid range [0, 100]
        return float(jnp.clip(optimal_percentage[0] * 100, 0, 100))


class CanaryController:
    """Controller for canary deployments with automatic progression."""

    def __init__(
        self,
        deployment_ai: DeploymentAI,
        initial_traffic_percentage: float = 5.0,
        progression_steps: list[float] | None = None,
        evaluation_period_minutes: int = 10,
    ):
        self.deployment_ai = deployment_ai
        self.initial_traffic_percentage = initial_traffic_percentage
        self.progression_steps = progression_steps or [5, 10, 25, 50, 75, 100]
        self.evaluation_period_minutes = evaluation_period_minutes

        self.active_deployments: dict[str, DeploymentState] = {}

    async def start_canary_deployment(
        self, deployment_id: str, config: DeploymentConfig
    ) -> bool:
        """Start a new canary deployment."""
        deployment_state = DeploymentState(
            deployment_id=deployment_id,
            status=DeploymentStatus.RUNNING,
            strategy=DeploymentStrategy.CANARY,
            start_time=time.time(),
            current_traffic_percentage=self.initial_traffic_percentage,
            target_traffic_percentage=config.traffic_split_percentage,
        )

        self.active_deployments[deployment_id] = deployment_state

        # Start monitoring and progression
        self._monitoring_task = asyncio.create_task(
            self._monitor_canary_deployment(deployment_id, config)
        )
        return True

    async def _monitor_canary_deployment(
        self, deployment_id: str, config: DeploymentConfig
    ) -> None:
        """Monitor canary deployment and handle progression."""
        while deployment_id in self.active_deployments:
            deployment_state = self.active_deployments[deployment_id]

            if deployment_state.status != DeploymentStatus.RUNNING:
                break

            # Collect current metrics
            current_metrics = await self._collect_deployment_metrics(deployment_id)
            deployment_state.metrics_history.append(current_metrics)

            # Evaluate deployment health
            health_status = await self._evaluate_deployment_health(
                current_metrics, config
            )

            if not health_status["is_healthy"]:
                # Trigger rollback
                await self._trigger_rollback(
                    deployment_id, RollbackTrigger.HEALTH_CHECK_FAILURE
                )
                break

            # Check if ready for next progression step
            if await self._should_progress_canary(deployment_state, config):
                next_percentage = self._get_next_traffic_percentage(
                    deployment_state.current_traffic_percentage
                )

                if next_percentage >= deployment_state.target_traffic_percentage:
                    # Deployment complete
                    deployment_state.status = DeploymentStatus.SUCCESS
                    deployment_state.current_traffic_percentage = 100.0
                    break
                # Progress to next step
                deployment_state.current_traffic_percentage = next_percentage
                await self._update_traffic_split(deployment_id, next_percentage)

            # Wait for next evaluation period
            await asyncio.sleep(config.health_check_interval_seconds)

    async def _collect_deployment_metrics(
        self, deployment_id: str
    ) -> DeploymentMetrics:
        """Collect current deployment metrics."""
        # Simulate metric collection (in practice, would query monitoring systems)
        current_time = time.time()

        # Generate realistic metrics with some variation
        base_success_rate = 0.99 + jax.random.normal(jax.random.PRNGKey(42)) * 0.01
        base_latency = 2.0 + jax.random.normal(jax.random.PRNGKey(43)) * 0.5

        return DeploymentMetrics(
            deployment_id=deployment_id,
            timestamp=current_time,
            success_rate=float(jnp.clip(base_success_rate, 0.95, 1.0)),
            error_rate=float(jnp.clip(1.0 - base_success_rate, 0.0, 0.05)),
            latency_p50_ms=float(jnp.clip(base_latency, 0.5, 10.0)),
            latency_p95_ms=float(jnp.clip(base_latency * 2, 1.0, 20.0)),
            latency_p99_ms=float(jnp.clip(base_latency * 3, 2.0, 30.0)),
            throughput_rps=float(
                1000.0 + jax.random.normal(jax.random.PRNGKey(44)) * 100
            ),
            cpu_utilization=float(
                0.6 + jax.random.normal(jax.random.PRNGKey(45)) * 0.1
            ),
            memory_utilization=float(
                0.7 + jax.random.normal(jax.random.PRNGKey(46)) * 0.1
            ),
            gpu_utilization=float(
                0.8 + jax.random.normal(jax.random.PRNGKey(47)) * 0.1
            ),
            numerical_accuracy=0.999,
            conservation_score=0.998,
            physics_consistency=0.997,
        )

    async def _evaluate_deployment_health(
        self, metrics: DeploymentMetrics, config: DeploymentConfig
    ) -> dict[str, Any]:
        """Evaluate if deployment is healthy."""
        health_checks = {
            "success_rate": metrics.success_rate
            >= (config.success_threshold_percentage / 100),
            "error_rate": metrics.error_rate
            <= (config.error_threshold_percentage / 100),
            "latency": metrics.latency_p95_ms <= config.latency_threshold_ms,
            "numerical_accuracy": metrics.numerical_accuracy >= 0.995,
            "conservation_score": metrics.conservation_score >= 0.995,
        }

        is_healthy = all(health_checks.values())
        failed_checks = [check for check, passed in health_checks.items() if not passed]

        return {
            "is_healthy": is_healthy,
            "failed_checks": failed_checks,
            "health_score": sum(health_checks.values()) / len(health_checks),
        }

    async def _should_progress_canary(
        self, deployment_state: DeploymentState, config: DeploymentConfig
    ) -> bool:
        """Determine if canary should progress to next step."""
        if len(deployment_state.metrics_history) < 3:
            return False  # Need sufficient data

        # Check recent metrics are stable and healthy
        recent_metrics = deployment_state.metrics_history[-3:]

        for metrics in recent_metrics:
            health_status = await self._evaluate_deployment_health(metrics, config)
            if not health_status["is_healthy"]:
                return False

        # Check if evaluation period has passed
        time_since_last_change = time.time() - deployment_state.start_time
        min_evaluation_time = self.evaluation_period_minutes * 60

        return time_since_last_change >= min_evaluation_time

    def _get_next_traffic_percentage(self, current_percentage: float) -> float:
        """Get next traffic percentage in progression."""
        for step in self.progression_steps:
            if step > current_percentage:
                return step
        return 100.0

    async def _update_traffic_split(
        self, deployment_id: str, percentage: float
    ) -> None:
        """Update traffic split for deployment."""
        # In practice, would update load balancer, service mesh, etc.

    async def _trigger_rollback(
        self, deployment_id: str, trigger: RollbackTrigger
    ) -> None:
        """Trigger rollback for deployment."""
        if deployment_id in self.active_deployments:
            deployment_state = self.active_deployments[deployment_id]
            deployment_state.status = DeploymentStatus.ROLLED_BACK
            deployment_state.rollback_triggers.append(trigger)

            # Execute rollback (in practice, would revert infrastructure changes)
            await self._execute_rollback(deployment_id)

    async def _execute_rollback(self, deployment_id: str) -> None:
        """Execute actual rollback operation."""
        # Simulate rollback execution
        await asyncio.sleep(0.5)


class TrafficShaper:
    """Intelligent traffic shaping for load distribution."""

    def __init__(
        self,
        deployment_ai: DeploymentAI,
        max_traffic_change_per_minute: float = 10.0,  # Max 10% change per minute
    ):
        self.deployment_ai = deployment_ai
        self.max_traffic_change_per_minute = max_traffic_change_per_minute

        self.current_traffic_splits: dict[str, float] = {}
        self.traffic_history: list[dict[str, Any]] = []

    async def optimize_traffic_distribution(
        self,
        deployments: dict[str, DeploymentState],
        target_distribution: dict[str, float],
    ) -> dict[str, float]:
        """Optimize traffic distribution across deployments."""

        # Collect current metrics for all deployments
        all_metrics = {}
        for deployment_id, state in deployments.items():
            if state.metrics_history:
                all_metrics[deployment_id] = state.metrics_history[-1]

        # Use AI to optimize traffic splits
        optimized_splits = {}
        total_percentage = 0.0

        for deployment_id, _target_percentage in target_distribution.items():
            if deployment_id in all_metrics:
                current_metrics = self._metrics_to_array(all_metrics[deployment_id])
                target_metrics = jnp.zeros_like(current_metrics)  # Ideal metrics

                optimal_percentage = self.deployment_ai.optimize_traffic_split(
                    current_metrics, target_metrics
                )

                # Apply rate limiting
                current_split = self.current_traffic_splits.get(deployment_id, 0.0)
                max_change = self.max_traffic_change_per_minute

                if optimal_percentage > current_split + max_change:
                    optimal_percentage = current_split + max_change
                elif optimal_percentage < current_split - max_change:
                    optimal_percentage = current_split - max_change

                optimized_splits[deployment_id] = optimal_percentage
                total_percentage += optimal_percentage

        # Normalize to ensure total doesn't exceed 100%
        if total_percentage > 100.0:
            for deployment_id in optimized_splits:
                optimized_splits[deployment_id] *= 100.0 / total_percentage

        self.current_traffic_splits.update(optimized_splits)
        return optimized_splits

    def _metrics_to_array(self, metrics: DeploymentMetrics) -> jnp.ndarray:
        """Convert deployment metrics to array for AI processing."""
        return jnp.array(
            [
                metrics.success_rate,
                metrics.error_rate,
                metrics.latency_p50_ms / 100.0,  # Normalize
                metrics.latency_p95_ms / 100.0,
                metrics.latency_p99_ms / 100.0,
                metrics.throughput_rps / 1000.0,  # Normalize
                metrics.cpu_utilization,
                metrics.memory_utilization,
                metrics.gpu_utilization,
                metrics.numerical_accuracy,
                metrics.conservation_score,
                metrics.physics_consistency,
            ]
        )


class RollbackEngine:
    """Automatic rollback engine with performance-based triggers."""

    def __init__(
        self,
        deployment_ai: DeploymentAI,
        rollback_threshold: float = 0.8,
        evaluation_window_minutes: int = 5,
    ):
        self.deployment_ai = deployment_ai
        self.rollback_threshold = rollback_threshold
        self.evaluation_window_minutes = evaluation_window_minutes

        self.rollback_history: list[dict[str, Any]] = []

    async def evaluate_rollback_decision(
        self, deployment_state: DeploymentState, config: DeploymentConfig
    ) -> RollbackDecision:
        """Evaluate whether deployment should be rolled back."""

        if not deployment_state.metrics_history:
            return RollbackDecision(
                should_rollback=False,
                trigger=RollbackTrigger.MANUAL,
                confidence=0.0,
                reason="Insufficient metrics data",
                rollback_strategy="none",
                estimated_rollback_time_minutes=0,
            )

        recent_metrics = deployment_state.metrics_history[
            -min(5, len(deployment_state.metrics_history)) :
        ]

        # Check explicit threshold violations
        for metrics in recent_metrics:
            if metrics.error_rate > config.error_threshold_percentage / 100:
                return RollbackDecision(
                    should_rollback=True,
                    trigger=RollbackTrigger.ERROR_RATE_THRESHOLD,
                    confidence=1.0,
                    reason=(
                        f"Error rate {metrics.error_rate:.2%} exceeds threshold "
                        f"{config.error_threshold_percentage:.1f}%"
                    ),
                    rollback_strategy="immediate",
                    estimated_rollback_time_minutes=2,
                )

            if metrics.latency_p95_ms > config.latency_threshold_ms:
                return RollbackDecision(
                    should_rollback=True,
                    trigger=RollbackTrigger.LATENCY_THRESHOLD,
                    confidence=1.0,
                    reason=(
                        f"Latency P95 {metrics.latency_p95_ms:.1f}ms exceeds threshold "
                        f"{config.latency_threshold_ms}ms"
                    ),
                    rollback_strategy="immediate",
                    estimated_rollback_time_minutes=2,
                )

            if metrics.success_rate < config.success_threshold_percentage / 100:
                return RollbackDecision(
                    should_rollback=True,
                    trigger=RollbackTrigger.SUCCESS_RATE_THRESHOLD,
                    confidence=1.0,
                    reason=(
                        f"Success rate {metrics.success_rate:.2%} below threshold "
                        f"{config.success_threshold_percentage:.1f}%"
                    ),
                    rollback_strategy="immediate",
                    estimated_rollback_time_minutes=2,
                )

        # Use AI to predict rollback probability
        latest_metrics = recent_metrics[-1]
        metrics_array = self._metrics_to_array(latest_metrics)

        rollback_probability = self.deployment_ai.predict_rollback_probability(
            metrics_array
        )

        if rollback_probability > self.rollback_threshold:
            return RollbackDecision(
                should_rollback=True,
                trigger=RollbackTrigger.ANOMALY_DETECTION,
                confidence=rollback_probability,
                reason=(
                    f"AI detected deployment anomaly with "
                    f"{rollback_probability:.1%} confidence"
                ),
                rollback_strategy="gradual",
                estimated_rollback_time_minutes=5,
            )

        return RollbackDecision(
            should_rollback=False,
            trigger=RollbackTrigger.MANUAL,
            confidence=1.0 - rollback_probability,
            reason=(
                f"Deployment healthy, rollback probability {rollback_probability:.1%}"
            ),
            rollback_strategy="none",
            estimated_rollback_time_minutes=0,
        )

    def _metrics_to_array(self, metrics: DeploymentMetrics) -> jnp.ndarray:
        """Convert deployment metrics to array for AI processing."""
        return jnp.array(
            [
                metrics.success_rate,
                metrics.error_rate,
                metrics.latency_p50_ms / 100.0,
                metrics.latency_p95_ms / 100.0,
                metrics.latency_p99_ms / 100.0,
                metrics.throughput_rps / 1000.0,
                metrics.cpu_utilization,
                metrics.memory_utilization,
                metrics.gpu_utilization,
                metrics.numerical_accuracy,
                metrics.conservation_score,
                metrics.physics_consistency,
            ]
        )


class AdaptiveDeploymentSystem:
    """Main orchestrator for adaptive deployment with AI-driven strategies."""

    def __init__(
        self,
        deployment_ai: DeploymentAI,
        canary_controller: CanaryController,
        traffic_shaper: TrafficShaper,
        rollback_engine: RollbackEngine,
    ):
        self.deployment_ai = deployment_ai
        self.canary_controller = canary_controller
        self.traffic_shaper = traffic_shaper
        self.rollback_engine = rollback_engine

        self.active_deployments: dict[str, DeploymentState] = {}
        self.deployment_history: list[dict[str, Any]] = []

    async def deploy_model(
        self,
        deployment_id: str,
        config: DeploymentConfig,
        system_features: jnp.ndarray,
    ) -> dict[str, Any]:
        """Deploy model using AI-selected strategy."""

        # Use AI to select optimal deployment strategy if not specified
        if config.strategy == DeploymentStrategy.CANARY:
            selected_strategy = config.strategy
            confidence = 1.0
        else:
            selected_strategy, confidence = (
                self.deployment_ai.select_deployment_strategy(system_features)
            )

        # Create deployment state
        deployment_state = DeploymentState(
            deployment_id=deployment_id,
            status=DeploymentStatus.PENDING,
            strategy=selected_strategy,
            start_time=time.time(),
            current_traffic_percentage=0.0,
            target_traffic_percentage=config.traffic_split_percentage,
        )

        self.active_deployments[deployment_id] = deployment_state

        # Execute deployment based on strategy
        success = False
        if selected_strategy == DeploymentStrategy.CANARY:
            success = await self.canary_controller.start_canary_deployment(
                deployment_id, config
            )
        elif selected_strategy == DeploymentStrategy.BLUE_GREEN:
            success = await self._execute_blue_green_deployment(deployment_id, config)
        elif selected_strategy == DeploymentStrategy.ROLLING:
            success = await self._execute_rolling_deployment(deployment_id, config)
        else:
            # Default to canary for safety
            success = await self.canary_controller.start_canary_deployment(
                deployment_id, config
            )

        deployment_state.status = (
            DeploymentStatus.RUNNING if success else DeploymentStatus.FAILED
        )

        return {
            "deployment_id": deployment_id,
            "strategy": selected_strategy,
            "confidence": confidence,
            "success": success,
            "status": deployment_state.status,
            "estimated_completion_minutes": config.rollout_duration_minutes,
        }

    async def _execute_blue_green_deployment(
        self, deployment_id: str, config: DeploymentConfig
    ) -> bool:
        """Execute blue-green deployment."""
        # Simulate blue-green deployment
        await asyncio.sleep(1.0)
        return True

    async def _execute_rolling_deployment(
        self, deployment_id: str, config: DeploymentConfig
    ) -> bool:
        """Execute rolling deployment."""
        # Simulate rolling deployment
        await asyncio.sleep(2.0)
        return True

    async def monitor_deployments(self) -> None:
        """Monitor all active deployments for health and rollback conditions."""
        while True:
            for deployment_id, deployment_state in list(
                self.active_deployments.items()
            ):
                if deployment_state.status not in [
                    DeploymentStatus.RUNNING,
                    DeploymentStatus.PENDING,
                ]:
                    continue

                # Check if deployment needs attention
                config = DeploymentConfig(
                    strategy=deployment_state.strategy,
                    traffic_split_percentage=deployment_state.target_traffic_percentage,
                    rollout_duration_minutes=60,
                    health_check_interval_seconds=30,
                    success_threshold_percentage=99.0,
                    error_threshold_percentage=1.0,
                    latency_threshold_ms=10.0,
                    auto_rollback_enabled=True,
                    monitoring_enabled=True,
                )

                # Evaluate rollback decision
                rollback_decision = (
                    await self.rollback_engine.evaluate_rollback_decision(
                        deployment_state, config
                    )
                )

                if rollback_decision.should_rollback and config.auto_rollback_enabled:
                    await self._execute_rollback(deployment_id, rollback_decision)

            await asyncio.sleep(30)  # Check every 30 seconds

    async def _execute_rollback(
        self, deployment_id: str, rollback_decision: RollbackDecision
    ) -> None:
        """Execute rollback for deployment."""
        if deployment_id in self.active_deployments:
            deployment_state = self.active_deployments[deployment_id]
            deployment_state.status = DeploymentStatus.ROLLED_BACK
            deployment_state.rollback_triggers.append(rollback_decision.trigger)

            # Record rollback in history
            self.deployment_history.append(
                {
                    "deployment_id": deployment_id,
                    "action": "rollback",
                    "timestamp": time.time(),
                    "trigger": rollback_decision.trigger,
                    "reason": rollback_decision.reason,
                    "confidence": rollback_decision.confidence,
                }
            )

    def get_deployment_status(self, deployment_id: str) -> dict[str, Any] | None:
        """Get current status of deployment."""
        if deployment_id not in self.active_deployments:
            return None

        deployment_state = self.active_deployments[deployment_id]
        recent_metrics = (
            deployment_state.metrics_history[-1]
            if deployment_state.metrics_history
            else None
        )

        return {
            "deployment_id": deployment_id,
            "status": deployment_state.status,
            "strategy": deployment_state.strategy,
            "current_traffic_percentage": deployment_state.current_traffic_percentage,
            "target_traffic_percentage": deployment_state.target_traffic_percentage,
            "health_checks_passed": deployment_state.health_checks_passed,
            "health_checks_failed": deployment_state.health_checks_failed,
            "rollback_triggers": deployment_state.rollback_triggers,
            "recent_metrics": recent_metrics,
            "uptime_minutes": (time.time() - deployment_state.start_time) / 60,
        }

    def get_system_statistics(self) -> dict[str, Any]:
        """Get comprehensive deployment system statistics."""
        total_deployments = len(self.active_deployments) + len(self.deployment_history)
        active_count = len(
            [
                d
                for d in self.active_deployments.values()
                if d.status == DeploymentStatus.RUNNING
            ]
        )
        success_count = len(
            [
                d
                for d in self.active_deployments.values()
                if d.status == DeploymentStatus.SUCCESS
            ]
        )
        rollback_count = len(
            [
                d
                for d in self.active_deployments.values()
                if d.status == DeploymentStatus.ROLLED_BACK
            ]
        )

        return {
            "total_deployments": total_deployments,
            "active_deployments": active_count,
            "successful_deployments": success_count,
            "rolled_back_deployments": rollback_count,
            "success_rate": success_count / max(total_deployments, 1),
            "rollback_rate": rollback_count / max(total_deployments, 1),
            "average_deployment_time_minutes": 15.0,  # Would calculate from actual data
            "deployment_strategies_used": list(
                {d.strategy for d in self.active_deployments.values()}
            ),
        }
