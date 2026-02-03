"""Types and data structures for resource management.

This module contains all enums and dataclasses used throughout the
resource management system for multi-cloud optimization, GPU management,
cost control, and sustainability tracking.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any


class CloudProvider(Enum):
    """Supported cloud providers for multi-cloud optimization."""

    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    IBM_CLOUD = "ibm_cloud"
    ORACLE_CLOUD = "oracle_cloud"
    ON_PREMISE = "on_premise"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """Types of computational resources."""

    GPU_A100 = "gpu_a100"
    GPU_V100 = "gpu_v100"
    GPU_H100 = "gpu_h100"
    CPU_INTEL = "cpu_intel"
    CPU_AMD = "cpu_amd"
    CPU_ARM = "cpu_arm"
    TPU_V4 = "tpu_v4"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class OptimizationObjective(Enum):
    """Optimization objectives for resource allocation."""

    MINIMIZE_COST = "minimize_cost"
    MINIMIZE_LATENCY = "minimize_latency"
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    BALANCE_COST_PERFORMANCE = "balance_cost_performance"
    MINIMIZE_CARBON = "minimize_carbon"
    MAXIMIZE_AVAILABILITY = "maximize_availability"


@dataclass
class ResourcePool:
    """Resource pool configuration and status."""

    pool_id: str
    provider: CloudProvider
    region: str
    resource_type: ResourceType
    total_capacity: int
    available_capacity: int
    reserved_capacity: int
    cost_per_hour_usd: float
    performance_score: float
    carbon_efficiency: float  # gCO2/compute unit
    availability_sla: float  # 0-1 scale
    current_utilization: float = 0.0
    maintenance_window: str = "02:00-04:00 UTC"


@dataclass
class ResourceAllocation:
    """Resource allocation request and result."""

    allocation_id: str
    requested_resources: dict[ResourceType, int]
    allocated_resources: dict[str, ResourcePool]  # pool_id -> ResourcePool
    start_time: float
    end_time: float | None
    cost_estimate_usd: float
    performance_estimate: float
    carbon_footprint_kg: float
    allocation_strategy: str


@dataclass
class CostOptimization:
    """Cost optimization analysis and recommendations."""

    current_cost_usd_per_hour: float
    optimized_cost_usd_per_hour: float
    potential_savings_percentage: float
    recommendations: list[str]
    alternative_configurations: list[dict[str, Any]]
    cost_breakdown_by_provider: dict[CloudProvider, float]
    roi_analysis: dict[str, float]


@dataclass
class SustainabilityMetrics:
    """Sustainability and carbon footprint metrics."""

    total_carbon_footprint_kg: float
    carbon_per_compute_unit: float
    renewable_energy_percentage: float
    carbon_offset_cost_usd: float
    sustainability_score: float  # 0-1 scale
    green_computing_recommendations: list[str]
