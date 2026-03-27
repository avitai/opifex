"""Kubernetes orchestration module for Opifex deployment.

Provides Kubernetes deployment capabilities including:
- Manifest generation for deployments, services, and ingress
- Auto-scaling configuration (HPA and VPA)
- Resource management (namespaces, quotas, limits)
- Orchestration for production deployments
"""

from opifex.deployment.kubernetes.autoscaler import AutoScaler
from opifex.deployment.kubernetes.manifest_generator import ManifestGenerator
from opifex.deployment.kubernetes.orchestrator import KubernetesOrchestrator
from opifex.deployment.kubernetes.resource_manager import ResourceManager


__all__ = [
    "AutoScaler",
    "KubernetesOrchestrator",
    "ManifestGenerator",
    "ResourceManager",
]
