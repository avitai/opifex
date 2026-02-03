"""
Kubernetes orchestration module for Opifex deployment.

This module provides comprehensive Kubernetes deployment capabilities including:
- Manifest generation for deployments, services, and ingress
- Auto-scaling configuration (HPA and VPA)
- Resource management (namespaces, quotas, limits)
- Complete orchestration for production deployments
"""

# Import modules that exist, handle missing modules gracefully
__all__ = []

try:
    from opifex.deployment.kubernetes.manifest_generator import ManifestGenerator

    __all__ += ["ManifestGenerator"]
except ImportError:
    pass

try:
    from opifex.deployment.kubernetes.autoscaler import AutoScaler

    __all__ += ["AutoScaler"]
except ImportError:
    pass

try:
    from opifex.deployment.kubernetes.resource_manager import ResourceManager

    __all__ += ["ResourceManager"]
except ImportError:
    pass

try:
    from opifex.deployment.kubernetes.orchestrator import KubernetesOrchestrator

    __all__ += ["KubernetesOrchestrator"]
except ImportError:
    pass
