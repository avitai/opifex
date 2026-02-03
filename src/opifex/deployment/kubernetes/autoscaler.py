"""
Kubernetes auto-scaling components for Opifex deployment.

This module provides Horizontal Pod Autoscaler (HPA) and Vertical Pod Autoscaler (VPA)
configurations for dynamic resource scaling based on performance metrics.
"""

from typing import Any


class AutoScaler:
    """
    Generate Kubernetes auto-scaling manifests for Opifex deployment infrastructure.

    Provides both horizontal scaling (HPA) for replica management and vertical
    scaling (VPA) for resource optimization based on workload demands.
    """

    def __init__(self, namespace: str, deployment_name: str):
        """
        Initialize auto-scaler generator.

        Args:
            namespace: Kubernetes namespace for auto-scaling resources
            deployment_name: Name of the deployment to scale
        """
        self.namespace = namespace
        self.deployment_name = deployment_name

    def generate_hpa(
        self,
        min_replicas: int = 2,
        max_replicas: int = 10,
        cpu_target_percentage: int = 70,
        memory_target_percentage: int | None = None,
        custom_metrics: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Generate Horizontal Pod Autoscaler manifest.

        Args:
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas
            cpu_target_percentage: Target CPU utilization percentage
            memory_target_percentage: Target memory utilization percentage
            custom_metrics: Additional custom metrics for scaling

        Returns:
            Dictionary representing the HPA manifest
        """
        # Build metrics list
        metrics = []

        # Add CPU metric
        metrics.append(
            {
                "type": "Resource",
                "resource": {
                    "name": "cpu",
                    "target": {
                        "type": "Utilization",
                        "averageUtilization": cpu_target_percentage,
                    },
                },
            }
        )

        # Add memory metric if specified
        if memory_target_percentage is not None:
            metrics.append(
                {
                    "type": "Resource",
                    "resource": {
                        "name": "memory",
                        "target": {
                            "type": "Utilization",
                            "averageUtilization": memory_target_percentage,
                        },
                    },
                }
            )

        # Add custom metrics if provided
        if custom_metrics:
            metrics.extend(custom_metrics)

        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.deployment_name}-hpa",
                "namespace": self.namespace,
                "labels": {
                    "app": self.deployment_name,
                    "component": "autoscaler",
                    "framework": "opifex",
                },
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.deployment_name,
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": metrics,
                "behavior": {
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [
                            {"type": "Percent", "value": 100, "periodSeconds": 60},
                            {"type": "Pods", "value": 2, "periodSeconds": 60},
                        ],
                        "selectPolicy": "Min",
                    },
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {"type": "Percent", "value": 50, "periodSeconds": 60}
                        ],
                    },
                },
            },
        }

    def generate_vpa(
        self,
        update_mode: str = "Auto",
        cpu_min: str = "100m",
        memory_min: str = "128Mi",
        cpu_max: str | None = None,
        memory_max: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate Vertical Pod Autoscaler manifest.

        Args:
            update_mode: VPA update mode ("Off", "Initial", "Auto")
            cpu_min: Minimum CPU request
            memory_min: Minimum memory request
            cpu_max: Maximum CPU limit (optional)
            memory_max: Maximum memory limit (optional)

        Returns:
            Dictionary representing the VPA manifest
        """
        resource_policy = {
            "containerPolicies": [
                {
                    "containerName": self.deployment_name,
                    "minAllowed": {"cpu": cpu_min, "memory": memory_min},
                }
            ]
        }

        # Add maximum limits if specified
        if cpu_max or memory_max:
            max_allowed = {}
            if cpu_max:
                max_allowed["cpu"] = cpu_max
            if memory_max:
                max_allowed["memory"] = memory_max
            resource_policy["containerPolicies"][0]["maxAllowed"] = max_allowed

        return {
            "apiVersion": "autoscaling.k8s.io/v1",
            "kind": "VerticalPodAutoscaler",
            "metadata": {
                "name": f"{self.deployment_name}-vpa",
                "namespace": self.namespace,
                "labels": {
                    "app": self.deployment_name,
                    "component": "autoscaler",
                    "framework": "opifex",
                },
            },
            "spec": {
                "targetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.deployment_name,
                },
                "updatePolicy": {"updateMode": update_mode},
                "resourcePolicy": resource_policy,
            },
        }

    def generate_pod_disruption_budget(
        self, min_available: int | None = None, max_unavailable: int | str | None = None
    ) -> dict[str, Any]:
        """
        Generate Pod Disruption Budget manifest for controlled scaling.

        Args:
            min_available: Minimum number of pods that must be available
            max_unavailable: Maximum number of pods that can be unavailable

        Returns:
            Dictionary representing the PDB manifest
        """
        if min_available is None and max_unavailable is None:
            # Default to allowing 25% unavailability
            max_unavailable = "25%"

        spec: dict[str, Any] = {
            "selector": {"matchLabels": {"app": self.deployment_name}}
        }

        if min_available is not None:
            spec["minAvailable"] = min_available
        if max_unavailable is not None:
            spec["maxUnavailable"] = max_unavailable

        return {
            "apiVersion": "policy/v1",
            "kind": "PodDisruptionBudget",
            "metadata": {
                "name": f"{self.deployment_name}-pdb",
                "namespace": self.namespace,
                "labels": {
                    "app": self.deployment_name,
                    "component": "disruption-budget",
                    "framework": "opifex",
                },
            },
            "spec": spec,
        }

    def generate_custom_metric_hpa(
        self,
        metric_name: str,
        metric_selector: dict[str, str],
        target_value: str,
        min_replicas: int = 2,
        max_replicas: int = 10,
    ) -> dict[str, Any]:
        """
        Generate HPA with custom metrics (e.g., inference requests per second).

        Args:
            metric_name: Name of the custom metric
            metric_selector: Labels to select the metric
            target_value: Target value for the metric
            min_replicas: Minimum number of replicas
            max_replicas: Maximum number of replicas

        Returns:
            Dictionary representing the custom metric HPA manifest
        """
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"{self.deployment_name}-custom-hpa",
                "namespace": self.namespace,
                "labels": {
                    "app": self.deployment_name,
                    "component": "custom-autoscaler",
                    "framework": "opifex",
                },
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": self.deployment_name,
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Pods",
                        "pods": {
                            "metric": {
                                "name": metric_name,
                                "selector": {"matchLabels": metric_selector},
                            },
                            "target": {
                                "type": "AverageValue",
                                "averageValue": target_value,
                            },
                        },
                    }
                ],
            },
        }
