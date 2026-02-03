"""
Kubernetes resource management for Opifex deployment.

This module provides resource management components including namespaces,
resource quotas, and limit ranges for controlled resource allocation.
"""

from typing import Any


class ResourceManager:
    """
    Generate Kubernetes resource management manifests for Opifex deployment.

    Provides comprehensive resource governance including namespace isolation,
    resource quotas, and default limits for enterprise-grade deployments.
    """

    def __init__(self, namespace: str):
        """
        Initialize resource manager.

        Args:
            namespace: Kubernetes namespace for resource management
        """
        self.namespace = namespace

    def generate_namespace(
        self,
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate Kubernetes namespace manifest.

        Args:
            labels: Labels for the namespace
            annotations: Annotations for the namespace

        Returns:
            Dictionary representing the namespace manifest
        """
        metadata = {
            "name": self.namespace,
            "labels": labels
            or {
                "name": self.namespace,
                "framework": "opifex",
                "managed-by": "opifex-deployment",
            },
        }

        if annotations:
            metadata["annotations"] = annotations

        return {"apiVersion": "v1", "kind": "Namespace", "metadata": metadata}

    def generate_resource_quota(
        self,
        cpu_request: str = "4",
        memory_request: str = "8Gi",
        cpu_limit: str = "8",
        memory_limit: str = "16Gi",
        pods: str = "20",
        storage_requests: str | None = None,
        persistent_volume_claims: str | None = None,
        services: str | None = None,
        secrets: str | None = None,
        config_maps: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate Kubernetes resource quota manifest.

        Args:
            cpu_request: Total CPU requests allowed
            memory_request: Total memory requests allowed
            cpu_limit: Total CPU limits allowed
            memory_limit: Total memory limits allowed
            pods: Maximum number of pods
            storage_requests: Total storage requests allowed
            persistent_volume_claims: Maximum number of PVCs
            services: Maximum number of services
            secrets: Maximum number of secrets
            config_maps: Maximum number of config maps

        Returns:
            Dictionary representing the resource quota manifest
        """
        hard_limits = {
            "requests.cpu": cpu_request,
            "requests.memory": memory_request,
            "limits.cpu": cpu_limit,
            "limits.memory": memory_limit,
            "pods": pods,
        }

        # Add optional limits
        if storage_requests:
            hard_limits["requests.storage"] = storage_requests
        if persistent_volume_claims:
            hard_limits["persistentvolumeclaims"] = persistent_volume_claims
        if services:
            hard_limits["services"] = services
        if secrets:
            hard_limits["secrets"] = secrets
        if config_maps:
            hard_limits["configmaps"] = config_maps

        return {
            "apiVersion": "v1",
            "kind": "ResourceQuota",
            "metadata": {
                "name": f"{self.namespace}-quota",
                "namespace": self.namespace,
                "labels": {
                    "namespace": self.namespace,
                    "component": "resource-quota",
                    "framework": "opifex",
                },
            },
            "spec": {"hard": hard_limits},
        }

    def generate_limit_range(
        self,
        default_cpu_request: str = "100m",
        default_memory_request: str = "128Mi",
        default_cpu_limit: str = "500m",
        default_memory_limit: str = "512Mi",
        max_cpu_limit: str | None = None,
        max_memory_limit: str | None = None,
        min_cpu_request: str | None = None,
        min_memory_request: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate Kubernetes limit range manifest.

        Args:
            default_cpu_request: Default CPU request for containers
            default_memory_request: Default memory request for containers
            default_cpu_limit: Default CPU limit for containers
            default_memory_limit: Default memory limit for containers
            max_cpu_limit: Maximum CPU limit allowed
            max_memory_limit: Maximum memory limit allowed
            min_cpu_request: Minimum CPU request required
            min_memory_request: Minimum memory request required

        Returns:
            Dictionary representing the limit range manifest
        """
        container_limit = {
            "type": "Container",
            "default": {"cpu": default_cpu_limit, "memory": default_memory_limit},
            "defaultRequest": {
                "cpu": default_cpu_request,
                "memory": default_memory_request,
            },
        }

        # Add maximum limits if specified
        if max_cpu_limit or max_memory_limit:
            max_limits = {}
            if max_cpu_limit:
                max_limits["cpu"] = max_cpu_limit
            if max_memory_limit:
                max_limits["memory"] = max_memory_limit
            container_limit["max"] = max_limits

        # Add minimum limits if specified
        if min_cpu_request or min_memory_request:
            min_limits = {}
            if min_cpu_request:
                min_limits["cpu"] = min_cpu_request
            if min_memory_request:
                min_limits["memory"] = min_memory_request
            container_limit["min"] = min_limits

        return {
            "apiVersion": "v1",
            "kind": "LimitRange",
            "metadata": {
                "name": f"{self.namespace}-limits",
                "namespace": self.namespace,
                "labels": {
                    "namespace": self.namespace,
                    "component": "limit-range",
                    "framework": "opifex",
                },
            },
            "spec": {"limits": [container_limit]},
        }

    def generate_network_policy(
        self,
        policy_name: str = "default-deny",
        allow_ingress_from: dict[str, Any] | None = None,
        allow_egress_to: dict[str, Any] | None = None,
        pod_selector: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate Kubernetes network policy manifest.

        Args:
            policy_name: Name of the network policy
            allow_ingress_from: Ingress rules (default: deny all)
            allow_egress_to: Egress rules (default: allow all)
            pod_selector: Pod selector for the policy

        Returns:
            Dictionary representing the network policy manifest
        """
        spec = {
            "podSelector": {"matchLabels": pod_selector or {"framework": "opifex"}},
            "policyTypes": [],
        }

        # Configure ingress rules
        if allow_ingress_from is not None:
            spec["policyTypes"].append("Ingress")
            spec["ingress"] = allow_ingress_from
        else:
            # Default deny all ingress
            spec["policyTypes"].append("Ingress")
            spec["ingress"] = []

        # Configure egress rules
        if allow_egress_to is not None:
            spec["policyTypes"].append("Egress")
            spec["egress"] = allow_egress_to

        return {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "NetworkPolicy",
            "metadata": {
                "name": f"{self.namespace}-{policy_name}",
                "namespace": self.namespace,
                "labels": {
                    "namespace": self.namespace,
                    "component": "network-policy",
                    "framework": "opifex",
                },
            },
            "spec": spec,
        }

    def generate_service_account(
        self,
        service_account_name: str = "opifex-service",
        automount_service_account_token: bool = False,
        image_pull_secrets: list | None = None,
    ) -> dict[str, Any]:
        """
        Generate Kubernetes service account manifest.

        Args:
            service_account_name: Name of the service account
            automount_service_account_token: Whether to automount SA token
            image_pull_secrets: List of image pull secrets

        Returns:
            Dictionary representing the service account manifest
        """
        manifest = {
            "apiVersion": "v1",
            "kind": "ServiceAccount",
            "metadata": {
                "name": service_account_name,
                "namespace": self.namespace,
                "labels": {
                    "namespace": self.namespace,
                    "component": "service-account",
                    "framework": "opifex",
                },
            },
            "automountServiceAccountToken": automount_service_account_token,
        }

        if image_pull_secrets:
            manifest["imagePullSecrets"] = [
                {"name": secret} for secret in image_pull_secrets
            ]

        return manifest

    def generate_security_context_constraints(
        self,
        scc_name: str = "opifex-scc",
        allow_privileged: bool = False,
        allow_privilege_escalation: bool = False,
        required_drop_capabilities: list | None = None,
        allowed_capabilities: list | None = None,
        run_as_user_strategy: str = "MustRunAsNonRoot",
    ) -> dict[str, Any]:
        """
        Generate security context constraints for OpenShift environments.

        Args:
            scc_name: Name of the security context constraint
            allow_privileged: Whether to allow privileged containers
            allow_privilege_escalation: Whether to allow privilege escalation
            required_drop_capabilities: Capabilities that must be dropped
            allowed_capabilities: Capabilities that are allowed
            run_as_user_strategy: Strategy for running as user

        Returns:
            Dictionary representing the SCC manifest
        """
        return {
            "apiVersion": "security.openshift.io/v1",
            "kind": "SecurityContextConstraints",
            "metadata": {
                "name": scc_name,
                "labels": {
                    "component": "security-context-constraints",
                    "framework": "opifex",
                },
            },
            "allowPrivilegedContainer": allow_privileged,
            "allowPrivilegeEscalation": allow_privilege_escalation,
            "requiredDropCapabilities": required_drop_capabilities or ["ALL"],
            "allowedCapabilities": allowed_capabilities or [],
            "runAsUser": {"type": run_as_user_strategy},
            "seLinuxContext": {"type": "MustRunAs"},
            "fsGroup": {"type": "MustRunAs"},
            "supplementalGroups": {"type": "MustRunAs"},
            "volumes": [
                "configMap",
                "emptyDir",
                "projected",
                "secret",
                "downwardAPI",
                "persistentVolumeClaim",
            ],
        }
