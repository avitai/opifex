"""
Kubernetes orchestrator for Opifex deployments.

This module provides a comprehensive orchestrator that combines all Kubernetes
components for complete deployment management.
"""

from pathlib import Path
from typing import Any

import yaml

from .autoscaler import AutoScaler
from .manifest_generator import ManifestGenerator
from .resource_manager import ResourceManager


class KubernetesOrchestrator:
    """
    Comprehensive Kubernetes orchestrator for Opifex deployments.

    Integrates manifest generation, auto-scaling, and resource management
    to provide complete deployment orchestration.
    """

    def __init__(
        self,
        namespace: str,
        app_name: str,
        image: str,
        service_port: int = 8000,
        container_port: int = 8000,
    ):
        """
        Initialize the Kubernetes orchestrator.

        Args:
            namespace: Kubernetes namespace for deployment
            app_name: Application name for labeling and naming
            image: Container image for deployment
            service_port: Service port (default: 8000)
            container_port: Container port (default: 8000)
        """
        self.namespace = namespace
        self.app_name = app_name
        self.image = image
        self.service_port = service_port
        self.container_port = container_port

        # Initialize component managers
        self.manifest_generator = ManifestGenerator(
            namespace=namespace,
            app_name=app_name,
            image=image,
        )
        self.autoscaler = AutoScaler(namespace=namespace, deployment_name=app_name)
        self.resource_manager = ResourceManager(namespace=namespace)

    def generate_complete_deployment(
        self,
        output_dir: Path,
        replicas: int = 3,
        cpu_request: str = "200m",
        memory_request: str = "512Mi",
        cpu_limit: str | None = None,
        memory_limit: str | None = None,
        enable_autoscaling: bool = False,
        min_replicas: int = 2,
        max_replicas: int = 10,
        target_cpu_utilization: int = 70,
        enable_resource_management: bool = False,
        enable_ingress: bool = True,
        ingress_host: str | None = None,
        **kwargs,
    ) -> dict[str, Path]:
        """
        Generate complete deployment manifest suite.

        Args:
            output_dir: Directory to write manifest files
            replicas: Number of deployment replicas
            cpu_request: CPU resource request
            memory_request: Memory resource request
            cpu_limit: CPU resource limit
            memory_limit: Memory resource limit
            enable_autoscaling: Whether to enable HPA
            min_replicas: Minimum replicas for autoscaling
            max_replicas: Maximum replicas for autoscaling
            target_cpu_utilization: Target CPU utilization for HPA
            enable_resource_management: Whether to create resource quotas
            enable_ingress: Whether to create ingress
            ingress_host: Ingress hostname
            **kwargs: Additional parameters

        Returns:
            Dict mapping manifest names to file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_files = {}

        # 1. Resource Management (namespace, quota, limits)
        if enable_resource_management:
            # Namespace
            namespace_manifest = self.resource_manager.generate_namespace(
                labels={"app": self.app_name, "env": "production"}
            )
            namespace_file = output_dir / "namespace.yaml"
            self._write_manifest(namespace_file, namespace_manifest)
            manifest_files["namespace"] = namespace_file

            # Resource Quota
            quota_manifest = self.resource_manager.generate_resource_quota(
                cpu_request="4",
                memory_request="8Gi",
                cpu_limit="8",
                memory_limit="16Gi",
                pods="20",
            )
            quota_file = output_dir / "resource-quota.yaml"
            self._write_manifest(quota_file, quota_manifest)
            manifest_files["resource_quota"] = quota_file

            # Limit Range
            limits_manifest = self.resource_manager.generate_limit_range(
                default_cpu_request="100m",
                default_memory_request="128Mi",
                default_cpu_limit="500m",
                default_memory_limit="512Mi",
            )
            limits_file = output_dir / "limit-range.yaml"
            self._write_manifest(limits_file, limits_manifest)
            manifest_files["limit_range"] = limits_file

        # 2. Core Deployment
        deployment_manifest = self.manifest_generator.generate_deployment(
            replicas=replicas,
            cpu_request=cpu_request,
            memory_request=memory_request,
            cpu_limit=cpu_limit or "500m",
            memory_limit=memory_limit or "1Gi",
            port=self.container_port,
        )
        deployment_file = output_dir / "deployment.yaml"
        self._write_manifest(deployment_file, deployment_manifest)
        manifest_files["deployment"] = deployment_file

        # 3. Service
        service_manifest = self.manifest_generator.generate_service()
        service_file = output_dir / "service.yaml"
        self._write_manifest(service_file, service_manifest)
        manifest_files["service"] = service_file

        # 4. Ingress
        if enable_ingress:
            ingress_manifest = self.manifest_generator.generate_ingress(
                host=ingress_host or f"{self.app_name}.example.com"
            )
            ingress_file = output_dir / "ingress.yaml"
            self._write_manifest(ingress_file, ingress_manifest)
            manifest_files["ingress"] = ingress_file

        # 5. Auto-scaling
        if enable_autoscaling:
            hpa_manifest = self.autoscaler.generate_hpa(
                min_replicas=min_replicas,
                max_replicas=max_replicas,
                cpu_target_percentage=target_cpu_utilization,
            )
            hpa_file = output_dir / "hpa.yaml"
            self._write_manifest(hpa_file, hpa_manifest)
            manifest_files["hpa"] = hpa_file

        return manifest_files

    def generate_gpu_deployment(
        self,
        replicas: int = 2,
        gpu_count: int = 1,
        gpu_type: str = "nvidia.com/gpu",
        cpu_request: str = "500m",
        memory_request: str = "2Gi",
        cpu_limit: str | None = None,
        memory_limit: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate GPU-enabled deployment manifest.

        Args:
            replicas: Number of deployment replicas
            gpu_count: Number of GPUs per pod
            gpu_type: GPU resource type
            cpu_request: CPU resource request
            memory_request: Memory resource request
            cpu_limit: CPU resource limit
            memory_limit: Memory resource limit

        Returns:
            Deployment manifest with GPU configuration
        """
        # Generate base deployment
        manifest = self.manifest_generator.generate_deployment(
            replicas=replicas,
            cpu_request=cpu_request,
            memory_request=memory_request,
            cpu_limit=cpu_limit or "2",
            memory_limit=memory_limit or "4Gi",
            port=self.container_port,
        )

        # Add GPU resources
        container = manifest["spec"]["template"]["spec"]["containers"][0]

        # Add GPU resource limits
        if "resources" not in container:
            container["resources"] = {}
        if "limits" not in container["resources"]:
            container["resources"]["limits"] = {}

        container["resources"]["limits"][gpu_type] = gpu_count

        # Add CUDA environment variables
        if "env" not in container:
            container["env"] = []

        cuda_env_vars = [
            {"name": "CUDA_VISIBLE_DEVICES", "value": "all"},
            {"name": "NVIDIA_VISIBLE_DEVICES", "value": "all"},
            {"name": "NVIDIA_DRIVER_CAPABILITIES", "value": "compute,utility"},
        ]

        for env_var in cuda_env_vars:
            # Check if env var already exists
            existing_names = [env["name"] for env in container["env"]]
            if env_var["name"] not in existing_names:
                container["env"].append(env_var)

        return manifest

    def _write_manifest(self, file_path: Path, manifest: dict[str, Any]) -> None:
        """
        Write manifest to YAML file.

        Args:
            file_path: Path to write manifest
            manifest: Manifest dictionary to write
        """
        with open(file_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
