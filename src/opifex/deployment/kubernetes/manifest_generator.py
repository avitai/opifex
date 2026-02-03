"""
Kubernetes manifest generator for Opifex deployment.

This module generates Kubernetes YAML manifests for deployment, service, and ingress
resources following production-ready patterns and Opifex framework requirements.
"""

from pathlib import Path
from typing import Any

import yaml


class ManifestGenerator:
    """
    Generate Kubernetes manifests for Opifex deployment infrastructure.

    Follows enterprise-grade patterns for containerized ML model serving
    with proper resource management and configuration.
    """

    def __init__(self, namespace: str, app_name: str, image: str):
        """
        Initialize manifest generator.

        Args:
            namespace: Kubernetes namespace for deployment
            app_name: Application name for labeling and naming
            image: Container image for deployment
        """
        self.namespace = namespace
        self.app_name = app_name
        self.image = image

    def generate_deployment(
        self,
        replicas: int = 3,
        cpu_request: str = "100m",
        memory_request: str = "256Mi",
        cpu_limit: str = "500m",
        memory_limit: str = "1Gi",
        port: int = 8000,
        environment_variables: dict[str, str] | None = None,
        node_selector: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Generate Kubernetes deployment manifest.

        Args:
            replicas: Number of pod replicas
            cpu_request: CPU resource request
            memory_request: Memory resource request
            cpu_limit: CPU resource limit
            memory_limit: Memory resource limit
            port: Container port for the application
            environment_variables: Environment variables for the container
            node_selector: Node selector constraints

        Returns:
            Dictionary representing the deployment manifest
        """
        env_vars = environment_variables or {}

        # Add default JAX environment variables for CPU deployment
        default_env = {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "True",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.75",
        }
        default_env.update(env_vars)

        env_list = [{"name": key, "value": value} for key, value in default_env.items()]

        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.app_name,
                "namespace": self.namespace,
                "labels": {
                    "app": self.app_name,
                    "component": "ml-server",
                    "framework": "opifex",
                },
            },
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": {"app": self.app_name}},
                "template": {
                    "metadata": {
                        "labels": {
                            "app": self.app_name,
                            "component": "ml-server",
                            "framework": "opifex",
                        }
                    },
                    "spec": {
                        "containers": [
                            {
                                "name": self.app_name,
                                "image": self.image,
                                "ports": [{"containerPort": port, "name": "http"}],
                                "env": env_list,
                                "resources": {
                                    "requests": {
                                        "cpu": cpu_request,
                                        "memory": memory_request,
                                    },
                                    "limits": {
                                        "cpu": cpu_limit,
                                        "memory": memory_limit,
                                    },
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": port},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10,
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/health", "port": port},
                                    "initialDelaySeconds": 5,
                                    "periodSeconds": 5,
                                },
                            }
                        ]
                    },
                },
            },
        }

        # Add node selector if specified
        if node_selector:
            manifest["spec"]["template"]["spec"]["nodeSelector"] = node_selector

        return manifest

    def generate_service(
        self, port: int = 8000, target_port: int = 8000, service_type: str = "ClusterIP"
    ) -> dict[str, Any]:
        """
        Generate Kubernetes service manifest.

        Args:
            port: Service port
            target_port: Target port on pods
            service_type: Kubernetes service type (ClusterIP, NodePort, LoadBalancer)

        Returns:
            Dictionary representing the service manifest
        """
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"{self.app_name}-service",
                "namespace": self.namespace,
                "labels": {
                    "app": self.app_name,
                    "component": "ml-server",
                    "framework": "opifex",
                },
            },
            "spec": {
                "type": service_type,
                "selector": {"app": self.app_name},
                "ports": [
                    {
                        "port": port,
                        "targetPort": target_port,
                        "protocol": "TCP",
                        "name": "http",
                    }
                ],
            },
        }

    def generate_ingress(
        self,
        host: str,
        path: str = "/",
        service_name: str | None = None,
        service_port: int = 8000,
        tls_enabled: bool = False,
        tls_secret_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Generate Kubernetes ingress manifest.

        Args:
            host: Hostname for the ingress
            path: Path for routing
            service_name: Name of the service (defaults to {app_name}-service)
            service_port: Service port
            tls_enabled: Whether to enable TLS
            tls_secret_name: Name of TLS secret

        Returns:
            Dictionary representing the ingress manifest
        """
        if service_name is None:
            service_name = f"{self.app_name}-service"

        manifest = {
            "apiVersion": "networking.k8s.io/v1",
            "kind": "Ingress",
            "metadata": {
                "name": f"{self.app_name}-ingress",
                "namespace": self.namespace,
                "labels": {
                    "app": self.app_name,
                    "component": "ml-server",
                    "framework": "opifex",
                },
                "annotations": {
                    "nginx.ingress.kubernetes.io/rewrite-target": "/",
                    "nginx.ingress.kubernetes.io/ssl-redirect": "false"
                    if not tls_enabled
                    else "true",
                },
            },
            "spec": {
                "rules": [
                    {
                        "host": host,
                        "http": {
                            "paths": [
                                {
                                    "path": path,
                                    "pathType": "Prefix",
                                    "backend": {
                                        "service": {
                                            "name": service_name,
                                            "port": {"number": service_port},
                                        }
                                    },
                                }
                            ]
                        },
                    }
                ]
            },
        }

        # Add TLS configuration if enabled
        if tls_enabled and tls_secret_name:
            manifest["spec"]["tls"] = [{"hosts": [host], "secretName": tls_secret_name}]

        return manifest

    def export_all_manifests(
        self,
        output_dir: Path,
        replicas: int = 2,
        cpu_request: str = "100m",
        memory_request: str = "256Mi",
        cpu_limit: str = "500m",
        memory_limit: str = "1Gi",
        service_port: int = 8000,
        ingress_host: str = "opifex.local",
        **kwargs,
    ) -> None:
        """
        Export all manifests to YAML files.

        Args:
            output_dir: Directory to save manifest files
            replicas: Number of pod replicas
            cpu_request: CPU resource request
            memory_request: Memory resource request
            cpu_limit: CPU resource limit
            memory_limit: Memory resource limit
            service_port: Service port
            ingress_host: Ingress hostname
            **kwargs: Additional arguments for manifest generation
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate deployment manifest
        deployment = self.generate_deployment(
            replicas=replicas,
            cpu_request=cpu_request,
            memory_request=memory_request,
            cpu_limit=cpu_limit,
            memory_limit=memory_limit,
            **kwargs,
        )

        # Generate service manifest
        service = self.generate_service(port=service_port, target_port=service_port)

        # Generate ingress manifest
        ingress = self.generate_ingress(host=ingress_host, service_port=service_port)

        # Export to files
        with open(output_dir / "deployment.yaml", "w") as f:
            yaml.dump(deployment, f, default_flow_style=False, sort_keys=False)

        with open(output_dir / "service.yaml", "w") as f:
            yaml.dump(service, f, default_flow_style=False, sort_keys=False)

        with open(output_dir / "ingress.yaml", "w") as f:
            yaml.dump(ingress, f, default_flow_style=False, sort_keys=False)
