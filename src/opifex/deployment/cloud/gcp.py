"""Google Cloud Platform deployment configurations for Opifex framework.

This module provides GCP-specific deployment configurations that leverage
managed Google Cloud services for security, scaling, and infrastructure.

Services integrated:
    - Google Kubernetes Engine (GKE) for container orchestration
    - Cloud IAM for authentication and authorization
    - VPC and firewall rules for network security
    - Secret Manager for secrets management
    - Cloud Monitoring and Logging for observability
    - Cloud Load Balancing for high availability
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class GCPConfig:
    """Configuration for GCP deployment."""

    project_id: str
    region: str = "us-central1"
    zone: str = "us-central1-a"
    cluster_name: str = "opifex-cluster"

    # GKE configuration
    node_pool_config: dict[str, Any] = field(
        default_factory=lambda: {
            "initial_node_count": 3,
            "machine_type": "n1-standard-4",
            "disk_size_gb": 100,
            "auto_scaling": {"min_node_count": 1, "max_node_count": 10},
            "gpu_config": {
                "accelerator_type": "nvidia-tesla-t4",
                "accelerator_count": 1,
            },
        }
    )

    # Network configuration
    network_config: dict[str, Any] = field(
        default_factory=lambda: {
            "network": "opifex-vpc",
            "subnetwork": "opifex-subnet",
            "authorized_networks": [],
            "enable_private_nodes": True,
        }
    )

    # Security configuration
    security_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_workload_identity": True,
            "enable_network_policy": True,
            "enable_pod_security_policy": True,
            "master_authorized_networks": [],
        }
    )


class GCPDeploymentManager:
    """Manages Opifex framework deployment on Google Cloud Platform."""

    def __init__(self, config: GCPConfig):
        """Initialize GCP deployment manager.

        Args:
            config: GCP deployment configuration
        """
        self.config = config

    def generate_gke_cluster_config(self) -> dict[str, Any]:
        """Generate GKE cluster configuration.

        Returns:
            GKE cluster configuration dictionary
        """
        return {
            "name": self.config.cluster_name,
            "location": self.config.zone,
            "initial_node_count": self.config.node_pool_config["initial_node_count"],
            "node_config": {
                "machine_type": self.config.node_pool_config["machine_type"],
                "disk_size_gb": self.config.node_pool_config["disk_size_gb"],
                "oauth_scopes": [
                    "https://www.googleapis.com/auth/cloud-platform",
                    "https://www.googleapis.com/auth/devstorage.read_only",
                    "https://www.googleapis.com/auth/logging.write",
                    "https://www.googleapis.com/auth/monitoring",
                ],
                "metadata": {"disable-legacy-endpoints": "true"},
                "workload_metadata_config": {"mode": "GKE_METADATA"},
            },
            "network": self.config.network_config["network"],
            "subnetwork": self.config.network_config["subnetwork"],
            "ip_allocation_policy": {"use_ip_aliases": True},
            "workload_identity_config": {
                "workload_pool": f"{self.config.project_id}.svc.id.goog"
            },
            "network_policy": {
                "enabled": self.config.security_config["enable_network_policy"]
            },
            "pod_security_policy_config": {
                "enabled": self.config.security_config["enable_pod_security_policy"]
            },
            "private_cluster_config": {
                "enable_private_nodes": self.config.network_config[
                    "enable_private_nodes"
                ],
                "enable_private_endpoint": False,
                "master_ipv4_cidr_block": "172.16.0.0/28",
            },
            "master_authorized_networks_config": {
                "enabled": len(
                    self.config.security_config["master_authorized_networks"]
                )
                > 0,
                "cidr_blocks": self.config.security_config[
                    "master_authorized_networks"
                ],
            },
        }

    def generate_iam_policy(self) -> dict[str, Any]:
        """Generate Cloud IAM policy for Opifex deployment.

        Returns:
            IAM policy configuration
        """
        return {
            "bindings": [
                {
                    "role": "roles/container.developer",
                    "members": [
                        f"serviceAccount:opifex-service@{self.config.project_id}.iam.gserviceaccount.com"
                    ],
                },
                {
                    "role": "roles/secretmanager.secretAccessor",
                    "members": [
                        f"serviceAccount:opifex-service@{self.config.project_id}.iam.gserviceaccount.com"
                    ],
                },
                {
                    "role": "roles/monitoring.editor",
                    "members": [
                        f"serviceAccount:opifex-service@{self.config.project_id}.iam.gserviceaccount.com"
                    ],
                },
                {
                    "role": "roles/logging.logWriter",
                    "members": [
                        f"serviceAccount:opifex-service@{self.config.project_id}.iam.gserviceaccount.com"
                    ],
                },
            ]
        }

    def generate_vpc_config(self) -> dict[str, Any]:
        """Generate VPC and firewall configuration.

        Returns:
            VPC configuration dictionary
        """
        return {
            "vpc": {
                "name": self.config.network_config["network"],
                "auto_create_subnetworks": False,
                "routing_mode": "REGIONAL",
            },
            "subnet": {
                "name": self.config.network_config["subnetwork"],
                "ip_cidr_range": "10.0.0.0/16",
                "region": self.config.region,
                "secondary_ip_ranges": [
                    {"range_name": "pods", "ip_cidr_range": "10.1.0.0/16"},
                    {"range_name": "services", "ip_cidr_range": "10.2.0.0/16"},
                ],
            },
            "firewall_rules": [
                {
                    "name": "opifex-allow-internal",
                    "direction": "INGRESS",
                    "priority": 1000,
                    "source_ranges": ["10.0.0.0/8"],
                    "allowed": [
                        {"IPProtocol": "tcp"},
                        {"IPProtocol": "udp"},
                        {"IPProtocol": "icmp"},
                    ],
                },
                {
                    "name": "opifex-allow-ssh",
                    "direction": "INGRESS",
                    "priority": 1000,
                    "source_ranges": ["0.0.0.0/0"],
                    "allowed": [{"IPProtocol": "tcp", "ports": ["22"]}],
                    "target_tags": ["opifex-ssh"],
                },
                {
                    "name": "opifex-allow-api",
                    "direction": "INGRESS",
                    "priority": 1000,
                    "source_ranges": ["0.0.0.0/0"],
                    "allowed": [{"IPProtocol": "tcp", "ports": ["80", "443", "8080"]}],
                    "target_tags": ["opifex-api"],
                },
            ],
        }

    def generate_secret_manager_config(self) -> dict[str, Any]:
        """Generate Secret Manager configuration for secure secrets storage.

        Returns:
            Secret Manager configuration
        """
        return {
            "secrets": [
                {
                    "secret_id": "opifex-database-password",
                    "replication": {"automatic": {}},
                },
                {
                    "secret_id": "opifex-api-keys",
                    "replication": {
                        "user_managed": {"replicas": [{"location": self.config.region}]}
                    },
                },
                {
                    "secret_id": "opifex-oauth-credentials",
                    "replication": {"automatic": {}},
                },
            ]
        }

    def generate_monitoring_config(self) -> dict[str, Any]:
        """Generate Cloud Monitoring and Logging configuration.

        Returns:
            Monitoring configuration
        """
        return {
            "notification_channels": [
                {
                    "type": "email",
                    "display_name": "Opifex Operations",
                    "labels": {"email_address": "ops@opifex.io"},
                }
            ],
            "alert_policies": [
                {
                    "display_name": "High CPU Usage",
                    "conditions": [
                        {
                            "display_name": "CPU usage",
                            "condition_threshold": {
                                "filter": 'resource.type="k8s_container"',
                                "comparison": "COMPARISON_GREATER_THAN",
                                "threshold_value": 0.8,
                                "duration": "300s",
                            },
                        }
                    ],
                },
                {
                    "display_name": "High Memory Usage",
                    "conditions": [
                        {
                            "display_name": "Memory usage",
                            "condition_threshold": {
                                "filter": 'resource.type="k8s_container"',
                                "comparison": "COMPARISON_GREATER_THAN",
                                "threshold_value": 0.9,
                                "duration": "300s",
                            },
                        }
                    ],
                },
            ],
        }

    def export_terraform_config(self, output_dir: str | Path) -> None:
        """Export complete GCP configuration as Terraform files.

        Args:
            output_dir: Directory to save Terraform configuration files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Main Terraform configuration
        main_tf = {
            "provider": {
                "google": {
                    "project": self.config.project_id,
                    "region": self.config.region,
                    "zone": self.config.zone,
                }
            },
            "resource": {
                "google_container_cluster": {
                    "opifex_cluster": self.generate_gke_cluster_config()
                },
                "google_compute_network": {
                    "opifex_vpc": self.generate_vpc_config()["vpc"]
                },
                "google_compute_subnetwork": {
                    "opifex_subnet": self.generate_vpc_config()["subnet"]
                },
            },
        }

        # Write Terraform files
        with open(output_path / "main.tf", "w") as f:
            json.dump(main_tf, f, indent=2)

        with open(output_path / "variables.tf", "w") as f:
            f.write(self._generate_terraform_variables())

        with open(output_path / "outputs.tf", "w") as f:
            f.write(self._generate_terraform_outputs())

    def _generate_terraform_variables(self) -> str:
        """Generate Terraform variables file content."""
        return """
variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP Zone"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "GKE Cluster Name"
  type        = string
  default     = "opifex-cluster"
}
"""

    def _generate_terraform_outputs(self) -> str:
        """Generate Terraform outputs file content."""
        return (
            'output "cluster_endpoint" {\n'
            '  description = "GKE Cluster Endpoint"\n'
            "  value       = google_container_cluster.opifex_cluster.endpoint\n"
            "}\n\n"
            'output "cluster_ca_certificate" {\n'
            '  description = "GKE Cluster CA Certificate"\n'
            "  value       = "
            "google_container_cluster.opifex_cluster.master_auth.0.cluster_ca_certificate\n"
            "  sensitive   = true\n"
            "}\n\n"
            'output "vpc_name" {\n'
            '  description = "VPC Network Name"\n'
            "  value       = google_compute_network.opifex_vpc.name\n"
            "}\n\n"
            'output "subnet_name" {\n'
            '  description = "Subnet Name"\n'
            "  value       = google_compute_subnetwork.opifex_subnet.name\n"
            "}\n"
        )
