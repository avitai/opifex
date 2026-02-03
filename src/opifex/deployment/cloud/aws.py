"""Amazon Web Services deployment configurations for Opifex framework.

This module provides AWS-specific deployment configurations that leverage
managed AWS services for security, scaling, and infrastructure.

Services integrated:
    - Amazon Elastic Kubernetes Service (EKS) for container orchestration
    - AWS Identity and Access Management (IAM) for authentication and authorization
    - Amazon VPC and Security Groups for network security
    - AWS Secrets Manager for secrets management
    - Amazon CloudWatch for monitoring and logging
    - Application Load Balancer for high availability
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AWSConfig:
    """Configuration for AWS deployment."""

    region: str = "us-east-1"
    cluster_name: str = "opifex-cluster"
    vpc_cidr: str = "10.0.0.0/16"

    # EKS configuration
    node_group_config: dict[str, Any] = field(
        default_factory=lambda: {
            "desired_size": 3,
            "max_size": 10,
            "min_size": 1,
            "instance_types": ["m5.xlarge"],
            "disk_size": 20,
            "capacity_type": "ON_DEMAND",
            "ami_type": "AL2_x86_64_GPU",
        }
    )

    # Network configuration
    network_config: dict[str, Any] = field(
        default_factory=lambda: {
            "availability_zones": ["us-east-1a", "us-east-1b", "us-east-1c"],
            "private_subnets": ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"],
            "public_subnets": ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"],
            "enable_nat_gateway": True,
            "enable_dns_hostnames": True,
        }
    )

    # Security configuration
    security_config: dict[str, Any] = field(
        default_factory=lambda: {
            "enable_logging": True,
            "log_types": [
                "api",
                "audit",
                "authenticator",
                "controllerManager",
                "scheduler",
            ],
            "enable_private_access": True,
            "enable_public_access": True,
            "public_access_cidrs": ["0.0.0.0/0"],
        }
    )


class AWSDeploymentManager:
    """Manages Opifex framework deployment on Amazon Web Services."""

    def __init__(self, config: AWSConfig):
        """Initialize AWS deployment manager.

        Args:
            config: AWS deployment configuration
        """
        self.config = config

    def generate_eks_cluster_config(self) -> dict[str, Any]:
        """Generate EKS cluster configuration.

        Returns:
            EKS cluster configuration dictionary
        """
        return {
            "name": self.config.cluster_name,
            "version": "1.28",
            "role_arn": (
                f"arn:aws:iam::ACCOUNT_ID:role/{self.config.cluster_name}-cluster-role"
            ),
            "vpc_config": {
                "subnet_ids": [
                    f"subnet-{self.config.cluster_name}-private-1",
                    f"subnet-{self.config.cluster_name}-private-2",
                    f"subnet-{self.config.cluster_name}-private-3",
                    f"subnet-{self.config.cluster_name}-public-1",
                    f"subnet-{self.config.cluster_name}-public-2",
                    f"subnet-{self.config.cluster_name}-public-3",
                ],
                "endpoint_config_private_access": self.config.security_config[
                    "enable_private_access"
                ],
                "endpoint_config_public_access": self.config.security_config[
                    "enable_public_access"
                ],
                "public_access_cidrs": self.config.security_config[
                    "public_access_cidrs"
                ],
            },
            "enabled_cluster_log_types": self.config.security_config["log_types"],
            "encryption_config": [
                {
                    "resources": ["secrets"],
                    "provider": {
                        "key_id": (
                            f"arn:aws:kms:{self.config.region}:ACCOUNT_ID:key/"
                            "12345678-1234-1234-1234-123456789012"
                        )
                    },
                }
            ],
            "tags": {
                "Environment": "production",
                "Project": "opifex",
                "ManagedBy": "terraform",
            },
        }

    def generate_node_group_config(self) -> dict[str, Any]:
        """Generate EKS node group configuration.

        Returns:
            Node group configuration dictionary
        """
        return {
            "cluster_name": self.config.cluster_name,
            "node_group_name": f"{self.config.cluster_name}-nodes",
            "node_role_arn": (
                f"arn:aws:iam::ACCOUNT_ID:role/{self.config.cluster_name}-node-role"
            ),
            "subnet_ids": [
                f"subnet-{self.config.cluster_name}-private-1",
                f"subnet-{self.config.cluster_name}-private-2",
                f"subnet-{self.config.cluster_name}-private-3",
            ],
            "scaling_config": {
                "desired_size": self.config.node_group_config["desired_size"],
                "max_size": self.config.node_group_config["max_size"],
                "min_size": self.config.node_group_config["min_size"],
            },
            "update_config": {"max_unavailable": 1},
            "instance_types": self.config.node_group_config["instance_types"],
            "ami_type": self.config.node_group_config["ami_type"],
            "capacity_type": self.config.node_group_config["capacity_type"],
            "disk_size": self.config.node_group_config["disk_size"],
            "remote_access": {"ec2_ssh_key": f"{self.config.cluster_name}-keypair"},
            "tags": {
                "Environment": "production",
                "Project": "opifex",
                "ManagedBy": "terraform",
            },
        }

    def generate_iam_roles(self) -> dict[str, Any]:
        """Generate IAM roles and policies for EKS.

        Returns:
            IAM configuration dictionary
        """
        return {
            "cluster_role": {
                "name": f"{self.config.cluster_name}-cluster-role",
                "assume_role_policy": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Action": "sts:AssumeRole",
                            "Effect": "Allow",
                            "Principal": {"Service": "eks.amazonaws.com"},
                        }
                    ],
                },
                "managed_policy_arns": [
                    "arn:aws:iam::aws:policy/AmazonEKSClusterPolicy"
                ],
            },
            "node_role": {
                "name": f"{self.config.cluster_name}-node-role",
                "assume_role_policy": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Action": "sts:AssumeRole",
                            "Effect": "Allow",
                            "Principal": {"Service": "ec2.amazonaws.com"},
                        }
                    ],
                },
                "managed_policy_arns": [
                    "arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy",
                    "arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy",
                    "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
                ],
            },
            "service_role": {
                "name": f"{self.config.cluster_name}-service-role",
                "assume_role_policy": {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Action": "sts:AssumeRole",
                            "Effect": "Allow",
                            "Principal": {"Service": "eks.amazonaws.com"},
                        }
                    ],
                },
                "inline_policies": [
                    {
                        "name": "SecretsManagerAccess",
                        "policy": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "secretsmanager:GetSecretValue",
                                        "secretsmanager:DescribeSecret",
                                    ],
                                    "Resource": (
                                        f"arn:aws:secretsmanager:{self.config.region}:"
                                        "ACCOUNT_ID:secret:opifex/*"
                                    ),
                                }
                            ],
                        },
                    },
                    {
                        "name": "CloudWatchAccess",
                        "policy": {
                            "Version": "2012-10-17",
                            "Statement": [
                                {
                                    "Effect": "Allow",
                                    "Action": [
                                        "cloudwatch:PutMetricData",
                                        "logs:CreateLogGroup",
                                        "logs:CreateLogStream",
                                        "logs:PutLogEvents",
                                    ],
                                    "Resource": "*",
                                }
                            ],
                        },
                    },
                ],
            },
        }

    def generate_vpc_config(self) -> dict[str, Any]:
        """Generate VPC and networking configuration.

        Returns:
            VPC configuration dictionary
        """
        return {
            "vpc": {
                "cidr_block": self.config.vpc_cidr,
                "enable_dns_hostnames": self.config.network_config[
                    "enable_dns_hostnames"
                ],
                "enable_dns_support": True,
                "tags": {
                    "Name": f"{self.config.cluster_name}-vpc",
                    "kubernetes.io/cluster/{self.config.cluster_name}": "shared",
                },
            },
            "internet_gateway": {"tags": {"Name": f"{self.config.cluster_name}-igw"}},
            "private_subnets": [
                {
                    "cidr_block": cidr,
                    "availability_zone": az,
                    "tags": {
                        "Name": f"{self.config.cluster_name}-private-{i + 1}",
                        "kubernetes.io/cluster/{self.config.cluster_name}": "owned",
                        "kubernetes.io/role/internal-elb": "1",
                    },
                }
                for i, (cidr, az) in enumerate(
                    zip(
                        self.config.network_config["private_subnets"],
                        self.config.network_config["availability_zones"],
                        strict=False,
                    )
                )
            ],
            "public_subnets": [
                {
                    "cidr_block": cidr,
                    "availability_zone": az,
                    "map_public_ip_on_launch": True,
                    "tags": {
                        "Name": f"{self.config.cluster_name}-public-{i + 1}",
                        "kubernetes.io/cluster/{self.config.cluster_name}": "owned",
                        "kubernetes.io/role/elb": "1",
                    },
                }
                for i, (cidr, az) in enumerate(
                    zip(
                        self.config.network_config["public_subnets"],
                        self.config.network_config["availability_zones"],
                        strict=False,
                    )
                )
            ],
            "nat_gateways": [
                {
                    "allocation_id": f"eip-{self.config.cluster_name}-nat-{i + 1}",
                    "subnet_id": f"subnet-{self.config.cluster_name}-public-{i + 1}",
                    "tags": {"Name": f"{self.config.cluster_name}-nat-{i + 1}"},
                }
                for i in range(len(self.config.network_config["availability_zones"]))
            ]
            if self.config.network_config["enable_nat_gateway"]
            else [],
            "security_groups": {
                "cluster_sg": {
                    "name": f"{self.config.cluster_name}-cluster-sg",
                    "description": "Security group for EKS cluster",
                    "ingress": [
                        {
                            "from_port": 443,
                            "to_port": 443,
                            "protocol": "tcp",
                            "cidr_blocks": ["0.0.0.0/0"],
                        }
                    ],
                    "egress": [
                        {
                            "from_port": 0,
                            "to_port": 0,
                            "protocol": "-1",
                            "cidr_blocks": ["0.0.0.0/0"],
                        }
                    ],
                },
                "node_sg": {
                    "name": f"{self.config.cluster_name}-node-sg",
                    "description": "Security group for EKS nodes",
                    "ingress": [
                        {
                            "from_port": 22,
                            "to_port": 22,
                            "protocol": "tcp",
                            "cidr_blocks": [self.config.vpc_cidr],
                        },
                        {
                            "from_port": 1025,
                            "to_port": 65535,
                            "protocol": "tcp",
                            "source_security_group_id": "cluster_sg",
                        },
                    ],
                    "egress": [
                        {
                            "from_port": 0,
                            "to_port": 0,
                            "protocol": "-1",
                            "cidr_blocks": ["0.0.0.0/0"],
                        }
                    ],
                },
            },
        }

    def generate_secrets_manager_config(self) -> dict[str, Any]:
        """Generate AWS Secrets Manager configuration.

        Returns:
            Secrets Manager configuration
        """
        return {
            "secrets": [
                {
                    "name": "opifex/database/password",
                    "description": "Database password for Opifex application",
                    "generate_secret_string": {
                        "password_length": 32,
                        "exclude_characters": '"@/\\',
                    },
                },
                {
                    "name": "opifex/api/keys",
                    "description": "API keys for Opifex external services",
                    "secret_string": json.dumps(
                        {"api_key_1": "PLACEHOLDER", "api_key_2": "PLACEHOLDER"}
                    ),
                },
                {
                    "name": "opifex/oauth/credentials",
                    "description": "OAuth credentials for Opifex authentication",
                    "secret_string": json.dumps(
                        {"client_id": "PLACEHOLDER", "client_secret": "PLACEHOLDER"}
                    ),
                },
            ]
        }

    def generate_cloudwatch_config(self) -> dict[str, Any]:
        """Generate CloudWatch monitoring and logging configuration.

        Returns:
            CloudWatch configuration
        """
        return {
            "log_groups": [
                {
                    "name": f"/aws/eks/{self.config.cluster_name}/cluster",
                    "retention_in_days": 30,
                },
                {
                    "name": f"/aws/eks/{self.config.cluster_name}/application",
                    "retention_in_days": 7,
                },
            ],
            "alarms": [
                {
                    "alarm_name": f"{self.config.cluster_name}-high-cpu",
                    "comparison_operator": "GreaterThanThreshold",
                    "evaluation_periods": 2,
                    "metric_name": "CPUUtilization",
                    "namespace": "AWS/EKS",
                    "period": 300,
                    "statistic": "Average",
                    "threshold": 80,
                    "alarm_description": "High CPU usage in EKS cluster",
                    "dimensions": {"ClusterName": self.config.cluster_name},
                },
                {
                    "alarm_name": f"{self.config.cluster_name}-high-memory",
                    "comparison_operator": "GreaterThanThreshold",
                    "evaluation_periods": 2,
                    "metric_name": "MemoryUtilization",
                    "namespace": "AWS/EKS",
                    "period": 300,
                    "statistic": "Average",
                    "threshold": 85,
                    "alarm_description": "High memory usage in EKS cluster",
                    "dimensions": {"ClusterName": self.config.cluster_name},
                },
            ],
            "dashboard": {
                "dashboard_name": f"{self.config.cluster_name}-dashboard",
                "dashboard_body": json.dumps(
                    {
                        "widgets": [
                            {
                                "type": "metric",
                                "properties": {
                                    "metrics": [
                                        [
                                            "AWS/EKS",
                                            "CPUUtilization",
                                            "ClusterName",
                                            self.config.cluster_name,
                                        ],
                                        [".", "MemoryUtilization", ".", "."],
                                    ],
                                    "period": 300,
                                    "stat": "Average",
                                    "region": self.config.region,
                                    "title": "EKS Cluster Metrics",
                                },
                            }
                        ]
                    }
                ),
            },
        }

    def export_terraform_config(self, output_dir: str | Path) -> None:
        """Export complete AWS configuration as Terraform files.

        Args:
            output_dir: Directory to save Terraform configuration files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Main Terraform configuration
        main_tf = {
            "provider": {"aws": {"region": self.config.region}},
            "resource": {
                "aws_eks_cluster": {
                    "opifex_cluster": self.generate_eks_cluster_config()
                },
                "aws_eks_node_group": {
                    "opifex_nodes": self.generate_node_group_config()
                },
                "aws_vpc": {"opifex_vpc": self.generate_vpc_config()["vpc"]},
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
variable "region" {
  description = "AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "cluster_name" {
  description = "EKS Cluster Name"
  type        = string
  default     = "opifex-cluster"
}

variable "node_instance_types" {
  description = "EC2 instance types for EKS nodes"
  type        = list(string)
  default     = ["m5.xlarge"]
}

variable "desired_capacity" {
  description = "Desired number of nodes"
  type        = number
  default     = 3
}
"""

    def _generate_terraform_outputs(self) -> str:
        """Generate Terraform outputs file content."""
        return """
output "cluster_endpoint" {
  description = "EKS Cluster Endpoint"
  value       = aws_eks_cluster.opifex_cluster.endpoint
}

output "cluster_ca_certificate" {
  description = "EKS Cluster CA Certificate"
  value       = aws_eks_cluster.opifex_cluster.certificate_authority.0.data
  sensitive   = true
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.opifex_vpc.id
}

output "cluster_security_group_id" {
  description = "Security group ID for the cluster"
  value       = aws_eks_cluster.opifex_cluster.vpc_config.0.cluster_security_group_id
}
"""
