"""Cloud deployment configurations for Opifex framework.

This package provides cloud-native deployment configurations that leverage
managed cloud services for security, scaling, and infrastructure instead of
custom implementations.

Supported Cloud Platforms:
    - Google Cloud Platform (GCP): GKE, Cloud IAM, VPC, Secret Manager, Cloud Monitoring
    - Amazon Web Services (AWS): EKS, IAM, VPC, Secrets Manager, CloudWatch

The configurations generate Terraform files for infrastructure as code deployment.
"""

# GCP deployment components
# AWS deployment components
from .aws import (
    AWSConfig,
    AWSDeploymentManager,
)
from .gcp import (
    GCPConfig,
    GCPDeploymentManager,
)


__all__ = [
    "AWSConfig",
    "AWSDeploymentManager",
    "GCPConfig",
    "GCPDeploymentManager",
]
