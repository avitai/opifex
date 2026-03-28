# AWS Deployment

This guide covers deploying Opifex models on Amazon Web Services. Opifex provides Python modules for generating AWS infrastructure configurations -- EKS clusters, IAM roles, VPC networking, Secrets Manager, and CloudWatch monitoring. You use these modules to produce configuration dictionaries or export Terraform files, then apply them with your own infrastructure tooling.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Container Image](#container-image)
4. [AWSDeploymentManager](#awsdeploymentmanager)
5. [Kubernetes Manifests](#kubernetes-manifests)
6. [Deployment Workflow](#deployment-workflow)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Overview

The deployment infrastructure is provided as Python modules, not as ready-made YAML manifests or Helm charts. The key components are:

| Module | Purpose |
|--------|---------|
| `opifex.deployment.cloud.aws` | `AWSDeploymentManager` and `AWSConfig` for EKS/VPC/IAM configuration generation |
| `opifex.deployment.kubernetes` | `ManifestGenerator` for producing K8s Deployment, Service, and Ingress YAML |
| `opifex.deployment.server` | FastAPI model serving server |
| `opifex.deployment.core_serving` | `InferenceEngine`, `ModelRegistry`, `DeploymentConfig` |

Source: [`src/opifex/deployment/cloud/aws.py`](../../src/opifex/deployment/cloud/aws.py)

## Prerequisites

- An AWS account with permissions for EKS, EC2, VPC, IAM, CloudWatch
- AWS CLI v2 configured (`aws configure`)
- `kubectl` and `eksctl` installed
- Docker for building container images
- Python 3.12+ with `uv` and `opifex` installed locally

## Container Image

The project `Dockerfile` builds a GPU-ready image based on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. It installs Python 3.12, `uv`, and all project dependencies.

```bash
# Build locally
docker build -t opifex:latest .

# Tag and push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
docker tag opifex:latest <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/opifex:latest
docker push <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/opifex:latest
```

The default `CMD` runs the test suite. Override at runtime to start the serving API:

```bash
docker run --rm --gpus all opifex:latest python -m opifex.deployment.server
```

## AWSDeploymentManager

`AWSDeploymentManager` generates configuration dictionaries for AWS infrastructure. It does not call AWS APIs directly -- you export the configurations and apply them with Terraform, CDK, or the AWS CLI.

### AWSConfig

```python
from opifex.deployment.cloud.aws import AWSConfig, AWSDeploymentManager

config = AWSConfig(
    region="us-east-1",
    cluster_name="opifex-cluster",
    vpc_cidr="10.0.0.0/16",
    # Override defaults:
    node_group_config={
        "desired_size": 3,
        "max_size": 10,
        "min_size": 1,
        "instance_types": ["p3.2xlarge"],  # GPU instances for model serving
        "disk_size": 100,
        "capacity_type": "ON_DEMAND",
        "ami_type": "AL2_x86_64_GPU",
    },
)
```

`AWSConfig` fields:

- **`region`** -- AWS region (default: `us-east-1`)
- **`cluster_name`** -- EKS cluster name (default: `opifex-cluster`)
- **`vpc_cidr`** -- VPC CIDR block (default: `10.0.0.0/16`)
- **`node_group_config`** -- dict with `desired_size`, `max_size`, `min_size`, `instance_types`, `disk_size`, `capacity_type`, `ami_type`
- **`network_config`** -- dict with `availability_zones`, `private_subnets`, `public_subnets`, `enable_nat_gateway`, `enable_dns_hostnames`
- **`security_config`** -- dict with `enable_logging`, `log_types`, `enable_private_access`, `enable_public_access`, `public_access_cidrs`

### Configuration Generation Methods

```python
manager = AWSDeploymentManager(config)

# Each method returns a dict suitable for Terraform, CloudFormation, or inspection
eks_config = manager.generate_eks_cluster_config()
node_config = manager.generate_node_group_config()
iam_roles = manager.generate_iam_roles()
vpc_config = manager.generate_vpc_config()
secrets_config = manager.generate_secrets_manager_config()
cloudwatch_config = manager.generate_cloudwatch_config()
```

| Method | Returns |
|--------|---------|
| `generate_eks_cluster_config()` | EKS cluster definition (version, VPC config, encryption, logging) |
| `generate_node_group_config()` | Node group with scaling, instance types, AMI |
| `generate_iam_roles()` | Cluster role, node role, and service role with inline policies |
| `generate_vpc_config()` | VPC, subnets (public/private), NAT gateways, security groups |
| `generate_secrets_manager_config()` | Secrets for database passwords, API keys, OAuth credentials |
| `generate_cloudwatch_config()` | Log groups, CPU/memory alarms, dashboard definition |

### Export as Terraform

```python
from pathlib import Path

manager.export_terraform_config(Path("./terraform-aws"))
# Creates:
#   terraform-aws/main.tf       (provider + resources as JSON)
#   terraform-aws/variables.tf  (region, cluster_name, instance_types, desired_capacity)
#   terraform-aws/outputs.tf    (cluster_endpoint, CA cert, VPC ID, security group ID)
```

Then apply with standard Terraform:

```bash
cd terraform-aws
terraform init
terraform plan
terraform apply
```

## Kubernetes Manifests

Use `ManifestGenerator` from `opifex.deployment.kubernetes` to programmatically generate Kubernetes YAML for Deployment, Service, and Ingress resources.

```python
from pathlib import Path
from opifex.deployment.kubernetes import ManifestGenerator

gen = ManifestGenerator(
    namespace="opifex",
    app_name="opifex-api",
    image="<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/opifex:latest",
)

# Generate individual manifests as dicts
deployment = gen.generate_deployment(
    replicas=3,
    cpu_request="1",
    memory_request="4Gi",
    cpu_limit="4",
    memory_limit="8Gi",
    port=8080,
    environment_variables={
        "JAX_PLATFORMS": "gpu",
        "OPIFEX_PORT": "8080",
        "OPIFEX_WORKERS": "2",
    },
)

service = gen.generate_service(port=8080, target_port=8080, service_type="LoadBalancer")
ingress = gen.generate_ingress(host="opifex.example.com", service_port=8080)

# Or export all manifests to YAML files at once
gen.export_all_manifests(Path("./k8s-manifests"), replicas=3, service_port=8080)
```

The generated deployment includes:

- Liveness and readiness probes pointing to `/health`
- Default JAX environment variables (`JAX_PLATFORMS`, `XLA_PYTHON_CLIENT_MEM_FRACTION`)
- Resource requests and limits
- Optional `nodeSelector` for GPU node targeting

Source: [`src/opifex/deployment/kubernetes/manifest_generator.py`](../../src/opifex/deployment/kubernetes/manifest_generator.py)

Additional Kubernetes modules:

- `AutoScaler` -- HPA/VPA configuration
- `ResourceManager` -- namespace and quota management
- `KubernetesOrchestrator` -- orchestration for production deployments

Source: [`src/opifex/deployment/kubernetes/__init__.py`](../../src/opifex/deployment/kubernetes/__init__.py)

## Deployment Workflow

A typical end-to-end deployment:

1. **Build and push the container image** to ECR (see [Container Image](#container-image))
2. **Generate AWS infrastructure** using `AWSDeploymentManager.export_terraform_config()` and apply with Terraform
3. **Configure kubectl** for your EKS cluster:
   ```bash
   aws eks update-kubeconfig --region us-east-1 --name opifex-cluster
   ```
4. **Generate and apply K8s manifests**:
   ```bash
   # Generate from Python
   python -c "
   from pathlib import Path
   from opifex.deployment.kubernetes import ManifestGenerator
   gen = ManifestGenerator('opifex', 'opifex-api', '<ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/opifex:latest')
   gen.export_all_manifests(Path('./k8s-manifests'), replicas=3, service_port=8080)
   "

   # Apply
   kubectl create namespace opifex
   kubectl apply -f k8s-manifests/
   ```
5. **Verify** the deployment:
   ```bash
   kubectl get pods -n opifex
   kubectl logs -l app=opifex-api -n opifex
   ```

### GPU Node Configuration

For GPU workloads on EKS, use a GPU-enabled node group (`p3.2xlarge`, `p3.8xlarge`, or `g4dn.*` instances) with the `AL2_x86_64_GPU` AMI type. Install the NVIDIA device plugin:

```bash
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

Then add GPU resource requests to your deployment manifest:

```python
deployment = gen.generate_deployment(
    environment_variables={"JAX_PLATFORMS": "gpu"},
    node_selector={"accelerator": "nvidia-gpu"},
)
# Manually add GPU limits to the generated dict if needed:
deployment["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = "1"
```

## Monitoring

`AWSDeploymentManager.generate_cloudwatch_config()` produces:

- **Log groups**: `/aws/eks/<cluster>/cluster` (30-day retention) and `/aws/eks/<cluster>/application` (7-day retention)
- **Alarms**: High CPU (>80%) and high memory (>85%) with 5-minute evaluation periods
- **Dashboard**: CloudWatch dashboard with CPU and memory utilization widgets

The FastAPI server also exposes a `/metrics` endpoint returning request count, average latency, throughput, and uptime.

## Troubleshooting

### EKS Cluster Access

```bash
# Verify AWS identity
aws sts get-caller-identity

# Update kubeconfig
aws eks update-kubeconfig --region us-east-1 --name opifex-cluster

# Check nodes are ready
kubectl get nodes -o wide
```

### GPU Not Available in Pods

```bash
# Verify NVIDIA device plugin is running
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Check GPU allocation on nodes
kubectl describe nodes | grep -A5 "nvidia.com/gpu"

# Verify JAX sees GPU inside a pod
kubectl exec -it <pod-name> -n opifex -- python -c "import jax; print(jax.devices())"
```

### Container Image Issues

```bash
# Test image locally before pushing
docker run --rm opifex:latest python -c "import opifex; print('OK')"

# Check ECR authentication
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com
```
