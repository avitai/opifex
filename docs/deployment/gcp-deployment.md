# GCP Deployment

This guide covers deploying Opifex models on Google Cloud Platform. Opifex provides Python modules for generating GCP infrastructure configurations -- GKE clusters, Cloud IAM, VPC networking, Secret Manager, and Cloud Monitoring. You use these modules to produce configuration dictionaries or export Terraform files, then apply them with your own infrastructure tooling.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Container Image](#container-image)
4. [GCPDeploymentManager](#gcpdeploymentmanager)
5. [Kubernetes Manifests](#kubernetes-manifests)
6. [Deployment Workflow](#deployment-workflow)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)

## Overview

The deployment infrastructure is provided as Python modules, not as ready-made YAML manifests or Helm charts. The key components are:

| Module | Purpose |
|--------|---------|
| `opifex.deployment.cloud.gcp` | `GCPDeploymentManager` and `GCPConfig` for GKE/VPC/IAM configuration generation |
| `opifex.deployment.kubernetes` | `ManifestGenerator` for producing K8s Deployment, Service, and Ingress YAML |
| `opifex.deployment.server` | FastAPI model serving server |
| `opifex.deployment.core_serving` | `InferenceEngine`, `ModelRegistry`, `DeploymentConfig` |

Source: [`src/opifex/deployment/cloud/gcp.py`](../../src/opifex/deployment/cloud/gcp.py)

## Prerequisites

- A GCP project with billing enabled
- Google Cloud SDK (`gcloud`) configured (`gcloud auth login`)
- `kubectl` installed (`gcloud components install kubectl`)
- Docker for building container images
- Python 3.12+ with `uv` and `opifex` installed locally

## Container Image

The project `Dockerfile` builds a GPU-ready image based on `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`. It installs Python 3.12, `uv`, and all project dependencies.

```bash
# Build locally
docker build -t opifex:latest .

# Tag and push to GCR
docker tag opifex:latest gcr.io/<PROJECT_ID>/opifex:latest
docker push gcr.io/<PROJECT_ID>/opifex:latest

# Or use Artifact Registry
docker tag opifex:latest <REGION>-docker.pkg.dev/<PROJECT_ID>/opifex/opifex:latest
docker push <REGION>-docker.pkg.dev/<PROJECT_ID>/opifex/opifex:latest
```

The default `CMD` runs the test suite. Override at runtime to start the serving API:

```bash
docker run --rm --gpus all opifex:latest python -m opifex.deployment.server
```

## GCPDeploymentManager

`GCPDeploymentManager` generates configuration dictionaries for GCP infrastructure. It does not call GCP APIs directly -- you export the configurations and apply them with Terraform, Deployment Manager, or `gcloud` commands.

### GCPConfig

```python
from opifex.deployment.cloud.gcp import GCPConfig, GCPDeploymentManager

config = GCPConfig(
    project_id="my-opifex-project",
    region="us-central1",
    zone="us-central1-a",
    cluster_name="opifex-cluster",
    # Override defaults:
    node_pool_config={
        "initial_node_count": 3,
        "machine_type": "n1-standard-8",
        "disk_size_gb": 100,
        "auto_scaling": {"min_node_count": 1, "max_node_count": 10},
        "gpu_config": {
            "accelerator_type": "nvidia-tesla-t4",
            "accelerator_count": 1,
        },
    },
)
```

`GCPConfig` fields:

- **`project_id`** (required) -- GCP project ID
- **`region`** -- GCP region (default: `us-central1`)
- **`zone`** -- GCP zone (default: `us-central1-a`)
- **`cluster_name`** -- GKE cluster name (default: `opifex-cluster`)
- **`node_pool_config`** -- dict with `initial_node_count`, `machine_type`, `disk_size_gb`, `auto_scaling`, `gpu_config`
- **`network_config`** -- dict with `network`, `subnetwork`, `authorized_networks`, `enable_private_nodes`
- **`security_config`** -- dict with `enable_workload_identity`, `enable_network_policy`, `enable_pod_security_policy`, `master_authorized_networks`

### GPU Recommendations

The Dockerfile uses CUDA 12.4, which requires compute capability 5.0+. Recommended GPUs:

| GPU | Accelerator Type | Use Case |
|-----|-----------------|----------|
| **T4** | `nvidia-tesla-t4` | Cost-effective inference |
| **V100** | `nvidia-tesla-v100` | Training and inference |
| **A100** | `nvidia-a100-80gb` | Large-scale training |

**Note**: K80 GPUs are incompatible with CUDA 12.x and should not be used.

### Configuration Generation Methods

```python
manager = GCPDeploymentManager(config)

# Each method returns a dict suitable for Terraform or inspection
gke_config = manager.generate_gke_cluster_config()
iam_policy = manager.generate_iam_policy()
vpc_config = manager.generate_vpc_config()
secrets_config = manager.generate_secret_manager_config()
monitoring_config = manager.generate_monitoring_config()
```

| Method | Returns |
|--------|---------|
| `generate_gke_cluster_config()` | GKE cluster definition (node config, networking, workload identity, private cluster) |
| `generate_iam_policy()` | IAM bindings for container developer, secret accessor, monitoring editor, log writer |
| `generate_vpc_config()` | VPC, subnet with secondary IP ranges, firewall rules (internal, SSH, API) |
| `generate_secret_manager_config()` | Secret definitions with replication policies |
| `generate_monitoring_config()` | Notification channels, alert policies (CPU >80%, memory >90%) |

### Export as Terraform

```python
from pathlib import Path

manager.export_terraform_config(Path("./terraform-gcp"))
# Creates:
#   terraform-gcp/main.tf       (provider + resources as JSON)
#   terraform-gcp/variables.tf  (project_id, region, zone, cluster_name)
#   terraform-gcp/outputs.tf    (cluster_endpoint, CA cert, VPC name, subnet name)
```

Then apply with standard Terraform:

```bash
cd terraform-gcp
terraform init
terraform plan -var="project_id=my-opifex-project"
terraform apply -var="project_id=my-opifex-project"
```

## Kubernetes Manifests

Use `ManifestGenerator` from `opifex.deployment.kubernetes` to programmatically generate Kubernetes YAML for Deployment, Service, and Ingress resources.

```python
from pathlib import Path
from opifex.deployment.kubernetes import ManifestGenerator

gen = ManifestGenerator(
    namespace="opifex",
    app_name="opifex-api",
    image="gcr.io/<PROJECT_ID>/opifex:latest",
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

1. **Enable required APIs**:
   ```bash
   gcloud services enable container.googleapis.com compute.googleapis.com
   ```

2. **Build and push the container image** to GCR or Artifact Registry (see [Container Image](#container-image))

3. **Generate GCP infrastructure** using `GCPDeploymentManager.export_terraform_config()` and apply with Terraform

4. **Configure kubectl** for your GKE cluster:
   ```bash
   gcloud container clusters get-credentials opifex-cluster --zone us-central1-a
   ```

5. **Install NVIDIA GPU device plugin** (for GPU node pools):
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
   ```

6. **Generate and apply K8s manifests**:
   ```bash
   python -c "
   from pathlib import Path
   from opifex.deployment.kubernetes import ManifestGenerator
   gen = ManifestGenerator('opifex', 'opifex-api', 'gcr.io/<PROJECT_ID>/opifex:latest')
   gen.export_all_manifests(Path('./k8s-manifests'), replicas=3, service_port=8080)
   "

   kubectl create namespace opifex
   kubectl apply -f k8s-manifests/
   ```

7. **Verify** the deployment:
   ```bash
   kubectl get pods -n opifex
   kubectl logs -l app=opifex-api -n opifex
   ```

## Monitoring

`GCPDeploymentManager.generate_monitoring_config()` produces:

- **Notification channels**: email-based alert routing
- **Alert policies**: High CPU (>80% for 5 minutes) and high memory (>90% for 5 minutes) on `k8s_container` resources

The FastAPI server also exposes a `/metrics` endpoint returning request count, average latency, throughput, and uptime.

For GCP-native monitoring:

```bash
# Set up log-based metrics
gcloud logging metrics create opifex_errors \
    --description="Opifex application errors" \
    --log-filter='resource.type="k8s_container" AND severity="ERROR"'
```

## Troubleshooting

### GKE Cluster Access

```bash
# Verify authentication
gcloud auth list

# Update kubeconfig
gcloud container clusters get-credentials opifex-cluster --zone us-central1-a

# Check nodes are ready
kubectl get nodes -o wide
```

### GPU Not Available in Pods

```bash
# Check GPU node pool exists
gcloud container node-pools list --cluster=opifex-cluster --zone=us-central1-a

# Verify NVIDIA device plugin is running
kubectl get pods -n kube-system -l k8s-app=nvidia-gpu-device-plugin

# Check GPU allocation on nodes
kubectl describe nodes | grep -A5 "nvidia.com/gpu"

# Verify JAX sees GPU inside a pod
kubectl exec -it <pod-name> -n opifex -- python -c "import jax; print(jax.devices())"
```

### Quota Issues

```bash
# Check project quotas
gcloud compute project-info describe --project=<PROJECT_ID>

# Check regional GPU quota
gcloud compute regions describe us-central1 --project=<PROJECT_ID>
```

### Container Image Issues

```bash
# Test image locally before pushing
docker run --rm opifex:latest python -c "import opifex; print('OK')"

# Authenticate to GCR
gcloud auth configure-docker
```
