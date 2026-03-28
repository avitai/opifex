# Deployment Troubleshooting Guide

This guide covers common issues when developing with or deploying the Opifex framework across local, Docker, and cloud environments.

## Table of Contents

1. [General Diagnostics](#general-diagnostics)
2. [Installation Issues](#installation-issues)
3. [JAX and GPU Issues](#jax-and-gpu-issues)
4. [Container Issues](#container-issues)
5. [Kubernetes Issues](#kubernetes-issues)
6. [Model Serving Issues](#model-serving-issues)
7. [Cloud-Specific Issues](#cloud-specific-issues)
8. [Performance Issues](#performance-issues)

## General Diagnostics

### System Information

```bash
# System resources
df -h                    # Disk usage
free -h                  # Memory usage
top                      # CPU usage
nvidia-smi               # GPU information (if available)

# Check environment
which python
python --version
uv --version
```

### Environment Check

```bash
# Activate environment first
source ./activate.sh

# Verify key environment variables
echo $VIRTUAL_ENV        # Should point to .venv
echo $JAX_PLATFORMS      # cpu, gpu, or unset (auto-detect)

# Verify Opifex is importable
python -c "import opifex; print('OK')"
python -c "import jax; print('JAX version:', jax.__version__); print('Devices:', jax.devices())"
```

## Installation Issues

### Virtual Environment Problems

**Symptom**: `Command not found: uv` or environment not activating.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Recreate the environment from scratch
./setup.sh --force-clean
source ./activate.sh
```

**Symptom**: `Permission denied` on `.venv/bin/activate`.

```bash
chmod +x .venv/bin/activate
source ./activate.sh
```

### Import Errors

**Symptom**: `ImportError: No module named 'jax'` or `No module named 'opifex'`.

```bash
# Ensure the environment is activated
source ./activate.sh

# Verify you are using the correct Python
which python  # Should be .venv/bin/python

# Reinstall all dependencies
./setup.sh --force-clean
source ./activate.sh

# Verify
python -c "import jax; print(jax.__version__)"
python -c "import opifex; print('OK')"
```

### Dependency Conflicts

```bash
# Check for dependency issues
uv pip check

# Force a clean sync
./setup.sh --recreate
source ./activate.sh

# Verify lock file is up to date
uv lock --check
```

## JAX and GPU Issues

### GPU Not Detected

**Symptom**: `jax.devices()` returns only CPU devices, or `RuntimeError: No GPU/TPU found`.

```bash
# Check NVIDIA driver
nvidia-smi

# Check what JAX sees
source ./activate.sh
python -c "import jax; print(jax.devices())"

# Reinstall with GPU support
./setup.sh --recreate --backend cuda12
source ./activate.sh
python -c "import jax; print(jax.devices())"
```

### JAX Version Mismatch

**Symptom**: `ImportError: jaxlib version X.X.X is too old for jax version Y.Y.Y`.

```bash
# The project pins compatible JAX/jaxlib versions via uv.lock.
# A clean sync resolves version mismatches.
./setup.sh --recreate
source ./activate.sh
```

### CUDA Errors at Runtime

**Symptom**: `XlaRuntimeError: CUDA not found` or CUDA library loading failures.

Opifex uses JAX's locally-bundled CUDA runtime (installed via the `gpu` extra). You do not need a system CUDA toolkit or `LD_LIBRARY_PATH` injection. If you see CUDA errors:

```bash
# Verify the GPU extra is installed
./setup.sh --backend cuda12
source ./activate.sh

# Check NVIDIA driver compatibility (driver must support CUDA 12.x)
nvidia-smi  # Check "CUDA Version" in the top-right of the output

# Force CPU mode as a workaround
JAX_PLATFORMS=cpu python your_script.py
```

### GPU Memory Issues

**Symptom**: `XLA_PYTHON_CLIENT_MEM_FRACTION` errors or out-of-memory.

```bash
# Prevent full GPU memory preallocation (set in Dockerfile by default)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75

# For very constrained GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
```

## Container Issues

### Docker Daemon Not Running

```bash
# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group (avoids needing sudo)
sudo usermod -aG docker $USER
newgrp docker
```

### Container Build Failures

```bash
# Build with full output for debugging
docker build --no-cache --progress=plain -t opifex:debug .

# Enter a failed build layer for inspection
docker run -it --rm opifex:debug /bin/bash
```

### Docker GPU Access

**Symptom**: `docker: Error response from daemon: could not select device driver`

```bash
# Install NVIDIA container toolkit
# Ubuntu/Debian:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

# Verify GPU access inside container
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### Docker Compose Services

The `docker-compose.yml` defines two services:

```bash
# GPU runtime
docker compose up opifex-gpu

# CPU-only runtime
docker compose up opifex-cpu

# Run tests
docker compose run opifex-cpu pytest tests/ -x -q

# View logs
docker compose logs opifex-gpu
docker compose logs opifex-cpu
```

## Kubernetes Issues

### kubectl Not Connecting

```bash
# GKE
gcloud container clusters get-credentials <cluster-name> --zone <zone>

# EKS
aws eks update-kubeconfig --region <region> --name <cluster-name>

# Verify
kubectl get nodes
```

### Pods Stuck in Pending

**Common causes**: insufficient resources, unschedulable GPU requests, or PVC binding failures.

```bash
kubectl describe pod <pod-name> -n <namespace>
kubectl get events --sort-by=.metadata.creationTimestamp -n <namespace>
kubectl top nodes
```

### Pods in CrashLoopBackOff

```bash
# Check current and previous logs
kubectl logs <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace> --previous

# Common fixes:
# - Increase memory limits (JAX can use significant memory at JIT compilation time)
# - Increase initialDelaySeconds on liveness probes (model loading takes time)
# - Verify the container image works locally first
```

### GPU Not Available in Pods

```bash
# Install NVIDIA device plugin (if not already installed)
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# Verify GPU allocation
kubectl describe nodes | grep -A5 "nvidia.com/gpu"

# Check pod GPU request
kubectl describe pod <pod-name> -n <namespace> | grep -A3 "Limits"
```

## Model Serving Issues

### Server Not Starting

The FastAPI server is at `opifex.deployment.server`:

```bash
source ./activate.sh

# Start with debug logging
OPIFEX_LOG_LEVEL=debug python -m opifex.deployment.server

# Check if port is in use
lsof -i :8080

# Use a different port
OPIFEX_PORT=9090 python -m opifex.deployment.server
```

### Health Check Failing

```bash
# Test the health endpoint
curl http://localhost:8080/health

# Expected response:
# {"status": "healthy", "service": "opifex-model-serving", "version": "1.0.0", ...}
```

If the health check returns unhealthy, check:
- Is the inference engine initialized? (requires loading a model via the API)
- Is the model registry path writable? (set via `OPIFEX_MODEL_REGISTRY`)

### Prediction Errors

```bash
# Test prediction endpoint
curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"data": [[1.0, 2.0, 3.0]]}'
```

Common errors:
- **503**: Model not loaded. The inference engine must have a model loaded via `InferenceEngine.load_model()` before predictions work.
- **400**: Missing `data` field in request body.
- **400**: Input shape mismatch with loaded model's expected input shape.

## Cloud-Specific Issues

### GCP: Quota Exceeded

```bash
gcloud compute project-info describe --project=<PROJECT_ID>
gcloud compute regions describe <REGION> --project=<PROJECT_ID>
# Request quota increase via the GCP Console
```

### GCP: Authentication

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <PROJECT_ID>
```

### AWS: Access Denied

```bash
# Verify identity
aws sts get-caller-identity

# Check IAM permissions
aws iam list-attached-user-policies --user-name <USER>

# Update kubeconfig for EKS
aws eks update-kubeconfig --region <REGION> --name <CLUSTER_NAME>
```

### AWS: ECR Push Failures

```bash
# Re-authenticate to ECR
aws ecr get-login-password --region <REGION> | \
    docker login --username AWS --password-stdin <ACCOUNT_ID>.dkr.ecr.<REGION>.amazonaws.com
```

## Performance Issues

### High Memory Usage

```bash
# Monitor system memory
free -h

# Monitor GPU memory
nvidia-smi

# Reduce JAX GPU memory fraction
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5

# In Kubernetes, increase pod memory limits:
# resources.limits.memory: "8Gi"
```

### Slow Inference

```bash
# Ensure JIT compilation has completed (first call is slow)
# The InferenceEngine warm-ups JIT on load_model()

# Verify GPU is being used
python -c "import jax; print(jax.devices())"

# Check the /metrics endpoint for latency data
curl http://localhost:8080/metrics
```

### JAX Compilation Overhead

The first inference call after loading a model triggers JIT compilation, which can take several seconds. Subsequent calls are fast. The `InferenceEngine` performs a warm-up call during `load_model()` to front-load this cost. If you change batch sizes at runtime, expect recompilation overhead.

## Diagnostic Bundle

For filing issues, collect this information:

```bash
#!/bin/bash
OUTPUT_DIR="opifex-diagnostics-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$OUTPUT_DIR"

# System info
python --version > "$OUTPUT_DIR/python-version.txt" 2>&1
uv pip list > "$OUTPUT_DIR/pip-list.txt" 2>&1
nvidia-smi > "$OUTPUT_DIR/nvidia-smi.txt" 2>&1 || echo "No GPU" > "$OUTPUT_DIR/nvidia-smi.txt"

# JAX info
python -c "import jax; print('JAX:', jax.__version__); print('Devices:', jax.devices())" > "$OUTPUT_DIR/jax-info.txt" 2>&1

# If using Kubernetes
kubectl get pods --all-namespaces -o wide > "$OUTPUT_DIR/pods.txt" 2>&1 || true
kubectl get nodes -o wide > "$OUTPUT_DIR/nodes.txt" 2>&1 || true

tar -czf "$OUTPUT_DIR.tar.gz" "$OUTPUT_DIR"
echo "Diagnostic bundle: $OUTPUT_DIR.tar.gz"
```
