# Opifex Deployment Troubleshooting Guide

This comprehensive troubleshooting guide helps you resolve common issues when deploying Opifex across different platforms (local, GCP, AWS, and other cloud providers).

## 📋 Table of Contents

1. [General Troubleshooting](#general-troubleshooting)
2. [Installation Issues](#installation-issues)
3. [Container Issues](#container-issues)
4. [Kubernetes Issues](#kubernetes-issues)
5. [Cloud-Specific Issues](#cloud-specific-issues)
6. [Performance Issues](#performance-issues)
7. [Security Issues](#security-issues)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [Recovery Procedures](#recovery-procedures)

## 🔧 General Troubleshooting

### Basic Diagnostic Commands

```bash
# Check system resources
df -h                    # Disk usage
free -h                  # Memory usage
top                      # CPU usage
lscpu                    # CPU information
nvidia-smi              # GPU information (if available)

# Check network connectivity
ping google.com
nslookup kubernetes.default.svc.cluster.local
curl -I http://localhost:8080/health

# Check service status
systemctl status docker
systemctl status kubelet
ps aux | grep -E "(python|java|node)"
```

### Environment Variables Check

```bash
# Check important environment variables
echo $PATH
echo $PYTHONPATH
echo $KUBECONFIG
echo $DOCKER_HOST
echo $JAX_PLATFORM_NAME

# Check Opifex-specific variables
echo $OPIFEX_ENV
echo $LOG_LEVEL
echo $DEBUG
```

### Log Collection

```bash
# System logs
journalctl -u docker.service --since "1 hour ago"
journalctl -u kubelet.service --since "1 hour ago"

# Application logs
docker logs <container-id>
kubectl logs <pod-name> -n <namespace>

# Export logs for analysis
kubectl logs -l app=opifex-api -n opifex --since=1h > opifex-logs.txt
docker logs opifex-api 2>&1 | tail -n 100 > docker-logs.txt
```

## 🐍 Installation Issues

### Python Environment Issues

#### Issue: Import Errors

```bash
# Symptoms
ImportError: No module named 'jax'
ImportError: No module named 'opifex'
ModuleNotFoundError: No module named 'flax'

# Diagnosis
python -c "import sys; print(sys.path)"
pip list | grep -E "(jax|flax|opifex)"
which python
which pip

# Solutions
# 1. Activate environment
source ./activate.sh

# 2. Reinstall dependencies
./setup.sh --force

# 3. Check Python version
python --version  # Should be 3.10+
```

#### Issue: JAX Installation Problems

```bash
# Symptoms
jax._src.lib.xla_bridge.XlaRuntimeError: CUDA not found
ImportError: jaxlib version X.X.X is too old for jax version Y.Y.Y

# Diagnosis
python -c "import jax; print(jax.__version__); print(jax.devices())"
nvidia-smi  # Check CUDA availability

# Solutions
# 1. Reinstall JAX with CUDA support
pip uninstall jax jaxlib
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 2. For CPU-only installation
pip install jax[cpu]

# 3. Check CUDA compatibility
nvcc --version
python -c "import jax; print(jax.local_devices())"
```

#### Issue: Virtual Environment Problems

```bash
# Symptoms
Command not found: uv
Permission denied: /usr/local/bin/python
Virtual environment not activating

# Solutions
# 1. Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Fix permissions
sudo chown -R $USER:$USER /usr/local/bin/python
chmod +x opifex-env/bin/activate

# 3. Recreate environment
./setup.sh --deep-clean
source ./activate.sh
```

### Dependency Conflicts

```bash
# Check for conflicts
pip check
uv pip check

# Resolve conflicts
pip install --upgrade --force-reinstall <package-name>
uv sync --force-reinstall

# Clean installation
pip freeze > requirements.txt
pip uninstall -r requirements.txt -y
uv sync
```

## 🐳 Container Issues

### Docker Problems

#### Issue: Docker Daemon Not Running

```bash
# Symptoms
Cannot connect to the Docker daemon
docker: command not found
permission denied while trying to connect to the Docker daemon socket

# Diagnosis
systemctl status docker
docker info
groups $USER

# Solutions
# 1. Start Docker service
sudo systemctl start docker
sudo systemctl enable docker

# 2. Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# 3. Fix socket permissions
sudo chmod 666 /var/run/docker.sock
```

#### Issue: Container Build Failures

```bash
# Symptoms
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
Step X/Y : RUN command failed

# Diagnosis
docker build --no-cache --progress=plain -t opifex:debug .
docker run -it --rm opifex:debug /bin/bash

# Solutions
# 1. Check Dockerfile syntax
docker build --dry-run -t opifex:test .

# 2. Use multi-stage build
FROM python:3.10-slim as builder
# ... build steps ...
FROM python:3.10-slim as runtime
COPY --from=builder /app /app

# 3. Fix base image
FROM python:3.10-slim
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
```

#### Issue: Container Runtime Problems

```bash
# Symptoms
Container exits immediately
Out of memory errors
Port binding failures

# Diagnosis
docker logs <container-id>
docker inspect <container-id>
docker stats <container-id>

# Solutions
# 1. Increase memory limits
docker run -m 4g opifex:latest

# 2. Fix port conflicts
docker run -p 8081:8080 opifex:latest
netstat -tlnp | grep :8080

# 3. Debug container
docker run -it --rm opifex:latest /bin/bash
docker exec -it <container-id> /bin/bash
```

### Docker Compose Issues

```bash
# Common issues and solutions
# 1. Service dependencies
depends_on:
  - redis
  - postgres
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
  interval: 30s
  timeout: 10s
  retries: 3

# 2. Volume mounting
volumes:
  - ./data:/app/data:rw
  - ./logs:/app/logs:rw

# 3. Network connectivity
networks:
  - opifex-network

# 4. Environment variables
env_file:
  - .env.development
environment:
  - DEBUG=true
```

## ☸️ Kubernetes Issues

### Cluster Connectivity

#### Issue: kubectl Not Working

```bash
# Symptoms
The connection to the server localhost:8080 was refused
Unable to connect to the server: dial tcp: lookup kubernetes.default.svc

# Diagnosis
kubectl cluster-info
kubectl config current-context
kubectl config view

# Solutions
# 1. Set correct context
kubectl config use-context <context-name>

# 2. Update kubeconfig
# For GKE
gcloud container clusters get-credentials <cluster-name> --zone <zone>

# For EKS
aws eks update-kubeconfig --region <region> --name <cluster-name>

# 3. Check cluster status
kubectl get nodes
kubectl get pods --all-namespaces
```

### Pod Issues

#### Issue: Pods Stuck in Pending State

```bash
# Symptoms
NAME                     READY   STATUS    RESTARTS   AGE
opifex-api-xxx-xxx        0/1     Pending   0          5m

# Diagnosis
kubectl describe pod <pod-name> -n <namespace>
kubectl get events --sort-by=.metadata.creationTimestamp -n <namespace>
kubectl top nodes

# Common causes and solutions
# 1. Insufficient resources
kubectl describe nodes
kubectl get pods --all-namespaces -o wide

# 2. Persistent volume issues
kubectl get pv,pvc -n <namespace>
kubectl describe pvc <pvc-name> -n <namespace>

# 3. Node selector issues
kubectl label nodes <node-name> workload-type=compute
kubectl get nodes --show-labels
```

#### Issue: Pods CrashLoopBackOff

```bash
# Symptoms
NAME                     READY   STATUS             RESTARTS   AGE
opifex-api-xxx-xxx        0/1     CrashLoopBackOff   5          5m

# Diagnosis
kubectl logs <pod-name> -n <namespace>
kubectl logs <pod-name> -n <namespace> --previous
kubectl describe pod <pod-name> -n <namespace>

# Solutions
# 1. Check resource limits
resources:
  limits:
    memory: "4Gi"
    cpu: "2"
  requests:
    memory: "2Gi"
    cpu: "1"

# 2. Fix liveness/readiness probes
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 60
  periodSeconds: 30

# 3. Debug container
kubectl run debug --image=busybox --rm -it -- sh
kubectl exec -it <pod-name> -- /bin/bash
```

### Service and Ingress Issues

#### Issue: Service Not Accessible

```bash
# Symptoms
curl: (7) Failed to connect to service.domain.com port 80: Connection refused
502 Bad Gateway

# Diagnosis
kubectl get services -n <namespace>
kubectl get endpoints -n <namespace>
kubectl describe service <service-name> -n <namespace>

# Solutions
# 1. Check service selector
kubectl get pods -l app=opifex-api -n <namespace>
kubectl describe service opifex-api-service -n <namespace>

# 2. Test internal connectivity
kubectl run test-pod --image=busybox --rm -it -- sh
# Inside pod: wget -qO- http://opifex-api-service:80/health

# 3. Check ingress configuration
kubectl get ingress -n <namespace>
kubectl describe ingress <ingress-name> -n <namespace>
```

### Storage Issues

```bash
# PVC stuck in Pending
kubectl get pvc -n <namespace>
kubectl describe pvc <pvc-name> -n <namespace>

# Solutions
# 1. Check storage class
kubectl get storageclass
kubectl describe storageclass <storage-class-name>

# 2. Create storage class if missing
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/gce-pd
parameters:
  type: pd-ssd
  zones: us-central1-a,us-central1-b

# 3. Check node affinity for local storage
kubectl get nodes --show-labels
kubectl describe pv <pv-name>
```

## ☁️ Cloud-Specific Issues

### GCP Issues

#### Issue: GKE Cluster Creation Fails

```bash
# Symptoms
ERROR: (gcloud.container.clusters.create) ResponseError: code=403
Insufficient quota

# Diagnosis
gcloud compute project-info describe --project=<project-id>
gcloud compute regions list
gcloud container clusters describe <cluster-name> --zone <zone>

# Solutions
# 1. Check quotas
gcloud compute project-info describe --project=<project-id> | grep -A 5 quotas

# 2. Request quota increase
gcloud compute regions describe <region>
# Use GCP Console to request quota increase

# 3. Use different machine types
gcloud container clusters create <cluster-name> \
    --machine-type=e2-standard-2 \
    --num-nodes=2
```

#### Issue: GCP Authentication Problems

```bash
# Symptoms
ERROR: (gcloud.auth.login) There was a problem with web authentication
Application Default Credentials not found

# Solutions
# 1. Re-authenticate
gcloud auth login
gcloud auth application-default login

# 2. Set service account
gcloud auth activate-service-account --key-file=<key-file>
export GOOGLE_APPLICATION_CREDENTIALS=<key-file>

# 3. Check permissions
gcloud projects get-iam-policy <project-id>
gcloud iam service-accounts list
```

### AWS Issues

#### Issue: EKS Cluster Access Denied

```bash
# Symptoms
error: You must be logged in to the server (Unauthorized)
An error occurred (AccessDenied) when calling the AssumeRole operation

# Diagnosis
aws sts get-caller-identity
aws eks describe-cluster --name <cluster-name> --region <region>
kubectl config view

# Solutions
# 1. Update kubeconfig
aws eks update-kubeconfig --region <region> --name <cluster-name>

# 2. Check IAM permissions
aws iam get-user
aws iam list-attached-user-policies --user-name <user-name>

# 3. Add user to cluster
eksctl create iamidentitymapping \
    --cluster <cluster-name> \
    --arn arn:aws:iam::<account-id>:user/<user-name> \
    --group system:masters \
    --username <user-name>
```

#### Issue: EKS Node Group Problems

```bash
# Symptoms
Nodes not joining cluster
NodeCreationFailure
InsufficientCapacity

# Diagnosis
aws eks describe-nodegroup --cluster-name <cluster-name> --nodegroup-name <nodegroup-name>
aws ec2 describe-instances --filters "Name=tag:kubernetes.io/cluster/<cluster-name>,Values=owned"

# Solutions
# 1. Check instance types availability
aws ec2 describe-instance-type-offerings --location-type availability-zone --filters Name=location,Values=<zone>

# 2. Update launch template
aws ec2 describe-launch-templates
aws ec2 modify-launch-template --launch-template-id <template-id>

# 3. Scale nodegroup
eksctl scale nodegroup --cluster=<cluster-name> --name=<nodegroup-name> --nodes=5
```

## 🚀 Performance Issues

### High Resource Usage

#### Issue: High Memory Usage

```bash
# Symptoms
OOMKilled pods
Node memory pressure
Slow application response

# Diagnosis
kubectl top nodes
kubectl top pods --all-namespaces
kubectl describe node <node-name>

# Solutions
# 1. Increase memory limits
resources:
  limits:
    memory: "8Gi"
  requests:
    memory: "4Gi"

# 2. Optimize application
# Add memory profiling
import psutil
import gc
gc.collect()

# 3. Add more nodes
kubectl scale deployment opifex-api --replicas=3
```

#### Issue: High CPU Usage

```bash
# Symptoms
CPU throttling
Slow processing
High load average

# Diagnosis
kubectl top pods -n <namespace>
kubectl describe pod <pod-name> -n <namespace>

# Solutions
# 1. Increase CPU limits
resources:
  limits:
    cpu: "4"
  requests:
    cpu: "2"

# 2. Optimize code
# Use JAX compilation
@jax.jit
def compute_function(x):
    return jax.numpy.sum(x**2)

# 3. Scale horizontally
kubectl autoscale deployment opifex-api --cpu-percent=70 --min=1 --max=10
```

### GPU Issues

#### Issue: GPU Not Available

```bash
# Symptoms
RuntimeError: No GPU/TPU found
jax._src.lib.xla_bridge.XlaRuntimeError: CUDA not found

# Diagnosis
nvidia-smi
kubectl get nodes -l accelerator=nvidia-tesla-k80
kubectl describe node <gpu-node-name>

# Solutions
# 1. Install GPU drivers
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# 2. Check GPU allocation
kubectl get nodes -o json | jq '.items[] | select(.status.capacity."nvidia.com/gpu" != null)'

# 3. Request GPU resources
resources:
  limits:
    nvidia.com/gpu: 1
  requests:
    nvidia.com/gpu: 1
```

### Network Performance

```bash
# Check network latency
kubectl run test-pod --image=busybox --rm -it -- sh
# Inside pod: ping <service-name>.<namespace>.svc.cluster.local

# Check bandwidth
kubectl run iperf-server --image=networkstatic/iperf3 -- iperf3 -s
kubectl run iperf-client --image=networkstatic/iperf3 --rm -it -- iperf3 -c iperf-server

# Optimize network
# Use nodeAffinity for co-location
affinity:
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: workload-type
          operator: In
          values: ["compute"]
```

## 🔒 Security Issues

### Authentication Problems

```bash
# RBAC issues
kubectl auth can-i create pods --as=system:serviceaccount:default:my-sa
kubectl get rolebindings,clusterrolebindings --all-namespaces

# Fix RBAC
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: opifex-role
rules:
- apiGroups: [""]
  resources: ["pods", "services"]
  verbs: ["get", "list", "create", "update", "delete"]
```

### Network Security

```bash
# Network policy issues
kubectl get networkpolicies -n <namespace>
kubectl describe networkpolicy <policy-name> -n <namespace>

# Test connectivity
kubectl run test-pod --image=busybox --rm -it -- sh
# Inside pod: nc -zv <service-name> <port>

# Fix network policies
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-opifex-api
spec:
  podSelector:
    matchLabels:
      app: opifex-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: opifex-worker
    ports:
    - protocol: TCP
      port: 8080
```

### Secrets Management

```bash
# Secrets not mounting
kubectl get secrets -n <namespace>
kubectl describe secret <secret-name> -n <namespace>

# Fix secrets mounting
volumeMounts:
- name: secret-volume
  mountPath: /etc/secrets
  readOnly: true
volumes:
- name: secret-volume
  secret:
    secretName: opifex-secrets
```

## 📊 Monitoring and Logging

### Metrics Collection Issues

```bash
# Prometheus not scraping
kubectl get servicemonitors -n monitoring
kubectl logs -l app=prometheus -n monitoring

# Fix service monitor
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: opifex-metrics
spec:
  selector:
    matchLabels:
      app: opifex-api
  endpoints:
  - port: metrics
    interval: 30s
```

### Log Aggregation Problems

```bash
# Logs not appearing
kubectl logs -l app=opifex-api -n <namespace>
kubectl describe pod <pod-name> -n <namespace>

# Fix logging
# Ensure proper log format
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Use structured logging
import structlog
logger = structlog.get_logger()
logger.info("Processing request", user_id=123, request_id="abc-123")
```

## 🔄 Recovery Procedures

### Cluster Recovery

```bash
# Backup cluster state
kubectl get all --all-namespaces -o yaml > cluster-backup.yaml
kubectl get pv -o yaml > pv-backup.yaml

# Restore from backup
kubectl apply -f cluster-backup.yaml
kubectl apply -f pv-backup.yaml

# Disaster recovery
# 1. Recreate cluster
eksctl create cluster -f cluster-config.yaml

# 2. Restore persistent volumes
kubectl apply -f pv-backup.yaml

# 3. Restore applications
kubectl apply -f opifex-deployment.yaml
```

### Data Recovery

```bash
# Backup persistent volumes
kubectl get pvc -n <namespace>
aws ec2 create-snapshot --volume-id <volume-id> --description "Backup"

# Restore from snapshot
aws ec2 create-volume --snapshot-id <snapshot-id> --availability-zone <zone>
kubectl apply -f restored-pvc.yaml
```

### Application Recovery

```bash
# Rollback deployment
kubectl rollout undo deployment/opifex-api -n <namespace>
kubectl rollout history deployment/opifex-api -n <namespace>

# Scale to zero and back
kubectl scale deployment opifex-api --replicas=0 -n <namespace>
kubectl scale deployment opifex-api --replicas=3 -n <namespace>

# Restart pods
kubectl delete pod -l app=opifex-api -n <namespace>
kubectl rollout restart deployment/opifex-api -n <namespace>
```

## 🆘 Getting Help

### Diagnostic Information Collection

```bash
# Create diagnostic bundle
#!/bin/bash
NAMESPACE="opifex"
OUTPUT_DIR="opifex-diagnostics-$(date +%Y%m%d-%H%M%S)"
mkdir -p $OUTPUT_DIR

# Cluster information
kubectl cluster-info > $OUTPUT_DIR/cluster-info.txt
kubectl get nodes -o wide > $OUTPUT_DIR/nodes.txt
kubectl get pods --all-namespaces -o wide > $OUTPUT_DIR/all-pods.txt

# Application-specific information
kubectl get all -n $NAMESPACE -o yaml > $OUTPUT_DIR/opifex-resources.yaml
kubectl logs -l app=opifex-api -n $NAMESPACE > $OUTPUT_DIR/opifex-api-logs.txt
kubectl describe pods -l app=opifex-api -n $NAMESPACE > $OUTPUT_DIR/opifex-api-describe.txt

# System information
kubectl get events --sort-by=.metadata.creationTimestamp -n $NAMESPACE > $OUTPUT_DIR/events.txt
kubectl top nodes > $OUTPUT_DIR/node-usage.txt
kubectl top pods -n $NAMESPACE > $OUTPUT_DIR/pod-usage.txt

# Create archive
tar -czf $OUTPUT_DIR.tar.gz $OUTPUT_DIR
echo "Diagnostic bundle created: $OUTPUT_DIR.tar.gz"
```

### Support Channels

1. **GitHub Issues**: [Opifex Issues](https://github.com/opifex-org/opifex/issues)
2. **Community Forum**: [GitHub Discussions](https://github.com/opifex-org/opifex/discussions)
3. **Documentation**: [Opifex Docs](https://opifex.readthedocs.io/)
4. **Cloud Provider Support**:
   - [GCP Support](https://cloud.google.com/support)
   - [AWS Support](https://aws.amazon.com/support/)

### Escalation Process

1. **Level 1**: Check this troubleshooting guide
2. **Level 2**: Search existing GitHub issues
3. **Level 3**: Create new GitHub issue with diagnostic bundle
4. **Level 4**: Contact enterprise support (if available)

---

**Remember**: Always include diagnostic information, error messages, and steps to reproduce when seeking help. This troubleshooting guide covers the most common issues, but every deployment is unique.
