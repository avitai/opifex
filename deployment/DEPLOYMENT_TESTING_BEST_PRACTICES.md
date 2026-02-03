# Opifex Deployment Testing Best Practices

**Version**: 1.0.0
**Foundation**: 158/158 L2O tests passing, enterprise-grade architecture

## 🎯 Overview

This document outlines comprehensive best practices for testing deployment configurations in the Opifex framework, ensuring reliable, secure, and scalable deployments across different environments.

## 🔍 **Dry Run Testing Strategies**

### 1. **Kubernetes Dry Run Validation**

The most important "dry run" approach for Kubernetes deployments:

```bash
# Validate individual configurations
kubectl apply --dry-run=client -f deployment/kubernetes/base/deployment-api.yaml

# Validate entire directories
kubectl apply --dry-run=client -k deployment/kubernetes/base/

# Validate with server-side validation
kubectl apply --dry-run=server -k deployment/kubernetes/base/
```

**Key Differences:**

- `--dry-run=client`: Validates locally without contacting the API server
- `--dry-run=server`: Validates against the actual cluster (requires cluster access)

### 2. **Kustomize Validation**

For Kustomize-based configurations:

```bash
# Build and validate Kustomize configurations
kustomize build deployment/kubernetes/base/ --dry-run

# Validate with kubectl
kubectl apply --dry-run=client -k deployment/kubernetes/base/
```

### 3. **Helm Chart Validation**

For Helm-based deployments:

```bash
# Lint Helm charts
helm lint deployment/kubernetes/gpu-operator/

# Template and validate
helm template deployment/kubernetes/gpu-operator/ | kubectl apply --dry-run=client -f -
```

## 🧪 **Comprehensive Testing Approaches**

### 1. **Syntax Validation**

```bash
# Validate YAML syntax
python3 -c "import yaml; yaml.safe_load(open('deployment/kubernetes/base/deployment-api.yaml'))"

# Validate all YAML files
find deployment/ -name "*.yaml" -exec python3 -c "import yaml; yaml.safe_load(open('{}'))" \;
```

### 2. **Resource Requirements Validation**

```bash
# Check for missing resource limits
find deployment/ -name "*.yaml" -exec grep -l "kind: Deployment" {} \; | \
xargs -I {} sh -c 'echo "Checking {}" && grep -A 10 "containers:" {} | grep -E "(resources:|limits:|requests:)" || echo "WARNING: Missing resource specifications in {}"'
```

### 3. **Security Configuration Validation**

```bash
# Check for security contexts
find deployment/ -name "*.yaml" -exec grep -l "kind: Deployment" {} \; | \
xargs -I {} sh -c 'grep -q "securityContext:" {} || echo "WARNING: Missing security context in {}"'

# Check for network policies
find deployment/ -name "*.yaml" | grep -E "network-policies" | wc -l
```

## 🏗️ **Environment-Specific Testing**

### 1. **Development Environment**

```bash
# Test development configuration
ENVIRONMENT=development ./deployment/test-deployment.sh

# Validate with minimal resources
kubectl apply --dry-run=client -k deployment/kubernetes/base/ \
  --set resources.requests.cpu=100m \
  --set resources.requests.memory=128Mi
```

### 2. **Staging Environment**

```bash
# Test staging configuration
ENVIRONMENT=staging ./deployment/test-deployment.sh

# Validate with production-like settings
kubectl apply --dry-run=client -k deployment/kubernetes/base/ \
  --set replicas=2 \
  --set resources.requests.cpu=500m
```

### 3. **Production Environment**

```bash
# Test production configuration
ENVIRONMENT=production ./deployment/test-deployment.sh

# Validate with full production settings
kubectl apply --dry-run=client -k deployment/kubernetes/base/ \
  --set replicas=3 \
  --set resources.requests.cpu=1000m \
  --set resources.requests.memory=2Gi
```

## 🔧 **Automated Testing Tools**

### 1. **Makefile Targets**

```bash
# Run all validations
make validate-complete

# Validate specific components
make validate-kustomize
make validate-helm
make validate-syntax
make validate-resources
make validate-security
```

### 2. **Custom Testing Script**

```bash
# Run comprehensive testing
./deployment/test-deployment.sh

# Test specific environment
ENVIRONMENT=production ./deployment/test-deployment.sh
```

## 📊 **Testing Phases**

### Phase 1: Pre-Deployment Validation

1. **Syntax Validation**

   ```bash
   make validate-syntax
   ```

2. **Configuration Validation**

   ```bash
   make validate-kustomize
   make validate-helm
   ```

3. **Resource Validation**

   ```bash
   make validate-resources
   ```

4. **Security Validation**

   ```bash
   make validate-security
   ```

### Phase 2: Dry Run Deployment

1. **Client-Side Validation**

   ```bash
   kubectl apply --dry-run=client -k deployment/kubernetes/base/
   ```

2. **Server-Side Validation** (if cluster available)

   ```bash
   kubectl apply --dry-run=server -k deployment/kubernetes/base/
   ```

3. **Environment-Specific Validation**

   ```bash
   kubectl apply --dry-run=client -k deployment/kubernetes/overlays/production/
   ```

### Phase 3: Post-Deployment Validation

1. **Health Checks**

   ```bash
   make test-deployment
   ```

2. **Resource Monitoring**

   ```bash
   make status
   ```

3. **Log Analysis**

   ```bash
   make logs
   ```

## 🛡️ **Security Testing**

### 1. **RBAC Validation**

```bash
# Validate RBAC configurations
kubectl apply --dry-run=client -k deployment/security/rbac/

# Check for excessive permissions
find deployment/security/rbac/ -name "*.yaml" -exec grep -l "rules:" {} \; | \
xargs -I {} sh -c 'echo "Checking {}" && grep -A 20 "rules:" {}'
```

### 2. **Network Policy Validation**

```bash
# Validate network policies
kubectl apply --dry-run=client -k deployment/security/network-policies/

# Check policy coverage
kubectl get networkpolicies --all-namespaces
```

### 3. **Secret Management**

```bash
# Validate secret configurations
kubectl apply --dry-run=client -f deployment/kubernetes/base/secret.yaml

# Check for hardcoded secrets
find deployment/ -name "*.yaml" -exec grep -l "password\|secret\|key" {} \; | \
xargs -I {} sh -c 'echo "Checking {}" && grep -E "(password|secret|key)" {}'
```

## 📈 **Performance Testing**

### 1. **Resource Usage Validation**

```bash
# Check resource requests and limits
find deployment/ -name "*.yaml" -exec grep -l "kind: Deployment" {} \; | \
xargs -I {} sh -c 'echo "Resource usage in {}:" && grep -A 10 "resources:" {}'
```

### 2. **Scalability Testing**

```bash
# Test horizontal pod autoscaler
kubectl apply --dry-run=client -k deployment/scalability/hpa/

# Test cluster autoscaler
kubectl apply --dry-run=client -k deployment/scalability/cluster-autoscaler/
```

## 🔍 **Monitoring and Observability Testing**

### 1. **Prometheus Configuration**

```bash
# Validate Prometheus configuration
kubectl apply --dry-run=client -k deployment/monitoring/prometheus/

# Check alerting rules
find deployment/monitoring/ -name "*.yaml" -exec grep -l "alert:" {} \;
```

### 2. **Grafana Configuration**

```bash
# Validate Grafana configuration
kubectl apply --dry-run=client -k deployment/monitoring/grafana/

# Check dashboard configurations
find deployment/monitoring/grafana/ -name "*.json" -exec echo "Dashboard: {}" \;
```

## 🚀 **GPU-Specific Testing**

### 1. **GPU Operator Validation**

```bash
# Validate GPU operator configuration
kubectl apply --dry-run=client -k deployment/kubernetes/gpu-operator/

# Test GPU resource requests
kubectl apply --dry-run=client -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
spec:
  containers:
  - name: gpu-test
    image: nvidia/cuda:11.8-base-ubuntu20.04
    resources:
      limits:
        nvidia.com/gpu: 1
EOF
```

### 2. **GPU Monitoring Validation**

```bash
# Validate GPU monitoring
kubectl apply --dry-run=client -k deployment/monitoring/prometheus/

# Check for GPU metrics
grep -r "nvidia" deployment/monitoring/
```

## 📋 **Testing Checklist**

### Pre-Deployment Checklist

- [ ] YAML syntax validation passed
- [ ] Kubernetes configuration validation passed
- [ ] Kustomize configuration validation passed
- [ ] Resource requirements defined
- [ ] Security contexts configured
- [ ] Network policies defined
- [ ] Environment-specific configuration validated
- [ ] GPU configuration validated (if applicable)

### Deployment Checklist

- [ ] Dry-run validation passed
- [ ] Namespace creation validated
- [ ] Service account creation validated
- [ ] ConfigMap and Secret creation validated
- [ ] Deployment creation validated
- [ ] Service creation validated
- [ ] Ingress creation validated (if applicable)

### Post-Deployment Checklist

- [ ] Pod status is Running
- [ ] Service endpoints are available
- [ ] Health checks are passing
- [ ] Monitoring is working
- [ ] Logs are accessible
- [ ] Metrics are being collected

## 🚨 **Common Issues and Solutions**

### 1. **Missing Resource Limits**

**Issue**: Deployments without resource limits can cause cluster instability.

**Solution**:

```yaml
resources:
  requests:
    cpu: "100m"
    memory: "128Mi"
  limits:
    cpu: "1000m"
    memory: "1Gi"
```

### 2. **Missing Security Contexts**

**Issue**: Pods running as root can pose security risks.

**Solution**:

```yaml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
```

### 3. **Incorrect Image Tags**

**Issue**: Using `latest` tags can cause deployment issues.

**Solution**: Always use specific version tags:

```yaml
image: opifex/framework:1.0.0
```

## 🧬 **Opifex-Specific Testing Practices**

### 1. **L2O Framework Validation**

The Learn-to-Optimize framework is the core of Opifex's optimization capabilities. Test it thoroughly:

```bash
# Test L2O framework availability
make test-l2o-framework

# Validate L2O components in development
find opifex/optimization/l2o/ -name "*.py" -exec python -m py_compile {} \;

# Test L2O integration with neural operators
kubectl apply --dry-run=client -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: l2o-integration-test
spec:
  containers:
  - name: test
    image: opifex/framework:latest
    command: ["python", "-c", "from opifex.optimization.l2o import L2OEngine; from opifex.neural.operators import FNO; print('L2O + Neural Operators Ready')"]
    resources:
      requests:
        cpu: "1"
        memory: "2Gi"
        nvidia.com/gpu: "1"
EOF
```

### 2. **Neural Operator Framework Testing**

Validate the neural operator ecosystem components:

```bash
# Test neural operator framework
make test-neural-operators

# Validate FNO, SFNO, and DeepONet availability
python -c "
from opifex.neural.operators import FNO, SFNO, DeepONet
from opifex.neural.operators.layers import SpectralConv2d
print('All neural operators available')
"

# Test GPU compatibility for neural operators
kubectl apply --dry-run=client -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: neural-ops-gpu-test
spec:
  containers:
  - name: test
    image: opifex/framework:latest
    command: ["python", "-c", "import jax; print(f'JAX devices: {jax.devices()}')"]
    resources:
      limits:
        nvidia.com/gpu: "1"
EOF
```

### 3. **Opifex Resource Profile Testing**

Test different workload profiles for various scientific computing scenarios:

```bash
# Test small workload profile (development/testing)
make deploy-small-workload
kubectl get pods -l profile=small-workload

# Test large workload profile (production L2O optimization)
make deploy-large-workload
kubectl get pods -l profile=large-workload

# Test research workload profile (multi-experiment workflows)
make deploy-research-workload
kubectl get pods -l profile=research-workload

# Validate resource allocations
kubectl describe pods -l app=opifex | grep -A 5 "Requests:"
```

### 4. **Scientific Computing Workload Validation**

```bash
# Validate GPU memory allocation for large neural operators
kubectl apply --dry-run=client -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-memory-test
spec:
  containers:
  - name: test
    image: opifex/framework:latest
    command: ["nvidia-smi", "--query-gpu=memory.total,memory.free", "--format=csv"]
    resources:
      limits:
        nvidia.com/gpu: "4"
EOF

# Test JAX X64 precision configuration
kubectl apply --dry-run=client -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: jax-precision-test
spec:
  containers:
  - name: test
    image: opifex/framework:latest
    command: ["python", "-c", "import jax; jax.config.update('jax_enable_x64', True); print(f'JAX X64 enabled: {jax.config.jax_enable_x64}')"]
    env:
    - name: JAX_ENABLE_X64
      value: "true"
EOF
```

### 5. **Benchmarking Infrastructure Testing**

```bash
# Test PDEBench integration
make test-benchmarking

# Validate benchmark data availability
kubectl apply --dry-run=client -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: benchmark-data-test
spec:
  containers:
  - name: test
    image: opifex/benchmarking:latest
    command: ["python", "-c", "from opifex.benchmarking import PDEBenchRunner; runner = PDEBenchRunner(); print('PDEBench ready')"]
    volumeMounts:
    - name: benchmark-data
      mountPath: /data
  volumes:
  - name: benchmark-data
    persistentVolumeClaim:
      claimName: pdebench-data
EOF
```

### 6. **Opifex Framework Integration Testing**

```bash
# Run complete Opifex framework validation
make test-opifex-complete

# Test framework component interactions
./deployment/test-deployment.sh

# Validate all Opifex capabilities in sequence
kubectl apply --dry-run=client -f - <<EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: opifex-integration-test
spec:
  template:
    spec:
      containers:
      - name: integration-test
        image: opifex/framework:latest
        command: ["python", "-c"]
        args:
          - |
            # Test L2O + Neural Operators integration
            from opifex.optimization.l2o import L2OEngine
            from opifex.neural.operators import FNO
            print("✅ L2O + Neural Operators integration")

            # Test benchmarking integration
            from opifex.benchmarking import EvaluationEngine
            print("✅ Benchmarking integration")

            # Test physics integration
            from opifex.physics.solvers import SpectralSolver
            print("✅ Physics solvers integration")

            print("🎉 Complete Opifex framework integration successful")
        resources:
          requests:
            cpu: "2"
            memory: "8Gi"
            nvidia.com/gpu: "2"
      restartPolicy: Never
EOF
```

### 7. **Performance and Scalability Testing**

```bash
# Test horizontal pod autoscaling for Opifex workloads
kubectl apply --dry-run=client -f - <<EOF
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: opifex-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: opifex-research-workload
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
EOF

# Test GPU utilization monitoring
kubectl apply --dry-run=client -k deployment/monitoring/prometheus/
```

## 📊 **Opifex Testing Checklist**

### Framework Component Checklist

- [ ] L2O Framework components available
- [ ] Neural Operator modules (FNO, SFNO, DeepONet) functional
- [ ] Benchmarking infrastructure (PDEBench) integrated
- [ ] Physics solvers accessible
- [ ] GPU acceleration configured
- [ ] JAX X64 precision enabled

### Resource Profile Checklist

- [ ] Small workload profile (2 CPU, 4Gi RAM, 1 GPU) validated
- [ ] Large workload profile (8 CPU, 32Gi RAM, 4 GPU) validated
- [ ] Research workload profile (4 CPU, 16Gi RAM, 2 GPU, 2 replicas) validated
- [ ] Resource limits and requests properly configured
- [ ] GPU allocation working correctly

### Integration Testing Checklist

- [ ] L2O + Neural Operators integration working
- [ ] Benchmarking + Framework integration functional
- [ ] Physics solvers + Optimization integration verified
- [ ] Monitoring + Opifex metrics collection active
- [ ] Community platform + Framework integration ready

### Performance Testing Checklist

- [ ] GPU utilization monitoring active
- [ ] Memory usage within expected bounds
- [ ] CPU utilization appropriate for workload
- [ ] Horizontal pod autoscaling functional
- [ ] Network performance adequate for distributed training

## 📚 **Additional Resources**

1. **Kubernetes Documentation**: <https://kubernetes.io/docs/>
2. **Kustomize Documentation**: <https://kustomize.io/>
3. **Helm Documentation**: <https://helm.sh/docs/>
4. **Prometheus Documentation**: <https://prometheus.io/docs/>
5. **Grafana Documentation**: <https://grafana.com/docs/>

## 🔄 **Continuous Integration**

### GitHub Actions Example

```yaml
name: Deploy Validation
on: [push, pull_request]
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Validate deployment
      run: |
        make validate-complete
        ./deployment/test-deployment.sh
```

## 📞 **Support and Troubleshooting**

For issues with deployment testing:

1. Check the logs: `make logs`
2. Validate configurations: `make validate-complete`
3. Test deployment health: `make test-deployment`
4. Check deployment status: `make status`

---

**Remember**: Always test in development/staging environments before deploying to production!
