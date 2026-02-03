# Opifex Scalability Infrastructure

This directory contains the Kubernetes-native scalability infrastructure for the Opifex framework, implementing horizontal scaling, load balancing, and resource optimization for scientific computing workloads.

## 🏗️ Architecture Overview

The scalability infrastructure provides:

- **Horizontal Pod Autoscaling (HPA)**: Dynamic pod scaling based on CPU, memory, GPU, and custom metrics
- **Cluster Autoscaling**: Automatic node provisioning and deprovisioning
- **Vertical Pod Autoscaling (VPA)**: Resource optimization for individual containers
- **Load Balancing**: Intelligent traffic distribution across multiple nodes
- **Resource Management**: Quotas, limits, and priority-based scheduling
- **GPU-Aware Scheduling**: Optimized placement for GPU-intensive workloads

## 📁 Directory Structure

```
scalability/
├── cluster-autoscaler/           # Cluster-level node scaling
│   ├── cluster-autoscaler.yaml   # Main deployment
│   ├── service-account.yaml      # RBAC service account
│   ├── cluster-role.yaml         # Required permissions
│   ├── cluster-role-binding.yaml # Permission binding
│   ├── config-map.yaml          # Configuration parameters
│   └── kustomization.yaml       # Component orchestration
├── hpa/                         # Horizontal Pod Autoscaling
│   ├── l2o-optimizer-hpa.yaml   # L2O workload scaling
│   ├── neural-operator-hpa.yaml # Neural operator scaling
│   ├── benchmarking-hpa.yaml    # Benchmarking scaling
│   ├── community-platform-hpa.yaml # Platform scaling
│   └── kustomization.yaml       # HPA orchestration
├── vpa/                         # Vertical Pod Autoscaling
│   ├── l2o-optimizer-vpa.yaml   # L2O resource optimization
│   └── neural-operator-vpa.yaml # Neural operator optimization
├── resource-quotas/             # Resource management
│   ├── gpu-resource-quota.yaml  # GPU resource limits
│   └── cpu-resource-quota.yaml  # CPU resource limits
├── node-affinity/               # Node scheduling rules
│   ├── gpu-node-selector.yaml   # GPU node placement
│   └── cpu-node-selector.yaml   # CPU node placement
├── priority-classes/            # Workload prioritization
│   ├── opifex-priority-classes.yaml # GPU workload priority
│   ├── high-priority.yaml       # Critical operations
│   ├── cpu-workload.yaml        # CPU-intensive tasks
│   ├── normal-priority.yaml     # Standard workloads
│   ├── low-priority.yaml        # Background tasks
│   └── kustomization.yaml       # Priority orchestration
├── load-balancing/              # Traffic distribution
│   ├── ingress-controller.yaml  # Main ingress configuration
│   └── service-monitor.yaml     # Load balancer metrics
├── kustomization.yaml           # Main orchestration
└── README.md                    # This documentation
```

## 🚀 Deployment

### Prerequisites

1. **Kubernetes Cluster**: v1.28+ with metrics server installed
2. **NVIDIA GPU Operator**: For GPU workload support
3. **Prometheus Operator**: For custom metrics collection
4. **Ingress Controller**: NGINX or similar for load balancing
5. **VPA Controller**: For vertical pod autoscaling

### Quick Deployment

```bash
# Deploy entire scalability infrastructure
kubectl apply -k deployment/scalability/

# Deploy specific components
kubectl apply -k deployment/scalability/cluster-autoscaler/
kubectl apply -k deployment/scalability/hpa/
kubectl apply -k deployment/scalability/vpa/
```

### Configuration

#### Cluster Autoscaler

```yaml
# Key parameters in cluster-autoscaler/config-map.yaml
data:
  nodes.max: "100"           # Maximum cluster nodes
  nodes.min: "3"             # Minimum cluster nodes
  gpu-nodes.max: "50"        # Maximum GPU nodes
  gpu-nodes.min: "1"         # Minimum GPU nodes
  cpu-nodes.max: "50"        # Maximum CPU nodes
  cpu-nodes.min: "2"         # Minimum CPU nodes
```

#### HPA Configuration

```yaml
# L2O Optimizer HPA thresholds
metrics:
- type: Resource
  resource:
    name: nvidia.com/gpu
    target:
      averageUtilization: 85    # GPU utilization target
- type: Pods
  pods:
    metric:
      name: opifex_l2o_queue_length
    target:
      averageValue: "10"        # Queue length target
```

#### Resource Quotas

```yaml
# GPU resource limits
spec:
  hard:
    nvidia.com/gpu: "50"       # Total GPU limit
    requests.cpu: "200"        # CPU request limit
    requests.memory: 800Gi     # Memory request limit
```

## 📊 Monitoring and Metrics

### Custom Metrics

The scalability infrastructure uses custom metrics for intelligent scaling:

- **L2O Metrics**:
  - `opifex_l2o_queue_length`: Optimization queue depth
  - `opifex_l2o_optimization_time`: Average optimization duration

- **Neural Operator Metrics**:
  - `opifex_training_queue_length`: Training queue depth
  - `opifex_neural_operator_batch_size`: Current batch size

- **Community Platform Metrics**:
  - `opifex_api_requests_per_second`: API request rate
  - `opifex_active_users`: Concurrent user count

### Load Balancer Metrics

```yaml
# Monitored NGINX metrics
metricRelabelings:
- sourceLabels: [__name__]
  regex: 'nginx_ingress_controller_(request_duration_seconds|requests|bytes_sent|bytes_received|connections|ssl_certificate_expiry_seconds)'
  action: keep
```

## 🎯 Scaling Behaviors

### L2O Optimizer Scaling

- **Scale Up**: 50% increase every 60s when GPU > 85% or queue > 10
- **Scale Down**: 10% decrease every 60s with 5min stabilization
- **Replicas**: 1-20 pods with GPU-aware scheduling

### Neural Operator Scaling

- **Scale Up**: 100% increase every 120s when GPU > 90%
- **Scale Down**: 20% decrease every 120s with 10min stabilization
- **Replicas**: 1-15 pods with high-memory preference

### Benchmarking Scaling

- **Scale Up**: 75% increase every 180s when queue > 3
- **Scale Down**: 25% decrease every 180s with 15min stabilization
- **Replicas**: 1-10 pods with CPU optimization

## 🔧 Node Affinity and Scheduling

### GPU Node Placement

```yaml
# Preferred GPU node types
nodeAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
  - weight: 100
    preference:
      matchExpressions:
      - key: instance-type
        operator: In
        values: ["p3.8xlarge", "p3.16xlarge", "p4d.24xlarge"]
```

### CPU Node Placement

```yaml
# Preferred CPU node types
nodeAffinity:
  preferredDuringSchedulingIgnoredDuringExecution:
  - weight: 100
    preference:
      matchExpressions:
      - key: instance-type
        operator: In
        values: ["c5.4xlarge", "c5.9xlarge", "c5.18xlarge"]
```

## 🔒 Priority Classes

| Priority Class | Value | Use Case |
|----------------|-------|----------|
| `gpu-workload` | 1000 | GPU-intensive L2O and neural operator training |
| `high-priority` | 900 | Critical operations and system components |
| `cpu-workload` | 500 | CPU-intensive computations and data processing |
| `normal-priority` | 100 | Standard Opifex workloads (default) |
| `low-priority` | 50 | Background tasks and cleanup jobs |

## 🌐 Load Balancing

### Ingress Configuration

```yaml
# API routing
- host: api.opifex.example.com
  http:
    paths:
    - path: /api/v1/l2o
      backend:
        service:
          name: l2o-optimizer-service
    - path: /api/v1/neural-operators
      backend:
        service:
          name: neural-operator-service
```

### TLS and Security

- **TLS Termination**: Let's Encrypt certificates
- **Rate Limiting**: 1000 requests/minute per client
- **Proxy Timeouts**: 300s for long-running operations
- **Body Size Limit**: 100MB for large model uploads

## 🧪 Testing and Validation

### Deployment Validation

```bash
# Validate YAML syntax
kubectl apply --dry-run=client -k deployment/scalability/

# Check resource creation
kubectl get hpa -n opifex-system
kubectl get vpa -n opifex-system
kubectl get priorityclass
kubectl get resourcequota -n opifex-system

# Monitor scaling events
kubectl get events -n opifex-system --field-selector reason=SuccessfulCreate
kubectl describe hpa l2o-optimizer-hpa -n opifex-system
```

### Load Testing

```bash
# Generate load for HPA testing
kubectl run load-generator --image=busybox --restart=Never -- /bin/sh -c "while true; do wget -q -O- http://l2o-optimizer-service.opifex-system.svc.cluster.local:8000/health; done"

# Monitor scaling behavior
watch kubectl get hpa -n opifex-system
watch kubectl get nodes
```

## 🔧 Troubleshooting

### Common Issues

1. **HPA Not Scaling**
   - Check metrics server: `kubectl top nodes`
   - Verify custom metrics: `kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1"`
   - Check resource requests: All containers must have resource requests

2. **Cluster Autoscaler Not Working**
   - Verify node group tags: `k8s.io/cluster-autoscaler/enabled=true`
   - Check cloud provider permissions
   - Review logs: `kubectl logs -n kube-system deployment/cluster-autoscaler`

3. **VPA Recommendations Not Applied**
   - Check VPA admission controller
   - Verify update mode: `Auto` vs `Off`
   - Review resource policies for conflicts

4. **Load Balancer Issues**
   - Check ingress controller logs
   - Verify DNS resolution
   - Test backend service connectivity

## 📈 Performance Optimization

### Scaling Efficiency

- **Predictive Scaling**: Custom metrics anticipate load patterns
- **Graceful Scale-Down**: Long stabilization windows prevent thrashing
- **Resource Right-Sizing**: VPA optimizes container resource allocation

### Cost Optimization

- **Spot Instance Support**: Cluster autoscaler uses cost-effective instances
- **Idle Resource Detection**: Automatic scale-down during low utilization
- **Priority-Based Preemption**: Efficient resource sharing

## 🔄 Maintenance

### Regular Tasks

1. **Update Autoscaler Images**: Monthly updates for security patches
2. **Review Scaling Metrics**: Analyze HPA effectiveness quarterly
3. **Resource Quota Adjustments**: Update limits based on usage patterns
4. **Node Pool Optimization**: Adjust instance types based on workload analysis

### Monitoring Alerts

```yaml
# Example Prometheus alerts
- alert: HPAScalingStuck
  expr: increase(kube_hpa_status_current_replicas[10m]) == 0 AND kube_hpa_status_desired_replicas > kube_hpa_status_current_replicas
  for: 5m

- alert: ClusterAutoscalerFailing
  expr: increase(cluster_autoscaler_errors_total[10m]) > 5
  for: 2m
```

## 🚀 Next Steps

1. **Implement Istio Service Mesh**: Advanced traffic management and security
2. **Add Predictive Scaling**: ML-based workload prediction
3. **Multi-Region Support**: Cross-region load balancing and failover
4. **Cost Optimization Dashboard**: Real-time cost tracking and optimization

For detailed implementation guidance, see the [Opifex Deployment Documentation](../README.md).
