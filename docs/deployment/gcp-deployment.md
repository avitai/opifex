# Opifex Deployment on Google Cloud Platform (GCP)

This comprehensive guide walks you through deploying the Opifex framework on Google Cloud Platform using Google Kubernetes Engine (GKE). This guide is designed for beginners and provides step-by-step instructions.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [GCP Account Setup](#gcp-account-setup)
3. [Environment Preparation](#environment-preparation)
4. [GKE Cluster Creation](#gke-cluster-creation)
5. [Opifex Deployment](#opifex-deployment)
6. [Monitoring Setup](#monitoring-setup)
7. [Security Configuration](#security-configuration)
8. [Verification and Testing](#verification-and-testing)
9. [Troubleshooting](#troubleshooting)
10. [Cost Optimization](#cost-optimization)

## 🚀 Prerequisites

Before starting, ensure you have:

### Required Tools

Install these tools on your local machine:

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Install kubectl
gcloud components install kubectl

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Verify installations
gcloud version
kubectl version --client
helm version
docker --version
```

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB free space
- **Network**: Stable internet connection

## 🔧 GCP Account Setup

### Step 1: Create GCP Account

1. **Visit Google Cloud Console**
   - Go to [https://console.cloud.google.com](https://console.cloud.google.com)
   - Sign in with your Google account or create a new one

2. **Accept Terms and Enable Billing**
   - Accept the Google Cloud Terms of Service
   - Set up billing (required for GKE clusters)
   - New users get $300 in free credits

3. **Create a New Project**

   ```bash
   # Set project name and ID
   export PROJECT_ID="opifex-deployment-$(date +%s)"
   export PROJECT_NAME="Opifex Deployment"

   # Create project
   gcloud projects create $PROJECT_ID --name="$PROJECT_NAME"

   # Set as default project
   gcloud config set project $PROJECT_ID
   ```

### Step 2: Enable Required APIs

```bash
# Enable required Google Cloud APIs
gcloud services enable \
    container.googleapis.com \
    compute.googleapis.com \
    storage.googleapis.com \
    cloudbuild.googleapis.com \
    containerregistry.googleapis.com \
    monitoring.googleapis.com \
    logging.googleapis.com \
    cloudresourcemanager.googleapis.com

# Verify APIs are enabled
gcloud services list --enabled
```

### Step 3: Set Up Authentication

```bash
# Authenticate with Google Cloud
gcloud auth login

# Create service account for deployment
gcloud iam service-accounts create opifex-deployment \
    --display-name="Opifex Deployment Service Account"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:opifex-deployment@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/container.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:opifex-deployment@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/compute.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:opifex-deployment@$PROJECT_ID.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

## 🌐 Environment Preparation

### Step 1: Set Environment Variables

```bash
# Project configuration
export PROJECT_ID="your-project-id"
export REGION="us-central1"
export ZONE="us-central1-a"
export CLUSTER_NAME="opifex-cluster"

# Cluster configuration
export MACHINE_TYPE="e2-standard-4"
export GPU_MACHINE_TYPE="n1-standard-4"
export GPU_TYPE="nvidia-tesla-k80"
export NUM_NODES="3"
export MAX_NODES="10"

# Application configuration
export NAMESPACE="opifex"
export RELEASE_NAME="opifex-release"

# Set default region and zone
gcloud config set compute/region $REGION
gcloud config set compute/zone $ZONE
```

### Step 2: Clone Opifex Repository

```bash
# Clone the Opifex repository
git clone https://github.com/opifex-org/opifex.git
cd opifex

# Verify repository structure
ls -la
```

### Step 3: Prepare Configuration Files

```bash
# Create deployment directory
mkdir -p gcp-deployment
cd gcp-deployment

# Create cluster configuration
cat > cluster-config.yaml << EOF
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://kubernetes.default.svc
  name: opifex-cluster
contexts:
- context:
    cluster: opifex-cluster
    user: opifex-user
  name: opifex-context
current-context: opifex-context
users:
- name: opifex-user
  user:
    token: REPLACE_WITH_ACTUAL_TOKEN
EOF
```

## ⚙️ GKE Cluster Creation

### Step 1: Create GKE Cluster (CPU-only)

For beginners, start with a CPU-only cluster:

```bash
# Create basic GKE cluster
gcloud container clusters create $CLUSTER_NAME \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --num-nodes=$NUM_NODES \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=$MAX_NODES \
    --enable-autorepair \
    --enable-autoupgrade \
    --disk-size=100GB \
    --disk-type=pd-standard \
    --enable-ip-alias \
    --network=default \
    --subnetwork=default \
    --addons=HorizontalPodAutoscaling,HttpLoadBalancing,GcePersistentDiskCsiDriver

# Get cluster credentials
gcloud container clusters get-credentials $CLUSTER_NAME --zone=$ZONE
```

### Step 2: Create GKE Cluster with GPU Support (Advanced)

For GPU-accelerated workloads:

```bash
# Create GPU-enabled node pool
gcloud container node-pools create gpu-pool \
    --cluster=$CLUSTER_NAME \
    --zone=$ZONE \
    --machine-type=$GPU_MACHINE_TYPE \
    --accelerator=type=$GPU_TYPE,count=1 \
    --num-nodes=1 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=3 \
    --enable-autorepair \
    --enable-autoupgrade \
    --disk-size=100GB \
    --disk-type=pd-ssd

# Install NVIDIA GPU device plugin
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

### Step 3: Verify Cluster

```bash
# Check cluster status
gcloud container clusters describe $CLUSTER_NAME --zone=$ZONE

# Check nodes
kubectl get nodes -o wide

# Check system pods
kubectl get pods --all-namespaces
```

## 🚀 Opifex Deployment

### Step 1: Create Namespace

```bash
# Create Opifex namespace
kubectl create namespace $NAMESPACE

# Set default namespace
kubectl config set-context --current --namespace=$NAMESPACE
```

### Step 2: Deploy Opifex Components

```bash
# Navigate to deployment directory
cd ../deployment

# Deploy base Kubernetes resources
kubectl apply -k kubernetes/base/

# Deploy monitoring stack
kubectl apply -k monitoring/prometheus/
kubectl apply -k monitoring/grafana/
kubectl apply -k monitoring/alertmanager/

# Deploy security components
kubectl apply -k security/rbac/

# Deploy Opifex application
kubectl apply -k kubernetes/overlays/production/
```

### Step 3: Configure Storage

```bash
# Create persistent volumes for data storage
cat > opifex-storage.yaml << EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: opifex-data
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: standard
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: opifex-models
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: standard
EOF

kubectl apply -f opifex-storage.yaml
```

### Step 4: Configure Services

```bash
# Create LoadBalancer service for external access
cat > opifex-service.yaml << EOF
apiVersion: v1
kind: Service
metadata:
  name: opifex-api-external
  namespace: $NAMESPACE
spec:
  type: LoadBalancer
  selector:
    app: opifex-api
  ports:
    - port: 80
      targetPort: 8080
      protocol: TCP
      name: http
    - port: 443
      targetPort: 8443
      protocol: TCP
      name: https
EOF

kubectl apply -f opifex-service.yaml
```

### Step 5: Deploy Application

```bash
# Deploy Opifex application
cat > opifex-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opifex-api
  namespace: $NAMESPACE
spec:
  replicas: 3
  selector:
    matchLabels:
      app: opifex-api
  template:
    metadata:
      labels:
        app: opifex-api
    spec:
      containers:
      - name: opifex-api
        image: gcr.io/$PROJECT_ID/opifex:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
        - name: models-volume
          mountPath: /app/models
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: opifex-data
      - name: models-volume
        persistentVolumeClaim:
          claimName: opifex-models
EOF

kubectl apply -f opifex-deployment.yaml
```

## 📊 Monitoring Setup

### Step 1: Access Grafana Dashboard

```bash
# Get Grafana service external IP
kubectl get service grafana -n monitoring

# Port forward to access locally (alternative)
kubectl port-forward svc/grafana 3000:3000 -n monitoring
```

### Step 2: Configure Prometheus

```bash
# Verify Prometheus is running
kubectl get pods -n monitoring -l app=prometheus

# Access Prometheus UI
kubectl port-forward svc/prometheus 9090:9090 -n monitoring
```

### Step 3: Set Up Alerts

```bash
# Create custom alert rules
cat > opifex-alerts.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: opifex-alerts
  namespace: monitoring
data:
  opifex.rules: |
    groups:
    - name: opifex
      rules:
      - alert: OpifexAPIDown
        expr: up{job="opifex-api"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Opifex API is down"
          description: "Opifex API has been down for more than 1 minute"

      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{pod=~"opifex-.*"} / container_spec_memory_limit_bytes > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage detected"
          description: "Pod {{ $labels.pod }} memory usage is above 80%"
EOF

kubectl apply -f opifex-alerts.yaml
```

## 🔒 Security Configuration

### Step 1: Enable RBAC

```bash
# Create service account for Opifex
cat > opifex-rbac.yaml << EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: opifex-service-account
  namespace: $NAMESPACE
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: opifex-role
  namespace: $NAMESPACE
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
- apiGroups: ["apps"]
  resources: ["deployments", "replicasets"]
  verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: opifex-role-binding
  namespace: $NAMESPACE
subjects:
- kind: ServiceAccount
  name: opifex-service-account
  namespace: $NAMESPACE
roleRef:
  kind: Role
  name: opifex-role
  apiGroup: rbac.authorization.k8s.io
EOF

kubectl apply -f opifex-rbac.yaml
```

### Step 2: Configure Network Policies

```bash
# Create network policy for Opifex
cat > opifex-network-policy.yaml << EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: opifex-network-policy
  namespace: $NAMESPACE
spec:
  podSelector:
    matchLabels:
      app: opifex-api
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: monitoring
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443
    - protocol: TCP
      port: 80
    - protocol: UDP
      port: 53
EOF

kubectl apply -f opifex-network-policy.yaml
```

### Step 3: Configure Secrets

```bash
# Create secrets for API keys and credentials
kubectl create secret generic opifex-secrets \
    --from-literal=api-key="your-api-key-here" \
    --from-literal=db-password="your-db-password" \
    --namespace=$NAMESPACE

# Create TLS secret for HTTPS
kubectl create secret tls opifex-tls \
    --cert=path/to/tls.crt \
    --key=path/to/tls.key \
    --namespace=$NAMESPACE
```

## ✅ Verification and Testing

### Step 1: Check Deployment Status

```bash
# Check all pods are running
kubectl get pods -n $NAMESPACE

# Check services
kubectl get services -n $NAMESPACE

# Check deployments
kubectl get deployments -n $NAMESPACE

# Check logs
kubectl logs -l app=opifex-api -n $NAMESPACE
```

### Step 2: Test Opifex API

```bash
# Get external IP
EXTERNAL_IP=$(kubectl get service opifex-api-external -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}')

# Test API endpoint
curl -X GET http://$EXTERNAL_IP/health

# Test with sample data
curl -X POST http://$EXTERNAL_IP/api/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"data": [1, 2, 3, 4, 5]}'
```

### Step 3: Run Comprehensive Tests

```bash
# Run deployment tests
cd ../scripts
./test-deployment.sh

# Run Opifex framework tests
python -m pytest tests/ -v
```

### Step 4: Performance Testing

```bash
# Install load testing tools
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.8.1/deploy/static/provider/cloud/deploy.yaml

# Create load test job
cat > load-test.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: load-test
  namespace: $NAMESPACE
spec:
  template:
    spec:
      containers:
      - name: load-test
        image: alpine/curl
        command:
        - /bin/sh
        - -c
        - |
          for i in {1..100}; do
            curl -s -o /dev/null -w "%{http_code}\n" http://opifex-api-external/health
            sleep 1
          done
      restartPolicy: Never
EOF

kubectl apply -f load-test.yaml
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue 1: Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n $NAMESPACE

# Check logs
kubectl logs <pod-name> -n $NAMESPACE

# Common solutions:
# 1. Check resource limits
# 2. Verify image availability
# 3. Check persistent volume claims
```

#### Issue 2: Service Not Accessible

```bash
# Check service endpoints
kubectl get endpoints -n $NAMESPACE

# Check ingress configuration
kubectl get ingress -n $NAMESPACE

# Test internal connectivity
kubectl run debug --image=alpine/curl --rm -it -- sh
```

#### Issue 3: GPU Not Available

```bash
# Check GPU nodes
kubectl get nodes -l cloud.google.com/gke-accelerator=nvidia-tesla-k80

# Check GPU device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Verify GPU allocation
kubectl describe node <gpu-node-name>
```

#### Issue 4: High Memory Usage

```bash
# Check memory usage
kubectl top pods -n $NAMESPACE

# Check resource limits
kubectl describe pod <pod-name> -n $NAMESPACE

# Adjust resource limits in deployment
kubectl patch deployment opifex-api -n $NAMESPACE -p '{"spec":{"template":{"spec":{"containers":[{"name":"opifex-api","resources":{"limits":{"memory":"8Gi"}}}]}}}}'
```

### Debugging Commands

```bash
# Get cluster info
kubectl cluster-info

# Check cluster events
kubectl get events --sort-by=.metadata.creationTimestamp

# Check node status
kubectl get nodes -o wide

# Check persistent volumes
kubectl get pv,pvc -n $NAMESPACE

# Check network policies
kubectl get networkpolicies -n $NAMESPACE

# Export logs
kubectl logs -l app=opifex-api -n $NAMESPACE > opifex-logs.txt
```

## 💰 Cost Optimization

### Step 1: Enable Cluster Autoscaling

```bash
# Update cluster with autoscaling
gcloud container clusters update $CLUSTER_NAME \
    --zone=$ZONE \
    --enable-autoscaling \
    --min-nodes=1 \
    --max-nodes=5
```

### Step 2: Use Preemptible Instances

```bash
# Create preemptible node pool
gcloud container node-pools create preemptible-pool \
    --cluster=$CLUSTER_NAME \
    --zone=$ZONE \
    --machine-type=e2-standard-2 \
    --preemptible \
    --num-nodes=2 \
    --enable-autoscaling \
    --min-nodes=0 \
    --max-nodes=5
```

### Step 3: Monitor Costs

```bash
# Set up billing alerts
gcloud alpha billing budgets create \
    --billing-account=$BILLING_ACCOUNT_ID \
    --display-name="Opifex Deployment Budget" \
    --budget-amount=100USD \
    --threshold-rule=percent=0.8,basis=CURRENT_SPEND
```

### Step 4: Optimize Resource Usage

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n $NAMESPACE

# Implement horizontal pod autoscaling
kubectl autoscale deployment opifex-api --cpu-percent=70 --min=1 --max=10 -n $NAMESPACE
```

## 🔄 Maintenance and Updates

### Regular Maintenance Tasks

```bash
# Update cluster
gcloud container clusters upgrade $CLUSTER_NAME --zone=$ZONE

# Update node pools
gcloud container node-pools upgrade default-pool --cluster=$CLUSTER_NAME --zone=$ZONE

# Update Opifex application
kubectl set image deployment/opifex-api opifex-api=gcr.io/$PROJECT_ID/opifex:v2.0.0 -n $NAMESPACE

# Backup persistent volumes
kubectl get pv -o yaml > pv-backup.yaml
```

### Monitoring and Alerting

```bash
# Set up log-based metrics
gcloud logging metrics create opifex_errors \
    --description="Opifex application errors" \
    --log-filter='resource.type="k8s_container" AND resource.labels.container_name="opifex-api" AND severity="ERROR"'

# Create alerting policy
gcloud alpha monitoring policies create \
    --policy-from-file=alerting-policy.yaml
```

## 📚 Next Steps

After successful deployment:

1. **Explore the Opifex Dashboard** - Access your deployed application
2. **Run Example Workloads** - Test with sample scientific computing tasks
3. **Scale Your Deployment** - Add more nodes or GPU resources as needed
4. **Set Up CI/CD** - Implement automated deployment pipelines
5. **Monitor Performance** - Use Grafana dashboards to track metrics
6. **Implement Security Best Practices** - Regular security audits and updates

## 🆘 Support and Resources

- **GCP Documentation**: [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine/docs)
- **Opifex Documentation**: [Framework Documentation](../index.md)
- **Community Support**: [GitHub Discussions](https://github.com/opifex-org/opifex/discussions)
- **Enterprise Support**: Contact for commercial support options

---

**Congratulations!** You have successfully deployed Opifex on Google Cloud Platform. Your scientific computing platform is now ready for production use.
