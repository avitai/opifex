# Opifex Deployment on Amazon Web Services (AWS)

This comprehensive guide walks you through deploying the Opifex framework on Amazon Web Services using Amazon Elastic Kubernetes Service (EKS). This guide is designed for beginners and provides step-by-step instructions.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [AWS Account Setup](#aws-account-setup)
3. [Environment Preparation](#environment-preparation)
4. [EKS Cluster Creation](#eks-cluster-creation)
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
# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install kubectl
curl -o kubectl https://s3.us-west-2.amazonaws.com/amazon-eks/1.28.3/2023-11-14/bin/linux/amd64/kubectl
chmod +x ./kubectl
sudo mv ./kubectl /usr/local/bin

# Install eksctl
curl --silent --location "https://github.com/weaveworks/eksctl/releases/latest/download/eksctl_$(uname -s)_amd64.tar.gz" | tar xz -C /tmp
sudo mv /tmp/eksctl /usr/local/bin

# Install Helm
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash

# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Verify installations
aws --version
kubectl version --client
eksctl version
helm version
docker --version
```

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB free space
- **Network**: Stable internet connection

## 🔧 AWS Account Setup

### Step 1: Create AWS Account

1. **Visit AWS Console**
   - Go to [https://aws.amazon.com](https://aws.amazon.com)
   - Click "Create an AWS Account"
   - Follow the registration process

2. **Set Up Billing**
   - Add a payment method
   - Consider setting up billing alerts
   - New users get 12 months of free tier

3. **Enable Required Services**
   - EKS (Elastic Kubernetes Service)
   - EC2 (Elastic Compute Cloud)
   - VPC (Virtual Private Cloud)
   - IAM (Identity and Access Management)

### Step 2: Configure AWS CLI

```bash
# Configure AWS credentials
aws configure

# You'll be prompted for:
# AWS Access Key ID: [Your Access Key]
# AWS Secret Access Key: [Your Secret Key]
# Default region name: us-west-2
# Default output format: json

# Verify configuration
aws sts get-caller-identity
```

### Step 3: Create IAM User and Roles

```bash
# Create IAM user for EKS deployment
aws iam create-user --user-name opifex-eks-user

# Create and attach policy for EKS access
cat > eks-policy.json << EOF
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "eks:*",
                "ec2:*",
                "iam:*",
                "cloudformation:*",
                "autoscaling:*",
                "elasticloadbalancing:*"
            ],
            "Resource": "*"
        }
    ]
}
EOF

aws iam create-policy --policy-name OpifexEKSPolicy --policy-document file://eks-policy.json

aws iam attach-user-policy --user-name opifex-eks-user --policy-arn arn:aws:iam::$(aws sts get-caller-identity --query Account --output text):policy/OpifexEKSPolicy

# Create access keys for the user
aws iam create-access-key --user-name opifex-eks-user
```

### Step 4: Create EKS Service Role

```bash
# Create EKS cluster service role
cat > eks-cluster-role-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "eks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
    --role-name opifex-eks-cluster-role \
    --assume-role-policy-document file://eks-cluster-role-trust-policy.json

# Attach required policies
aws iam attach-role-policy \
    --role-name opifex-eks-cluster-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonEKSClusterPolicy

# Create EKS node group service role
cat > eks-nodegroup-role-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ec2.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
    --role-name opifex-eks-nodegroup-role \
    --assume-role-policy-document file://eks-nodegroup-role-trust-policy.json

# Attach required policies for node group
aws iam attach-role-policy \
    --role-name opifex-eks-nodegroup-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonEKSWorkerNodePolicy

aws iam attach-role-policy \
    --role-name opifex-eks-nodegroup-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonEKS_CNI_Policy

aws iam attach-role-policy \
    --role-name opifex-eks-nodegroup-role \
    --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly
```

## 🌐 Environment Preparation

### Step 1: Set Environment Variables

```bash
# AWS configuration
export AWS_REGION="us-west-2"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export CLUSTER_NAME="opifex-cluster"
export CLUSTER_VERSION="1.28"

# Node group configuration
export NODE_GROUP_NAME="opifex-nodes"
export NODE_INSTANCE_TYPE="m5.large"
export GPU_NODE_GROUP_NAME="opifex-gpu-nodes"
export GPU_INSTANCE_TYPE="p3.2xlarge"
export MIN_NODES="1"
export MAX_NODES="10"
export DESIRED_NODES="3"

# Application configuration
export NAMESPACE="opifex"
export RELEASE_NAME="opifex-release"

# Set default region
aws configure set default.region $AWS_REGION
```

### Step 2: Clone Opifex Repository

```bash
# Clone the Opifex repository
git clone https://github.com/opifex-org/opifex.git
cd opifex

# Verify repository structure
ls -la
```

### Step 3: Create VPC and Subnets

```bash
# Create VPC for EKS cluster
aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --tag-specifications 'ResourceType=vpc,Tags=[{Key=Name,Value=opifex-vpc}]'

# Get VPC ID
VPC_ID=$(aws ec2 describe-vpcs \
    --filters "Name=tag:Name,Values=opifex-vpc" \
    --query 'Vpcs[0].VpcId' \
    --output text)

# Create public subnets
aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.1.0/24 \
    --availability-zone ${AWS_REGION}a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=opifex-public-subnet-1}]'

aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.2.0/24 \
    --availability-zone ${AWS_REGION}b \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=opifex-public-subnet-2}]'

# Create private subnets
aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.3.0/24 \
    --availability-zone ${AWS_REGION}a \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=opifex-private-subnet-1}]'

aws ec2 create-subnet \
    --vpc-id $VPC_ID \
    --cidr-block 10.0.4.0/24 \
    --availability-zone ${AWS_REGION}b \
    --tag-specifications 'ResourceType=subnet,Tags=[{Key=Name,Value=opifex-private-subnet-2}]'

# Create internet gateway
aws ec2 create-internet-gateway \
    --tag-specifications 'ResourceType=internet-gateway,Tags=[{Key=Name,Value=opifex-igw}]'

IGW_ID=$(aws ec2 describe-internet-gateways \
    --filters "Name=tag:Name,Values=opifex-igw" \
    --query 'InternetGateways[0].InternetGatewayId' \
    --output text)

# Attach internet gateway to VPC
aws ec2 attach-internet-gateway \
    --internet-gateway-id $IGW_ID \
    --vpc-id $VPC_ID
```

### Step 4: Create EKS Cluster Configuration

```bash
# Create eksctl configuration file
cat > opifex-cluster-config.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $AWS_REGION
  version: "$CLUSTER_VERSION"

vpc:
  id: "$VPC_ID"
  subnets:
    public:
      ${AWS_REGION}a:
        id: $(aws ec2 describe-subnets --filters "Name=tag:Name,Values=opifex-public-subnet-1" --query 'Subnets[0].SubnetId' --output text)
      ${AWS_REGION}b:
        id: $(aws ec2 describe-subnets --filters "Name=tag:Name,Values=opifex-public-subnet-2" --query 'Subnets[0].SubnetId' --output text)
    private:
      ${AWS_REGION}a:
        id: $(aws ec2 describe-subnets --filters "Name=tag:Name,Values=opifex-private-subnet-1" --query 'Subnets[0].SubnetId' --output text)
      ${AWS_REGION}b:
        id: $(aws ec2 describe-subnets --filters "Name=tag:Name,Values=opifex-private-subnet-2" --query 'Subnets[0].SubnetId' --output text)

managedNodeGroups:
  - name: $NODE_GROUP_NAME
    instanceType: $NODE_INSTANCE_TYPE
    minSize: $MIN_NODES
    maxSize: $MAX_NODES
    desiredCapacity: $DESIRED_NODES
    volumeSize: 100
    volumeType: gp3
    privateNetworking: true
    ssh:
      allow: true
    labels:
      workload-type: compute
    tags:
      Environment: production
      Application: opifex

addons:
  - name: vpc-cni
    version: latest
  - name: coredns
    version: latest
  - name: kube-proxy
    version: latest
  - name: aws-ebs-csi-driver
    version: latest

cloudWatch:
  clusterLogging:
    enable: true
    types: ["api", "audit", "authenticator", "controllerManager", "scheduler"]

iam:
  withOIDC: true
  serviceAccounts:
    - metadata:
        name: aws-load-balancer-controller
        namespace: kube-system
      wellKnownPolicies:
        awsLoadBalancerController: true
    - metadata:
        name: cluster-autoscaler
        namespace: kube-system
      wellKnownPolicies:
        autoScaler: true
EOF
```

## ⚙️ EKS Cluster Creation

### Step 1: Create EKS Cluster

```bash
# Create EKS cluster using eksctl
eksctl create cluster -f opifex-cluster-config.yaml

# This process takes 15-20 minutes
# Wait for cluster creation to complete

# Verify cluster creation
aws eks describe-cluster --name $CLUSTER_NAME --query cluster.status
```

### Step 2: Configure kubectl

```bash
# Update kubeconfig
aws eks update-kubeconfig --region $AWS_REGION --name $CLUSTER_NAME

# Verify connection
kubectl get nodes
kubectl get pods --all-namespaces
```

### Step 3: Create GPU Node Group (Optional)

For GPU-accelerated workloads:

```bash
# Create GPU node group configuration
cat > gpu-nodegroup-config.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $AWS_REGION

managedNodeGroups:
  - name: $GPU_NODE_GROUP_NAME
    instanceType: $GPU_INSTANCE_TYPE
    minSize: 0
    maxSize: 3
    desiredCapacity: 1
    volumeSize: 200
    volumeType: gp3
    privateNetworking: true
    ssh:
      allow: true
    labels:
      workload-type: gpu
      nvidia.com/gpu: "true"
    tags:
      Environment: production
      Application: opifex
      NodeType: gpu
    preBootstrapCommands:
      - /etc/eks/bootstrap.sh $CLUSTER_NAME
EOF

# Create GPU node group
eksctl create nodegroup -f gpu-nodegroup-config.yaml

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
```

### Step 4: Install AWS Load Balancer Controller

```bash
# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm repo update

helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
    -n kube-system \
    --set clusterName=$CLUSTER_NAME \
    --set serviceAccount.create=false \
    --set serviceAccount.name=aws-load-balancer-controller

# Verify installation
kubectl get deployment -n kube-system aws-load-balancer-controller
```

## 🚀 Opifex Deployment

### Step 1: Create Namespace

```bash
# Create Opifex namespace
kubectl create namespace $NAMESPACE

# Set default namespace
kubectl config set-context --current --namespace=$NAMESPACE
```

### Step 2: Create Storage Classes

```bash
# Create storage class for EBS volumes
cat > opifex-storage-class.yaml << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: opifex-gp3
provisioner: ebs.csi.aws.com
parameters:
  type: gp3
  iops: "3000"
  throughput: "125"
  encrypted: "true"
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF

kubectl apply -f opifex-storage-class.yaml
```

### Step 3: Configure Persistent Storage

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
  storageClassName: opifex-gp3
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
  storageClassName: opifex-gp3
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: opifex-cache
  namespace: $NAMESPACE
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
  storageClassName: opifex-gp3
EOF

kubectl apply -f opifex-storage.yaml
```

### Step 4: Deploy Opifex Application

```bash
# Navigate to deployment directory
cd deployment

# Deploy base Kubernetes resources
kubectl apply -k kubernetes/base/

# Deploy Opifex application with AWS-specific configurations
cat > opifex-aws-deployment.yaml << EOF
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
      serviceAccountName: opifex-service-account
      containers:
      - name: opifex-api
        image: public.ecr.aws/opifex/opifex:latest
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: LOG_LEVEL
          value: "INFO"
        - name: AWS_REGION
          value: "$AWS_REGION"
        - name: CLUSTER_NAME
          value: "$CLUSTER_NAME"
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
        - name: cache-volume
          mountPath: /app/cache
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: data-volume
        persistentVolumeClaim:
          claimName: opifex-data
      - name: models-volume
        persistentVolumeClaim:
          claimName: opifex-models
      - name: cache-volume
        persistentVolumeClaim:
          claimName: opifex-cache
EOF

kubectl apply -f opifex-aws-deployment.yaml
```

### Step 5: Configure Services and Ingress

```bash
# Create service for Opifex API
cat > opifex-service.yaml << EOF
apiVersion: v1
kind: Service
metadata:
  name: opifex-api-service
  namespace: $NAMESPACE
spec:
  selector:
    app: opifex-api
  ports:
    - port: 80
      targetPort: 8080
      protocol: TCP
      name: http
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: opifex-api-ingress
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: alb
    alb.ingress.kubernetes.io/scheme: internet-facing
    alb.ingress.kubernetes.io/target-type: ip
    alb.ingress.kubernetes.io/healthcheck-path: /health
    alb.ingress.kubernetes.io/listen-ports: '[{"HTTP": 80}, {"HTTPS": 443}]'
    alb.ingress.kubernetes.io/certificate-arn: arn:aws:acm:$AWS_REGION:$AWS_ACCOUNT_ID:certificate/your-certificate-id
spec:
  rules:
  - host: opifex.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: opifex-api-service
            port:
              number: 80
EOF

kubectl apply -f opifex-service.yaml
```

### Step 6: Deploy Worker Nodes

```bash
# Deploy Opifex worker nodes for distributed computation
cat > opifex-workers.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opifex-workers
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: opifex-worker
  template:
    metadata:
      labels:
        app: opifex-worker
    spec:
      containers:
      - name: opifex-worker
        image: public.ecr.aws/opifex/opifex-worker:latest
        env:
        - name: WORKER_TYPE
          value: "neural-operator"
        - name: API_ENDPOINT
          value: "http://opifex-api-service:80"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
        volumeMounts:
        - name: shared-data
          mountPath: /app/data
      volumes:
      - name: shared-data
        persistentVolumeClaim:
          claimName: opifex-data
      nodeSelector:
        workload-type: compute
EOF

kubectl apply -f opifex-workers.yaml
```

## 📊 Monitoring Setup

### Step 1: Install Prometheus and Grafana

```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus
helm install prometheus prometheus-community/kube-prometheus-stack \
    --namespace monitoring \
    --create-namespace \
    --set grafana.adminPassword=admin123 \
    --set grafana.service.type=LoadBalancer \
    --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
    --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=opifex-gp3

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/prometheus-grafana -n monitoring
```

### Step 2: Configure AWS CloudWatch Integration

```bash
# Install CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cloudwatch-namespace.yaml

kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-serviceaccount.yaml

# Create CloudWatch config
cat > cloudwatch-config.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: cwagentconfig
  namespace: amazon-cloudwatch
data:
  cwagentconfig.json: |
    {
      "logs": {
        "metrics_collected": {
          "kubernetes": {
            "cluster_name": "$CLUSTER_NAME",
            "metrics_collection_interval": 60
          }
        },
        "force_flush_interval": 5
      }
    }
EOF

kubectl apply -f cloudwatch-config.yaml

# Deploy CloudWatch agent
kubectl apply -f https://raw.githubusercontent.com/aws-samples/amazon-cloudwatch-container-insights/latest/k8s-deployment-manifest-templates/deployment-mode/daemonset/container-insights-monitoring/cwagent/cwagent-daemonset.yaml
```

### Step 3: Set Up Custom Metrics

```bash
# Create custom metrics for Opifex
cat > opifex-servicemonitor.yaml << EOF
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: opifex-metrics
  namespace: monitoring
spec:
  selector:
    matchLabels:
      app: opifex-api
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
EOF

kubectl apply -f opifex-servicemonitor.yaml
```

### Step 4: Access Grafana Dashboard

```bash
# Get Grafana LoadBalancer URL
kubectl get service prometheus-grafana -n monitoring

# Get admin password
kubectl get secret prometheus-grafana -n monitoring -o jsonpath="{.data.admin-password}" | base64 --decode
```

## 🔒 Security Configuration

### Step 1: Configure RBAC

```bash
# Create service account and RBAC
cat > opifex-rbac.yaml << EOF
apiVersion: v1
kind: ServiceAccount
metadata:
  name: opifex-service-account
  namespace: $NAMESPACE
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::$AWS_ACCOUNT_ID:role/opifex-pod-role
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
- apiGroups: ["batch"]
  resources: ["jobs", "cronjobs"]
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
# Create network policies for security
cat > opifex-network-policies.yaml << EOF
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: opifex-api-policy
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
          name: kube-system
    - podSelector:
        matchLabels:
          app: aws-load-balancer-controller
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
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: opifex-worker-policy
  namespace: $NAMESPACE
spec:
  podSelector:
    matchLabels:
      app: opifex-worker
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: opifex-api
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: opifex-api
    ports:
    - protocol: TCP
      port: 8080
EOF

kubectl apply -f opifex-network-policies.yaml
```

### Step 3: Configure Secrets Management

```bash
# Create secrets for API keys and credentials
kubectl create secret generic opifex-secrets \
    --from-literal=api-key="your-api-key-here" \
    --from-literal=db-password="your-db-password" \
    --from-literal=aws-access-key="your-aws-access-key" \
    --from-literal=aws-secret-key="your-aws-secret-key" \
    --namespace=$NAMESPACE

# Create AWS Secrets Manager integration
cat > opifex-secrets-store.yaml << EOF
apiVersion: secrets-store.csi.x-k8s.io/v1
kind: SecretProviderClass
metadata:
  name: opifex-secrets-store
  namespace: $NAMESPACE
spec:
  provider: aws
  parameters:
    objects: |
      - objectName: "opifex/api-keys"
        objectType: "secretsmanager"
        jmesPath:
          - path: "api_key"
            objectAlias: "api-key"
          - path: "db_password"
            objectAlias: "db-password"
EOF

kubectl apply -f opifex-secrets-store.yaml
```

### Step 4: Enable Pod Security Standards

```bash
# Configure pod security standards
cat > opifex-pod-security.yaml << EOF
apiVersion: v1
kind: Namespace
metadata:
  name: $NAMESPACE
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
EOF

kubectl apply -f opifex-pod-security.yaml
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

# Check persistent volumes
kubectl get pv,pvc -n $NAMESPACE

# Check logs
kubectl logs -l app=opifex-api -n $NAMESPACE --tail=50
```

### Step 2: Test Opifex API

```bash
# Get ALB endpoint
ALB_ENDPOINT=$(kubectl get ingress opifex-api-ingress -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].hostname}')

# Test health endpoint
curl -X GET http://$ALB_ENDPOINT/health

# Test API endpoint with sample data
curl -X POST http://$ALB_ENDPOINT/api/v1/neural-operator/predict \
    -H "Content-Type: application/json" \
    -d '{
        "model_type": "fno",
        "input_data": [[1, 2, 3, 4, 5]],
        "parameters": {
            "modes": 12,
            "width": 64
        }
    }'

# Test worker connectivity
curl -X GET http://$ALB_ENDPOINT/api/v1/workers/status
```

### Step 3: Run Comprehensive Tests

```bash
# Run deployment validation tests
cd ../scripts
./test-deployment.sh

# Run Opifex framework tests
python -m pytest tests/ -v --tb=short

# Run integration tests
python -m pytest tests/integration/ -v
```

### Step 4: Performance and Load Testing

```bash
# Install K6 for load testing
kubectl apply -f https://raw.githubusercontent.com/grafana/k6-operator/main/bundle.yaml

# Create load test configuration
cat > load-test-config.yaml << EOF
apiVersion: v1
kind: ConfigMap
metadata:
  name: opifex-load-test
  namespace: $NAMESPACE
data:
  test.js: |
    import http from 'k6/http';
    import { check } from 'k6';

    export let options = {
      stages: [
        { duration: '2m', target: 10 },
        { duration: '5m', target: 10 },
        { duration: '2m', target: 20 },
        { duration: '5m', target: 20 },
        { duration: '2m', target: 0 },
      ],
    };

    export default function() {
      let response = http.get('http://$ALB_ENDPOINT/health');
      check(response, {
        'status is 200': (r) => r.status === 200,
        'response time < 500ms': (r) => r.timings.duration < 500,
      });
    }
EOF

kubectl apply -f load-test-config.yaml

# Run load test
cat > load-test.yaml << EOF
apiVersion: k6.io/v1alpha1
kind: K6
metadata:
  name: opifex-load-test
  namespace: $NAMESPACE
spec:
  parallelism: 2
  script:
    configMap:
      name: opifex-load-test
      file: test.js
EOF

kubectl apply -f load-test.yaml
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue 1: Pods Stuck in Pending State

```bash
# Check node capacity
kubectl describe nodes

# Check pod events
kubectl describe pod <pod-name> -n $NAMESPACE

# Check resource quotas
kubectl describe resourcequota -n $NAMESPACE

# Common solutions:
# 1. Scale node group
eksctl scale nodegroup --cluster=$CLUSTER_NAME --name=$NODE_GROUP_NAME --nodes=5

# 2. Check persistent volume claims
kubectl get pvc -n $NAMESPACE
```

#### Issue 2: Load Balancer Not Accessible

```bash
# Check ALB controller logs
kubectl logs -n kube-system deployment/aws-load-balancer-controller

# Check ingress status
kubectl describe ingress opifex-api-ingress -n $NAMESPACE

# Check security groups
aws ec2 describe-security-groups --filters "Name=group-name,Values=*$CLUSTER_NAME*"

# Verify target groups
aws elbv2 describe-target-groups --names opifex-api-targets
```

#### Issue 3: GPU Nodes Not Working

```bash
# Check GPU nodes
kubectl get nodes -l workload-type=gpu

# Check NVIDIA device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Check GPU allocation
kubectl describe node <gpu-node-name>

# Verify GPU resources
kubectl get nodes -o json | jq '.items[] | select(.status.capacity."nvidia.com/gpu" != null) | {name: .metadata.name, gpu: .status.capacity."nvidia.com/gpu"}'
```

#### Issue 4: High Costs

```bash
# Check resource usage
kubectl top nodes
kubectl top pods -n $NAMESPACE

# Check EBS volumes
aws ec2 describe-volumes --filters "Name=tag:kubernetes.io/cluster/$CLUSTER_NAME,Values=owned"

# Optimize node groups
eksctl get nodegroup --cluster=$CLUSTER_NAME
```

### Debugging Commands

```bash
# Get cluster information
kubectl cluster-info
aws eks describe-cluster --name $CLUSTER_NAME

# Check node status
kubectl get nodes -o wide
kubectl describe nodes

# Check system pods
kubectl get pods -n kube-system

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp -n $NAMESPACE

# Export logs
kubectl logs -l app=opifex-api -n $NAMESPACE > opifex-api-logs.txt
kubectl logs -l app=opifex-worker -n $NAMESPACE > opifex-worker-logs.txt

# Check AWS resources
aws eks list-clusters
aws ec2 describe-instances --filters "Name=tag:kubernetes.io/cluster/$CLUSTER_NAME,Values=owned"
aws elbv2 describe-load-balancers
```

## 💰 Cost Optimization

### Step 1: Enable Cluster Autoscaler

```bash
# Install cluster autoscaler
kubectl apply -f https://raw.githubusercontent.com/kubernetes/autoscaler/master/cluster-autoscaler/cloudprovider/aws/examples/cluster-autoscaler-autodiscover.yaml

# Configure cluster autoscaler
kubectl -n kube-system annotate deployment.apps/cluster-autoscaler cluster-autoscaler.kubernetes.io/safe-to-evict="false"

kubectl -n kube-system edit deployment.apps/cluster-autoscaler

# Add the following to the command section:
# - --balance-similar-node-groups
# - --skip-nodes-with-system-pods=false
# - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled,k8s.io/cluster-autoscaler/$CLUSTER_NAME
```

### Step 2: Use Spot Instances

```bash
# Create spot instance node group
cat > spot-nodegroup.yaml << EOF
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: $CLUSTER_NAME
  region: $AWS_REGION

managedNodeGroups:
  - name: spot-nodes
    instanceTypes: ["m5.large", "m5.xlarge", "m4.large"]
    spot: true
    minSize: 0
    maxSize: 10
    desiredCapacity: 2
    volumeSize: 100
    volumeType: gp3
    labels:
      workload-type: spot
    tags:
      Environment: production
      NodeType: spot
EOF

eksctl create nodegroup -f spot-nodegroup.yaml
```

### Step 3: Implement Resource Quotas

```bash
# Create resource quotas
cat > resource-quotas.yaml << EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: opifex-quota
  namespace: $NAMESPACE
spec:
  hard:
    requests.cpu: "20"
    requests.memory: 40Gi
    limits.cpu: "40"
    limits.memory: 80Gi
    persistentvolumeclaims: "10"
    pods: "20"
EOF

kubectl apply -f resource-quotas.yaml
```

### Step 4: Monitor and Optimize Costs

```bash
# Set up cost monitoring
aws budgets create-budget \
    --account-id $AWS_ACCOUNT_ID \
    --budget '{
        "BudgetName": "Opifex-EKS-Budget",
        "BudgetLimit": {
            "Amount": "200",
            "Unit": "USD"
        },
        "TimeUnit": "MONTHLY",
        "BudgetType": "COST"
    }'

# Check current costs
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-31 \
    --granularity MONTHLY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE
```

## 🔄 Maintenance and Updates

### Regular Maintenance Tasks

```bash
# Update EKS cluster
eksctl update cluster --name $CLUSTER_NAME --region $AWS_REGION

# Update node groups
eksctl update nodegroup --cluster=$CLUSTER_NAME --name=$NODE_GROUP_NAME

# Update add-ons
eksctl update addon --cluster $CLUSTER_NAME --name vpc-cni --force
eksctl update addon --cluster $CLUSTER_NAME --name coredns --force
eksctl update addon --cluster $CLUSTER_NAME --name kube-proxy --force

# Update Opifex application
kubectl set image deployment/opifex-api opifex-api=public.ecr.aws/opifex/opifex:v2.0.0 -n $NAMESPACE

# Backup persistent volumes
kubectl get pv -o yaml > pv-backup.yaml
aws ec2 create-snapshot --volume-id vol-xxxxxxxxx --description "Opifex backup"
```

### Automated Backup Strategy

```bash
# Create backup script
cat > backup-script.sh << '#!/bin/bash
# Backup Kubernetes resources
kubectl get all --all-namespaces -o yaml > k8s-resources-backup.yaml

# Backup persistent volumes
kubectl get pv -o yaml > pv-backup.yaml
kubectl get pvc --all-namespaces -o yaml > pvc-backup.yaml

# Create EBS snapshots
aws ec2 describe-volumes --filters "Name=tag:kubernetes.io/cluster/$CLUSTER_NAME,Values=owned" --query "Volumes[].VolumeId" --output text | xargs -I {} aws ec2 create-snapshot --volume-id {} --description "Automated backup $(date)"

# Upload to S3
aws s3 cp k8s-resources-backup.yaml s3://opifex-backups/$(date +%Y-%m-%d)/
aws s3 cp pv-backup.yaml s3://opifex-backups/$(date +%Y-%m-%d)/
aws s3 cp pvc-backup.yaml s3://opifex-backups/$(date +%Y-%m-%d)/
EOF

chmod +x backup-script.sh

# Create cron job for automated backups
kubectl create configmap backup-script --from-file=backup-script.sh -n $NAMESPACE

cat > backup-cronjob.yaml << EOF
apiVersion: batch/v1
kind: CronJob
metadata:
  name: opifex-backup
  namespace: $NAMESPACE
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: amazon/aws-cli:latest
            command:
            - /bin/bash
            - /scripts/backup-script.sh
            volumeMounts:
            - name: backup-script
              mountPath: /scripts
            env:
            - name: AWS_DEFAULT_REGION
              value: $AWS_REGION
          volumes:
          - name: backup-script
            configMap:
              name: backup-script
              defaultMode: 0755
          restartPolicy: OnFailure
EOF

kubectl apply -f backup-cronjob.yaml
```

## 📚 Next Steps

After successful deployment:

1. **Access Opifex Dashboard** - Use the ALB endpoint to access your application
2. **Run Scientific Workloads** - Test with neural operators and physics simulations
3. **Scale Your Infrastructure** - Add GPU nodes and increase capacity as needed
4. **Implement CI/CD** - Set up automated deployment pipelines with AWS CodePipeline
5. **Monitor Performance** - Use CloudWatch and Grafana for comprehensive monitoring
6. **Optimize Costs** - Implement spot instances and resource optimization strategies

## 🆘 Support and Resources

- **AWS Documentation**: [Amazon EKS User Guide](https://docs.aws.amazon.com/eks/latest/userguide/)
- **Opifex Documentation**: [Framework Documentation](../index.md)
- **Community Support**: [GitHub Discussions](https://github.com/opifex-org/opifex/discussions)
- **AWS Support**: [AWS Support Plans](https://aws.amazon.com/support/)
- **Enterprise Support**: Contact for commercial support options

## 🎯 Production Checklist

Before going to production, ensure:

- [ ] EKS cluster is running with latest version
- [ ] All pods are healthy and running
- [ ] Monitoring and alerting are configured
- [ ] Security policies are in place
- [ ] Backup strategy is implemented
- [ ] Cost monitoring is enabled
- [ ] Load balancer is accessible
- [ ] SSL/TLS certificates are configured
- [ ] Network policies are applied
- [ ] Resource quotas are set
- [ ] Autoscaling is configured
- [ ] Disaster recovery plan is documented

---

**Congratulations!** You have successfully deployed Opifex on Amazon Web Services. Your scientific computing platform is now ready for production use with enterprise-grade scalability and reliability.
