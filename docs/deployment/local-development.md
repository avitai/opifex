# Opifex Local Development Setup

This guide helps you set up a local development environment for the Opifex framework. Perfect for development, testing, and learning before deploying to the cloud.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Installation](#local-installation)
3. [Docker Setup](#docker-setup)
4. [Local Kubernetes Setup](#local-kubernetes-setup)
5. [Development Workflow](#development-workflow)
6. [Testing](#testing)
7. [Troubleshooting](#troubleshooting)

## 🚀 Prerequisites

### System Requirements

- **Operating System**: Linux, macOS, or Windows 10/11
- **RAM**: 16GB minimum, 32GB recommended
- **Storage**: 100GB free space
- **CPU**: 8 cores minimum, 16 cores recommended
- **GPU**: Optional NVIDIA GPU for acceleration

### Required Software

```bash
# Install Python 3.10+
# Ubuntu/Debian
sudo apt update
sudo apt install python3.10 python3.10-venv python3.10-dev

# macOS (using Homebrew)
brew install python@3.10

# Windows (using Chocolatey)
choco install python --version=3.10.0

# Install Git
# Ubuntu/Debian
sudo apt install git

# macOS
brew install git

# Windows
choco install git

# Install Docker
# Ubuntu/Debian
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# macOS
brew install --cask docker

# Windows
choco install docker-desktop

# Install Docker Compose
# Ubuntu/Debian
sudo apt install docker-compose-plugin

# macOS/Windows (included with Docker Desktop)
```

## 🔧 Local Installation

### Step 1: Clone Repository

```bash
# Clone the Opifex repository
git clone https://github.com/opifex-org/opifex.git
cd opifex

# Verify repository structure
ls -la
```

### Step 2: Set Up Development Environment

```bash
# Run setup script (auto-detects GPU/CPU and installs all dependencies)
./setup.sh

# Activate environment
source ./activate.sh
```

### Step 3: Install Pre-commit Hooks

```bash
# Install pre-commit hooks
uv run pre-commit install

# Run pre-commit on all files
uv run pre-commit run --all-files
```

### Step 4: Verify Installation

```bash
# Run basic tests
uv run pytest tests/ -v

# Check code formatting
uv run ruff format --check .

# Check linting
uv run ruff check .

# Type checking
uv run pyright

# Verify JAX installation
python -c "import jax; print('JAX version:', jax.__version__); print('JAX devices:', jax.devices())"
```

## 🐳 Docker Setup

### Step 1: Build Docker Images

```bash
# Build main Opifex image
docker build -t opifex:local .

# Build development image with additional tools
docker build -t opifex:dev -f Dockerfile.dev .

# Verify images
docker images | grep opifex
```

### Step 2: Run with Docker Compose

```bash
# Create docker-compose.yml for local development
cat > docker-compose.dev.yml << EOF
version: '3.8'

services:
  opifex-api:
    build: .
    ports:
      - "8080:8080"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    environment:
      - ENVIRONMENT=development
      - LOG_LEVEL=DEBUG
    depends_on:
      - redis
      - postgres

  opifex-worker:
    build: .
    command: python -m opifex.workers.neural_operator
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - ENVIRONMENT=development
      - WORKER_TYPE=neural_operator
      - API_ENDPOINT=http://opifex-api:8080
    depends_on:
      - opifex-api
      - redis

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=opifex
      - POSTGRES_USER=opifex
      - POSTGRES_PASSWORD=opifex_dev_password
    volumes:
      - postgres_data:/var/lib/postgresql/data

  jupyter:
    build: .
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    ports:
      - "8888:8888"
    volumes:
      - ./notebooks:/app/notebooks
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - JUPYTER_ENABLE_LAB=yes

volumes:
  redis_data:
  postgres_data:
EOF

# Start services
docker-compose -f docker-compose.dev.yml up -d

# Check services
docker-compose -f docker-compose.dev.yml ps
```

### Step 3: Test Docker Setup

```bash
# Test API endpoint
curl http://localhost:8080/health

# Test with sample data
curl -X POST http://localhost:8080/api/v1/neural-operator/predict \
    -H "Content-Type: application/json" \
    -d '{
        "model_type": "fno",
        "input_data": [[1, 2, 3, 4, 5]],
        "parameters": {
            "modes": 12,
            "width": 64
        }
    }'

# Access Jupyter notebook
echo "Jupyter Lab available at: http://localhost:8888"
```

## ☸️ Local Kubernetes Setup

### Step 1: Install Local Kubernetes

Choose one of the following options:

#### Option A: Minikube

```bash
# Install Minikube
# Linux
curl -Lo minikube https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
chmod +x minikube
sudo mv minikube /usr/local/bin/

# macOS
brew install minikube

# Windows
choco install minikube

# Start Minikube with sufficient resources
minikube start --cpus=4 --memory=8192 --disk-size=50g

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server
minikube addons enable dashboard
```

#### Option B: Kind (Kubernetes in Docker)

```bash
# Install Kind
# Linux
curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64
chmod +x ./kind
sudo mv ./kind /usr/local/bin/kind

# macOS
brew install kind

# Windows
choco install kind

# Create cluster configuration
cat > kind-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
- role: worker
- role: worker
EOF

# Create cluster
kind create cluster --config kind-config.yaml --name opifex-local

# Install ingress controller
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
```

### Step 2: Deploy Opifex to Local Kubernetes

```bash
# Create namespace
kubectl create namespace opifex-local

# Set default namespace
kubectl config set-context --current --namespace=opifex-local

# Create local storage class
cat > local-storage.yaml << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: local-storage
provisioner: kubernetes.io/no-provisioner
volumeBindingMode: WaitForFirstConsumer
---
apiVersion: v1
kind: PersistentVolume
metadata:
  name: opifex-data-pv
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteOnce
  persistentVolumeReclaimPolicy: Retain
  storageClassName: local-storage
  local:
    path: /tmp/opifex-data
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - minikube  # or kind-worker if using Kind
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: opifex-data-pvc
  namespace: opifex-local
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: local-storage
EOF

kubectl apply -f local-storage.yaml

# Deploy Opifex application
cat > opifex-local-deployment.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: opifex-api
  namespace: opifex-local
spec:
  replicas: 1
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
        image: opifex:local
        imagePullPolicy: Never
        ports:
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "development"
        - name: LOG_LEVEL
          value: "DEBUG"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1"
        volumeMounts:
        - name: data-volume
          mountPath: /app/data
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
          claimName: opifex-data-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: opifex-api-service
  namespace: opifex-local
spec:
  selector:
    app: opifex-api
  ports:
    - port: 80
      targetPort: 8080
  type: ClusterIP
---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: opifex-api-ingress
  namespace: opifex-local
spec:
  rules:
  - host: opifex.local
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

kubectl apply -f opifex-local-deployment.yaml

# Wait for deployment
kubectl wait --for=condition=available --timeout=300s deployment/opifex-api -n opifex-local

# Add local DNS entry
echo "127.0.0.1 opifex.local" | sudo tee -a /etc/hosts
```

### Step 3: Test Local Kubernetes Deployment

```bash
# Check pod status
kubectl get pods -n opifex-local

# Check services
kubectl get services -n opifex-local

# Test API
curl http://opifex.local/health

# Port forward for direct access
kubectl port-forward service/opifex-api-service 8080:80 -n opifex-local &

# Test forwarded port
curl http://localhost:8080/health
```

## 🔄 Development Workflow

### Step 1: Set Up Development Environment

```bash
# Create development configuration
cat > .env.development << EOF
ENVIRONMENT=development
LOG_LEVEL=DEBUG
DEBUG=true
RELOAD=true
DATABASE_URL=postgresql://opifex:opifex_dev_password@localhost:5432/opifex
REDIS_URL=redis://localhost:6379
JUPYTER_ENABLE_LAB=yes
EOF

# Load environment variables
source .env.development
```

### Step 2: Run Development Server

```bash
# Start development server with hot reload
uv run python -m opifex.api.server --reload --debug

# Or use the development script
uv run python scripts/dev-server.py

# Run in background
nohup uv run python -m opifex.api.server --reload --debug > dev-server.log 2>&1 &
```

### Step 3: Run Jupyter for Interactive Development

```bash
# Start Jupyter Lab
uv run jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# Or use the development script
uv run python scripts/start-jupyter.py
```

### Step 4: Code Development Workflow

```bash
# Make changes to code
# ... edit files ...

# Run tests
uv run pytest tests/ -v

# Run specific test
uv run pytest tests/test_neural_operators.py::test_fno_forward -v

# Run with coverage
uv run pytest tests/ --cov=opifex --cov-report=html

# Check code quality
uv run ruff format .
uv run ruff check .
uv run pyright

# Run pre-commit hooks
uv run pre-commit run --all-files

# Commit changes
git add .
git commit -m "Add new feature"
```

### Step 5: Testing Different Components

```bash
# Test neural operators
uv run python examples/neural_operators_comprehensive_demo.py

# Test L2O optimization
uv run python examples/l2o_optimization_demo.py

# Test benchmarking
uv run python examples/benchmarking_demo.py

# Test with different backends
JAX_PLATFORM_NAME=cpu uv run python examples/cpu_demo.py
JAX_PLATFORM_NAME=gpu uv run python examples/gpu_demo.py
```

## 🧪 Testing

### Step 1: Run Unit Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test modules
uv run pytest tests/test_core/ -v
uv run pytest tests/test_neural/ -v
uv run pytest tests/test_optimization/ -v

# Run with markers
uv run pytest tests/ -m "not slow" -v
uv run pytest tests/ -m "gpu" -v
uv run pytest tests/ -m "integration" -v
```

### Step 2: Run Integration Tests

```bash
# Run integration tests
uv run pytest tests/integration/ -v

# Test API endpoints
uv run pytest tests/integration/test_api.py -v

# Test end-to-end workflows
uv run pytest tests/integration/test_workflows.py -v
```

### Step 3: Performance Testing

```bash
# Run benchmarks
uv run python benchmarks/neural_operators_benchmark.py

# Profile code
uv run python -m cProfile -o profile.stats examples/neural_operators_demo.py
uv run python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"

# Memory profiling
uv run python -m memory_profiler examples/neural_operators_demo.py
```

### Step 4: Load Testing

```bash
# Install load testing tools
uv add locust

# Create load test
cat > load_test.py << EOF
from locust import HttpUser, task, between

class OpifexUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def health_check(self):
        self.client.get("/health")

    @task(1)
    def predict(self):
        self.client.post("/api/v1/neural-operator/predict", json={
            "model_type": "fno",
            "input_data": [[1, 2, 3, 4, 5]],
            "parameters": {"modes": 12, "width": 64}
        })
EOF

# Run load test
uv run locust -f load_test.py --host=http://localhost:8080
```

## 🛠️ Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Reinstall dependencies
uv sync --force-reinstall

# Check virtual environment
which python
which pip
```

#### Issue 2: JAX/GPU Issues

```bash
# Check JAX installation
python -c "import jax; print(jax.__version__); print(jax.devices())"

# Check CUDA availability
nvidia-smi

# Reinstall JAX with CUDA support
uv remove jax jaxlib
uv add jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

#### Issue 3: Docker Issues

```bash
# Check Docker daemon
docker info

# Rebuild images
docker-compose -f docker-compose.dev.yml build --no-cache

# Check container logs
docker-compose -f docker-compose.dev.yml logs opifex-api

# Clean up Docker resources
docker system prune -a
```

#### Issue 4: Kubernetes Issues

```bash
# Check cluster status
kubectl cluster-info

# Check pod logs
kubectl logs -l app=opifex-api -n opifex-local

# Describe pod for events
kubectl describe pod <pod-name> -n opifex-local

# Check resource usage
kubectl top nodes
kubectl top pods -n opifex-local
```

#### Issue 5: Port Conflicts

```bash
# Check port usage
netstat -tlnp | grep :8080
lsof -i :8080

# Kill process using port
kill -9 $(lsof -t -i:8080)

# Use different port
export PORT=8081
uv run python -m opifex.api.server --port=$PORT
```

### Debugging Tips

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG=true

# Use Python debugger
python -m pdb examples/neural_operators_demo.py

# Use IPython for interactive debugging
uv add ipython
uv run ipython

# Check memory usage
uv add psutil
python -c "import psutil; print(f'Memory: {psutil.virtual_memory().percent}%')"

# Monitor GPU usage
watch -n 1 nvidia-smi
```

## 📚 Development Resources

### Useful Commands

```bash
# Quick development setup
make dev-setup

# Run all quality checks
make check

# Build documentation
make docs

# Clean up generated files
make clean

# Run specific example
make run-example EXAMPLE=neural_operators_demo
```

### IDE Configuration

#### VS Code

```json
{
    "python.defaultInterpreterPath": "./opifex-env/bin/python",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "ruff",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.associations": {
        "*.md": "markdown"
    }
}
```

#### PyCharm

1. Set interpreter to `./opifex-env/bin/python`
2. Enable pytest as test runner
3. Configure ruff as formatter
4. Set up run configurations for common tasks

### Development Best Practices

1. **Always activate virtual environment** before development
2. **Run tests before committing** changes
3. **Use pre-commit hooks** for code quality
4. **Write tests** for new features
5. **Update documentation** when needed
6. **Follow code style guidelines**
7. **Use meaningful commit messages**

## 🚀 Next Steps

After setting up your local development environment:

1. **Explore Examples** - Run the provided examples to understand the framework
2. **Read Documentation** - Familiarize yourself with the API and concepts
3. **Write Tests** - Add tests for any new features you develop
4. **Contribute** - Submit pull requests for improvements
5. **Deploy to Cloud** - Use the cloud deployment guides when ready

---

**Happy Coding!** Your local Opifex development environment is now ready for scientific machine learning research and development.
