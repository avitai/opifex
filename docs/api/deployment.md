```python
from typing import Union, Optional, List, Dict, Any
from flax import nnx
# Deployment API Reference

The `opifex.deployment` package provides enterprise-grade deployment capabilities for serving Opifex models in production.

## Overview

The deployment module offers:

- **Model Serving**: High-performance REST API for model inference
- **Inference Engine**: Optimized inference with batching and caching
- **Cloud Deployment**: AWS and GCP integration
- **Kubernetes Orchestration**: Auto-scaling and resource management
- **Monitoring**: Health checks, metrics, and logging
- **Model Registry**: Integration with model versioning

## Model Serving

### ModelServer

High-performance model serving infrastructure.

```python
from opifex.deployment import ModelServer, DeploymentConfig

class ModelServer:
    """
    Production model serving with REST API.

    Provides HTTP endpoints for model inference with automatic
    batching, caching, and performance optimization.

    Args:
        model: Trained model or model ID from registry
        config: Deployment configuration
        enable_batching: Enable request batching
        batch_size: Maximum batch size
        timeout_ms: Request timeout in milliseconds

    Example:
        >>> from opifex.neural.operators.fno import FNO
        >>> model = FNO.load_from_checkpoint('model.ckpt')
        >>> config = DeploymentConfig(
        ...     host='0.0.0.0',
        ...     port=8000,
        ...     workers=4,
        ...     enable_gpu=True
        ... )
        >>> server = ModelServer(model, config)
        >>> server.start()  # Starts serving on port 8000
    """

    def __init__(
        self,
        model: Union[nnx.Module, str],
        config: DeploymentConfig,
        enable_batching: bool = True,
        batch_size: int = 32,
        timeout_ms: int = 5000
    ):
        """Initialize model server."""

    def start(self):
        """
        Start serving model.

        Creates REST API with endpoints:
        - POST /predict: Run inference
        - GET /health: Health check
        - GET /metrics: Prometheus metrics
        - GET /model/info: Model metadata

        Example:
            >>> server.start()
            >>> # Server now accepting requests at http://localhost:8000
        """

    def stop(self):
        """Stop server gracefully."""

    def reload_model(self, model_path: str):
        """
        Hot-reload model without downtime.

        Args:
            model_path: Path to new model checkpoint

        Example:
            >>> # Deploy new model version
            >>> server.reload_model('model_v2.ckpt')
        """
```

### DeploymentConfig

Configuration for model deployment.

```python
from opifex.deployment import DeploymentConfig

@dataclass
class DeploymentConfig:
    """
    Configuration for model deployment.

    Attributes:
        host: Server host address
        port: Server port
        workers: Number of worker processes
        enable_gpu: Use GPU for inference
        max_batch_size: Maximum batch size
        timeout_ms: Request timeout
        enable_caching: Cache frequent requests
        cache_size: Maximum cached items
        log_level: Logging level
        cors_origins: Allowed CORS origins
        api_key_required: Require API key authentication
    """

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    enable_gpu: bool = True
    max_batch_size: int = 32
    timeout_ms: int = 5000
    enable_caching: bool = True
    cache_size: int = 1000
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_required: bool = False
```

## Inference Engine

### InferenceEngine

Optimized inference with batching and model optimization.

```python
from opifex.deployment import InferenceEngine

class InferenceEngine:
    """
    High-performance inference engine.

    Features:
    - Automatic request batching
    - Model optimization (quantization, pruning)
    - Multi-device support
    - Cached predictions
    - Performance monitoring

    Args:
        model: Model to serve
        device: Target device ('cpu', 'cuda', 'tpu')
        optimize: Apply model optimizations
        precision: Inference precision ('fp32', 'fp16', 'bf16')

    Example:
        >>> engine = InferenceEngine(
        ...     model=fno_model,
        ...     device='cuda',
        ...     optimize=True,
        ...     precision='fp16'
        ... )
        >>> # Run inference
        >>> predictions = engine.predict(inputs)
    """

    def predict(
        self,
        inputs: Array,
        batch_size: Optional[int] = None
    ) -> Array:
        """
        Run inference on inputs.

        Args:
            inputs: Input data
            batch_size: Override default batch size

        Returns:
            Model predictions

        Example:
            >>> inputs = jnp.array([...])  # Shape: (1000, 64, 64)
            >>> # Automatically batched
            >>> predictions = engine.predict(inputs, batch_size=32)
        """

    def predict_async(
        self,
        inputs: Array
    ) -> asyncio.Future:
        """
        Async inference for concurrent requests.

        Args:
            inputs: Input data

        Returns:
            Future for prediction result

        Example:
            >>> async def process_batch(batch):
            ...     result = await engine.predict_async(batch)
            ...     return result
        """

    def optimize_model(
        self,
        optimization_level: int = 1
    ):
        """
        Apply model optimizations.

        Args:
            optimization_level: Optimization level (0-3):
                - 0: No optimization
                - 1: Basic (JIT compilation)
                - 2: Standard (+ operator fusion)
                - 3: Aggressive (+ quantization)

        Example:
            >>> engine.optimize_model(optimization_level=2)
        """

    def benchmark(
        self,
        test_inputs: Array,
        num_iterations: int = 100
    ) -> Dict[str, float]:
        """
        Benchmark inference performance.

        Args:
            test_inputs: Sample inputs for benchmarking
            num_iterations: Number of benchmark iterations

        Returns:
            Performance metrics:
                - throughput: Samples/second
                - latency_p50: Median latency (ms)
                - latency_p95: 95th percentile latency
                - latency_p99: 99th percentile latency

        Example:
            >>> metrics = engine.benchmark(test_data, num_iterations=1000)
            >>> print(f"Throughput: {metrics['throughput']:.1f} samples/s")
            >>> print(f"P95 latency: {metrics['latency_p95']:.2f} ms")
        """
```

## Cloud Deployment

### AWS Deployment

Deploy models to AWS infrastructure.

```python
from opifex.deployment.cloud import AWSDeploymentManager, AWSConfig

class AWSDeploymentManager:
    """
    Manage model deployment on AWS.

    Supports:
    - EC2 instances
    - SageMaker endpoints
    - Lambda functions
    - ECS containers

    Args:
        config: AWS configuration
        region: AWS region

    Example:
        >>> aws_config = AWSConfig(
        ...     access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        ...     secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        ...     instance_type='ml.g4dn.xlarge',
        ...     endpoint_name='opifex-fno-prod'
        ... )
        >>> manager = AWSDeploymentManager(aws_config, region='us-east-1')
    """

    def deploy_sagemaker(
        self,
        model: nnx.Module,
        deployment_config: DeploymentConfig
    ) -> str:
        """
        Deploy model to SageMaker endpoint.

        Args:
            model: Model to deploy
            deployment_config: Deployment configuration

        Returns:
            Endpoint URL

        Example:
            >>> endpoint_url = manager.deploy_sagemaker(
            ...     model=fno_model,
            ...     deployment_config=config
            ... )
            >>> print(f"Model deployed at: {endpoint_url}")
        """

    def deploy_lambda(
        self,
        model: nnx.Module,
        memory_mb: int = 3008,
        timeout_seconds: int = 300
    ) -> str:
        """
        Deploy model as AWS Lambda function.

        Args:
            model: Model to deploy
            memory_mb: Lambda memory allocation
            timeout_seconds: Function timeout

        Returns:
            Lambda function ARN

        Example:
            >>> # Deploy lightweight model to Lambda
            >>> function_arn = manager.deploy_lambda(
            ...     model=small_model,
            ...     memory_mb=1024,
            ...     timeout_seconds=60
            ... )
        """

    def create_auto_scaling(
        self,
        endpoint_name: str,
        min_instances: int = 1,
        max_instances: int = 10,
        target_metric: str = 'InvocationsPerInstance',
        target_value: float = 1000.0
    ):
        """
        Configure auto-scaling for endpoint.

        Args:
            endpoint_name: SageMaker endpoint name
            min_instances: Minimum instance count
            max_instances: Maximum instance count
            target_metric: Scaling metric
            target_value: Target metric value

        Example:
            >>> manager.create_auto_scaling(
            ...     endpoint_name='opifex-fno-prod',
            ...     min_instances=2,
            ...     max_instances=20,
            ...     target_value=500.0
            ... )
        """
```

### GCP Deployment

Deploy models to Google Cloud Platform.

```python
from opifex.deployment.cloud import GCPDeploymentManager, GCPConfig

class GCPDeploymentManager:
    """
    Manage model deployment on GCP.

    Supports:
    - Vertex AI endpoints
    - Cloud Run
    - Cloud Functions
    - GKE clusters

    Args:
        config: GCP configuration
        project_id: GCP project ID

    Example:
        >>> gcp_config = GCPConfig(
        ...     credentials_path='credentials.json',
        ...     machine_type='n1-standard-4-k80',
        ...     endpoint_name='opifex-model'
        ... )
        >>> manager = GCPDeploymentManager(gcp_config, project_id='my-project')
    """

    def deploy_vertex_ai(
        self,
        model: nnx.Module,
        deployment_config: DeploymentConfig
    ) -> str:
        """
        Deploy to Vertex AI endpoint.

        Args:
            model: Model to deploy
            deployment_config: Deployment configuration

        Returns:
            Endpoint URL
        """

    def deploy_cloud_run(
        self,
        model: nnx.Module,
        min_instances: int = 0,
        max_instances: int = 10,
        concurrency: int = 80
    ) -> str:
        """
        Deploy to Cloud Run (serverless).

        Args:
            model: Model to deploy
            min_instances: Minimum instances
            max_instances: Maximum instances
            concurrency: Requests per instance

        Returns:
            Service URL
        """
```

## Kubernetes Orchestration

### Kubernetes Manifest Generator

Generate Kubernetes manifests for model deployment.

```python
from opifex.deployment.kubernetes import ManifestGenerator

class ManifestGenerator:
    """
    Generate Kubernetes deployment manifests.

    Creates complete k8s configuration including:
    - Deployment
    - Service
    - HorizontalPodAutoscaler
    - Ingress
    - ConfigMap
    """

    def generate_deployment(
        self,
        model_name: str,
        image: str,
        replicas: int = 3,
        resources: Optional[Dict] = None
    ) -> str:
        """
        Generate deployment manifest.

        Args:
            model_name: Name for deployment
            image: Container image
            replicas: Number of replicas
            resources: Resource requests/limits

        Returns:
            YAML manifest

        Example:
            >>> generator = ManifestGenerator()
            >>> manifest = generator.generate_deployment(
            ...     model_name='opifex-fno',
            ...     image='gcr.io/my-project/opifex-fno:v1',
            ...     replicas=5,
            ...     resources={
            ...         'requests': {'memory': '2Gi', 'cpu': '1'},
            ...         'limits': {'memory': '4Gi', 'cpu': '2', 'nvidia.com/gpu': '1'}
            ...     }
            ... )
            >>> # Apply to cluster
            >>> with open('deployment.yaml', 'w') as f:
            ...     f.write(manifest)
        """

    def generate_autoscaler(
        self,
        deployment_name: str,
        min_replicas: int = 2,
        max_replicas: int = 20,
        target_cpu: int = 70
    ) -> str:
        """
        Generate HorizontalPodAutoscaler manifest.

        Args:
            deployment_name: Target deployment
            min_replicas: Minimum pods
            max_replicas: Maximum pods
            target_cpu: Target CPU utilization (%)

        Returns:
            YAML manifest
        """
```

### Kubernetes Orchestrator

Manage Kubernetes deployments.

```python
from opifex.deployment.kubernetes import KubernetesOrchestrator

class KubernetesOrchestrator:
    """
    Orchestrate model deployment on Kubernetes.

    Args:
        kubeconfig_path: Path to kubeconfig
        namespace: Kubernetes namespace

    Example:
        >>> orchestrator = KubernetesOrchestrator(
        ...     kubeconfig_path='~/.kube/config',
        ...     namespace='ml-models'
        ... )
    """

    def deploy(
        self,
        model: nnx.Module,
        deployment_name: str,
        image: str,
        replicas: int = 3
    ):
        """
        Deploy model to Kubernetes.

        Args:
            model: Model to deploy
            deployment_name: Deployment name
            image: Container image
            replicas: Number of replicas

        Example:
            >>> orchestrator.deploy(
            ...     model=fno_model,
            ...     deployment_name='fno-production',
            ...     image='myregistry/fno:latest',
            ...     replicas=5
            ... )
        """

    def scale(
        self,
        deployment_name: str,
        replicas: int
    ):
        """
        Scale deployment to specified replicas.

        Args:
            deployment_name: Deployment to scale
            replicas: Target replica count

        Example:
            >>> orchestrator.scale('fno-production', replicas=10)
        """

    def rolling_update(
        self,
        deployment_name: str,
        new_image: str
    ):
        """
        Perform rolling update to new model version.

        Args:
            deployment_name: Deployment to update
            new_image: New container image

        Example:
            >>> orchestrator.rolling_update(
            ...     'fno-production',
            ...     'myregistry/fno:v2'
            ... )
        """
```

## Monitoring

### Health Monitoring

Monitor deployment health and performance.

```python
from opifex.deployment.monitoring import HealthMonitor

class HealthMonitor:
    """
    Monitor deployment health and performance.

    Tracks:
    - Request latency
    - Error rates
    - Model performance metrics
    - Resource utilization
    """

    def check_health(self) -> Dict[str, Any]:
        """
        Perform health check.

        Returns:
            Health status dictionary

        Example:
            >>> monitor = HealthMonitor(server)
            >>> status = monitor.check_health()
            >>> if status['healthy']:
            ...     print("System healthy")
            >>> else:
            ...     print(f"Issues: {status['issues']}")
        """

    def get_metrics(self) -> Dict[str, float]:
        """
        Get current performance metrics.

        Returns:
            Metrics dictionary:
                - requests_per_second
                - average_latency_ms
                - error_rate
                - p95_latency_ms
                - p99_latency_ms
                - cpu_usage_percent
                - memory_usage_mb
                - gpu_utilization_percent (if GPU)
        """
```

## Complete Deployment Examples

### Local Development Deployment

```python
from opifex.deployment import ModelServer, DeploymentConfig
from opifex.neural.operators.fno import FNO

# Load trained model
model = FNO.load_from_checkpoint('checkpoints/fno_best.ckpt')

# Configure server
config = DeploymentConfig(
    host='localhost',
    port=8000,
    workers=2,
    enable_gpu=False,  # CPU for local dev
    log_level='DEBUG'
)

# Start server
server = ModelServer(model, config)
server.start()

# Make prediction (from client)
import requests
response = requests.post(
    'http://localhost:8000/predict',
    json={'input': input_data.tolist()}
)
prediction = response.json()['prediction']
```

### Production AWS Deployment

```python
from opifex.deployment.cloud import AWSDeploymentManager, AWSConfig
from opifex.deployment import DeploymentConfig

# Configure AWS
aws_config = AWSConfig(
    instance_type='ml.g4dn.4xlarge',  # GPU instance
    endpoint_name='opifex-fno-production',
    initial_instance_count=3
)

# Configure deployment
deploy_config = DeploymentConfig(
    enable_gpu=True,
    max_batch_size=64,
    enable_caching=True,
    cache_size=10000
)

# Deploy
manager = AWSDeploymentManager(aws_config, region='us-east-1')
endpoint_url = manager.deploy_sagemaker(model, deploy_config)

# Setup auto-scaling
manager.create_auto_scaling(
    endpoint_name='opifex-fno-production',
    min_instances=3,
    max_instances=20,
    target_value=500.0  # Target 500 requests/instance
)

print(f"Model deployed at: {endpoint_url}")
```

### Kubernetes Production Deployment

```python
from opifex.deployment.kubernetes import KubernetesOrchestrator, ManifestGenerator

# Generate manifests
generator = ManifestGenerator()

deployment = generator.generate_deployment(
    model_name='opifex-fno',
    image='gcr.io/my-project/opifex-fno:v1.0',
    replicas=5,
    resources={
        'requests': {'memory': '4Gi', 'cpu': '2'},
        'limits': {'memory': '8Gi', 'cpu': '4', 'nvidia.com/gpu': '1'}
    }
)

autoscaler = generator.generate_autoscaler(
    deployment_name='opifex-fno',
    min_replicas=3,
    max_replicas=20,
    target_cpu=70
)

# Save manifests
with open('k8s/deployment.yaml', 'w') as f:
    f.write(deployment)
with open('k8s/autoscaler.yaml', 'w') as f:
    f.write(autoscaler)

# Deploy
orchestrator = KubernetesOrchestrator(namespace='ml-production')
orchestrator.deploy(
    model=fno_model,
    deployment_name='opifex-fno',
    image='gcr.io/my-project/opifex-fno:v1.0',
    replicas=5
)

# Monitor
from opifex.deployment.monitoring import HealthMonitor
monitor = HealthMonitor(orchestrator)
metrics = monitor.get_metrics()
print(f"Current RPS: {metrics['requests_per_second']}")
print(f"P95 latency: {metrics['p95_latency_ms']} ms")
```

## See Also

- [Platform API](platform.md): Model registry and versioning
- [MLOps API](mlops.md): Experiment tracking
- [Cloud Deployment Guide](../deployment/aws-deployment.md): Detailed AWS setup
- [Kubernetes Deployment](../deployment/aws-deployment.md): Cloud deployment guide
