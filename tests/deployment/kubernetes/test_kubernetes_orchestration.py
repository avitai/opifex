"""
Tests for Kubernetes orchestration components.

This module tests the Kubernetes deployment manifests, auto-scaling, and
resource management for the Opifex deployment infrastructure.
"""

import tempfile
from pathlib import Path

import pytest
import yaml


# Import the modules we'll implement
try:
    from opifex.deployment.kubernetes.autoscaler import AutoScaler
    from opifex.deployment.kubernetes.manifest_generator import ManifestGenerator
    from opifex.deployment.kubernetes.resource_manager import ResourceManager
except ImportError:
    # These will fail until we implement them - that's expected for TDD
    pass


class TestManifestGenerator:
    """Test Kubernetes manifest generation."""

    def test_manifest_generator_initialization(self):
        """Test that ManifestGenerator can be initialized."""
        try:
            generator = ManifestGenerator(
                namespace="opifex-production",
                app_name="opifex-server",
                image="opifex:latest",
            )
            assert generator.namespace == "opifex-production"
            assert generator.app_name == "opifex-server"
            assert generator.image == "opifex:latest"
        except NameError:
            pytest.skip("ManifestGenerator not implemented yet - TDD phase")

    def test_deployment_manifest_generation(self):
        """Test generation of deployment manifest."""
        try:
            generator = ManifestGenerator(
                namespace="opifex-production",
                app_name="opifex-server",
                image="opifex:latest",
            )
            manifest = generator.generate_deployment(
                replicas=3,
                cpu_request="100m",
                memory_request="256Mi",
                cpu_limit="500m",
                memory_limit="1Gi",
            )

            # Validate YAML structure
            assert isinstance(manifest, dict)
            assert manifest["kind"] == "Deployment"
            assert manifest["metadata"]["name"] == "opifex-server"
            assert manifest["metadata"]["namespace"] == "opifex-production"
            assert manifest["spec"]["replicas"] == 3

            # Validate container spec
            container = manifest["spec"]["template"]["spec"]["containers"][0]
            assert container["image"] == "opifex:latest"
            assert container["resources"]["requests"]["cpu"] == "100m"
            assert container["resources"]["requests"]["memory"] == "256Mi"
            assert container["resources"]["limits"]["cpu"] == "500m"
            assert container["resources"]["limits"]["memory"] == "1Gi"

        except NameError:
            pytest.skip("ManifestGenerator not implemented yet - TDD phase")

    def test_service_manifest_generation(self):
        """Test generation of service manifest."""
        try:
            generator = ManifestGenerator(
                namespace="opifex-production",
                app_name="opifex-server",
                image="opifex:latest",
            )
            manifest = generator.generate_service(
                port=8000, target_port=8000, service_type="ClusterIP"
            )

            # Validate YAML structure
            assert isinstance(manifest, dict)
            assert manifest["kind"] == "Service"
            assert manifest["metadata"]["name"] == "opifex-server-service"
            assert manifest["metadata"]["namespace"] == "opifex-production"
            assert manifest["spec"]["type"] == "ClusterIP"
            assert manifest["spec"]["ports"][0]["port"] == 8000
            assert manifest["spec"]["ports"][0]["targetPort"] == 8000

        except NameError:
            pytest.skip("ManifestGenerator not implemented yet - TDD phase")

    def test_ingress_manifest_generation(self):
        """Test generation of ingress manifest."""
        try:
            generator = ManifestGenerator(
                namespace="opifex-production",
                app_name="opifex-server",
                image="opifex:latest",
            )
            manifest = generator.generate_ingress(
                host="opifex.example.com",
                path="/",
                service_name="opifex-server-service",
                service_port=8000,
            )

            # Validate YAML structure
            assert isinstance(manifest, dict)
            assert manifest["kind"] == "Ingress"
            assert manifest["metadata"]["name"] == "opifex-server-ingress"
            assert manifest["metadata"]["namespace"] == "opifex-production"

            # Validate ingress rules
            rule = manifest["spec"]["rules"][0]
            assert rule["host"] == "opifex.example.com"
            assert rule["http"]["paths"][0]["path"] == "/"
            assert (
                rule["http"]["paths"][0]["backend"]["service"]["name"]
                == "opifex-server-service"
            )
            assert (
                rule["http"]["paths"][0]["backend"]["service"]["port"]["number"] == 8000
            )

        except NameError:
            pytest.skip("ManifestGenerator not implemented yet - TDD phase")

    def test_manifest_export_to_file(self):
        """Test exporting manifests to YAML files."""
        try:
            generator = ManifestGenerator(
                namespace="opifex-production",
                app_name="opifex-server",
                image="opifex:latest",
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)

                # Generate and export all manifests
                generator.export_all_manifests(
                    output_dir=output_dir,
                    replicas=2,
                    cpu_request="100m",
                    memory_request="256Mi",
                )

                # Verify files exist
                assert (output_dir / "deployment.yaml").exists()
                assert (output_dir / "service.yaml").exists()
                assert (output_dir / "ingress.yaml").exists()

                # Verify YAML content
                with open(output_dir / "deployment.yaml") as f:
                    deployment = yaml.safe_load(f)
                    assert deployment["kind"] == "Deployment"
                    assert deployment["spec"]["replicas"] == 2

        except NameError:
            pytest.skip("ManifestGenerator not implemented yet - TDD phase")


class TestAutoScaler:
    """Test Kubernetes auto-scaling components."""

    def test_autoscaler_initialization(self):
        """Test that AutoScaler can be initialized."""
        try:
            autoscaler = AutoScaler(
                namespace="opifex-production", deployment_name="opifex-server"
            )
            assert autoscaler.namespace == "opifex-production"
            assert autoscaler.deployment_name == "opifex-server"
        except NameError:
            pytest.skip("AutoScaler not implemented yet - TDD phase")

    def test_hpa_manifest_generation(self):
        """Test generation of Horizontal Pod Autoscaler manifest."""
        try:
            autoscaler = AutoScaler(
                namespace="opifex-production", deployment_name="opifex-server"
            )
            manifest = autoscaler.generate_hpa(
                min_replicas=2,
                max_replicas=10,
                cpu_target_percentage=70,
                memory_target_percentage=80,
            )

            # Validate HPA structure
            assert isinstance(manifest, dict)
            assert manifest["kind"] == "HorizontalPodAutoscaler"
            assert manifest["metadata"]["name"] == "opifex-server-hpa"
            assert manifest["metadata"]["namespace"] == "opifex-production"

            # Validate scaling parameters
            spec = manifest["spec"]
            assert spec["minReplicas"] == 2
            assert spec["maxReplicas"] == 10
            assert spec["scaleTargetRef"]["name"] == "opifex-server"

            # Validate metrics
            metrics = spec["metrics"]
            cpu_metric = next(
                m
                for m in metrics
                if m["type"] == "Resource" and m["resource"]["name"] == "cpu"
            )
            assert cpu_metric["resource"]["target"]["averageUtilization"] == 70

        except NameError:
            pytest.skip("AutoScaler not implemented yet - TDD phase")

    def test_vpa_manifest_generation(self):
        """Test generation of Vertical Pod Autoscaler manifest."""
        try:
            autoscaler = AutoScaler(
                namespace="opifex-production", deployment_name="opifex-server"
            )
            manifest = autoscaler.generate_vpa(
                update_mode="Auto", cpu_min="100m", memory_min="128Mi"
            )

            # Validate VPA structure
            assert isinstance(manifest, dict)
            assert manifest["kind"] == "VerticalPodAutoscaler"
            assert manifest["metadata"]["name"] == "opifex-server-vpa"
            assert manifest["spec"]["updatePolicy"]["updateMode"] == "Auto"

        except NameError:
            pytest.skip("AutoScaler not implemented yet - TDD phase")


class TestResourceManager:
    """Test Kubernetes resource management components."""

    def test_resource_manager_initialization(self):
        """Test that ResourceManager can be initialized."""
        try:
            manager = ResourceManager(namespace="opifex-production")
            assert manager.namespace == "opifex-production"
        except NameError:
            pytest.skip("ResourceManager not implemented yet - TDD phase")

    def test_namespace_manifest_generation(self):
        """Test generation of namespace manifest."""
        try:
            manager = ResourceManager(namespace="opifex-production")
            manifest = manager.generate_namespace(
                labels={"env": "production", "app": "opifex"}
            )

            # Validate namespace structure
            assert isinstance(manifest, dict)
            assert manifest["kind"] == "Namespace"
            assert manifest["metadata"]["name"] == "opifex-production"
            assert manifest["metadata"]["labels"]["env"] == "production"
            assert manifest["metadata"]["labels"]["app"] == "opifex"

        except NameError:
            pytest.skip("ResourceManager not implemented yet - TDD phase")

    def test_resource_quota_generation(self):
        """Test generation of resource quota manifest."""
        try:
            manager = ResourceManager(namespace="opifex-production")
            manifest = manager.generate_resource_quota(
                cpu_request="4",
                memory_request="8Gi",
                cpu_limit="8",
                memory_limit="16Gi",
                pods="20",
            )

            # Validate resource quota structure
            assert isinstance(manifest, dict)
            assert manifest["kind"] == "ResourceQuota"
            assert manifest["metadata"]["name"] == "opifex-production-quota"
            assert manifest["spec"]["hard"]["requests.cpu"] == "4"
            assert manifest["spec"]["hard"]["requests.memory"] == "8Gi"
            assert manifest["spec"]["hard"]["limits.cpu"] == "8"
            assert manifest["spec"]["hard"]["limits.memory"] == "16Gi"
            assert manifest["spec"]["hard"]["pods"] == "20"

        except NameError:
            pytest.skip("ResourceManager not implemented yet - TDD phase")

    def test_limit_range_generation(self):
        """Test generation of limit range manifest."""
        try:
            manager = ResourceManager(namespace="opifex-production")
            manifest = manager.generate_limit_range(
                default_cpu_request="100m",
                default_memory_request="128Mi",
                default_cpu_limit="500m",
                default_memory_limit="512Mi",
            )

            # Validate limit range structure
            assert isinstance(manifest, dict)
            assert manifest["kind"] == "LimitRange"
            assert manifest["metadata"]["name"] == "opifex-production-limits"

            limit = manifest["spec"]["limits"][0]
            assert limit["type"] == "Container"
            assert limit["defaultRequest"]["cpu"] == "100m"
            assert limit["defaultRequest"]["memory"] == "128Mi"
            assert limit["default"]["cpu"] == "500m"
            assert limit["default"]["memory"] == "512Mi"

        except NameError:
            pytest.skip("ResourceManager not implemented yet - TDD phase")


class TestKubernetesIntegration:
    """Test integration between all Kubernetes components."""

    def test_complete_manifest_suite_generation(self):
        """Test generation of complete manifest suite."""
        try:
            from opifex.deployment.kubernetes.orchestrator import KubernetesOrchestrator

            orchestrator = KubernetesOrchestrator(
                namespace="opifex-production",
                app_name="opifex-server",
                image="opifex:latest",
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)

                # Generate complete manifest suite
                orchestrator.generate_complete_deployment(
                    output_dir=output_dir,
                    replicas=3,
                    cpu_request="200m",
                    memory_request="512Mi",
                    enable_autoscaling=True,
                    min_replicas=2,
                    max_replicas=10,
                    enable_resource_management=True,
                )

                # Verify all manifest files exist
                expected_files = [
                    "namespace.yaml",
                    "resource-quota.yaml",
                    "limit-range.yaml",
                    "deployment.yaml",
                    "service.yaml",
                    "ingress.yaml",
                    "hpa.yaml",
                ]

                for filename in expected_files:
                    file_path = output_dir / filename
                    assert file_path.exists(), f"Missing manifest file: {filename}"

                    # Verify each file contains valid YAML
                    with open(file_path) as f:
                        manifest = yaml.safe_load(f)
                        assert isinstance(manifest, dict)
                        assert "kind" in manifest
                        assert "metadata" in manifest

        except ImportError:
            pytest.skip("KubernetesOrchestrator not implemented yet - TDD phase")

    def test_gpu_resource_configuration(self):
        """Test GPU resource configuration for CUDA workloads."""
        try:
            from opifex.deployment.kubernetes.orchestrator import KubernetesOrchestrator

            orchestrator = KubernetesOrchestrator(
                namespace="opifex-production",
                app_name="opifex-server",
                image="opifex:latest-gpu",
            )

            manifest = orchestrator.generate_gpu_deployment(
                replicas=2,
                gpu_count=1,
                gpu_type="nvidia.com/gpu",
                cpu_request="500m",
                memory_request="2Gi",
            )

            # Validate GPU configuration
            container = manifest["spec"]["template"]["spec"]["containers"][0]
            assert container["resources"]["limits"]["nvidia.com/gpu"] == 1
            assert "CUDA_VISIBLE_DEVICES" in [
                env["name"] for env in container.get("env", [])
            ]

        except ImportError:
            pytest.skip("GPU deployment not implemented yet - TDD phase")


if __name__ == "__main__":
    pytest.main([__file__])
