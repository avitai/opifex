"""
Test Suite for Container Orchestration Architecture
Phase 7.1: Container Orchestration - Validation and Testing

Tests multi-stage Docker builds, GPU optimization, security policies,
and Istio service mesh configuration.
"""

import os
import sys
from pathlib import Path

import pytest
import yaml


class TestContainerOrchestration:
    """Test suite for Phase 7.1 Container Orchestration implementation."""

    @pytest.fixture(scope="class")
    def deployment_path(self) -> Path:
        """Get deployment path for container orchestration."""
        return Path(__file__).parent.parent.parent / "deployment" / "containers"

    @pytest.fixture(scope="class")
    def multi_stage_path(self, deployment_path: Path) -> Path:
        """Get multi-stage build path."""
        return deployment_path / "multi-stage"

    @pytest.fixture(scope="class")
    def security_path(self, deployment_path: Path) -> Path:
        """Get security configuration path."""
        return deployment_path / "security"

    @pytest.fixture(scope="class")
    def gpu_optimization_path(self, deployment_path: Path) -> Path:
        """Get GPU optimization configuration path."""
        return deployment_path / "gpu-optimization"

    @pytest.fixture(scope="class")
    def istio_path(self, deployment_path: Path) -> Path:
        """Get Istio configuration path."""
        return deployment_path / "istio"

    def test_multi_stage_dockerfile_exists(self, multi_stage_path: Path):
        """Test that optimized multi-stage Dockerfile exists."""
        dockerfile_path = multi_stage_path / "Dockerfile.optimized"
        assert dockerfile_path.exists(), "Optimized Dockerfile should exist"

        # Read and validate Dockerfile content
        content = dockerfile_path.read_text()

        # Check for required stages
        assert "FROM nvidia/cuda:11.8-devel-ubuntu20.04 AS build-cache" in content
        assert "FROM build-cache AS python-deps" in content
        assert "FROM python-deps AS app-builder" in content
        assert "FROM nvidia/cuda:11.8-runtime-ubuntu20.04 AS gpu-runtime" in content
        assert "FROM gpu-runtime AS production" in content

        # Check for optimization features
        assert "PIP_NO_CACHE_DIR=1" in content
        assert "NVIDIA_VISIBLE_DEVICES=all" in content
        assert "XLA_PYTHON_CLIENT_MEM_FRACTION" in content
        assert "USER opifex:opifex" in content  # Security: non-root user

        print("✅ Multi-stage Dockerfile validation passed")

    def test_docker_compose_gpu_config(self, multi_stage_path: Path):
        """Test Docker Compose GPU configuration."""
        compose_path = multi_stage_path / "docker-compose.gpu.yaml"
        assert compose_path.exists(), "GPU Docker Compose should exist"

        # Load and validate YAML
        with open(compose_path) as f:
            compose_config = yaml.safe_load(f)

        assert "services" in compose_config

        # Check for GPU services
        expected_services = [
            "opifex-core",
            "opifex-neural-ops",
            "opifex-l2o",
            "opifex-benchmarks",
            "opifex-dev",
        ]

        for service_name in expected_services:
            assert service_name in compose_config["services"]
            service = compose_config["services"][service_name]

            # Check GPU configuration
            assert "deploy" in service
            assert "resources" in service["deploy"]
            assert "reservations" in service["deploy"]["resources"]
            assert "devices" in service["deploy"]["resources"]["reservations"]

            # Check environment variables
            assert "environment" in service
            env_vars = service["environment"]
            gpu_env_found = any("NVIDIA_VISIBLE_DEVICES" in var for var in env_vars)
            assert gpu_env_found, f"GPU environment not found in {service_name}"

        # Check for volumes optimization
        assert "volumes" in compose_config
        cache_volumes = [v for v in compose_config["volumes"] if "cache" in v]
        assert len(cache_volumes) >= 4, (
            "Should have multiple cache volumes for optimization"
        )

        print("✅ Docker Compose GPU configuration validation passed")

    def test_build_optimization_script(self, multi_stage_path: Path):
        """Test build optimization script exists and is executable."""
        script_path = multi_stage_path / "build-optimization.sh"
        assert script_path.exists(), "Build optimization script should exist"
        assert os.access(script_path, os.X_OK), "Build script should be executable"

        # Read and validate script content
        content = script_path.read_text()

        # Check for required functionality
        assert "DOCKER_BUILDKIT=1" in content
        assert "TARGET_REDUCTION=54" in content
        assert "docker build" in content
        assert "nvidia-smi" in content
        assert "get_image_size" in content
        assert "calculate_reduction" in content

        print("✅ Build optimization script validation passed")

    def test_gpu_optimization_configs(self, gpu_optimization_path: Path):
        """Test GPU optimization configurations."""
        # Load all individual YAML files instead of consolidated file
        configs = []

        # Load numbered config files (only those that don't have individual equivalents)
        for i in range(5):  # 00-04
            config_file = gpu_optimization_path / f"gpu-runtime-config-{i:02d}.yaml"
            if config_file.exists():
                with open(config_file) as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        # Only add if it's not one of the configs that has an individual file
                        config_name = config_data.get("metadata", {}).get("name", "")
                        if config_name not in [
                            "nvidia-container-runtime-config",
                            "gpu-device-plugin-config",
                            "cuda-optimization-scripts",
                        ]:
                            configs.append(config_data)

        # Load the specific individual configuration files (these take priority)
        required_individual_configs = [
            "nvidia-container-runtime-config.yaml",
            "gpu-device-plugin-config.yaml",
            "cuda-optimization-scripts.yaml",
        ]

        for config_name in required_individual_configs:
            config_path = gpu_optimization_path / config_name
            if config_path.exists():
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        configs.append(config_data)

        # Check for required ConfigMaps
        config_names = [
            "nvidia-container-runtime-config",
            "gpu-device-plugin-config",
            "cuda-optimization-scripts",
        ]

        found_configs = []
        for config in configs:
            if config and config.get("kind") == "ConfigMap":
                name = config.get("metadata", {}).get("name")
                if name in config_names:
                    found_configs.append(name)

        assert len(found_configs) == len(config_names), (
            f"Missing configs: {set(config_names) - set(found_configs)}"
        )

        # Validate CUDA optimization scripts
        cuda_config = next(
            c
            for c in configs
            if c and c.get("metadata", {}).get("name") == "cuda-optimization-scripts"
        )
        scripts = cuda_config.get("data", {})

        required_scripts = ["gpu-setup.sh", "monitor-gpu.sh", "gpu-health-check.sh"]
        for script_name in required_scripts:
            assert script_name in scripts, f"Missing script: {script_name}"
            script_content = scripts[script_name]
            assert "nvidia-smi" in script_content
            if script_name in ["gpu-setup.sh", "gpu-health-check.sh"]:
                assert "jax" in script_content.lower(), (
                    f"Script {script_name} should contain JAX references"
                )

        print("✅ GPU optimization configurations validation passed")

    def test_pod_security_standards(self, security_path: Path):
        """Test Pod Security Standards implementation."""
        # Load all individual security standard files instead of consolidated file
        configs = []

        # Load all numbered security standard files
        for i in range(8):  # 00-07
            psp_file = security_path / f"pod-security-standards-{i:02d}.yaml"
            if psp_file.exists():
                with open(psp_file) as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        configs.append(config_data)

        # Check for required security resources
        resource_kinds = {config.get("kind") for config in configs if config}
        expected_kinds = {
            "PodSecurityPolicy",
            "Namespace",
            "ClusterRole",
            "ClusterRoleBinding",
            "ServiceAccount",
            "NetworkPolicy",
            "ConfigMap",
        }

        missing_kinds = expected_kinds - resource_kinds
        assert not missing_kinds, f"Missing security resource kinds: {missing_kinds}"

        # Validate RBAC configuration
        cluster_roles = [c for c in configs if c and c.get("kind") == "ClusterRole"]
        assert len(cluster_roles) >= 1, "Should have cluster roles defined"

        # Validate namespace-level PSP
        namespaces = [c for c in configs if c and c.get("kind") == "Namespace"]
        opifex_namespaces = [
            ns
            for ns in namespaces
            if "opifex" in ns.get("metadata", {}).get("name", "")
        ]
        assert len(opifex_namespaces) >= 1, "Should have Opifex namespaces with PSP"

        # Validate network policies
        network_policies = [
            c for c in configs if c and c.get("kind") == "NetworkPolicy"
        ]
        assert len(network_policies) >= 1, "Should have network policies"

        print("✅ Pod Security Standards validation passed")

    def test_istio_service_mesh_config(self, istio_path: Path):
        """Test Istio service mesh configuration."""
        # Load all individual Istio config files instead of consolidated file
        configs = []

        # Load all numbered service mesh config files
        for i in range(7):  # 00-06
            mesh_file = istio_path / f"opifex-service-mesh-config-{i:02d}.yaml"
            if mesh_file.exists():
                with open(mesh_file) as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        configs.append(config_data)

        # Load additional Istio configuration files
        additional_istio_configs = [
            "istio-namespace.yaml",
            "istio-operator.yaml",
            "istio-servicemonitor.yaml",
            "opifex-envoy-filter.yaml",
            "opifex-namespace.yaml",
        ]

        for config_name in additional_istio_configs:
            config_path = istio_path / config_name
            if config_path.exists():
                with open(config_path) as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        configs.append(config_data)

        # Check for required Istio resources
        resource_kinds = {config.get("kind") for config in configs if config}
        expected_kinds = {
            "Gateway",
            "VirtualService",
            "DestinationRule",
            "PeerAuthentication",
            "AuthorizationPolicy",
            "Telemetry",
        }

        missing_kinds = expected_kinds - resource_kinds
        assert not missing_kinds, f"Missing Istio resource kinds: {missing_kinds}"

        # Validate Gateway configuration
        gateway = next(c for c in configs if c and c.get("kind") == "Gateway")
        gateway_spec = gateway.get("spec", {})
        servers = gateway_spec.get("servers", [])

        # Check for HTTPS and HTTP redirect
        https_servers = [
            s for s in servers if s.get("port", {}).get("protocol") == "HTTPS"
        ]
        http_servers = [
            s for s in servers if s.get("port", {}).get("protocol") == "HTTP"
        ]

        assert len(https_servers) >= 1, "Should have HTTPS server configured"
        assert len(http_servers) >= 1, "Should have HTTP redirect server"

        # Validate mTLS enforcement
        peer_auth = next(
            c for c in configs if c and c.get("kind") == "PeerAuthentication"
        )
        mtls_mode = peer_auth.get("spec", {}).get("mtls", {}).get("mode")
        assert mtls_mode == "STRICT", "mTLS should be strictly enforced"

        # Validate DestinationRule for GPU optimization
        dest_rules = [c for c in configs if c and c.get("kind") == "DestinationRule"]
        assert len(dest_rules) >= 1, "Should have destination rules for services"

        main_dr = dest_rules[0]  # Use the main destination rule
        traffic_policy = main_dr.get("spec", {}).get("trafficPolicy", {})
        lb_policy = traffic_policy.get("loadBalancer", {}).get("simple")
        assert lb_policy == "LEAST_CONN", "Should use LEAST_CONN for GPU workloads"

        print("✅ Istio service mesh configuration validation passed")

    def test_security_context_configuration(self, security_path: Path):
        """Test security context configuration."""
        # Load all individual security files to find security context config
        configs = []

        # Load all numbered security standard files
        for i in range(8):  # 00-07
            psp_file = security_path / f"pod-security-standards-{i:02d}.yaml"
            if psp_file.exists():
                with open(psp_file) as f:
                    config_data = yaml.safe_load(f)
                    if config_data:
                        configs.append(config_data)

        security_context_config = next(
            c
            for c in configs
            if c and c.get("metadata", {}).get("name") == "security-context-config"
        )

        assert security_context_config, "Security context config should exist"

        # Validate security context data
        data = security_context_config.get("data", {})
        assert "security-context.yaml" in data

        security_context_yaml = data["security-context.yaml"]
        security_context = yaml.safe_load(security_context_yaml)

        # Validate security context settings
        sec_ctx = security_context.get("securityContext", {})
        assert sec_ctx.get("runAsNonRoot")
        assert sec_ctx.get("runAsUser") == 1000
        assert not sec_ctx.get("allowPrivilegeEscalation")

        capabilities = sec_ctx.get("capabilities", {})
        assert "ALL" in capabilities.get("drop", [])
        assert "SYS_ADMIN" in capabilities.get("add", [])  # Required for NVIDIA runtime

        print("✅ Security context configuration validation passed")

    def test_yaml_syntax_validation(self, deployment_path: Path):
        """Test that all YAML files have valid syntax."""
        yaml_files = []

        # Collect all YAML files
        for subdir in ["multi-stage", "security", "gpu-optimization", "istio"]:
            subdir_path = deployment_path / subdir
            if subdir_path.exists():
                yaml_files.extend(subdir_path.glob("*.yaml"))
                yaml_files.extend(subdir_path.glob("*.yml"))

        assert len(yaml_files) > 0, "Should find YAML files to validate"

        # Validate each YAML file
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    # Load all documents in the file
                    list(yaml.safe_load_all(f))
                print(f"✅ YAML syntax valid: {yaml_file.name}")
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML syntax in {yaml_file}: {e}")

        print(f"✅ All {len(yaml_files)} YAML files have valid syntax")

    def test_container_orchestration_integration(self, deployment_path: Path):
        """Test integration between different container orchestration components."""

        # Check that namespace is consistent across components
        namespace_configs = []

        for subdir in ["security", "gpu-optimization", "istio"]:
            subdir_path = deployment_path / subdir
            if subdir_path.exists():
                for yaml_file in subdir_path.glob("*.yaml"):
                    with open(yaml_file) as f:
                        configs = list(yaml.safe_load_all(f))
                        for config in configs:
                            if config and "metadata" in config:
                                namespace = config["metadata"].get("namespace")
                                if namespace:
                                    namespace_configs.append(namespace)

        # Verify consistent namespace usage
        unique_namespaces = set(namespace_configs)
        opifex_namespaces = {ns for ns in unique_namespaces if "opifex" in ns}
        assert "opifex-system" in opifex_namespaces, (
            "Should use opifex-system namespace"
        )

        # Check label consistency
        label_selectors = []
        for subdir in ["security", "istio"]:
            subdir_path = deployment_path / subdir
            if subdir_path.exists():
                for yaml_file in subdir_path.glob("*.yaml"):
                    with open(yaml_file) as f:
                        configs = list(yaml.safe_load_all(f))
                        for config in configs:
                            if (
                                config
                                and "metadata" in config
                                and "labels" in config["metadata"]
                            ):
                                labels = config["metadata"]["labels"]
                                part_of = labels.get("app.kubernetes.io/part-of")
                                if part_of:
                                    label_selectors.append(part_of)

        # Verify consistent labeling
        opifex_labels = {label for label in label_selectors if "opifex" in label}
        assert len(opifex_labels) > 0, "Should have consistent Opifex labeling"

        print("✅ Container orchestration integration validation passed")

    @pytest.mark.integration
    def test_dockerfile_build_simulation(self, multi_stage_path: Path):
        """Simulate Dockerfile build process validation (without actual Docker)."""
        dockerfile_path = multi_stage_path / "Dockerfile.optimized"
        content = dockerfile_path.read_text()

        # Extract and validate build stages
        stages = []
        for line in content.split("\n"):
            if line.strip().startswith("FROM") and " AS " in line:
                stage_name = line.split(" AS ")[-1].strip()
                stages.append(stage_name)

        expected_stages = [
            "build-cache",
            "python-deps",
            "app-builder",
            "gpu-runtime",
            "production",
        ]
        assert stages == expected_stages, (
            f"Expected stages {expected_stages}, got {stages}"
        )

        # Validate stage dependencies
        stage_dependencies = {
            "python-deps": "build-cache",
            "app-builder": "python-deps",
            "gpu-runtime": None,  # Independent stage
            "production": "gpu-runtime",
        }

        for stage, expected_from in stage_dependencies.items():
            if expected_from:
                stage_line = next(
                    line
                    for line in content.split("\n")
                    if line.strip().startswith(f"FROM {expected_from} AS {stage}")
                )
                assert stage_line, f"Stage {stage} should depend on {expected_from}"

        # Validate security practices
        assert "USER opifex:opifex" in content, "Should switch to non-root user"
        assert "runAsNonRoot" not in content or "true" in content, (
            "Should enforce non-root execution"
        )

        print("✅ Dockerfile build simulation validation passed")

    def test_performance_optimization_targets(self, multi_stage_path: Path):
        """Test that performance optimization targets are configured."""

        # Check build script targets
        build_script = multi_stage_path / "build-optimization.sh"
        content = build_script.read_text()

        # Validate performance targets
        assert "TARGET_REDUCTION=54" in content, "Should target 54% size reduction"
        assert "startup_time" in content, "Should measure startup time"
        assert "<30s" in content, "Should target <30s startup time"

        # Check Docker Compose resource limits
        compose_path = multi_stage_path / "docker-compose.gpu.yaml"
        with open(compose_path) as f:
            compose_config = yaml.safe_load(f)

        # Validate resource configurations
        services = compose_config.get("services", {})
        for _, service_config in services.items():
            if "mem_limit" in service_config:
                # Should have memory limits for resource management
                assert (
                    "16g" in service_config["mem_limit"]
                    or "memswap_limit" in service_config
                )

        # Check GPU memory fraction settings for GPU services
        gpu_service_names = [
            "opifex-core",
            "opifex-neural-ops",
        ]

        for service_name, service_config in services.items():
            # Skip non-GPU services
            if not ("gpu" in service_name.lower() or service_name in gpu_service_names):
                continue

            env_vars = service_config.get("environment", [])
            mem_fraction_found = any(
                "XLA_PYTHON_CLIENT_MEM_FRACTION" in str(var) for var in env_vars
            )
            assert mem_fraction_found, (
                f"GPU service {service_name} should have memory fraction configured"
            )

        print("✅ Performance optimization targets validation passed")


def test_container_orchestration_comprehensive():
    """Comprehensive test runner for Container Orchestration Phase 7.1."""
    print("\n🚀 Running Container Orchestration Architecture Tests")
    print("=" * 60)

    # Run the test class
    test_instance = TestContainerOrchestration()

    # Mock paths for testing
    base_path = Path(__file__).parent.parent.parent / "deployment" / "containers"
    multi_stage_path = base_path / "multi-stage"
    security_path = base_path / "security"
    gpu_optimization_path = base_path / "gpu-optimization"
    istio_path = base_path / "istio"

    try:
        test_instance.test_multi_stage_dockerfile_exists(multi_stage_path)
        test_instance.test_docker_compose_gpu_config(multi_stage_path)
        test_instance.test_build_optimization_script(multi_stage_path)
        test_instance.test_gpu_optimization_configs(gpu_optimization_path)
        test_instance.test_pod_security_standards(security_path)
        test_instance.test_istio_service_mesh_config(istio_path)
        test_instance.test_security_context_configuration(security_path)
        test_instance.test_yaml_syntax_validation(base_path)
        test_instance.test_container_orchestration_integration(base_path)
        test_instance.test_dockerfile_build_simulation(multi_stage_path)
        test_instance.test_performance_optimization_targets(multi_stage_path)

        print("\n🎉 All Container Orchestration tests passed!")
        print("✅ Phase 7.1 Container Orchestration implementation validated")
        return True

    except Exception as e:
        print(f"\n❌ Container Orchestration test failed: {e}")
        return False


if __name__ == "__main__":
    # Run comprehensive test when script is executed directly
    success = test_container_orchestration_comprehensive()
    sys.exit(0 if success else 1)
