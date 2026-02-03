"""Tests for Google Cloud Platform deployment configurations."""

import json

import pytest

from opifex.deployment.cloud.gcp import GCPConfig, GCPDeploymentManager


class TestGCPConfig:
    """Test GCP configuration class."""

    def test_default_config(self):
        """Test default GCP configuration values."""
        config = GCPConfig(project_id="test-project")

        assert config.project_id == "test-project"
        assert config.region == "us-central1"
        assert config.zone == "us-central1-a"
        assert config.cluster_name == "opifex-cluster"

        # Test default node pool configuration
        assert config.node_pool_config["initial_node_count"] == 3
        assert config.node_pool_config["machine_type"] == "n1-standard-4"
        assert config.node_pool_config["disk_size_gb"] == 100

        # Test default network configuration
        assert config.network_config["network"] == "opifex-vpc"
        assert config.network_config["subnetwork"] == "opifex-subnet"
        assert config.network_config["enable_private_nodes"] is True

        # Test default security configuration
        assert config.security_config["enable_workload_identity"] is True
        assert config.security_config["enable_network_policy"] is True
        assert config.security_config["enable_pod_security_policy"] is True

    def test_custom_config(self):
        """Test custom GCP configuration values."""
        custom_node_config = {
            "initial_node_count": 5,
            "machine_type": "n1-standard-8",
            "disk_size_gb": 200,
        }

        config = GCPConfig(
            project_id="custom-project",
            region="us-west1",
            zone="us-west1-a",
            cluster_name="custom-cluster",
            node_pool_config=custom_node_config,
        )

        assert config.project_id == "custom-project"
        assert config.region == "us-west1"
        assert config.zone == "us-west1-a"
        assert config.cluster_name == "custom-cluster"
        assert config.node_pool_config["initial_node_count"] == 5
        assert config.node_pool_config["machine_type"] == "n1-standard-8"


class TestGCPDeploymentManager:
    """Test GCP deployment manager functionality."""

    @pytest.fixture
    def gcp_config(self):
        """Create a test GCP configuration."""
        return GCPConfig(project_id="test-opifex-project")

    @pytest.fixture
    def deployment_manager(self, gcp_config):
        """Create a test GCP deployment manager."""
        return GCPDeploymentManager(gcp_config)

    def test_deployment_manager_initialization(self, gcp_config):
        """Test deployment manager initialization."""
        manager = GCPDeploymentManager(gcp_config)
        assert manager.config == gcp_config

    def test_generate_gke_cluster_config(self, deployment_manager):
        """Test GKE cluster configuration generation."""
        config = deployment_manager.generate_gke_cluster_config()

        # Test basic cluster properties
        assert config["name"] == "opifex-cluster"
        assert config["location"] == "us-central1-a"
        assert config["initial_node_count"] == 3

        # Test node configuration
        node_config = config["node_config"]
        assert node_config["machine_type"] == "n1-standard-4"
        assert node_config["disk_size_gb"] == 100
        assert (
            "https://www.googleapis.com/auth/cloud-platform"
            in node_config["oauth_scopes"]
        )

        # Test network configuration
        assert config["network"] == "opifex-vpc"
        assert config["subnetwork"] == "opifex-subnet"

        # Test security settings
        assert (
            config["workload_identity_config"]["workload_pool"]
            == "test-opifex-project.svc.id.goog"
        )
        assert config["network_policy"]["enabled"] is True
        assert config["pod_security_policy_config"]["enabled"] is True

    def test_generate_iam_policy(self, deployment_manager):
        """Test IAM policy generation."""
        policy = deployment_manager.generate_iam_policy()

        assert "bindings" in policy
        bindings = policy["bindings"]

        # Check for required roles
        role_names = [binding["role"] for binding in bindings]
        assert "roles/container.developer" in role_names
        assert "roles/secretmanager.secretAccessor" in role_names
        assert "roles/monitoring.editor" in role_names
        assert "roles/logging.logWriter" in role_names

        # Check service account references
        for binding in bindings:
            members = binding["members"]
            assert any(
                "opifex-service@test-opifex-project.iam.gserviceaccount.com" in member
                for member in members
            )

    def test_generate_vpc_config(self, deployment_manager):
        """Test VPC configuration generation."""
        vpc_config = deployment_manager.generate_vpc_config()

        # Test VPC settings
        vpc = vpc_config["vpc"]
        assert vpc["name"] == "opifex-vpc"
        assert vpc["auto_create_subnetworks"] is False
        assert vpc["routing_mode"] == "REGIONAL"

        # Test subnet settings
        subnet = vpc_config["subnet"]
        assert subnet["name"] == "opifex-subnet"
        assert subnet["ip_cidr_range"] == "10.0.0.0/16"
        assert subnet["region"] == "us-central1"

        # Test secondary IP ranges for pods and services
        secondary_ranges = subnet["secondary_ip_ranges"]
        assert len(secondary_ranges) == 2
        pods_range = next(r for r in secondary_ranges if r["range_name"] == "pods")
        services_range = next(
            r for r in secondary_ranges if r["range_name"] == "services"
        )
        assert pods_range["ip_cidr_range"] == "10.1.0.0/16"
        assert services_range["ip_cidr_range"] == "10.2.0.0/16"

        # Test firewall rules
        firewall_rules = vpc_config["firewall_rules"]
        assert len(firewall_rules) == 3

        rule_names = [rule["name"] for rule in firewall_rules]
        assert "opifex-allow-internal" in rule_names
        assert "opifex-allow-ssh" in rule_names
        assert "opifex-allow-api" in rule_names

    def test_generate_secret_manager_config(self, deployment_manager):
        """Test Secret Manager configuration generation."""
        secrets_config = deployment_manager.generate_secret_manager_config()

        secrets = secrets_config["secrets"]
        assert len(secrets) == 3

        secret_ids = [secret["secret_id"] for secret in secrets]
        assert "opifex-database-password" in secret_ids
        assert "opifex-api-keys" in secret_ids
        assert "opifex-oauth-credentials" in secret_ids

        # Test replication settings
        for secret in secrets:
            assert "replication" in secret

    def test_generate_monitoring_config(self, deployment_manager):
        """Test Cloud Monitoring configuration generation."""
        monitoring_config = deployment_manager.generate_monitoring_config()

        # Test notification channels
        channels = monitoring_config["notification_channels"]
        assert len(channels) == 1
        assert channels[0]["type"] == "email"
        assert channels[0]["display_name"] == "Opifex Operations"

        # Test alert policies
        policies = monitoring_config["alert_policies"]
        assert len(policies) == 2

        policy_names = [policy["display_name"] for policy in policies]
        assert "High CPU Usage" in policy_names
        assert "High Memory Usage" in policy_names

    def test_export_terraform_config(self, deployment_manager, tmp_path):
        """Test Terraform configuration export."""
        output_dir = tmp_path / "terraform"
        deployment_manager.export_terraform_config(output_dir)

        # Check that files were created
        assert (output_dir / "main.tf").exists()
        assert (output_dir / "variables.tf").exists()
        assert (output_dir / "outputs.tf").exists()

        # Check main.tf content
        with open(output_dir / "main.tf") as f:
            main_config = json.load(f)

        # Test provider configuration
        assert "provider" in main_config
        assert "google" in main_config["provider"]
        provider_config = main_config["provider"]["google"]
        assert provider_config["project"] == "test-opifex-project"
        assert provider_config["region"] == "us-central1"

        # Test resource configuration
        assert "resource" in main_config
        resources = main_config["resource"]
        assert "google_container_cluster" in resources
        assert "google_compute_network" in resources
        assert "google_compute_subnetwork" in resources

    def test_terraform_variables_content(self, deployment_manager):
        """Test Terraform variables file content."""
        variables_content = deployment_manager._generate_terraform_variables()

        assert 'variable "project_id"' in variables_content
        assert 'variable "region"' in variables_content
        assert 'variable "zone"' in variables_content
        assert 'variable "cluster_name"' in variables_content

        # Check default values
        assert 'default     = "us-central1"' in variables_content
        assert 'default     = "us-central1-a"' in variables_content
        assert 'default     = "opifex-cluster"' in variables_content

    def test_terraform_outputs_content(self, deployment_manager):
        """Test Terraform outputs file content."""
        outputs_content = deployment_manager._generate_terraform_outputs()

        assert 'output "cluster_endpoint"' in outputs_content
        assert 'output "cluster_ca_certificate"' in outputs_content
        assert 'output "vpc_name"' in outputs_content
        assert 'output "subnet_name"' in outputs_content

        # Check sensitive marking
        assert "sensitive   = true" in outputs_content

    def test_custom_configuration_propagation(self):
        """Test that custom configuration values propagate through all generated configs."""
        custom_config = GCPConfig(
            project_id="custom-project",
            region="europe-west1",
            zone="europe-west1-b",
            cluster_name="custom-opifex",
        )

        manager = GCPDeploymentManager(custom_config)

        # Test GKE config
        gke_config = manager.generate_gke_cluster_config()
        assert gke_config["name"] == "custom-opifex"
        assert gke_config["location"] == "europe-west1-b"

        # Test VPC config
        vpc_config = manager.generate_vpc_config()
        subnet = vpc_config["subnet"]
        assert subnet["region"] == "europe-west1"

        # Test IAM policy
        iam_policy = manager.generate_iam_policy()
        for binding in iam_policy["bindings"]:
            for member in binding["members"]:
                if "serviceAccount:" in member:
                    assert "custom-project" in member

    def test_security_configuration_validation(self, deployment_manager):
        """Test that security configurations are properly applied."""
        cluster_config = deployment_manager.generate_gke_cluster_config()

        # Test private cluster configuration
        private_config = cluster_config["private_cluster_config"]
        assert private_config["enable_private_nodes"] is True
        assert private_config["enable_private_endpoint"] is False
        assert private_config["master_ipv4_cidr_block"] == "172.16.0.0/28"

        # Test workload identity
        workload_identity = cluster_config["workload_identity_config"]
        assert "workload_pool" in workload_identity

        # Test network policy
        network_policy = cluster_config["network_policy"]
        assert network_policy["enabled"] is True
