"""Tests for Amazon Web Services deployment configurations."""

import json

import pytest

from opifex.deployment.cloud.aws import AWSConfig, AWSDeploymentManager


class TestAWSConfig:
    """Test AWS configuration class."""

    def test_default_config(self):
        """Test default AWS configuration values."""
        config = AWSConfig()

        assert config.region == "us-east-1"
        assert config.cluster_name == "opifex-cluster"
        assert config.vpc_cidr == "10.0.0.0/16"

        # Test default node group configuration
        assert config.node_group_config["desired_size"] == 3
        assert config.node_group_config["max_size"] == 10
        assert config.node_group_config["min_size"] == 1
        assert config.node_group_config["instance_types"] == ["m5.xlarge"]
        assert config.node_group_config["ami_type"] == "AL2_x86_64_GPU"

        # Test default network configuration
        assert len(config.network_config["availability_zones"]) == 3
        assert len(config.network_config["private_subnets"]) == 3
        assert len(config.network_config["public_subnets"]) == 3
        assert config.network_config["enable_nat_gateway"] is True

        # Test default security configuration
        assert config.security_config["enable_logging"] is True
        assert len(config.security_config["log_types"]) == 5
        assert config.security_config["enable_private_access"] is True
        assert config.security_config["enable_public_access"] is True

    def test_custom_config(self):
        """Test custom AWS configuration values."""
        custom_node_config = {
            "desired_size": 5,
            "max_size": 15,
            "min_size": 2,
            "instance_types": ["m5.2xlarge"],
        }

        config = AWSConfig(
            region="us-west-2",
            cluster_name="custom-cluster",
            vpc_cidr="172.16.0.0/16",
            node_group_config=custom_node_config,
        )

        assert config.region == "us-west-2"
        assert config.cluster_name == "custom-cluster"
        assert config.vpc_cidr == "172.16.0.0/16"
        assert config.node_group_config["desired_size"] == 5
        assert config.node_group_config["instance_types"] == ["m5.2xlarge"]


class TestAWSDeploymentManager:
    """Test AWS deployment manager functionality."""

    @pytest.fixture
    def aws_config(self):
        """Create a test AWS configuration."""
        return AWSConfig(cluster_name="test-opifex-cluster")

    @pytest.fixture
    def deployment_manager(self, aws_config):
        """Create a test AWS deployment manager."""
        return AWSDeploymentManager(aws_config)

    def test_deployment_manager_initialization(self, aws_config):
        """Test deployment manager initialization."""
        manager = AWSDeploymentManager(aws_config)
        assert manager.config == aws_config

    def test_generate_eks_cluster_config(self, deployment_manager):
        """Test EKS cluster configuration generation."""
        config = deployment_manager.generate_eks_cluster_config()

        # Test basic cluster properties
        assert config["name"] == "test-opifex-cluster"
        assert config["version"] == "1.28"
        assert "test-opifex-cluster-cluster-role" in config["role_arn"]

        # Test VPC configuration
        vpc_config = config["vpc_config"]
        assert len(vpc_config["subnet_ids"]) == 6  # 3 private + 3 public
        assert vpc_config["endpoint_config_private_access"] is True
        assert vpc_config["endpoint_config_public_access"] is True

        # Test logging configuration
        assert config["enabled_cluster_log_types"] == [
            "api",
            "audit",
            "authenticator",
            "controllerManager",
            "scheduler",
        ]

        # Test encryption configuration
        assert len(config["encryption_config"]) == 1
        assert config["encryption_config"][0]["resources"] == ["secrets"]

        # Test tags
        tags = config["tags"]
        assert tags["Environment"] == "production"
        assert tags["Project"] == "opifex"
        assert tags["ManagedBy"] == "terraform"

    def test_generate_node_group_config(self, deployment_manager):
        """Test EKS node group configuration generation."""
        config = deployment_manager.generate_node_group_config()

        # Test basic node group properties
        assert config["cluster_name"] == "test-opifex-cluster"
        assert config["node_group_name"] == "test-opifex-cluster-nodes"
        assert "test-opifex-cluster-node-role" in config["node_role_arn"]

        # Test scaling configuration
        scaling_config = config["scaling_config"]
        assert scaling_config["desired_size"] == 3
        assert scaling_config["max_size"] == 10
        assert scaling_config["min_size"] == 1

        # Test instance configuration
        assert config["instance_types"] == ["m5.xlarge"]
        assert config["ami_type"] == "AL2_x86_64_GPU"
        assert config["capacity_type"] == "ON_DEMAND"
        assert config["disk_size"] == 20

        # Test update configuration
        assert config["update_config"]["max_unavailable"] == 1

        # Test remote access
        assert config["remote_access"]["ec2_ssh_key"] == "test-opifex-cluster-keypair"

    def test_generate_iam_roles(self, deployment_manager):
        """Test IAM roles generation."""
        iam_config = deployment_manager.generate_iam_roles()

        # Test cluster role
        cluster_role = iam_config["cluster_role"]
        assert cluster_role["name"] == "test-opifex-cluster-cluster-role"
        assert "eks.amazonaws.com" in str(cluster_role["assume_role_policy"])
        assert "AmazonEKSClusterPolicy" in cluster_role["managed_policy_arns"][0]

        # Test node role
        node_role = iam_config["node_role"]
        assert node_role["name"] == "test-opifex-cluster-node-role"
        assert "ec2.amazonaws.com" in str(node_role["assume_role_policy"])
        managed_policies = node_role["managed_policy_arns"]
        assert any("AmazonEKSWorkerNodePolicy" in policy for policy in managed_policies)
        assert any("AmazonEKS_CNI_Policy" in policy for policy in managed_policies)
        assert any(
            "AmazonEC2ContainerRegistryReadOnly" in policy
            for policy in managed_policies
        )

        # Test service role with inline policies
        service_role = iam_config["service_role"]
        assert service_role["name"] == "test-opifex-cluster-service-role"
        inline_policies = service_role["inline_policies"]
        assert len(inline_policies) == 2

        policy_names = [policy["name"] for policy in inline_policies]
        assert "SecretsManagerAccess" in policy_names
        assert "CloudWatchAccess" in policy_names

    def test_generate_vpc_config(self, deployment_manager):
        """Test VPC configuration generation."""
        vpc_config = deployment_manager.generate_vpc_config()

        # Test VPC settings
        vpc = vpc_config["vpc"]
        assert vpc["cidr_block"] == "10.0.0.0/16"
        assert vpc["enable_dns_hostnames"] is True
        assert vpc["enable_dns_support"] is True
        assert vpc["tags"]["Name"] == "test-opifex-cluster-vpc"

        # Test Internet Gateway
        igw = vpc_config["internet_gateway"]
        assert igw["tags"]["Name"] == "test-opifex-cluster-igw"

        # Test private subnets
        private_subnets = vpc_config["private_subnets"]
        assert len(private_subnets) == 3
        for i, subnet in enumerate(private_subnets):
            assert f"10.0.{i + 1}.0/24" == subnet["cidr_block"]
            assert f"us-east-1{chr(97 + i)}" == subnet["availability_zone"]
            assert "kubernetes.io/role/internal-elb" in subnet["tags"]

        # Test public subnets
        public_subnets = vpc_config["public_subnets"]
        assert len(public_subnets) == 3
        for i, subnet in enumerate(public_subnets):
            assert f"10.0.{101 + i}.0/24" == subnet["cidr_block"]
            assert subnet["map_public_ip_on_launch"] is True
            assert "kubernetes.io/role/elb" in subnet["tags"]

        # Test NAT gateways
        nat_gateways = vpc_config["nat_gateways"]
        assert len(nat_gateways) == 3

        # Test security groups
        security_groups = vpc_config["security_groups"]
        assert "cluster_sg" in security_groups
        assert "node_sg" in security_groups

        cluster_sg = security_groups["cluster_sg"]
        assert cluster_sg["name"] == "test-opifex-cluster-cluster-sg"
        assert len(cluster_sg["ingress"]) == 1
        assert cluster_sg["ingress"][0]["from_port"] == 443

    def test_generate_secrets_manager_config(self, deployment_manager):
        """Test Secrets Manager configuration generation."""
        secrets_config = deployment_manager.generate_secrets_manager_config()

        secrets = secrets_config["secrets"]
        assert len(secrets) == 3

        # Test secret names and configurations
        secret_names = [secret["name"] for secret in secrets]
        assert "opifex/database/password" in secret_names
        assert "opifex/api/keys" in secret_names
        assert "opifex/oauth/credentials" in secret_names

        # Test password generation config
        db_secret = next(s for s in secrets if s["name"] == "opifex/database/password")
        gen_config = db_secret["generate_secret_string"]
        assert gen_config["password_length"] == 32
        assert gen_config["exclude_characters"] == '"@/\\'

        # Test JSON secrets
        api_secret = next(s for s in secrets if s["name"] == "opifex/api/keys")
        api_data = json.loads(api_secret["secret_string"])
        assert "api_key_1" in api_data
        assert "api_key_2" in api_data

    def test_generate_cloudwatch_config(self, deployment_manager):
        """Test CloudWatch configuration generation."""
        cloudwatch_config = deployment_manager.generate_cloudwatch_config()

        # Test log groups
        log_groups = cloudwatch_config["log_groups"]
        assert len(log_groups) == 2

        log_group_names = [lg["name"] for lg in log_groups]
        assert "/aws/eks/test-opifex-cluster/cluster" in log_group_names
        assert "/aws/eks/test-opifex-cluster/application" in log_group_names

        # Test retention periods
        cluster_log = next(lg for lg in log_groups if "cluster" in lg["name"])
        app_log = next(lg for lg in log_groups if "application" in lg["name"])
        assert cluster_log["retention_in_days"] == 30
        assert app_log["retention_in_days"] == 7

        # Test alarms
        alarms = cloudwatch_config["alarms"]
        assert len(alarms) == 2

        alarm_names = [alarm["alarm_name"] for alarm in alarms]
        assert "test-opifex-cluster-high-cpu" in alarm_names
        assert "test-opifex-cluster-high-memory" in alarm_names

        # Test alarm configurations
        cpu_alarm = next(a for a in alarms if "cpu" in a["alarm_name"])
        assert cpu_alarm["threshold"] == 80
        assert cpu_alarm["metric_name"] == "CPUUtilization"
        assert cpu_alarm["namespace"] == "AWS/EKS"

        # Test dashboard
        dashboard = cloudwatch_config["dashboard"]
        assert dashboard["dashboard_name"] == "test-opifex-cluster-dashboard"
        dashboard_body = json.loads(dashboard["dashboard_body"])
        assert "widgets" in dashboard_body

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
        assert "aws" in main_config["provider"]
        provider_config = main_config["provider"]["aws"]
        assert provider_config["region"] == "us-east-1"

        # Test resource configuration
        assert "resource" in main_config
        resources = main_config["resource"]
        assert "aws_eks_cluster" in resources
        assert "aws_eks_node_group" in resources
        assert "aws_vpc" in resources

    def test_terraform_variables_content(self, deployment_manager):
        """Test Terraform variables file content."""
        variables_content = deployment_manager._generate_terraform_variables()

        assert 'variable "region"' in variables_content
        assert 'variable "cluster_name"' in variables_content
        assert 'variable "node_instance_types"' in variables_content
        assert 'variable "desired_capacity"' in variables_content

        # Check default values
        assert 'default     = "us-east-1"' in variables_content
        assert 'default     = "opifex-cluster"' in variables_content
        assert 'default     = ["m5.xlarge"]' in variables_content

    def test_terraform_outputs_content(self, deployment_manager):
        """Test Terraform outputs file content."""
        outputs_content = deployment_manager._generate_terraform_outputs()

        assert 'output "cluster_endpoint"' in outputs_content
        assert 'output "cluster_ca_certificate"' in outputs_content
        assert 'output "vpc_id"' in outputs_content
        assert 'output "cluster_security_group_id"' in outputs_content

        # Check sensitive marking
        assert "sensitive   = true" in outputs_content

    def test_custom_configuration_propagation(self):
        """Test that custom configuration values propagate through all generated configs."""
        custom_config = AWSConfig(
            region="eu-west-1", cluster_name="custom-opifex", vpc_cidr="172.16.0.0/16"
        )

        manager = AWSDeploymentManager(custom_config)

        # Test EKS config
        eks_config = manager.generate_eks_cluster_config()
        assert eks_config["name"] == "custom-opifex"

        # Test VPC config
        vpc_config = manager.generate_vpc_config()
        vpc = vpc_config["vpc"]
        assert vpc["cidr_block"] == "172.16.0.0/16"
        assert vpc["tags"]["Name"] == "custom-opifex-vpc"

        # Test CloudWatch config
        cloudwatch_config = manager.generate_cloudwatch_config()
        dashboard = cloudwatch_config["dashboard"]
        assert dashboard["dashboard_name"] == "custom-opifex-dashboard"

    def test_security_configuration_validation(self, deployment_manager):
        """Test that security configurations are properly applied."""
        eks_config = deployment_manager.generate_eks_cluster_config()

        # Test encryption configuration
        encryption_config = eks_config["encryption_config"]
        assert len(encryption_config) == 1
        assert encryption_config[0]["resources"] == ["secrets"]
        assert "key_id" in encryption_config[0]["provider"]

        # Test logging enabled
        log_types = eks_config["enabled_cluster_log_types"]
        expected_logs = [
            "api",
            "audit",
            "authenticator",
            "controllerManager",
            "scheduler",
        ]
        assert log_types == expected_logs

        # Test private/public access
        vpc_config = eks_config["vpc_config"]
        assert vpc_config["endpoint_config_private_access"] is True
        assert vpc_config["endpoint_config_public_access"] is True

    def test_iam_policy_security_validation(self, deployment_manager):
        """Test that IAM policies follow security best practices."""
        iam_config = deployment_manager.generate_iam_roles()

        # Test service role inline policies
        service_role = iam_config["service_role"]
        inline_policies = service_role["inline_policies"]

        # Check Secrets Manager policy
        secrets_policy = next(
            p for p in inline_policies if p["name"] == "SecretsManagerAccess"
        )
        policy_doc = secrets_policy["policy"]
        statements = policy_doc["Statement"]

        secrets_statement = statements[0]
        assert secrets_statement["Effect"] == "Allow"
        assert "secretsmanager:GetSecretValue" in secrets_statement["Action"]
        assert "opifex/*" in secrets_statement["Resource"]

        # Check CloudWatch policy
        cloudwatch_policy = next(
            p for p in inline_policies if p["name"] == "CloudWatchAccess"
        )
        policy_doc = cloudwatch_policy["policy"]
        statements = policy_doc["Statement"]

        cloudwatch_statement = statements[0]
        assert cloudwatch_statement["Effect"] == "Allow"
        assert "cloudwatch:PutMetricData" in cloudwatch_statement["Action"]
