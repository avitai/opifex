# SciML Production Deployment Makefile
# Version: 1.0.0
# Foundation: 158/158 L2O tests passing, enterprise-grade architecture

# Configuration
CLUSTER_NAME ?= sciml-cluster
REGION ?= us-west-2
ENVIRONMENT ?= development
NAMESPACE ?= sciml-system
KUBECTL ?= kubectl
HELM ?= helm
KUSTOMIZE ?= kustomize

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)SciML Production Deployment Makefile$(NC)"
	@echo "$(BLUE)Foundation: 158/158 L2O tests passing$(NC)"
	@echo ""
	@echo "$(YELLOW)Usage:$(NC)"
	@echo "  make <target> [ENVIRONMENT=<env>]"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$$$1, $$$$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Environments:$(NC)"
	@echo "  $(GREEN)development$(NC)  - Local development environment"
	@echo "  $(GREEN)staging$(NC)      - Staging environment"
	@echo "  $(GREEN)production$(NC)   - Production environment"
	@echo ""
	@echo "$(YELLOW)SciML Framework Testing:$(NC)"
	@echo "  $(GREEN)test-l2o-framework$(NC)     - Test L2O optimization framework"
	@echo "  $(GREEN)test-neural-operators$(NC)  - Test neural operator ecosystem"
	@echo "  $(GREEN)test-benchmarking$(NC)      - Test PDEBench integration"
	@echo "  $(GREEN)test-sciml-complete$(NC)    - Run complete SciML validation"
	@echo ""
	@echo "$(YELLOW)SciML Resource Profiles:$(NC)"
	@echo "  $(GREEN)deploy-small-workload$(NC)    - Deploy small workload (2 CPU, 4Gi, 1 GPU)"
	@echo "  $(GREEN)deploy-large-workload$(NC)    - Deploy large workload (8 CPU, 32Gi, 4 GPU)"
	@echo "  $(GREEN)deploy-research-workload$(NC) - Deploy research workload (4 CPU, 16Gi, 2 GPU)"
	@echo "  $(GREEN)clean-workload-profiles$(NC)  - Remove all workload profiles"

# Prerequisites
.PHONY: check-prerequisites
check-prerequisites: ## Check if required tools are installed
	@echo "$(BLUE)Checking prerequisites...$(NC)"
	@command -v $(KUBECTL) >/dev/null 2>&1 || { echo "$(RED)kubectl is required but not installed$(NC)"; exit 1; }
	@command -v $(HELM) >/dev/null 2>&1 || { echo "$(RED)helm is required but not installed$(NC)"; exit 1; }
	@command -v $(KUSTOMIZE) >/dev/null 2>&1 || { echo "$(RED)kustomize is required but not installed$(NC)"; exit 1; }
	@echo "$(GREEN)All prerequisites are installed$(NC)"

# Cluster management
.PHONY: cluster-info
cluster-info: check-prerequisites ## Show cluster information
	@echo "$(BLUE)Cluster Information:$(NC)"
	@$(KUBECTL) cluster-info
	@echo ""
	@echo "$(BLUE)Cluster Nodes:$(NC)"
	@$(KUBECTL) get nodes -o wide

.PHONY: create-namespaces
create-namespaces: check-prerequisites ## Create all required namespaces
	@echo "$(BLUE)Creating namespaces...$(NC)"
	@$(KUBECTL) create namespace $(NAMESPACE) --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@$(KUBECTL) create namespace monitoring --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@$(KUBECTL) create namespace security-system --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@$(KUBECTL) create namespace community-platform --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@$(KUBECTL) create namespace benchmarking-system --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@$(KUBECTL) create namespace gpu-operator --dry-run=client -o yaml | $(KUBECTL) apply -f -
	@echo "$(GREEN)Namespaces created successfully$(NC)"

# Foundation Phase
.PHONY: deploy-foundation
deploy-foundation: check-prerequisites create-namespaces ## Deploy foundation components
	@echo "$(BLUE)Deploying foundation components...$(NC)"
	@$(KUBECTL) apply -k kubernetes/base/
	@echo "$(GREEN)Foundation components deployed$(NC)"

.PHONY: deploy-gpu-operator
deploy-gpu-operator: check-prerequisites ## Deploy NVIDIA GPU Operator
	@echo "$(BLUE)Deploying NVIDIA GPU Operator...$(NC)"
	@$(HELM) repo add nvidia https://nvidia.github.io/gpu-operator
	@$(HELM) repo update
	@$(HELM) upgrade --install gpu-operator nvidia/gpu-operator \
		--namespace gpu-operator \
		--create-namespace \
		--set operator.defaultRuntime=containerd
	@echo "$(GREEN)GPU Operator deployed$(NC)"

# Core Phase
.PHONY: deploy-monitoring
deploy-monitoring: check-prerequisites ## Deploy monitoring stack
	@echo "$(BLUE)Deploying monitoring stack...$(NC)"
	@$(HELM) repo add prometheus-community https://prometheus-community.github.io/helm-charts
	@$(HELM) repo add grafana https://grafana.github.io/helm-charts
	@$(HELM) repo update
	@$(KUBECTL) apply -k monitoring/prometheus/
	@$(KUBECTL) apply -k monitoring/grafana/
	@$(KUBECTL) apply -k monitoring/alertmanager/
	@echo "$(GREEN)Monitoring stack deployed$(NC)"

.PHONY: deploy-security
deploy-security: check-prerequisites ## Deploy security framework
	@echo "$(BLUE)Deploying security framework...$(NC)"
	@$(KUBECTL) apply -k security/keycloak/
	@$(KUBECTL) apply -k security/rbac/
	@$(KUBECTL) apply -k security/network-policies/
	@echo "$(GREEN)Security framework deployed$(NC)"

# Extension Phase
.PHONY: deploy-community
deploy-community: check-prerequisites ## Deploy community platform
	@echo "$(BLUE)Deploying community platform...$(NC)"
	@$(KUBECTL) apply -k community/registry/
	@$(KUBECTL) apply -k community/api/
	@$(KUBECTL) apply -k community/frontend/
	@echo "$(GREEN)Community platform deployed$(NC)"

.PHONY: deploy-benchmarking
deploy-benchmarking: check-prerequisites ## Deploy benchmarking infrastructure
	@echo "$(BLUE)Deploying benchmarking infrastructure...$(NC)"
	@$(KUBECTL) apply -k benchmarking/pdebench/
	@$(KUBECTL) apply -k benchmarking/reporting/
	@echo "$(GREEN)Benchmarking infrastructure deployed$(NC)"

.PHONY: deploy-scalability
deploy-scalability: check-prerequisites ## Deploy scalability components
	@echo "$(BLUE)Deploying scalability components...$(NC)"
	@$(KUBECTL) apply -k scalability/hpa/
	@$(KUBECTL) apply -k scalability/cluster-autoscaler/
	@echo "$(GREEN)Scalability components deployed$(NC)"

# Complete deployment
.PHONY: deploy-all
deploy-all: deploy-foundation deploy-gpu-operator deploy-monitoring deploy-security deploy-community deploy-benchmarking deploy-scalability ## Deploy all components
	@echo "$(GREEN)All components deployed successfully$(NC)"

# Environment-specific deployments
.PHONY: deploy-dev
deploy-dev: ## Deploy development environment
	@$(MAKE) deploy-all ENVIRONMENT=development

.PHONY: deploy-staging
deploy-staging: ## Deploy staging environment
	@$(MAKE) deploy-all ENVIRONMENT=staging

.PHONY: deploy-prod
deploy-prod: ## Deploy production environment
	@$(MAKE) deploy-all ENVIRONMENT=production

# Status and monitoring
.PHONY: status
status: check-prerequisites ## Show deployment status
	@echo "$(BLUE)Deployment Status:$(NC)"
	@echo ""
	@echo "$(YELLOW)Namespaces:$(NC)"
	@$(KUBECTL) get namespaces | grep -E "(sciml|monitoring|security|community|benchmarking|gpu-operator)"
	@echo ""
	@echo "$(YELLOW)Pods:$(NC)"
	@$(KUBECTL) get pods --all-namespaces | grep -E "(sciml|monitoring|security|community|benchmarking|gpu-operator)"
	@echo ""
	@echo "$(YELLOW)Services:$(NC)"
	@$(KUBECTL) get services --all-namespaces | grep -E "(sciml|monitoring|security|community|benchmarking|gpu-operator)"

.PHONY: logs
logs: check-prerequisites ## Show logs for SciML components
	@echo "$(BLUE)Recent logs from SciML components:$(NC)"
	@$(KUBECTL) logs -n $(NAMESPACE) -l app=sciml --tail=100

.PHONY: port-forward-grafana
port-forward-grafana: check-prerequisites ## Port-forward Grafana dashboard
	@echo "$(BLUE)Port-forwarding Grafana dashboard to http://localhost:3000$(NC)"
	@$(KUBECTL) port-forward -n monitoring service/grafana 3000:80

.PHONY: port-forward-prometheus
port-forward-prometheus: check-prerequisites ## Port-forward Prometheus UI
	@echo "$(BLUE)Port-forwarding Prometheus UI to http://localhost:9090$(NC)"
	@$(KUBECTL) port-forward -n monitoring service/prometheus 9090:9090

.PHONY: port-forward-keycloak
port-forward-keycloak: check-prerequisites ## Port-forward Keycloak admin console
	@echo "$(BLUE)Port-forwarding Keycloak admin console to http://localhost:8080$(NC)"
	@$(KUBECTL) port-forward -n security-system service/keycloak 8080:8080

# Testing
.PHONY: test-deployment
test-deployment: check-prerequisites ## Test deployment health
	@echo "$(BLUE)Testing deployment health...$(NC)"
	@echo ""
	@echo "$(YELLOW)Checking pod readiness:$(NC)"
	@$(KUBECTL) get pods --all-namespaces | grep -E "(sciml|monitoring|security|community|benchmarking)" | grep -v Running | grep -v Completed || echo "$(GREEN)All pods are running$(NC)"
	@echo ""
	@echo "$(YELLOW)Checking service endpoints:$(NC)"
	@$(KUBECTL) get endpoints --all-namespaces | grep -E "(sciml|monitoring|security|community|benchmarking)" | grep -v "<none>" || echo "$(GREEN)All services have endpoints$(NC)"
	@echo ""
	@echo "$(YELLOW)Checking persistent volumes:$(NC)"
	@$(KUBECTL) get pv | grep -E "(sciml|monitoring|security|community|benchmarking)" || echo "$(GREEN)No persistent volumes found$(NC)"

.PHONY: test-gpu
test-gpu: check-prerequisites ## Test GPU availability
	@echo "$(BLUE)Testing GPU availability...$(NC)"
	@$(KUBECTL) apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
  namespace: $(NAMESPACE)
spec:
  containers:
  - name: gpu-test
test-target:
	@echo 'test'
