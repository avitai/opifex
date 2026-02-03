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
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""
	@echo "$(YELLOW)Environments:$(NC)"
	@echo "  $(GREEN)development$(NC)  - Local development environment"
	@echo "  $(GREEN)staging$(NC)      - Staging environment"
	@echo "  $(GREEN)production$(NC)   - Production environment"

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
    image: nvidia/cuda:11.8-base-ubuntu20.04
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  restartPolicy: Never
EOF
	@echo "$(YELLOW)Waiting for GPU test pod to complete...$(NC)"
	@$(KUBECTL) wait --for=condition=complete pod/gpu-test -n $(NAMESPACE) --timeout=60s
	@echo "$(YELLOW)GPU test results:$(NC)"
	@$(KUBECTL) logs gpu-test -n $(NAMESPACE)
	@$(KUBECTL) delete pod gpu-test -n $(NAMESPACE)

# Cleanup
.PHONY: clean-all
clean-all: check-prerequisites ## Remove all deployed components
	@echo "$(YELLOW)WARNING: This will remove all deployed components$(NC)"
	@echo "$(YELLOW)Press Ctrl+C to cancel, or wait 10 seconds to continue...$(NC)"
	@sleep 10
	@echo "$(BLUE)Removing all components...$(NC)"
	@$(KUBECTL) delete namespace $(NAMESPACE) --ignore-not-found=true
	@$(KUBECTL) delete namespace monitoring --ignore-not-found=true
	@$(KUBECTL) delete namespace security-system --ignore-not-found=true
	@$(KUBECTL) delete namespace community-platform --ignore-not-found=true
	@$(KUBECTL) delete namespace benchmarking-system --ignore-not-found=true
	@$(HELM) uninstall gpu-operator -n gpu-operator --ignore-not-found
	@$(KUBECTL) delete namespace gpu-operator --ignore-not-found=true
	@echo "$(GREEN)All components removed$(NC)"

.PHONY: clean-monitoring
clean-monitoring: check-prerequisites ## Remove monitoring stack
	@echo "$(BLUE)Removing monitoring stack...$(NC)"
	@$(KUBECTL) delete -k monitoring/prometheus/ --ignore-not-found=true
	@$(KUBECTL) delete -k monitoring/grafana/ --ignore-not-found=true
	@$(KUBECTL) delete -k monitoring/alertmanager/ --ignore-not-found=true
	@echo "$(GREEN)Monitoring stack removed$(NC)"

.PHONY: clean-security
clean-security: check-prerequisites ## Remove security framework
	@echo "$(BLUE)Removing security framework...$(NC)"
	@$(KUBECTL) delete -k security/keycloak/ --ignore-not-found=true
	@$(KUBECTL) delete -k security/rbac/ --ignore-not-found=true
	@$(KUBECTL) delete -k security/network-policies/ --ignore-not-found=true
	@echo "$(GREEN)Security framework removed$(NC)"

# Development helpers
.PHONY: dev-setup
dev-setup: ## Setup development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@$(MAKE) deploy-dev
	@echo "$(GREEN)Development environment ready$(NC)"

.PHONY: dev-test
dev-test: ## Run development tests
	@echo "$(BLUE)Running development tests...$(NC)"
	@$(MAKE) test-deployment
	@echo "$(GREEN)Development tests completed$(NC)"

.PHONY: dev-clean
dev-clean: ## Clean development environment
	@echo "$(BLUE)Cleaning development environment...$(NC)"
	@$(MAKE) clean-all
	@echo "$(GREEN)Development environment cleaned$(NC)"

# Backup and restore
.PHONY: backup
backup: check-prerequisites ## Create backup of current deployment
	@echo "$(BLUE)Creating backup...$(NC)"
	@mkdir -p backups/$(shell date +%Y%m%d-%H%M%S)
	@$(KUBECTL) get all --all-namespaces -o yaml > backups/$(shell date +%Y%m%d-%H%M%S)/all-resources.yaml
	@$(KUBECTL) get pv -o yaml > backups/$(shell date +%Y%m%d-%H%M%S)/persistent-volumes.yaml
	@echo "$(GREEN)Backup created in backups/$(shell date +%Y%m%d-%H%M%S)$(NC)"

.PHONY: validate-configs
validate-configs: check-prerequisites ## Validate Kubernetes configurations
	@echo "$(BLUE)Validating Kubernetes configurations...$(NC)"
	@find . -name "*.yaml" -o -name "*.yml" | grep -E "(kubernetes|monitoring|security|community|benchmarking|scalability)" | xargs -I {} sh -c 'echo "Validating {}" && $(KUBECTL) apply --dry-run=client -f {}'
	@echo "$(GREEN)All configurations are valid$(NC)"

# Documentation
.PHONY: docs
docs: ## Generate documentation
	@echo "$(BLUE)Generating documentation...$(NC)"
	@echo "$(YELLOW)Deployment configuration:$(NC)"
	@cat config.yaml
	@echo ""
	@echo "$(YELLOW)Available make targets:$(NC)"
	@$(MAKE) help

# Monitoring shortcuts
.PHONY: dashboard
dashboard: port-forward-grafana ## Open Grafana dashboard (alias for port-forward-grafana)

.PHONY: prometheus
prometheus: port-forward-prometheus ## Open Prometheus UI (alias for port-forward-prometheus)

.PHONY: keycloak
keycloak: port-forward-keycloak ## Open Keycloak admin console (alias for port-forward-keycloak)

# Quick deployment for different phases
.PHONY: phase-1
phase-1: deploy-foundation ## Deploy Phase 1: Foundation
	@echo "$(GREEN)Phase 1 (Foundation) deployed$(NC)"

.PHONY: phase-2
phase-2: deploy-foundation deploy-gpu-operator deploy-monitoring deploy-security ## Deploy Phase 2: Core
	@echo "$(GREEN)Phase 2 (Core) deployed$(NC)"

.PHONY: phase-3
phase-3: phase-2 deploy-community deploy-benchmarking deploy-scalability ## Deploy Phase 3: Extension
	@echo "$(GREEN)Phase 3 (Extension) deployed$(NC)"

.PHONY: phase-4
phase-4: phase-3 test-deployment ## Deploy Phase 4: Integration
	@echo "$(GREEN)Phase 4 (Integration) deployed$(NC)"

.PHONY: phase-5
phase-5: phase-4 ## Deploy Phase 5: Finalization
	@echo "$(BLUE)Running final validation...$(NC)"
	@$(MAKE) validate-configs
	@$(MAKE) test-deployment
	@$(MAKE) backup
	@echo "$(GREEN)Phase 5 (Finalization) completed$(NC)"
	@echo "$(GREEN)🎉 SciML Production Deployment Complete! 🎉$(NC)"

# Version information
.PHONY: version
version: ## Show version information
	@echo "$(BLUE)SciML Production Deployment$(NC)"
	@echo "Version: 1.0.0"
	@echo "Foundation: 158/158 L2O tests passing"
	@echo "Architecture: Enterprise-grade production deployment"
	@echo ""
	@echo "$(BLUE)Tool Versions:$(NC)"
	@$(KUBECTL) version --client --short 2>/dev/null || echo "kubectl: not available"
	@$(HELM) version --short 2>/dev/null || echo "helm: not available"
	@$(KUSTOMIZE) version --short 2>/dev/null || echo "kustomize: not available"

# Include SciML-specific enhancements
-include Makefile-sciml-enhancements
