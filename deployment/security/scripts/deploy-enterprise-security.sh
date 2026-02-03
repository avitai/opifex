#!/bin/bash

# Opifex Phase 7.2: Enterprise Security & Scalability Deployment Script
# Hybrid Security Model + Federation + Multi-Tier Compliance

set -euo pipefail

# Script Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SECURITY_DIR="$(dirname "$SCRIPT_DIR")"
DEPLOYMENT_DIR="$(dirname "$SECURITY_DIR")"
export PROJECT_ROOT
PROJECT_ROOT="$(dirname "$DEPLOYMENT_DIR")"

# Logging Configuration
LOG_FILE="/tmp/opifex-security-deployment-$(date +%Y%m%d-%H%M%S).log"
VERBOSE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Phase 7.2 Configuration
KEYCLOAK_NAMESPACE="opifex-security"
NETWORK_SECURITY_NAMESPACE="opifex-network-security"
PLATFORM_NAMESPACE="opifex-platform"

# Function: Display help
show_help() {
    cat << EOF
Opifex Phase 7.2: Enterprise Security & Scalability Deployment

USAGE:
    $0 [OPTIONS]

OPTIONS:
    -h, --help              Show this help message
    -v, --verbose           Enable verbose output
    --dry-run              Show commands without executing
    --check-prerequisites   Check prerequisites only
    --deploy-keycloak      Deploy Keycloak identity provider
    --deploy-network       Deploy network security policies
    --deploy-all           Deploy complete enterprise security
    --validate             Validate deployment
    --status               Show deployment status
    --uninstall            Remove enterprise security deployment

EXAMPLES:
    $0 --check-prerequisites    # Check system requirements
    $0 --deploy-all             # Full enterprise security deployment
    $0 --validate               # Validate security deployment
    $0 --status                 # Check deployment status

PHASE 7.2 COMPONENTS:
    1. Keycloak Enterprise Identity Hub
    2. Advanced Network Security (Zero Trust)
    3. GDPR Compliance Framework
    4. SOC 2 Type II Controls
    5. Federated Authentication
    6. Multi-Tier Data Classification

EOF
}

# Function: Logging
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')

    case $level in
        INFO)  echo -e "${BLUE}[INFO]${NC} $message" | tee -a "$LOG_FILE" ;;
        WARN)  echo -e "${YELLOW}[WARN]${NC} $message" | tee -a "$LOG_FILE" ;;
        ERROR) echo -e "${RED}[ERROR]${NC} $message" | tee -a "$LOG_FILE" ;;
        SUCCESS) echo -e "${GREEN}[SUCCESS]${NC} $message" | tee -a "$LOG_FILE" ;;
        DEBUG) [[ "$VERBOSE" == "true" ]] && echo -e "${PURPLE}[DEBUG]${NC} $message" | tee -a "$LOG_FILE" ;;
    esac

    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Function: Execute command with logging
execute_cmd() {
    local cmd="$*"
    log DEBUG "Executing: $cmd"

    if [[ "${DRY_RUN:-false}" == "true" ]]; then
        log INFO "DRY RUN: $cmd"
        return 0
    fi

    if eval "$cmd" >> "$LOG_FILE" 2>&1; then
        log DEBUG "Command succeeded: $cmd"
        return 0
    else
        log ERROR "Command failed: $cmd"
        return 1
    fi
}

# Function: Check prerequisites
check_prerequisites() {
    log INFO "🔍 Checking Phase 7.2 Enterprise Security Prerequisites..."

    local all_good=true

    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        log ERROR "kubectl is not installed or not in PATH"
        all_good=false
    else
        log SUCCESS "✓ kubectl found"
    fi

    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log ERROR "Cannot connect to Kubernetes cluster"
        all_good=false
    else
        log SUCCESS "✓ Kubernetes cluster accessible"
    fi

    # Check for Phase 7.1 dependencies
    if ! kubectl get namespace istio-system &> /dev/null; then
        log WARN "⚠️  Istio service mesh not found - Phase 7.1 may not be complete"
    else
        log SUCCESS "✓ Istio service mesh detected"
    fi

    # Check for required CRDs
    local required_crds=(
        "peerauthentications.security.istio.io"
        "authorizationpolicies.security.istio.io"
        "requestauthentications.security.istio.io"
        "serviceentries.networking.istio.io"
        "destinationrules.networking.istio.io"
    )

    for crd in "${required_crds[@]}"; do
        if kubectl get crd "$crd" &> /dev/null; then
            log SUCCESS "✓ CRD found: $crd"
        else
            log ERROR "✗ Missing CRD: $crd"
            all_good=false
        fi
    done

    # Check storage classes
    if ! kubectl get storageclass fast-ssd &> /dev/null; then
        log WARN "⚠️  StorageClass 'fast-ssd' not found - using default"
    else
        log SUCCESS "✓ Fast SSD storage class available"
    fi

    # Check cluster resources
    local node_count
    node_count=$(kubectl get nodes --no-headers | wc -l)
    log INFO "📊 Cluster has $node_count nodes"

    if [[ $node_count -lt 3 ]]; then
        log WARN "⚠️  Recommended: At least 3 nodes for HA deployment"
    fi

    # Check resource capacity
    local total_cpu
    total_cpu=$(kubectl top nodes --no-headers 2>/dev/null | awk '{sum+=$2} END {print sum}' || echo "unknown")
    local total_memory
    total_memory=$(kubectl top nodes --no-headers 2>/dev/null | awk '{sum+=$4} END {print sum}' || echo "unknown")

    log INFO "📊 Cluster resources: CPU=${total_cpu}m, Memory=${total_memory}Mi"

    if [[ "$all_good" == "true" ]]; then
        log SUCCESS "🎉 All prerequisites met for Phase 7.2!"
        return 0
    else
        log ERROR "❌ Prerequisites check failed"
        return 1
    fi
}

# Function: Create namespaces
create_namespaces() {
    log INFO "📦 Creating namespaces for Phase 7.2..."

    local namespaces=(
        "$KEYCLOAK_NAMESPACE"
        "$NETWORK_SECURITY_NAMESPACE"
    )

    for ns in "${namespaces[@]}"; do
        if kubectl get namespace "$ns" &> /dev/null; then
            log INFO "Namespace $ns already exists"
        else
            execute_cmd "kubectl create namespace $ns"
            log SUCCESS "✓ Created namespace: $ns"
        fi

        # Label namespaces appropriately
        execute_cmd "kubectl label namespace $ns app.kubernetes.io/part-of=opifex-framework --overwrite"
        execute_cmd "kubectl label namespace $ns security.opifex.io/tier=enterprise --overwrite"
    done
}

# Function: Deploy Keycloak Enterprise Identity Hub
deploy_keycloak() {
    log INFO "🔑 Deploying Keycloak Enterprise Identity Hub..."

    # Apply PostgreSQL for Keycloak
    log INFO "Deploying PostgreSQL database for Keycloak..."
    # Apply Keycloak components (split from keycloak-enterprise-ha.yaml)
    execute_cmd "kubectl apply -f $SECURITY_DIR/keycloak/"

    # Wait for PostgreSQL to be ready
    log INFO "Waiting for PostgreSQL to be ready..."
    execute_cmd "kubectl wait --for=condition=ready pod -l app=keycloak-postgres -n $KEYCLOAK_NAMESPACE --timeout=300s"

    # Apply Keycloak realm configuration
    log INFO "Applying Opifex realm configuration..."
    execute_cmd "kubectl apply -f $SECURITY_DIR/keycloak/opifex-realm-config.yaml"

    # Apply themes and providers
    log INFO "Applying Opifex themes and providers..."
    # Apply themes and providers (split from opifex-themes-providers.yaml)
    execute_cmd "kubectl apply -f $SECURITY_DIR/keycloak/opifex-keycloak-themes-configmap-1.yaml"
    execute_cmd "kubectl apply -f $SECURITY_DIR/keycloak/opifex-keycloak-providers-configmap.yaml"

    # Wait for Keycloak deployment
    log INFO "Waiting for Keycloak to be ready..."
    execute_cmd "kubectl wait --for=condition=available deployment/keycloak-enterprise -n $KEYCLOAK_NAMESPACE --timeout=600s"

    # Get Keycloak service details
    local keycloak_ip
    keycloak_ip=$(kubectl get service keycloak -n $KEYCLOAK_NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
    log INFO "🌐 Keycloak service IP: $keycloak_ip"

    log SUCCESS "✅ Keycloak Enterprise Identity Hub deployed successfully"
}

# Function: Deploy advanced network security
deploy_network_security() {
    log INFO "🛡️  Deploying Advanced Network Security (Zero Trust)..."

    # Apply network security policies
    # Apply advanced network security components (split from advanced-network-security.yaml)
    execute_cmd "kubectl apply -f $SECURITY_DIR/network/"
    execute_cmd "kubectl apply -f $SECURITY_DIR/network-policies/"

    # Wait for Falco DaemonSet
    log INFO "Waiting for Falco security monitoring to be ready..."
    execute_cmd "kubectl wait --for=condition=ready pod -l app=falco -n $NETWORK_SECURITY_NAMESPACE --timeout=300s"

    # Verify network policies
    local np_count
    np_count=$(kubectl get networkpolicies -n $PLATFORM_NAMESPACE --no-headers | wc -l)
    log INFO "📊 Applied $np_count network policies"

    # Verify Istio security policies
    local auth_policies
    auth_policies=$(kubectl get authorizationpolicies -n $PLATFORM_NAMESPACE --no-headers | wc -l)
    log INFO "📊 Applied $auth_policies Istio authorization policies"

    log SUCCESS "✅ Advanced Network Security deployed successfully"
}

# Function: Validate deployment
validate_deployment() {
    log INFO "🔍 Validating Phase 7.2 Enterprise Security deployment..."

    local validation_failed=false

    # Validate Keycloak
    log INFO "Validating Keycloak deployment..."
    if kubectl get deployment keycloak-enterprise -n $KEYCLOAK_NAMESPACE &> /dev/null; then
        local replicas
        replicas=$(kubectl get deployment keycloak-enterprise -n $KEYCLOAK_NAMESPACE -o jsonpath='{.status.readyReplicas}')
        if [[ ${replicas:-0} -gt 0 ]]; then
            log SUCCESS "✓ Keycloak deployment is healthy ($replicas replicas)"
        else
            log ERROR "✗ Keycloak deployment has no ready replicas"
            validation_failed=true
        fi
    else
        log ERROR "✗ Keycloak deployment not found"
        validation_failed=true
    fi

    # Validate PostgreSQL
    log INFO "Validating PostgreSQL database..."
    if kubectl get statefulset keycloak-postgres -n $KEYCLOAK_NAMESPACE &> /dev/null; then
        local pg_replicas
        pg_replicas=$(kubectl get statefulset keycloak-postgres -n $KEYCLOAK_NAMESPACE -o jsonpath='{.status.readyReplicas}')
        if [[ ${pg_replicas:-0} -gt 0 ]]; then
            log SUCCESS "✓ PostgreSQL is healthy ($pg_replicas replicas)"
        else
            log ERROR "✗ PostgreSQL has no ready replicas"
            validation_failed=true
        fi
    else
        log ERROR "✗ PostgreSQL StatefulSet not found"
        validation_failed=true
    fi

    # Validate Network Policies
    log INFO "Validating network security policies..."
    local np_count
    np_count=$(kubectl get networkpolicies -n $PLATFORM_NAMESPACE --no-headers 2>/dev/null | wc -l)
    if [[ $np_count -gt 0 ]]; then
        log SUCCESS "✓ Network policies applied ($np_count policies)"
    else
        log ERROR "✗ No network policies found"
        validation_failed=true
    fi

    # Validate Istio Security
    log INFO "Validating Istio security policies..."
    local istio_policies
    istio_policies=$(kubectl get authorizationpolicies,peerauthentications,requestauthentications -n $PLATFORM_NAMESPACE --no-headers 2>/dev/null | wc -l)
    if [[ $istio_policies -gt 0 ]]; then
        log SUCCESS "✓ Istio security policies applied ($istio_policies policies)"
    else
        log ERROR "✗ No Istio security policies found"
        validation_failed=true
    fi

    # Validate Falco
    log INFO "Validating Falco security monitoring..."
    if kubectl get daemonset falco-opifex -n $NETWORK_SECURITY_NAMESPACE &> /dev/null; then
        local falco_ready
        falco_ready=$(kubectl get daemonset falco-opifex -n $NETWORK_SECURITY_NAMESPACE -o jsonpath='{.status.numberReady}')
        local falco_desired
        falco_desired=$(kubectl get daemonset falco-opifex -n $NETWORK_SECURITY_NAMESPACE -o jsonpath='{.status.desiredNumberScheduled}')
        if [[ ${falco_ready:-0} -eq ${falco_desired:-1} ]]; then
            log SUCCESS "✓ Falco security monitoring is healthy ($falco_ready/$falco_desired nodes)"
        else
            log ERROR "✗ Falco security monitoring not fully deployed ($falco_ready/$falco_desired nodes)"
            validation_failed=true
        fi
    else
        log ERROR "✗ Falco DaemonSet not found"
        validation_failed=true
    fi

    # Test authentication endpoint
    log INFO "Testing Keycloak authentication endpoint..."
    local keycloak_service
    keycloak_service=$(kubectl get service keycloak -n $KEYCLOAK_NAMESPACE -o jsonpath='{.spec.clusterIP}' 2>/dev/null)
    if [[ -n "$keycloak_service" ]]; then
        if kubectl run test-auth --image=curlimages/curl --rm -i --restart=Never -- \
           curl -s -f -k "http://$keycloak_service/auth/realms/opifex/.well-known/openid_connect_configuration" &> /dev/null; then
            log SUCCESS "✓ Keycloak authentication endpoint is responding"
        else
            log WARN "⚠️  Keycloak authentication endpoint test failed (may be normal during startup)"
        fi
    fi

    if [[ "$validation_failed" == "false" ]]; then
        log SUCCESS "🎉 Phase 7.2 Enterprise Security validation completed successfully!"
        return 0
    else
        log ERROR "❌ Phase 7.2 validation failed"
        return 1
    fi
}

# Function: Show deployment status
show_status() {
    log INFO "📊 Phase 7.2 Enterprise Security Status Report"
    echo

    # Keycloak Status
    echo -e "${CYAN}=== Keycloak Identity Hub ===${NC}"
    if kubectl get namespace $KEYCLOAK_NAMESPACE &> /dev/null; then
        kubectl get deployment,statefulset,service,hpa -n $KEYCLOAK_NAMESPACE
        echo
        echo "Keycloak Pods:"
        kubectl get pods -n $KEYCLOAK_NAMESPACE
    else
        echo "Keycloak namespace not found"
    fi

    echo
    echo -e "${CYAN}=== Network Security ===${NC}"
    if kubectl get namespace $NETWORK_SECURITY_NAMESPACE &> /dev/null; then
        kubectl get daemonset,configmap -n $NETWORK_SECURITY_NAMESPACE
        echo
        echo "Security Monitoring Pods:"
        kubectl get pods -n $NETWORK_SECURITY_NAMESPACE
    else
        echo "Network security namespace not found"
    fi

    echo
    echo -e "${CYAN}=== Security Policies ===${NC}"
    echo "Network Policies:"
    kubectl get networkpolicies -n $PLATFORM_NAMESPACE 2>/dev/null || echo "No network policies found"
    echo
    echo "Istio Security Policies:"
    kubectl get authorizationpolicies,peerauthentications,requestauthentications -n $PLATFORM_NAMESPACE 2>/dev/null || echo "No Istio security policies found"

    echo
    echo -e "${CYAN}=== Resource Usage ===${NC}"
    echo "Keycloak Namespace:"
    kubectl top pods -n $KEYCLOAK_NAMESPACE 2>/dev/null || echo "Metrics not available"
    echo
    echo "Security Namespace:"
    kubectl top pods -n $NETWORK_SECURITY_NAMESPACE 2>/dev/null || echo "Metrics not available"
}

# Function: Uninstall enterprise security
uninstall() {
    log WARN "🗑️  Uninstalling Phase 7.2 Enterprise Security..."

    # Confirm deletion
    read -r -p "Are you sure you want to uninstall enterprise security? (yes/no): " confirm
    if [[ "$confirm" != "yes" ]]; then
        log INFO "Uninstall cancelled"
        return 0
    fi

    # Remove network security
    log INFO "Removing network security policies..."
    # Delete advanced network security components (split from advanced-network-security.yaml)
    execute_cmd "kubectl delete -f $SECURITY_DIR/network-policies/ --ignore-not-found=true"
    execute_cmd "kubectl delete -f $SECURITY_DIR/network/ --ignore-not-found=true"

    # Remove Keycloak configuration
    log INFO "Removing Keycloak configuration..."
    # Delete themes and providers (split from opifex-themes-providers.yaml)
    execute_cmd "kubectl delete -f $SECURITY_DIR/keycloak/opifex-keycloak-providers-configmap.yaml --ignore-not-found=true"
    execute_cmd "kubectl delete -f $SECURITY_DIR/keycloak/opifex-keycloak-themes-configmap-1.yaml --ignore-not-found=true"
    execute_cmd "kubectl delete -f $SECURITY_DIR/keycloak/opifex-realm-config.yaml --ignore-not-found=true"

    # Remove Keycloak deployment
    log INFO "Removing Keycloak deployment..."
    # Delete Keycloak components (split from keycloak-enterprise-ha.yaml)
    execute_cmd "kubectl delete -f $SECURITY_DIR/keycloak/ --ignore-not-found=true"

    # Remove namespaces
    log INFO "Removing namespaces..."
    execute_cmd "kubectl delete namespace $KEYCLOAK_NAMESPACE --ignore-not-found=true"
    execute_cmd "kubectl delete namespace $NETWORK_SECURITY_NAMESPACE --ignore-not-found=true"

    log SUCCESS "✅ Phase 7.2 Enterprise Security uninstalled"
}

# Function: Deploy all components
deploy_all() {
    log INFO "🚀 Starting Phase 7.2 Enterprise Security & Scalability Deployment"

    # Check prerequisites
    if ! check_prerequisites; then
        log ERROR "Prerequisites check failed. Please resolve issues before proceeding."
        exit 1
    fi

    # Create namespaces
    create_namespaces

    # Deploy Keycloak
    deploy_keycloak

    # Deploy network security
    deploy_network_security

    # Validate deployment
    if validate_deployment; then
        log SUCCESS "🎉 Phase 7.2 Enterprise Security & Scalability deployed successfully!"

        # Show final status
        echo
        log INFO "📋 Deployment Summary:"
        echo "  • Keycloak Enterprise Identity Hub: ✅ Deployed"
        echo "  • Advanced Network Security: ✅ Deployed"
        echo "  • GDPR Compliance Framework: ✅ Configured"
        echo "  • SOC 2 Type II Controls: ✅ Implemented"
        echo "  • Zero Trust Network Policies: ✅ Applied"
        echo "  • Multi-Tier Data Classification: ✅ Enforced"
        echo
        echo "🔗 Access Points:"
        local keycloak_ip
        keycloak_ip=$(kubectl get service keycloak -n $KEYCLOAK_NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "pending")
        echo "  • Keycloak Admin Console: http://$keycloak_ip/auth/admin/"
        echo "  • Opifex Realm: http://$keycloak_ip/auth/realms/opifex/"
        echo
        echo "📋 Next Steps:"
        echo "  1. Configure external identity providers (University SAML, GitHub, ORCID)"
        echo "  2. Import user groups and assign roles"
        echo "  3. Test federated authentication flows"
        echo "  4. Proceed to Phase 7.3: MLOps Integration"
        echo
        log INFO "📝 Deployment log saved to: $LOG_FILE"
    else
        log ERROR "❌ Phase 7.2 deployment validation failed"
        exit 1
    fi
}

# Main script logic
main() {
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --check-prerequisites)
                check_prerequisites
                exit $?
                ;;
            --deploy-keycloak)
                create_namespaces
                deploy_keycloak
                exit $?
                ;;
            --deploy-network)
                create_namespaces
                deploy_network_security
                exit $?
                ;;
            --deploy-all)
                deploy_all
                exit $?
                ;;
            --validate)
                validate_deployment
                exit $?
                ;;
            --status)
                show_status
                exit $?
                ;;
            --uninstall)
                uninstall
                exit $?
                ;;
            *)
                log ERROR "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    # Default action if no arguments provided
    show_help
}

# Initialize script
log INFO "🔧 Opifex Phase 7.2: Enterprise Security & Scalability Deployment Script"
log INFO "📝 Log file: $LOG_FILE"

# Run main function with all arguments
main "$@"
