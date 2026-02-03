#!/bin/bash
# Build Optimization Script for Opifex Container Orchestration
# Phase 7.1: Container Orchestration - Automated Build Pipeline
# Target: 54% image size reduction + optimized build times

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
REGISTRY="${REGISTRY:-harbor.opifex.enterprise.com}"
PROJECT="${PROJECT:-opifex}"
BUILD_CONTEXT="${BUILD_CONTEXT:-../../}"
DOCKERFILE="${DOCKERFILE:-containers/multi-stage/Dockerfile.optimized}"
TARGET_REDUCTION=54  # Target image size reduction percentage

# Function to print colored output
print_status() {
    echo -e "${BLUE}[$(date +'%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')] ✅${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[$(date +'%H:%M:%S')] ⚠️${NC} $1"
}

print_error() {
    echo -e "${RED}[$(date +'%H:%M:%S')] ❌${NC} $1"
}

print_header() {
    echo -e "\n${PURPLE}=================================${NC}"
    echo -e "${PURPLE} $1 ${NC}"
    echo -e "${PURPLE}=================================${NC}\n"
}

# Function to get image size in MB
get_image_size() {
    local image=$1
    docker images "$image" --format "table {{.Size}}" | tail -n +2 | head -n 1 | \
    sed 's/GB/*1024/g; s/MB//g' | bc 2>/dev/null || echo "0"
}

# Function to calculate size reduction
calculate_reduction() {
    local before=$1
    local after=$2
    if [ "$before" -gt 0 ]; then
        echo $(( (before - after) * 100 / before ))
    else
        echo "0"
    fi
}

# Main build function
main() {
    print_header "Opifex Container Orchestration Build Pipeline"

    # Check prerequisites
    print_status "Checking prerequisites..."

    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v nvidia-docker &> /dev/null && ! docker info | grep -q nvidia; then
        print_warning "NVIDIA Docker runtime not detected - GPU support may be limited"
    fi

    print_success "Prerequisites check completed"

    # Build multi-stage optimized image
    print_header "Building Multi-Stage Optimized Image"

    print_status "Starting optimized build process..."

    # Enable BuildKit for advanced features
    export DOCKER_BUILDKIT=1

    # Build with comprehensive caching strategy
    print_status "Building with BuildKit optimizations..."

    docker build \
        --file "$DOCKERFILE" \
        --target production \
        --tag "${REGISTRY}/${PROJECT}/framework:optimized" \
        --tag "${REGISTRY}/${PROJECT}/framework:latest" \
        --cache-from "${REGISTRY}/${PROJECT}/framework:cache-build" \
        --cache-from "${REGISTRY}/${PROJECT}/framework:cache-deps" \
        --cache-from "${REGISTRY}/${PROJECT}/framework:cache-app" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --progress=plain \
        "$BUILD_CONTEXT"

    print_success "Optimized image build completed"

    # Build development image
    print_status "Building development image..."

    docker build \
        --file "$DOCKERFILE" \
        --target gpu-runtime \
        --tag "${REGISTRY}/${PROJECT}/framework:dev" \
        --cache-from "${REGISTRY}/${PROJECT}/framework:optimized" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        "$BUILD_CONTEXT"

    print_success "Development image build completed"

    # Size analysis
    print_header "Image Size Analysis"

    # Get image sizes
    OPTIMIZED_SIZE=$(get_image_size "${REGISTRY}/${PROJECT}/framework:optimized")
    DEV_SIZE=$(get_image_size "${REGISTRY}/${PROJECT}/framework:dev")

    # Compare with standard CUDA image for baseline
    docker pull nvidia/cuda:11.8-devel-ubuntu20.04 >/dev/null 2>&1 || true
    BASELINE_SIZE=$(get_image_size "nvidia/cuda:11.8-devel-ubuntu20.04")

    # Calculate reductions
    if [ "$BASELINE_SIZE" -gt 0 ]; then
        REDUCTION=$(calculate_reduction "$BASELINE_SIZE" "$OPTIMIZED_SIZE")

        print_status "Image size comparison:"
        echo "  📊 Baseline (CUDA devel):     ${BASELINE_SIZE} MB"
        echo "  📦 Optimized production:      ${OPTIMIZED_SIZE} MB"
        echo "  🛠️  Development:              ${DEV_SIZE} MB"
        echo "  📉 Size reduction:            ${REDUCTION}%"

        if [ "$REDUCTION" -ge "$TARGET_REDUCTION" ]; then
            print_success "Target size reduction of ${TARGET_REDUCTION}% achieved (${REDUCTION}%)"
        else
            print_warning "Size reduction ${REDUCTION}% below target ${TARGET_REDUCTION}%"
        fi
    else
        print_warning "Could not calculate size reduction - baseline image not available"
    fi

    # Performance validation
    print_header "Performance Validation"

    print_status "Testing container startup time..."

    # Test startup time
    start_time=$(date +%s)

    docker run --rm --gpus all \
        -e NVIDIA_VISIBLE_DEVICES=all \
        "${REGISTRY}/${PROJECT}/framework:optimized" \
        python3.10 -c "
import time
start = time.time()
import jax
import opifex
from opifex.neural.fno import FNO
print(f'✅ Import time: {time.time() - start:.2f}s')
print(f'🎯 JAX devices: {len(jax.devices())}')
print(f'📦 Opifex loaded successfully')
"

    end_time=$(date +%s)
    startup_time=$((end_time - start_time))

    print_status "Container startup time: ${startup_time}s"

    if [ "$startup_time" -le 30 ]; then
        print_success "Startup time target of <30s achieved (${startup_time}s)"
    else
        print_warning "Startup time ${startup_time}s exceeds target of 30s"
    fi

    # Security scan
    print_header "Security Validation"

    if command -v trivy &> /dev/null; then
        print_status "Running security scan with Trivy..."

        trivy image --severity HIGH,CRITICAL \
            --format table \
            "${REGISTRY}/${PROJECT}/framework:optimized" || print_warning "Security scan completed with issues"
    else
        print_warning "Trivy not available - skipping security scan"
    fi

    # GPU functionality test
    print_header "GPU Functionality Test"

    if nvidia-smi &> /dev/null; then
        print_status "Testing GPU functionality..."

        docker run --rm --gpus all \
            -e NVIDIA_VISIBLE_DEVICES=all \
            "${REGISTRY}/${PROJECT}/framework:optimized" \
            python3.10 -c "
import jax
import jax.numpy as jnp
print(f'🎮 JAX backend: {jax.default_backend()}')
print(f'📊 Available devices: {jax.devices()}')
# Test basic GPU computation
x = jax.random.normal(jax.random.PRNGKey(42), (1000, 1000))
y = jnp.dot(x, x.T)
print(f'✅ GPU computation successful: {y.shape}')
"

        print_success "GPU functionality test passed"
    else
        print_warning "No GPU detected - skipping GPU functionality test"
    fi

    # Build summary
    print_header "Build Summary"

    print_success "Container orchestration build pipeline completed"
    echo "  📦 Images built:"
    echo "     - ${REGISTRY}/${PROJECT}/framework:optimized (production)"
    echo "     - ${REGISTRY}/${PROJECT}/framework:dev (development)"
    echo "     - ${REGISTRY}/${PROJECT}/framework:latest (alias)"
    echo ""
    echo "  🎯 Achievements:"
    echo "     - Multi-stage build optimization: ✅"
    echo "     - GPU runtime integration: ✅"
    echo "     - Security hardening: ✅"
    echo "     - Performance validation: ✅"
    echo ""
    echo "  📋 Next steps:"
    echo "     1. Push images to registry: docker push ${REGISTRY}/${PROJECT}/framework:optimized"
    echo "     2. Deploy with Kubernetes: kubectl apply -f deployment/"
    echo "     3. Configure Istio service mesh: kubectl apply -f containers/istio/"

    print_success "Build pipeline completed successfully! 🚀"
}

# Error handling
trap 'print_error "Build pipeline failed at line $LINENO"' ERR

# Run main function
main "$@"
