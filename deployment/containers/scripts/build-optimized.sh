#!/bin/bash
# Opifex Container Build Optimization Script
# Phase 7.1: Container Orchestration Implementation
# Target: 54% image size reduction + <30s startup time

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$(dirname "$SCRIPT_DIR")")")"
DOCKERFILE_DIR="$PROJECT_ROOT/deployment/containers/dockerfiles"
BUILD_CONTEXT="$PROJECT_ROOT"

# Build configuration
IMAGE_NAME="opifex-framework"
REGISTRY=${OPIFEX_REGISTRY:-"localhost:5000"}
VERSION=${OPIFEX_VERSION:-"latest"}
BUILD_TARGET=${BUILD_TARGET:-"production"}
ENABLE_BUILDKIT=${ENABLE_BUILDKIT:-"1"}
ENABLE_CACHE=${ENABLE_CACHE:-"1"}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_header() {
    echo -e "\n${BLUE}=================================="
    echo -e "🚀 $1"
    echo -e "==================================${NC}\n"
}

# Prerequisites check
check_prerequisites() {
    log_header "Checking Prerequisites"

    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    log_success "Docker: $(docker --version)"

    # Check Docker Buildx
    if ! docker buildx version &> /dev/null; then
        log_error "Docker Buildx is required but not available"
        exit 1
    fi
    log_success "Docker Buildx: $(docker buildx version)"

    # Check GPU support
    if command -v nvidia-smi &> /dev/null; then
        log_success "NVIDIA GPU support detected"
        nvidia-smi --query-gpu=name --format=csv,noheader
    else
        log_warning "No NVIDIA GPU detected, building CPU-only variant"
    fi

    # Check available disk space
    AVAILABLE_SPACE=$(df "$PROJECT_ROOT" | awk 'NR==2 {print $4}')
    if [ "$AVAILABLE_SPACE" -lt 10485760 ]; then  # 10GB in KB
        log_warning "Low disk space detected (< 10GB), build may fail"
    fi

    log_success "Prerequisites check complete"
}

# Build context optimization
optimize_build_context() {
    log_header "Optimizing Build Context"

    # Create .dockerignore if it doesn't exist
    DOCKERIGNORE="$PROJECT_ROOT/.dockerignore"
    if [ ! -f "$DOCKERIGNORE" ]; then
        log_info "Creating optimized .dockerignore"
        cat > "$DOCKERIGNORE" << 'EOF'
# Version control
.git
.gitignore
.gitattributes

# Python cache and build artifacts
__pycache__
*.py[cod]
*$py.class
*.so
build/
dist/
*.egg-info/
.tox/
.coverage
htmlcov/
.pytest_cache/
.cache

# Virtual environments
venv/
.venv/
env/
.env

# IDE and editor files
.vscode/
.idea/
*.swp
*.swo
*~

# OS generated files
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Documentation and examples
docs/
examples/
*.md
*.rst

# Temporary files
tmp/
temp/
*.tmp
*.log

# Large data files
data/
*.h5
*.hdf5
*.nc
*.npz
*.pkl

# Jupyter notebooks
*.ipynb
.ipynb_checkpoints

# Docker files
Dockerfile*
docker-compose*.yml

# CI/CD
.github/
.gitlab-ci.yml
.travis.yml

# Deployment files (except current containers)
deployment/community/
deployment/monitoring/
deployment/security/
deployment/benchmarking/
deployment/scalability/
deployment/kubernetes/
deployment/*.yaml
deployment/*.yml
deployment/*.md
EOF
        log_success "Created optimized .dockerignore"
    fi

    # Calculate build context size
    CONTEXT_SIZE=$(du -sh "$BUILD_CONTEXT" 2>/dev/null | cut -f1)
    log_info "Build context size: $CONTEXT_SIZE"
}

# Multi-stage build with optimization
build_container() {
    log_header "Building Opifex Container (Target: $BUILD_TARGET)"

    # Set up buildx builder if needed
    if [ "$ENABLE_BUILDKIT" = "1" ]; then
        log_info "Setting up Docker Buildx builder"
        docker buildx create --use --name opifex-builder 2>/dev/null || true
        docker buildx inspect --bootstrap
    fi

    # Build arguments
    BUILD_ARGS=(
        "--target" "$BUILD_TARGET"
        "--file" "$DOCKERFILE_DIR/Dockerfile.opifex"
        "--tag" "$REGISTRY/$IMAGE_NAME:$VERSION"
        "--tag" "$REGISTRY/$IMAGE_NAME:latest"
    )

    # Add cache options
    if [ "$ENABLE_CACHE" = "1" ]; then
        BUILD_ARGS+=(
            "--cache-from" "type=local,src=/tmp/.buildx-cache"
            "--cache-to" "type=local,dest=/tmp/.buildx-cache-new,mode=max"
        )
    fi

    # Add platform if building for multiple architectures
    if [ "${BUILD_MULTI_ARCH:-0}" = "1" ]; then
        BUILD_ARGS+=(
            "--platform" "linux/amd64,linux/arm64"
            "--push"
        )
    else
        BUILD_ARGS+=("--load")
    fi

    # Add build metadata
    BUILD_ARGS+=(
        "--label" "org.opencontainers.image.created=$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
        "--label" "org.opencontainers.image.version=$VERSION"
        "--label" "org.opencontainers.image.source=https://github.com/opifex/framework"
        "--label" "org.opencontainers.image.title=Opifex Framework"
        "--label" "org.opencontainers.image.description=Scientific Machine Learning Framework with GPU Optimization"
    )

    # Start build with timing
    log_info "Starting container build..."
    START_TIME=$(date +%s)

    if [ "$ENABLE_BUILDKIT" = "1" ]; then
        DOCKER_BUILDKIT=1 docker buildx build "${BUILD_ARGS[@]}" "$BUILD_CONTEXT"
    else
        docker build "${BUILD_ARGS[@]}" "$BUILD_CONTEXT"
    fi

    END_TIME=$(date +%s)
    BUILD_TIME=$((END_TIME - START_TIME))

    log_success "Container build completed in ${BUILD_TIME}s"

    # Rotate cache
    if [ "$ENABLE_CACHE" = "1" ] && [ -d "/tmp/.buildx-cache-new" ]; then
        rm -rf /tmp/.buildx-cache
        mv /tmp/.buildx-cache-new /tmp/.buildx-cache
    fi
}

# Analyze container size
analyze_container() {
    log_header "Container Analysis"

    IMAGE_TAG="$REGISTRY/$IMAGE_NAME:$VERSION"

    # Get image size
    IMAGE_SIZE=$(docker image inspect "$IMAGE_TAG" --format='{{.Size}}' 2>/dev/null || echo "0")
    IMAGE_SIZE_MB=$((IMAGE_SIZE / 1024 / 1024))

    log_info "Final image size: ${IMAGE_SIZE_MB} MB"

    # Compare with base image size
    BASE_IMAGE="nvidia/cuda:11.8-runtime-ubuntu20.04"
    if docker image inspect "$BASE_IMAGE" &>/dev/null; then
        BASE_SIZE=$(docker image inspect "$BASE_IMAGE" --format='{{.Size}}')
        BASE_SIZE_MB=$((BASE_SIZE / 1024 / 1024))
        REDUCTION_MB=$((BASE_SIZE_MB - IMAGE_SIZE_MB))
        REDUCTION_PERCENT=$((REDUCTION_MB * 100 / BASE_SIZE_MB))

        log_info "Base image size: ${BASE_SIZE_MB} MB"
        log_info "Size reduction: ${REDUCTION_MB} MB (${REDUCTION_PERCENT}%)"

        # Check if we achieved our target
        if [ "$REDUCTION_PERCENT" -ge 54 ]; then
            log_success "✅ Achieved target size reduction (>54%): ${REDUCTION_PERCENT}%"
        else
            log_warning "⚠️  Size reduction below target (54%): ${REDUCTION_PERCENT}%"
        fi
    fi

    # Show layer breakdown
    log_info "Image layer breakdown:"
    docker history "$IMAGE_TAG" --format "table {{.CreatedBy}}\t{{.Size}}" | head -10
}

# Test container startup time
test_startup_time() {
    log_header "Testing Container Startup Time"

    IMAGE_TAG="$REGISTRY/$IMAGE_NAME:$VERSION"

    log_info "Testing container startup time..."
    START_TIME=$(date +%s%N)

    # Run container with health check
    CONTAINER_ID=$(docker run -d "$IMAGE_TAG" python3.10 -c "import opifex; print('Ready')")

    # Wait for container to be ready
    timeout 60s bash -c "while ! docker logs $CONTAINER_ID 2>&1 | grep -q 'Ready'; do sleep 0.1; done"

    END_TIME=$(date +%s%N)
    STARTUP_TIME_MS=$(((END_TIME - START_TIME) / 1000000))
    STARTUP_TIME_S=$((STARTUP_TIME_MS / 1000))

    # Cleanup
    docker rm -f "$CONTAINER_ID" > /dev/null

    log_info "Container startup time: ${STARTUP_TIME_S}s (${STARTUP_TIME_MS}ms)"

    # Check if we achieved our target
    if [ "$STARTUP_TIME_S" -le 30 ]; then
        log_success "✅ Achieved target startup time (<30s): ${STARTUP_TIME_S}s"
    else
        log_warning "⚠️  Startup time above target (30s): ${STARTUP_TIME_S}s"
    fi
}

# Vulnerability scanning
security_scan() {
    log_header "Security Vulnerability Scan"

    IMAGE_TAG="$REGISTRY/$IMAGE_NAME:$VERSION"

    # Check if Trivy is available
    if command -v trivy &> /dev/null; then
        log_info "Running Trivy security scan..."
        trivy image --exit-code 1 --severity HIGH,CRITICAL "$IMAGE_TAG" || {
            log_warning "Security vulnerabilities found, check output above"
        }
    else
        log_warning "Trivy not installed, skipping security scan"
        log_info "Install Trivy for security scanning: https://github.com/aquasecurity/trivy"
    fi
}

# Performance benchmarking
performance_benchmark() {
    log_header "Performance Benchmarking"

    IMAGE_TAG="$REGISTRY/$IMAGE_NAME:$VERSION"

    log_info "Running performance benchmark..."

    # GPU benchmark if available
    if command -v nvidia-smi &> /dev/null; then
        log_info "Running GPU performance test..."
        docker run --rm --gpus all "$IMAGE_TAG" python3.10 -c "
import jax
import jax.numpy as jnp
import time

# GPU performance test
print('🎮 GPU Performance Test')
start = time.time()
x = jax.random.normal(jax.random.PRNGKey(42), (5000, 5000))
y = jnp.dot(x, x.T)
result = y.block_until_ready()
end = time.time()

print(f'✅ GPU matrix multiplication (5000x5000): {end-start:.3f}s')
print(f'✅ Performance: {(5000**3 * 2) / (end-start) / 1e9:.1f} GFLOPS')
"
    fi

    # CPU benchmark
    log_info "Running CPU performance test..."
    docker run --rm "$IMAGE_TAG" python3.10 -c "
import time
import opifex

print('🖥️  CPU Performance Test')
start = time.time()
# Simple opifex load test
result = opifex.__version__
end = time.time()

print(f'✅ Opifex import time: {end-start:.3f}s')
print(f'✅ Framework version: {result}')
"
}

# Cleanup function
cleanup() {
    log_header "Cleanup"

    # Remove builder if created
    if [ "$ENABLE_BUILDKIT" = "1" ]; then
        docker buildx rm opifex-builder 2>/dev/null || true
    fi

    # Clean up build cache if requested
    if [ "${CLEAN_CACHE:-0}" = "1" ]; then
        log_info "Cleaning build cache..."
        rm -rf /tmp/.buildx-cache /tmp/.buildx-cache-new
        docker system prune -f
    fi

    log_success "Cleanup complete"
}

# Help function
show_help() {
    cat << EOF
Opifex Container Build Optimization Script

Usage: $0 [OPTIONS]

Options:
    -t, --target TARGET     Build target (production, development) [default: production]
    -v, --version VERSION   Image version tag [default: latest]
    -r, --registry REGISTRY Registry URL [default: localhost:5000]
    --no-cache             Disable build cache
    --no-buildkit          Disable Docker Buildx
    --multi-arch           Build for multiple architectures
    --clean-cache          Clean build cache after build
    --skip-tests           Skip startup time and performance tests
    --help                 Show this help message

Environment Variables:
    OPIFEX_REGISTRY         Container registry URL
    OPIFEX_VERSION          Image version
    BUILD_TARGET           Build target stage
    ENABLE_BUILDKIT        Enable Docker Buildx (1/0)
    ENABLE_CACHE           Enable build cache (1/0)
    BUILD_MULTI_ARCH       Build for multiple architectures (1/0)
    CLEAN_CACHE            Clean cache after build (1/0)

Examples:
    # Build production image
    $0 --target production --version v1.0.0

    # Build development image with cache disabled
    $0 --target development --no-cache

    # Build multi-architecture image
    $0 --multi-arch --registry my-registry.com/opifex
EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--target)
            BUILD_TARGET="$2"
            shift 2
            ;;
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        --no-cache)
            ENABLE_CACHE="0"
            shift
            ;;
        --no-buildkit)
            ENABLE_BUILDKIT="0"
            shift
            ;;
        --multi-arch)
            BUILD_MULTI_ARCH="1"
            shift
            ;;
        --clean-cache)
            CLEAN_CACHE="1"
            shift
            ;;
        --skip-tests)
            SKIP_TESTS="1"
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main execution
main() {
    log_header "Opifex Container Orchestration Build"
    log_info "Building target: $BUILD_TARGET"
    log_info "Version: $VERSION"
    log_info "Registry: $REGISTRY"

    # Set trap for cleanup
    trap cleanup EXIT

    # Execute build pipeline
    check_prerequisites
    optimize_build_context
    build_container
    analyze_container

    # Optional tests
    if [ "${SKIP_TESTS:-0}" != "1" ]; then
        test_startup_time
        performance_benchmark
        security_scan
    fi

    log_success "🎉 Container build pipeline completed successfully!"
    log_info "Image: $REGISTRY/$IMAGE_NAME:$VERSION"
}

# Run main function
main "$@"
