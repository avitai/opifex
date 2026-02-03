#!/bin/bash
# Opifex CUDA Test Runner
# This script runs tests with proper CUDA environment setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default values
VERBOSE=false
HELP=false
FORCE_CPU=false
TEST_PATH="tests/"
PYTEST_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-cpu)
            FORCE_CPU=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        --*)
            # Pass through pytest arguments
            PYTEST_ARGS="$PYTEST_ARGS $1"
            shift
            ;;
        *)
            # Assume it's a test path
            TEST_PATH="$1"
            shift
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    cat << 'EOF'
🚀 Opifex CUDA Test Runner
=======================

Runs Opifex tests with proper CUDA environment setup and GPU detection.

USAGE:
    ./scripts/run_tests_with_cuda.sh [TEST_PATH] [OPTIONS]

ARGUMENTS:
    TEST_PATH           Path to tests (default: tests/)

OPTIONS:
    --force-cpu         Force CPU-only testing even if GPU is available
    --verbose, -v       Enable verbose pytest output
    --help, -h          Show this help message

PYTEST OPTIONS:
    Any pytest options (--cov, --benchmark-skip, etc.) are passed through

EXAMPLES:
    ./scripts/run_tests_with_cuda.sh                           # Run all tests
    ./scripts/run_tests_with_cuda.sh tests/neural/            # Run specific test directory
    ./scripts/run_tests_with_cuda.sh --verbose                # Run with verbose output
    ./scripts/run_tests_with_cuda.sh --force-cpu              # Force CPU-only mode
    ./scripts/run_tests_with_cuda.sh --cov=opifex              # Run with coverage
    ./scripts/run_tests_with_cuda.sh tests/ --benchmark-skip  # Skip benchmarks

ENVIRONMENT:
    This script automatically:
    - Detects GPU availability
    - Sets up appropriate CUDA environment variables
    - Configures JAX backend (cuda,cpu or cpu-only)
    - Manages GPU memory settings
    - Provides detailed diagnostics

EOF
    exit 0
fi

# Utility functions
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

log_step() {
    echo -e "${PURPLE}🔧 $1${NC}"
}

verbose_log() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}   → $1${NC}"
    fi
}

# Function to detect CUDA availability
detect_cuda_support() {
    if [ "$FORCE_CPU" = true ]; then
        log_info "CPU-only mode requested, skipping GPU detection"
        return 1
    fi

    if command -v nvidia-smi &> /dev/null; then
        local gpu_info
        if gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) && [ -n "$gpu_info" ]; then
            log_success "NVIDIA GPU detected: $gpu_info"
            return 0
        fi
    fi

    log_info "No NVIDIA GPU detected - using CPU-only mode"
    return 1
}

# Function to setup CUDA environment
setup_cuda_environment() {
    local has_cuda=$1

    log_step "Setting up test environment..."

    if [ "$has_cuda" = true ]; then
        # Setup CUDA environment variables
        export CUDA_ROOT="/usr/local/cuda"
        export CUDA_HOME="/usr/local/cuda"

        # JAX CUDA configuration
        export JAX_PLATFORMS="cuda,cpu"
        export XLA_PYTHON_CLIENT_PREALLOCATE="false"
        export XLA_PYTHON_CLIENT_MEM_FRACTION="0.75"
        export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

        # JAX CUDA Plugin Configuration
        export JAX_CUDA_PLUGIN_VERIFY="false"

        # Use venv CUDA libraries if available
        if [ -d ".venv/lib" ]; then
            # Find Python version dynamically
            PYTHON_VERSION=$(find .venv/lib -name "python3.*" -type d | head -1 | xargs basename 2>/dev/null || echo "python3.12")
            VENV_CUDA_BASE=".venv/lib/${PYTHON_VERSION}/site-packages/nvidia"

            if [ -d "$VENV_CUDA_BASE" ]; then
                verbose_log "Using venv CUDA libraries"
                export LD_LIBRARY_PATH="${VENV_CUDA_BASE}/cublas/lib:${VENV_CUDA_BASE}/cusolver/lib:${VENV_CUDA_BASE}/cusparse/lib:${VENV_CUDA_BASE}/cudnn/lib:${VENV_CUDA_BASE}/cufft/lib:${VENV_CUDA_BASE}/curand/lib:${VENV_CUDA_BASE}/nccl/lib:${VENV_CUDA_BASE}/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
            else
                verbose_log "Using system CUDA libraries"
                export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
            fi
        else
            verbose_log "Using system CUDA libraries"
            export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
        fi

        log_success "GPU environment configured"
    else
        # CPU-only configuration
        export JAX_PLATFORMS="cpu"
        export JAX_ENABLE_X64="0"
        log_success "CPU-only environment configured"
    fi

    # Common settings
    export TF_CPP_MIN_LOG_LEVEL="1"
    export PYTEST_CUDA_ENABLED="$has_cuda"
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
}

# Function to run environment diagnostics
run_diagnostics() {
    log_step "Running environment diagnostics..."

    # Python version
    echo -e "${CYAN}Python: $(python --version)${NC}"

    # Virtual environment
    if [[ -n "$VIRTUAL_ENV" ]]; then
        echo -e "${CYAN}Virtual Environment: Active ($VIRTUAL_ENV)${NC}"
    else
        log_warning "No virtual environment detected"
    fi

    # JAX configuration
    python << 'PYTHON_EOF' || log_warning "JAX diagnostic failed"
try:
    import jax
    import jax.numpy as jnp

    print(f"JAX version: {jax.__version__}")
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX platforms: {jax.config.jax_platforms}")

    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    cpu_devices = [d for d in devices if d.platform == 'cpu']

    print(f"Available devices: {len(devices)} total")
    if gpu_devices:
        print(f"  🎮 GPU devices: {len(gpu_devices)} ({[str(d) for d in gpu_devices]})")
        # Quick GPU test
        try:
            x = jnp.array([1., 2., 3.])
            with jax.default_device(gpu_devices[0]):
                y = jnp.sum(x**2)
            print(f"  ✅ GPU computation test: {float(y)}")
        except Exception as e:
            print(f"  ⚠️  GPU test warning: {e}")
    else:
        print(f"  💻 CPU devices: {len(cpu_devices)}")

    # Test basic computation
    x = jnp.linspace(0, 1, 100)
    y = jnp.sin(2 * jnp.pi * x)
    print("✅ JAX functionality verified")

except ImportError as e:
    print(f"❌ JAX not available: {e}")
except Exception as e:
    print(f"⚠️  JAX diagnostic issue: {e}")
PYTHON_EOF

    echo ""
}

# Function to run the tests
run_tests() {
    log_step "Running tests with CUDA support..."

    # Ensure we're in a virtual environment
    if [[ -z "$VIRTUAL_ENV" ]]; then
        log_error "No virtual environment active!"
        log_info "Please run: source ./activate.sh"
        exit 1
    fi

    # Build pytest command
    local pytest_cmd="python -m pytest"

    # Add verbose flag if requested
    if [ "$VERBOSE" = true ]; then
        pytest_cmd="$pytest_cmd -v"
    fi

    # Add test path
    pytest_cmd="$pytest_cmd $TEST_PATH"

    # Add any additional pytest arguments
    if [ -n "$PYTEST_ARGS" ]; then
        pytest_cmd="$pytest_cmd $PYTEST_ARGS"
    fi

    # Add standard Opifex test options
    pytest_cmd="$pytest_cmd --tb=short --color=yes"

    log_info "Executing: $pytest_cmd"
    echo ""

    # Run the tests
    eval "$pytest_cmd"
    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        log_success "All tests passed!"
    else
        log_error "Some tests failed (exit code: $exit_code)"
    fi

    return $exit_code
}

# Main execution
main() {
    echo -e "${PURPLE}🚀 Opifex CUDA Test Runner${NC}"
    echo "=========================="
    echo ""

    # Detect CUDA support
    HAS_CUDA=false
    if detect_cuda_support; then
        HAS_CUDA=true
    fi

    # Setup environment
    setup_cuda_environment "$HAS_CUDA"

    # Run diagnostics
    run_diagnostics

    # Run tests
    run_tests
    local test_exit=$?

    # Summary
    echo ""
    echo "=========================="
    if [ "$HAS_CUDA" = true ]; then
        log_success "Tests completed with GPU support"
    else
        log_success "Tests completed in CPU-only mode"
    fi

    exit $test_exit
}

# Run main function
main "$@"
