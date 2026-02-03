#!/bin/bash

# Opifex Test Runner - GPU-Required Testing
# This script provides test configurations that require GPU and fail appropriately when GPU is not available

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if we're in a CI environment
is_ci_environment() {
    [[ -n "$CI" || -n "$GITHUB_ACTIONS" || -n "$GITLAB_CI" || -n "$CIRCLECI" ]]
}

# Function to check GPU availability
check_gpu_availability() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0
        fi
    fi
    return 1
}

# Function to run tests with specific configuration
run_tests() {
    local config_name="$1"
    local extra_args="$2"

    print_status "Running tests with configuration: $config_name"

    # Set environment variables for this configuration
    case "$config_name" in
        "gpu_required")
            # Ensure GPU is required - no fallback
            unset JAX_PLATFORM_NAME
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            ;;
        "gpu_safe")
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.5
            export XLA_PYTHON_CLIENT_PREALLOCATE=false
            ;;
        "sequential")
            export XLA_PYTHON_CLIENT_MEM_FRACTION=0.75
            ;;
    esac

    # Run the tests
    if uv run pytest --config="$config_name" "$extra_args"; then
        print_success "Tests passed with $config_name configuration"
        return 0
    else
        print_error "Tests failed with $config_name configuration"
        return 1
    fi
}

# Main script logic
main() {
    print_status "Opifex Test Runner - GPU-Required Testing"
    print_status "========================================="

    # Parse command line arguments
    CONFIG="auto"
    EXTRA_ARGS=""

    while [[ $# -gt 0 ]]; do
        case $1 in
            --config)
                CONFIG="$2"
                shift 2
                ;;
            --gpu-required)
                CONFIG="gpu_required"
                shift
                ;;
            --gpu-safe)
                CONFIG="gpu_safe"
                shift
                ;;
            --sequential)
                CONFIG="sequential"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --config CONFIG     Test configuration (auto, gpu_required, gpu_safe, sequential)"
                echo "  --gpu-required      Require GPU (tests will fail if GPU unavailable)"
                echo "  --gpu-safe          Use GPU-safe configuration (single worker)"
                echo "  --sequential        Use sequential testing (no parallel execution)"
                echo "  --help, -h          Show this help message"
                echo ""
                echo "Test Configurations:"
                echo "  auto        - Automatically detect best configuration"
                echo "  gpu_required - Require GPU (tests will fail if GPU unavailable)"
                echo "  gpu_safe    - GPU with single worker to avoid memory conflicts"
                echo "  sequential  - No parallel execution for maximum stability"
                echo ""
                echo "Examples:"
                echo "  $0                    # Auto-detect configuration"
                echo "  $0 --gpu-required     # Require GPU (fail if unavailable)"
                echo "  $0 --gpu-safe         # GPU-safe configuration"
                echo "  $0 --sequential       # Sequential execution"
                echo ""
                echo "Note: This test runner requires GPU. Tests will fail with clear"
                echo "error messages if GPU is not available or not functioning properly."
                exit 0
                ;;
            *)
                EXTRA_ARGS="$EXTRA_ARGS $1"
                shift
                ;;
        esac
    done

    # Check if we're in a CI environment
    if is_ci_environment; then
        print_warning "Detected CI environment - using GPU-required configuration"
        CONFIG="gpu_required"
    fi

    # Auto-detect configuration if not specified
    if [[ "$CONFIG" == "auto" ]]; then
        print_status "Auto-detecting optimal test configuration..."

        if check_gpu_availability; then
            print_status "GPU detected - checking memory availability..."

            # Check GPU memory
            if command -v nvidia-smi &> /dev/null; then
                GPU_MEMORY=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
                if [[ $GPU_MEMORY -gt 8000 ]]; then  # More than 8GB free
                    print_status "Sufficient GPU memory available - using GPU-required configuration"
                    CONFIG="gpu_required"
                else
                    print_warning "Limited GPU memory - using GPU-safe configuration"
                    CONFIG="gpu_safe"
                fi
            else
                print_warning "Cannot check GPU memory - using GPU-safe configuration"
                CONFIG="gpu_safe"
            fi
        else
            print_error "No GPU detected - tests will fail"
            print_error "This test suite requires GPU. Please ensure GPU is available and properly configured."
            exit 1
        fi
    fi

    print_status "Selected configuration: $CONFIG"

    # Verify GPU is available before running tests
    if ! check_gpu_availability; then
        print_error "GPU is not available. This test suite requires GPU."
        print_error "Please ensure:"
        print_error "  1. NVIDIA GPU is installed and working"
        print_error "  2. NVIDIA drivers are properly installed"
        print_error "  3. CUDA toolkit is installed"
        print_error "  4. JAX with CUDA support is installed"
        exit 1
    fi

    # Run tests with selected configuration
    case "$CONFIG" in
        "gpu_required")
            run_tests "gpu_required" "$EXTRA_ARGS"
            ;;
        "gpu_safe")
            run_tests "gpu_safe" "$EXTRA_ARGS"
            ;;
        "sequential")
            run_tests "sequential" "$EXTRA_ARGS"
            ;;
        *)
            print_error "Unknown configuration: $CONFIG"
            exit 1
            ;;
    esac

    # Clean up environment variables
    unset JAX_PLATFORM_NAME
    unset XLA_PYTHON_CLIENT_MEM_FRACTION
    unset XLA_PYTHON_CLIENT_PREALLOCATE
}

# Run main function
main "$@"
