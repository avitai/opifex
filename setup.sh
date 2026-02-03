#!/bin/bash
# Unified Opifex Development Environment Setup
# Consolidated script for creating, building, and activating venv for both CPU and GPU development
# Supports Linux (CUDA), macOS (Intel), and macOS (Apple Silicon with Metal)

set -e  # Exit on any error

# Platform detection
OS_TYPE=$(uname -s)
ARCH_TYPE=$(uname -m)
IS_MACOS=false
IS_APPLE_SILICON=false

if [ "$OS_TYPE" = "Darwin" ]; then
    IS_MACOS=true
    if [ "$ARCH_TYPE" = "arm64" ]; then
        IS_APPLE_SILICON=true
    fi
fi

# Default values
DEEP_CLEAN=false
CPU_ONLY=false
HELP=false
VERBOSE=false
FORCE_REINSTALL=false
ENABLE_METAL=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deep-clean)
            DEEP_CLEAN=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --metal)
            ENABLE_METAL=true
            shift
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            export VERBOSE=true  # Export for Python scripts
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    cat << 'EOF'
🚀 Unified Opifex Development Environment Setup
==============================================

Creates, builds, and prepares the virtual environment for Opifex development
with automatic GPU/CPU detection and optimal configuration.

USAGE:
    ./setup.sh [OPTIONS]

OPTIONS:
    --deep-clean     Perform comprehensive cleaning (JAX cache, pip cache, etc.)
    --cpu-only       Force CPU-only setup (skip GPU detection)
    --metal          Enable Metal acceleration on Apple Silicon Macs
    --force          Force reinstallation even if environment exists
    --verbose, -v    Show detailed output during setup
    --help, -h       Show this help message

PLATFORM SUPPORT:
    Linux (CUDA)     Automatic NVIDIA GPU detection with CUDA acceleration
    macOS (Intel)    CPU-only mode (no GPU acceleration)
    macOS (Apple M)  Optional Metal acceleration with --metal flag

EXAMPLES:
    ./setup.sh                    # Standard setup with auto GPU detection
    ./setup.sh --deep-clean       # Clean setup with cache clearing
    ./setup.sh --cpu-only         # Force CPU-only development setup
    ./setup.sh --metal            # macOS Apple Silicon with Metal acceleration
    ./setup.sh --force --verbose  # Verbose forced reinstallation

ACTIVATION:
    After setup, activate the environment with:
    source ./activate.sh

FILES CREATED:
    .venv/           Virtual environment directory
    .env             Environment variables and CUDA configuration
    activate.sh      Unified activation script
    uv.lock          Dependency lock file

REQUIREMENTS:
    - uv package manager (installed automatically if missing)
    - Python 3.10+ (handled by uv)
    - NVIDIA drivers (for GPU support on Linux)
    - Xcode Command Line Tools (for macOS)

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

# Function to check and install uv if needed
ensure_uv_installed() {
    if ! command -v uv &> /dev/null; then
        log_warning "uv not found. Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        if ! command -v uv &> /dev/null; then
            log_error "Failed to install uv. Please install manually:"
            echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
        log_success "uv installed successfully"
    else
        verbose_log "uv already installed: $(uv --version)"
    fi
}

# Function to detect CUDA availability (Linux only)
detect_cuda_support() {
    if [ "$CPU_ONLY" = true ]; then
        verbose_log "CPU-only mode requested, skipping GPU detection"
        return 1
    fi

    # CUDA is only available on Linux
    if [ "$IS_MACOS" = true ]; then
        verbose_log "macOS detected - CUDA not available"
        return 1
    fi

    if command -v nvidia-smi &> /dev/null; then
        local gpu_info
        if gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) && [ -n "$gpu_info" ]; then
            log_success "NVIDIA GPU detected: $gpu_info"

            # Check CUDA installation
            if [ -d "/usr/local/cuda" ] || [ -n "$CUDA_HOME" ]; then
                verbose_log "CUDA installation found"
                return 0
            else
                log_warning "GPU detected but CUDA not found in standard locations"
                log_info "Will attempt GPU setup anyway"
                return 0
            fi
        fi
    fi

    log_info "No NVIDIA GPU detected - setting up CPU-only environment"
    return 1
}

# Function to detect Metal support (macOS Apple Silicon only)
detect_metal_support() {
    if [ "$IS_MACOS" != true ]; then
        verbose_log "Not macOS - Metal not available"
        return 1
    fi

    if [ "$CPU_ONLY" = true ]; then
        verbose_log "CPU-only mode requested, skipping Metal detection"
        return 1
    fi

    if [ "$ENABLE_METAL" != true ]; then
        verbose_log "Metal not enabled (use --metal flag to enable)"
        return 1
    fi

    if [ "$IS_APPLE_SILICON" = true ]; then
        log_success "Apple Silicon detected - Metal acceleration available"
        return 0
    fi

    log_warning "Metal requested but not on Apple Silicon - Metal not available"
    return 1
}

# Function to perform cleaning
perform_cleaning() {
    log_step "Cleaning existing environment..."

    # Remove virtual environment
    if [ -d ".venv" ]; then
        verbose_log "Removing virtual environment (.venv)"
        rm -rf .venv
    fi

    # Remove lock files if force reinstall
    if [ "$FORCE_REINSTALL" = true ] && [ -f "uv.lock" ]; then
        verbose_log "Removing lock file (uv.lock)"
        rm -f uv.lock
    fi

    # Clean uv cache to avoid package conflicts
    verbose_log "Cleaning uv cache"
    uv cache clean 2>/dev/null || true

    # Remove existing environment files
    if [ -f ".env" ]; then
        verbose_log "Removing existing .env file"
        rm -f .env
    fi

    # Remove old activation scripts
    for script in activate_opifex.sh activate_env.sh activate_opifex_venv.sh setup_dev.sh; do
        if [ -f "$script" ]; then
            verbose_log "Removing old script: $script"
            rm -f "$script"
        fi
    done

    # Clean Python cache files
    verbose_log "Cleaning Python cache files"
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true

    # Deep cleaning if requested
    if [ "$DEEP_CLEAN" = true ]; then
        log_step "Performing deep cleaning..."

        # Clean JAX compilation cache
        if [ -d "$HOME/.cache/jax" ]; then
            verbose_log "Removing JAX compilation cache"
            rm -rf "$HOME/.cache/jax"
        fi

        # Clean pip cache
        verbose_log "Cleaning pip cache"
        python -m pip cache purge 2>/dev/null || pip cache purge 2>/dev/null || true

        # Clean pytest cache
        if [ -d ".pytest_cache" ]; then
            verbose_log "Removing pytest cache"
            rm -rf .pytest_cache
        fi

        # Clean coverage files
        for file in .coverage .coverage.*; do
            if [ -f "$file" ]; then
                verbose_log "Removing coverage file: $file"
                rm -f "$file"
            fi
        done
        if [ -d "htmlcov" ]; then
            verbose_log "Removing HTML coverage directory"
            rm -rf htmlcov
        fi

        # Clean benchmark results
        if [ -d "benchmark_results" ]; then
            verbose_log "Cleaning benchmark results"
            find benchmark_results -name "*.json" -delete 2>/dev/null || true
        fi

        # Clean temp directory
        if [ -d "temp" ]; then
            verbose_log "Cleaning temp directory"
            rm -rf temp/*
        fi

        # Clean temporary files
        verbose_log "Cleaning temporary files"
        find . -name "tmp*" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name ".tmp*" -type f -delete 2>/dev/null || true
    fi

    log_success "Environment cleaned"
}

# Function to create .env file
create_env_file() {
    local has_cuda=$1
    local has_metal=$2

    log_step "Creating environment configuration..."

    # Create cache directories
    mkdir -p .cache/jax .cache/xla 2>/dev/null || true

    if [ "$has_metal" = true ]; then
        # Metal configuration for Apple Silicon
        cat > .env << 'EOF'
# Opifex Environment Configuration - Metal (Apple Silicon)
# Auto-generated by unified setup script

# JAX Configuration for Metal
export JAX_PLATFORMS="metal,cpu"
export JAX_ENABLE_X64="0"
export METAL_DEVICE_WRAPPER_TYPE="1"

# TensorFlow macOS ARM64 compatibility settings
export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTEROP_THREADS="1"
export TF_NUM_INTRAOP_THREADS="1"
export TF_METAL_DEVICE_SELECTOR=""
export TF_DISABLE_MLC_BRIDGE="1"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Testing configuration
export PYTEST_CUDA_ENABLED="false"

# Performance settings
export TF_CPP_MIN_LOG_LEVEL="1"

# Protobuf Configuration to fix compatibility issues
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
EOF
        verbose_log "Created Metal-enabled .env configuration"
    elif [ "$has_cuda" = true ]; then
        # Check if .env.example template exists and use it
        if [ -f ".env.example" ]; then
            # Copy template - it uses $(pwd) which will be evaluated when sourced
            cp .env.example .env
            verbose_log "Created GPU-enabled .env configuration from template"
        elif [ -f "dot_env_template" ]; then
            # Legacy fallback: use dot_env_template if it exists
            cp dot_env_template .env
            verbose_log "Created GPU-enabled .env configuration from legacy template"
        else
            # Fallback to embedded template if dot_env_template is missing
            # Detect Python version dynamically
            PYTHON_VERSION=$(python -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "python3.11")

            cat > .env << EOF
# Opifex Environment Configuration - GPU Enabled
# Auto-generated by unified setup script

# CUDA Library Configuration - Use local venv CUDA installation
# Point to locally installed CUDA libraries in venv
export LD_LIBRARY_PATH="\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cublas/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cusolver/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cusparse/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cudnn/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/cufft/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/curand/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/nccl/lib:\$(pwd)/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia/nvjitlink/lib:\${LD_LIBRARY_PATH}"

# JAX Configuration for CUDA
export JAX_PLATFORMS="cuda,cpu"
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.85"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_latency_hiding_scheduler=true"

# JAX CUDA Plugin Configuration
export JAX_CUDA_PLUGIN_VERIFY="false"
export JAX_SKIP_CUDA_CONSTRAINTS_CHECK="1"

# Reduce CUDA warnings
export TF_CPP_MIN_LOG_LEVEL="1"

# Performance settings
export JAX_ENABLE_X64="0"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Testing configuration
export PYTEST_CUDA_ENABLED="true"

# Protobuf Configuration to fix compatibility issues
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
EOF
            verbose_log "Created GPU-enabled .env configuration from embedded template"
        fi
    elif [ "$IS_MACOS" = true ]; then
        # macOS CPU-only configuration with TensorFlow compatibility settings
        cat > .env << 'EOF'
# Opifex Environment Configuration - macOS CPU Only
# Auto-generated by unified setup script

# JAX Configuration for CPU
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="0"

# macOS-specific TensorFlow compatibility settings
export CUDA_VISIBLE_DEVICES=""
export TF_NUM_INTEROP_THREADS="1"
export TF_NUM_INTRAOP_THREADS="1"
export TF_METAL_DEVICE_SELECTOR=""
export TF_DISABLE_MLC_BRIDGE="1"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Testing configuration
export PYTEST_CUDA_ENABLED="false"

# Performance settings
export TF_CPP_MIN_LOG_LEVEL="1"

# Protobuf Configuration to fix compatibility issues
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
EOF
        verbose_log "Created macOS CPU-only .env configuration"
    else
        # Linux CPU-only configuration
        cat > .env << 'EOF'
# Opifex Environment Configuration - CPU Only
# Auto-generated by unified setup script

# JAX Configuration for CPU
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="0"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}$(pwd)"

# Testing configuration
export PYTEST_CUDA_ENABLED="false"

# Performance settings
export TF_CPP_MIN_LOG_LEVEL="1"

# Protobuf Configuration to fix compatibility issues
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
EOF
        verbose_log "Created CPU-only .env configuration"
    fi

    log_success "Environment configuration created"
}

# Function to create unified activation script with enhanced process detection
create_activation_script() {
    log_step "Creating unified activation script with enhanced process detection..."

    cat > activate.sh << 'EOF'
#!/bin/bash
# Unified Opifex Environment Activation Script
# Created by unified setup - replaces all previous activation scripts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}🚀 Activating Opifex Development Environment${NC}"
echo "============================================="

# Check if already activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${YELLOW}⚠️  Virtual environment already active: $VIRTUAL_ENV${NC}"
    echo "Deactivating current environment..."

    # Improved process detection and user feedback
    # This prevents hanging by informing users about blocking processes
    # Check for processes using the virtual environment
    VENV_PROCESSES=$(pgrep -f "$VIRTUAL_ENV" | xargs -I {} ps -p {} -o pid,etime,args --no-headers 2>/dev/null || true)

    if [[ -n "$VENV_PROCESSES" ]]; then
        echo -e "${YELLOW}🔍 Checking for processes using the virtual environment...${NC}"

        # Count processes
        PROCESS_COUNT=$(echo "$VENV_PROCESSES" | wc -l)
        if [[ $PROCESS_COUNT -gt 0 ]]; then
            echo -e "${YELLOW}⏳ Found $PROCESS_COUNT process(es) using the virtual environment:${NC}"
            echo ""

            # Show process details in a readable format
            echo "$VENV_PROCESSES" | while IFS= read -r line; do
                if [[ -n "$line" ]]; then
                    PID=$(echo "$line" | awk '{print $1}')
                    ETIME=$(echo "$line" | awk '{print $2}')
                    CMD=$(echo "$line" | awk '{for(i=3;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/[[:space:]]*$//')
                    echo -e "${CYAN}   • PID $PID (running for $ETIME): ${NC}$CMD"
                fi
            done

            echo ""
            echo -e "${YELLOW}💡 Options:${NC}"
            echo -e "${CYAN}   1. Wait for processes to complete naturally${NC}"
            echo -e "${CYAN}   2. Press Ctrl+C to cancel activation${NC}"
            echo -e "${CYAN}   3. In another terminal, stop processes manually:${NC}"
            echo -e "${CYAN}      pkill -f pytest  # Stop test processes${NC}"
            echo -e "${CYAN}      pkill -f jupyter # Stop Jupyter processes${NC}"
            echo ""
            echo -e "${YELLOW}⏳ Waiting for processes to complete (this may take a while)...${NC}"
            echo -e "${CYAN}   You can press Ctrl+C to cancel and handle processes manually${NC}"
        fi
    fi

    # Attempt deactivation with timeout and progress indication
    echo -e "${YELLOW}🔄 Attempting environment deactivation...${NC}"

    # Function to show waiting animation
    show_waiting() {
        local delay=1
        local spinstr="|/-\\"
        local temp
        while true; do
            temp=${spinstr#?}
            printf "\r%s   [%c] Waiting for environment deactivation...%s" "${CYAN}" "$spinstr" "${NC}"
            spinstr=$temp${spinstr%"$temp"}
            sleep $delay
        done
    }

        # Start background spinner
    show_waiting &
    SPINNER_PID=$!

    # Setup cleanup function for spinner
    cleanup_spinner() {
        if [[ -n "$SPINNER_PID" ]]; then
            kill $SPINNER_PID 2>/dev/null || true
            wait $SPINNER_PID 2>/dev/null || true
            printf "\r                                                    \r"  # Clear spinner line
        fi
    }

    # Setup trap to ensure spinner cleanup on script exit
    trap cleanup_spinner EXIT INT TERM

    # Try to deactivate with timeout
    if timeout 30 bash -c 'deactivate 2>/dev/null || true'; then
        cleanup_spinner
        printf "\r%s✅ Environment deactivation completed%s\n" "${GREEN}" "${NC}"
    else
        cleanup_spinner
        printf "\r%s❌ Environment deactivation timed out after 30 seconds%s\n" "${RED}" "${NC}"
        echo -e "${YELLOW}💡 This usually means processes are still using the environment.${NC}"
        echo -e "${CYAN}   Run this command to force cleanup:${NC}"
        echo -e "${CYAN}   pkill -f '$VIRTUAL_ENV'${NC}"
        echo ""
        echo -e "${YELLOW}⚠️  Proceeding with activation anyway...${NC}"
        # Clear VIRTUAL_ENV to force reactivation
        unset VIRTUAL_ENV
    fi

    # Clear the trap
    trap - EXIT INT TERM
fi

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo -e "${GREEN}✅ Virtual environment activated${NC}"
else
    echo -e "${RED}❌ Virtual environment not found!${NC}"
    echo "Run './setup.sh' to create the environment first."
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    source .env
    echo -e "${GREEN}✅ Environment configuration loaded${NC}"

    # Show configuration based on JAX_PLATFORMS
    if [[ "$JAX_PLATFORMS" == *"cuda"* ]]; then
        echo -e "${CYAN}   🎮 GPU Mode: CUDA enabled${NC}"
        echo -e "${CYAN}   📍 CUDA_HOME: ${CUDA_HOME:-not set}${NC}"
    elif [[ "$JAX_PLATFORMS" == *"metal"* ]]; then
        echo -e "${CYAN}   🍎 GPU Mode: Metal enabled (Apple Silicon)${NC}"
    else
        echo -e "${CYAN}   💻 CPU Mode: CPU-only configuration${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  .env file not found - using minimal setup${NC}"
    export JAX_PLATFORMS="cpu"
fi

# Display system information
echo ""
echo -e "${BLUE}🔍 Environment Status:${NC}"
echo -e "${CYAN}   Python: $(python --version)${NC}"
echo -e "${CYAN}   Working Directory: $(pwd)${NC}"
echo -e "${CYAN}   Virtual Environment: $VIRTUAL_ENV${NC}"

# Check JAX installation and display configuration
echo ""
echo -e "${BLUE}🧪 JAX Configuration:${NC}"

# JAX configuration check with error handling
python << 'PYTHON_EOF'
try:
    import jax
    import jax.numpy as jnp

    print(f"   JAX version: {jax.__version__}")
    print(f"   Default backend: {jax.default_backend()}")

    devices = jax.devices()
    print(f"   Available devices: {len(devices)} total")

    # Check for GPU devices
    gpu_devices = [d for d in devices if d.platform == 'gpu']
    metal_devices = [d for d in devices if d.platform == 'METAL']
    cpu_devices = [d for d in devices if d.platform == 'cpu']

    if gpu_devices:
        print(f"   🎉 GPU devices: {len(gpu_devices)} ({[str(d) for d in gpu_devices]})")
        print("   ✅ CUDA acceleration ready!")

        # Quick GPU test
        try:
            x = jnp.array([1., 2., 3.])
            y = jnp.sum(x**2)
            print(f"   🧮 GPU test successful: {float(y)}")
        except Exception as e:
            print(f"   ⚠️  GPU test warning: {e}")
    elif metal_devices:
        print(f"   🍎 Metal devices: {len(metal_devices)} ({[str(d) for d in metal_devices]})")
        print("   ✅ Metal acceleration ready!")

        # Quick Metal test
        try:
            x = jnp.array([1., 2., 3.])
            y = jnp.sum(x**2)
            print(f"   🧮 Metal test successful: {float(y)}")
        except Exception as e:
            print(f"   ⚠️  Metal test warning: {e}")
    else:
        print(f"   💻 CPU devices: {len(cpu_devices)} ({[str(d) for d in cpu_devices]})")
        print("   📱 Running in CPU-only mode")

    # Quick functionality test
    try:
        x = jnp.linspace(0, 1, 100)
        y = jnp.sin(2 * jnp.pi * x)
        print(f"   ✅ JAX functionality verified")
    except Exception as e:
        print(f"   ❌ JAX functionality test failed: {e}")

except ImportError as e:
    print(f"   ❌ JAX not installed properly: {e}")
    print("   Run './setup.sh' to reinstall dependencies")
except Exception as e:
    print(f"   ⚠️  JAX configuration issue: {e}")
PYTHON_EOF

# Display usage information
echo ""
echo -e "${BLUE}🚀 Ready for Development!${NC}"
echo "========================="
echo ""
echo -e "${GREEN}📝 Common Commands:${NC}"
echo -e "${CYAN}   uv run pytest tests/ -v              ${NC}# Run all tests"
echo -e "${CYAN}   uv run pytest tests/neural/ -v       ${NC}# Run neural network tests"
echo -e "${CYAN}   uv run pytest tests/benchmarks/ -v   ${NC}# Run benchmarking tests"
echo -e "${CYAN}   uv run python your_script.py         ${NC}# Run your code"
echo -e "${CYAN}   uv run jupyter lab                   ${NC}# Start Jupyter lab"
echo ""
echo -e "${GREEN}🔧 Development Tools:${NC}"
echo -e "${CYAN}   uv add package_name                  ${NC}# Add new dependency"
echo -e "${CYAN}   uv run pre-commit run --all-files    ${NC}# Run code quality checks"
echo -e "${CYAN}   uv run pytest --cov=opifex tests/    ${NC}# Run tests with coverage"
echo ""
echo -e "${GREEN}📊 Benchmarking:${NC}"
echo -e "${CYAN}   uv run python -m opifex.benchmarking  ${NC}# Run benchmarks"
echo -e "${CYAN}   ./scripts/gpu_test_manager.py        ${NC}# GPU testing utilities"
echo ""
echo -e "${YELLOW}💡 To deactivate: ${NC}deactivate"
EOF

    chmod +x activate.sh
    log_success "Unified activation script created: ./activate.sh"
}

# Function to create virtual environment and install dependencies
setup_environment() {
    local has_cuda=$1
    local has_metal=$2

    log_step "Creating virtual environment..."
    uv venv

    # Activate the environment for installation
    source .venv/bin/activate
    source .env

    log_step "Installing dependencies..."

    # First, create lock file if it doesn't exist
    if [ ! -f "uv.lock" ]; then
        log_info "Creating dependency lock file..."
        uv lock
    fi

    if [ "$has_metal" = true ]; then
        log_info "Installing with Metal support for Apple Silicon..."
        log_info "This may take several minutes on first install..."
        verbose_log "Installing complete package with macOS Metal dependencies"

        echo -e "${CYAN}   → Running: uv sync --extra all-macos${NC}"
        if uv sync --extra all-macos; then
            log_success "Complete installation with Metal support successful"
        else
            log_warning "Metal installation failed, falling back to CPU-only"
            echo -e "${CYAN}   → Running: uv sync --extra all-cpu${NC}"
            uv sync --extra all-cpu
            has_metal=false
            # Update .env file for CPU-only
            create_env_file false false
        fi
    elif [ "$has_cuda" = true ]; then
        log_info "Installing with CUDA support..."
        log_info "This may take several minutes on first install..."
        verbose_log "Installing complete package with all dependencies (dev, docs, gpu, test)"

        # Install complete package with all dependency groups
        echo -e "${CYAN}   → Running: uv sync --extra all${NC}"
        if uv sync --extra all; then
            log_success "Complete installation with GPU support successful"

            # Ensure matching JAX CUDA plugin versions
            log_info "Ensuring correct JAX CUDA plugin versions..."
            JAX_VERSION=$(python -c "import jax; print(jax.__version__)" 2>/dev/null || echo "0.8.0")

            # Install matching CUDA plugins for JAX if needed
            verbose_log "Checking JAX CUDA plugins version $JAX_VERSION"
            # For JAX 0.8.x, try to ensure CUDA plugins match
            if [[ "$JAX_VERSION" == 0.8.* ]]; then
                verbose_log "JAX 0.8.x detected, verifying CUDA plugin compatibility"
                # The CUDA plugins are included in jax[cuda12_local] for 0.8.x
            fi
        else
            log_warning "Full installation with GPU failed, falling back to CPU-only"
            echo -e "${CYAN}   → Running: uv sync --extra all-cpu${NC}"
            uv sync --extra all-cpu
            has_cuda=false
            # Update .env file for CPU-only
            create_env_file false false
        fi
    elif [ "$IS_MACOS" = true ]; then
        log_info "Installing CPU-only version for macOS..."
        log_info "This may take several minutes on first install..."
        echo -e "${CYAN}   → Running: uv sync --extra all-cpu${NC}"
        uv sync --extra all-cpu
    else
        log_info "Installing CPU-only version with all dependencies..."
        log_info "This may take several minutes on first install..."
        echo -e "${CYAN}   → Running: uv sync --extra all-cpu${NC}"
        uv sync --extra all-cpu
    fi

    log_success "Dependencies installed successfully"
    return 0
}

# Function to verify installation
verify_installation() {
    local has_cuda=$1
    local has_metal=$2

    log_step "Verifying installation..."

    # Create a temporary verification script and run it
    if [ -f .venv/bin/python ]; then
        # Create temporary verification script
        cat > /tmp/opifex_verify.py << 'PYTHON_EOF'
import sys
import traceback
import os

try:
    # Verify environment setup
    print("🔍 Environment verification:")
    print(f"   LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH', 'Not set')[:100]}...")
    print(f"   CUDA_HOME: {os.environ.get('CUDA_HOME', 'Not set')}")
    print(f"   JAX_PLATFORMS: {os.environ.get('JAX_PLATFORMS', 'Not set')}")

    import jax
    import jax.numpy as jnp
    import flax
    import optax

    print(f"✅ Core dependencies verified:")
    print(f"   JAX: {jax.__version__}")
    print(f"   Flax: {flax.__version__}")
    print(f"   Optax: {optax.__version__}")

    # Test basic functionality
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x**2)
    print(f"✅ Basic computation test: {float(y)}")

    # Test devices with better error handling
    try:
        devices = jax.devices()
        gpu_devices = [d for d in devices if d.platform == 'gpu']
        metal_devices = [d for d in devices if d.platform == 'METAL']

        print(f"✅ Available devices: {len(devices)} total")
        if gpu_devices:
            print(f"✅ GPU devices detected: {len(gpu_devices)}")
            for i, device in enumerate(gpu_devices):
                print(f"   GPU {i}: {device}")

            # Simple GPU test with more conservative approach
            try:
                # Test GPU computation
                z = jnp.array([1., 2., 3.])
                with jax.default_device(gpu_devices[0]):
                    w = jnp.dot(z, z)
                print(f"✅ GPU computation test: {float(w)}")

                # Test GPU memory allocation
                test_array = jnp.ones((100, 100))
                result = jnp.sum(test_array)
                print(f"✅ GPU memory test: {float(result)}")

            except Exception as gpu_e:
                print(f"⚠️  GPU test warning: {gpu_e}")
                print("ℹ️  GPU detected but computation failed - checking CUDA libraries...")
                # Try to provide more diagnostic info
                try:
                    import subprocess
                    nvidia_smi = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
                    if nvidia_smi.returncode == 0:
                        print("ℹ️  nvidia-smi output shows GPU is available")
                    else:
                        print("⚠️  nvidia-smi failed - GPU driver issues possible")
                except:
                    print("ℹ️  Could not run nvidia-smi for diagnostics")
        elif metal_devices:
            print(f"✅ Metal devices detected: {len(metal_devices)}")
            for i, device in enumerate(metal_devices):
                print(f"   Metal {i}: {device}")

            # Simple Metal test
            try:
                z = jnp.array([1., 2., 3.])
                w = jnp.dot(z, z)
                print(f"✅ Metal computation test: {float(w)}")
            except Exception as metal_e:
                print(f"⚠️  Metal test warning: {metal_e}")
        else:
            print("ℹ️  No GPU/Metal devices detected (CPU-only mode)")

    except Exception as device_e:
        print(f"⚠️  Device detection error: {device_e}")
        print("ℹ️  This may indicate library path issues")
        traceback.print_exc()

    print("✅ Installation verification complete!")

except ImportError as e:
    print(f"❌ Import error: {e}")
    print("ℹ️  Installation may be incomplete - missing Python packages")
    print("💡 Try: uv sync --extra all")
    sys.exit(2)  # Different exit code for import errors
except Exception as e:
    print(f"❌ Verification error: {e}")
    print("ℹ️  This may be due to library configuration issues")
    print("💡 Check environment variables and library paths")
    # Only show full traceback in verbose mode or for unexpected errors
    if "VERBOSE" in os.environ or "segmentation fault" in str(e).lower():
        traceback.print_exc()
        sys.exit(3)  # Different exit code for runtime errors
PYTHON_EOF

                # Run the verification script in a new bash session
        bash -c "cd '$PWD' && source ./activate.sh > /dev/null 2>&1 && python /tmp/opifex_verify.py"
        local python_exit_code=$?

        # Clean up temporary script
        rm -f /tmp/opifex_verify.py

        # Handle the exit code properly
        case $python_exit_code in
            0)
                log_success "Installation verified successfully"
                return 0
                ;;
            2)
                log_error "Installation verification failed - missing dependencies (exit code $python_exit_code)"
                echo -e "${YELLOW}🔧 Dependency Issue - Try:${NC}"
                echo -e "${CYAN}   uv sync --extra all${NC}"
                return 2
                ;;
            3)
                log_error "Installation verification failed - runtime error (exit code $python_exit_code)"
                echo -e "${YELLOW}🔧 Runtime Issue - Try:${NC}"
                echo -e "${CYAN}   Check libraries and drivers${NC}"
                return 3
                ;;
            *)
                log_error "Installation verification failed - unexpected error (exit code $python_exit_code)"
                echo -e "${YELLOW}🔧 General troubleshooting:${NC}"
                echo -e "${CYAN}   1. Check if drivers are properly installed${NC}"
                echo -e "${CYAN}   2. Try manual activation: source ./activate.sh${NC}"
                echo -e "${CYAN}   3. Test JAX manually: python -c 'import jax; print(jax.devices())'${NC}"
                return 1
                ;;
        esac
    else
        log_error "Virtual environment not found at .venv/bin/python"
        return 1
    fi
}

# Function to display setup summary
display_summary() {
    local has_cuda=$1
    local has_metal=$2

    echo ""
    echo -e "${GREEN}🎉 Opifex Development Environment Setup Complete!${NC}"
    echo "================================================="
    echo ""
    echo -e "${BLUE}📁 Files Created:${NC}"
    echo -e "${CYAN}   .venv/                 Virtual environment${NC}"
    echo -e "${CYAN}   .env                   Environment configuration${NC}"
    echo -e "${CYAN}   activate.sh            Unified activation script${NC}"
    echo -e "${CYAN}   uv.lock                Dependency lock file${NC}"
    echo ""
    echo -e "${BLUE}🚀 Quick Start:${NC}"
    echo -e "${YELLOW}   source ./activate.sh   ${NC}# Activate environment (use 'source'!)"
    echo -e "${CYAN}   uv run pytest tests/   ${NC}# Run tests to verify setup"
    echo ""

    if [ "$has_metal" = true ]; then
        echo -e "${GREEN}🍎 GPU Support: ✅ Metal Enabled (Apple Silicon)${NC}"
        echo "   Your environment is ready for Metal-accelerated development!"
    elif [ "$has_cuda" = true ]; then
        echo -e "${GREEN}🎮 GPU Support: ✅ CUDA Enabled${NC}"
        echo "   Your environment is ready for GPU-accelerated development!"
    else
        echo -e "${BLUE}💻 GPU Support: ❌ CPU-Only Mode${NC}"
        if [ "$IS_MACOS" = true ]; then
            if [ "$IS_APPLE_SILICON" = true ]; then
                echo "   For Metal support on Apple Silicon, re-run with: ./setup.sh --metal --force"
            else
                echo "   macOS Intel detected - GPU acceleration not available"
            fi
        else
            echo "   For GPU support, ensure NVIDIA drivers and CUDA are installed,"
            echo "   then re-run with: ./setup.sh --force"
        fi
    fi
    echo ""
    echo -e "${PURPLE}📖 For more information, see docs/guides/development.md${NC}"
}

# Main execution function
main() {
    echo -e "${PURPLE}🚀 Unified Opifex Development Environment Setup${NC}"
    echo "==============================================="
    echo ""

    # Display platform information
    if [ "$IS_MACOS" = true ]; then
        if [ "$IS_APPLE_SILICON" = true ]; then
            log_info "Platform: macOS (Apple Silicon - $ARCH_TYPE)"
        else
            log_info "Platform: macOS (Intel - $ARCH_TYPE)"
        fi
    else
        log_info "Platform: $OS_TYPE ($ARCH_TYPE)"
    fi

    # Pre-flight checks
    ensure_uv_installed

    # Detect GPU capability
    HAS_CUDA=false
    HAS_METAL=false

    if detect_metal_support; then
        HAS_METAL=true
    elif detect_cuda_support; then
        HAS_CUDA=true
    fi

    # Check if environment already exists and handle appropriately
    if [ -d ".venv" ] && [ "$FORCE_REINSTALL" != true ]; then
        log_warning "Virtual environment already exists"
        echo "Use --force to reinstall or source ./activate.sh to use existing environment"
        exit 1
    fi

    # Perform cleanup
    perform_cleaning

    # Create configuration files
    create_env_file "$HAS_CUDA" "$HAS_METAL"
    create_activation_script

    # Setup environment and install dependencies
    if ! setup_environment "$HAS_CUDA" "$HAS_METAL"; then
        log_error "Failed to setup environment"
        exit 1
    fi

    # Verify installation works - run in a clean environment
    echo ""
    echo -e "${BLUE}🔧 Verifying installation in clean environment...${NC}"
    if ! verify_installation "$HAS_CUDA" "$HAS_METAL"; then
        echo -e "${RED}❌ Setup completed but verification failed${NC}"
        echo -e "${YELLOW}ℹ️  The environment may still be usable - try: source ./activate.sh${NC}"
        echo -e "${YELLOW}ℹ️  Then test manually: python -c 'import jax; print(jax.devices())'${NC}"
        exit 1
    fi

    # Show summary
    display_summary "$HAS_CUDA" "$HAS_METAL"
}

# Run main function with all arguments
main "$@"
