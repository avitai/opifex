# Opifex Deployment Enhancements - Quick Start Guide

## 🚀 **Immediate Usage**

### **1. Show Opifex Framework Status**

```bash
cd deployment/
make -f Makefile-opifex-enhancements opifex-status
```

### **2. Get Opifex-Specific Help**

```bash
make -f Makefile-opifex-enhancements opifex-help
```

### **3. Validate Resource Profiles**

```bash
# Validate all workload profiles
make -f Makefile-opifex-enhancements validate-profiles

# Or validate individually:
make -f Makefile-opifex-enhancements validate-small-profile
make -f Makefile-opifex-enhancements validate-large-profile
make -f Makefile-opifex-enhancements validate-research-profile
```

### **4. Test Framework Structure (Development)**

```bash
# Test L2O framework structure
make -f Makefile-opifex-enhancements test-l2o-simple

# Test neural operators structure
make -f Makefile-opifex-enhancements test-neural-ops-simple

# Test benchmarking infrastructure
make -f Makefile-opifex-enhancements test-benchmarking-simple

# Run all structure tests
make -f Makefile-opifex-enhancements test-opifex-simple
```

### **5. Run Enhanced Deployment Testing**

```bash
# Run full deployment validation
./test-deployment.sh

# This will generate an enhanced test report with Opifex-specific results
```

## 📋 **Key Files**

- **`Makefile-opifex-enhancements`**: Opifex-specific deployment targets
- **`test-deployment.sh`**: Enhanced testing script with Opifex validation
- **`DEPLOYMENT_TESTING_BEST_PRACTICES.md`**: Full testing guidelines
- **`OPIFEX_ENHANCEMENTS_SUMMARY.md`**: Complete implementation summary

## 🎯 **Resource Profiles**

| Profile | CPU | Memory | GPU | Replicas | Use Case |
|---------|-----|--------|-----|----------|----------|
| **Small** | 2 | 4Gi | 1 | 1 | Development, small neural operator training |
| **Large** | 8 | 32Gi | 4 | 1 | Production L2O optimization |
| **Research** | 4 | 16Gi | 2 | 2 | Research workflows, multi-experiments |

## ⚡ **Prerequisites**

- **kubectl**: Kubernetes command-line tool
- **helm**: Kubernetes package manager
- **kustomize**: Kubernetes configuration management

*Note: Framework structure tests work without cluster access*

## 🔧 **Integration**

To integrate with existing Makefile:

```makefile
# Add to end of main Makefile
-include Makefile-opifex-enhancements
```

## 📊 **Expected Results**

✅ **Framework Status Display**

```
Framework Version: Opifex v1.0.0
L2O Tests: 158/158 passing
Pre-commit Hooks: 19/19 passing
Framework Completion: 99%+
Framework ready for production deployment
```

✅ **Profile Validation**

```
✓ Small workload profile configuration valid
✓ Large workload profile configuration valid
✓ Research workload profile configuration valid
🎉 All Opifex resource profiles validated!
```
