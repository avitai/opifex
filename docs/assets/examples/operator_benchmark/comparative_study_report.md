# Neural Operator Comparative Benchmarking Study

**Generated**: 2026-02-06 08:59:03

## Executive Summary

This report presents a comprehensive comparative analysis of 3 neural operators across 1 datasets and 1 resolutions.

## Neural Operators Analyzed

- **UNO**: U-Net Neural Operator (Multi-scale CNN + Fourier layers)
- **FNO**: Fourier Neural Operator (Spectral convolutions)
- **SFNO**: Spherical Fourier Neural Operator (Spherical harmonics)

## Datasets Evaluated

- **Darcy**: 3 benchmark runs

## Multi-Resolution Analysis

**Resolutions tested**: 32

## Performance Summary

### UNO
- **Mean MSE**: 0.187617
- **MSE Std**: 0.000000
- **Mean Execution Time**: 0.0010s
- **Successful Runs**: 1

### FNO
- **Mean MSE**: 0.006594
- **MSE Std**: 0.000000
- **Mean Execution Time**: 0.0004s
- **Successful Runs**: 1

### SFNO
- **Mean MSE**: 0.002312
- **MSE Std**: 0.000000
- **Mean Execution Time**: 0.0004s
- **Successful Runs**: 1

## Key Findings

- **Best Overall Accuracy**: SFNO (MSE: 0.002312)
- **Fastest Execution**: FNO (0.0004s average)

## Conclusions

This comparative study provides insights into the relative performance of different neural operator architectures across multiple scientific computing scenarios. Results should be interpreted in the context of specific application requirements.

## Generated Files

- `mse_comparison.png`: MSE vs resolution plots
- `execution_time_comparison.png`: Execution time distributions
- `statistical_analysis.json`: Detailed statistical comparisons
- Individual benchmark result files in results directory
