# GPU Development

## Overview

Guidelines for GPU-accelerated development with Opifex using JAX and CUDA.

## GPU Setup

### CUDA Installation

```bash
# Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run

# Verify installation
nvcc --version
nvidia-smi
```

### JAX GPU Configuration

```python
import jax
print("Available devices:", jax.devices())
print("Default backend:", jax.default_backend())

# Force GPU usage
jax.config.update('jax_platform_name', 'gpu')
```

## Memory Management

### GPU Memory Optimization

```python
# Enable memory preallocation
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

# Monitor memory usage
print(f"GPU memory: {jax.extend.backend.get_backend().get_device_memory_info()}")
```

### Batch Size Tuning

```python
# Auto-tune batch size for memory
def find_optimal_batch_size(model, max_size=1024):
    for batch_size in [32, 64, 128, 256, 512, 1024]:
        try:
            test_batch = jnp.ones((batch_size, input_dim))
            _ = model(test_batch)
            print(f"Batch size {batch_size}: OK")
        except jax.errors.OutOfMemoryError:
            print(f"Batch size {batch_size}: OOM")
            return batch_size // 2
    return max_size
```

## Performance Optimization

### JIT Compilation

```python
@jax.jit
def optimized_training_step(params, batch):
    """JIT-compiled training step for speed."""
    loss, grads = jax.value_and_grad(loss_fn)(params, batch)
    return loss, grads

# Compile once, run many times
compiled_step = jax.jit(training_step)
```

### PMAP for Multi-GPU

```python
# Parallel computation across GPUs
@jax.pmap
def parallel_training_step(params, batch):
    """Multi-GPU training step."""
    return jax.value_and_grad(loss_fn)(params, batch)

# Replicate across devices
n_devices = jax.local_device_count()
replicated_params = jax.tree_map(
    lambda x: jnp.stack([x] * n_devices), params
)
```

## Profiling and Debugging

### Performance Profiling

```python
# Profile GPU kernels
with jax.profiler.trace("/tmp/tensorboard"):
    for i in range(100):
        result = model(batch)

# View in TensorBoard
# tensorboard --logdir=/tmp/tensorboard
```

### Memory Profiling

```python
# Track memory allocation
def memory_usage():
    device = jax.devices()[0]
    stats = device.memory_stats()
    return stats['bytes_in_use'] / 1e9  # GB

print(f"Memory before: {memory_usage():.2f} GB")
result = large_computation()
print(f"Memory after: {memory_usage():.2f} GB")
```

## Best Practices

### Code Patterns

1. **Vectorization**: Use `jnp.vectorize` for element-wise operations
2. **Broadcasting**: Leverage JAX broadcasting for efficiency
3. **Tree Operations**: Use `jax.tree_map` for nested structures
4. **Gradient Checkpointing**: Save memory with `jax.checkpoint`

### Common Pitfalls

- Host-device transfers (minimize)
- Shape mismatches (check dimensions)
- Memory leaks (clear unused arrays)
- Sequential operations (vectorize when possible)
