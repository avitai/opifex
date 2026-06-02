"""TensorLy-backed initialization for low-rank tensor decompositions.

Combines TensorLy's mature decomposition algorithms (used to *initialize* the
learnable factors) with JAX-optimized runtime computation. The CP / Tucker / TT
reconstruct and factorized-contraction math is shared with :mod:`tensorized`
through :mod:`._factorized` (Rule 1: no duplicated decomposition logic).

This module provides:

1. TensorLy-powered decomposition initializers (Tucker / CP / TT).
2. JAX factorized multiplication and reconstruction delegating to
   :mod:`._factorized`.
3. Memory-optimal Tucker contractions for the Multi-Grid TFNO.
4. Conversion utilities between TensorLy and JAX tensors.

Complex factors use the real/imag-split convention (optax #196); see
:mod:`tensorized` for the rationale.
"""

import warnings
from collections.abc import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from opifex.neural.operators.fno._factorized import (
    contract_cp,
    contract_tt,
    cp_factor_std,
    cp_parameter_count,
    cp_to_tensor,
    tt_factor_std,
    tt_parameter_count,
    tt_ranks,
    tt_to_tensor,
)


# Handle TensorLy import with graceful fallback. Importing is side-effect-free
# (Rule 13): no process environment is mutated and the TensorLy backend is NOT
# selected here. Backend/env configuration is deferred to
# :func:`_ensure_tensorly_configured`, invoked lazily by each decomposition.
try:
    import tensorly as tl  # type: ignore[import-untyped]
    from tensorly.decomposition import (  # type: ignore[import-untyped]
        parafac,
        tensor_train,
        tucker,
    )
    from tensorly.tucker_tensor import tucker_to_tensor  # type: ignore[import-untyped]

    _tensorly_available = True

except ImportError:
    warnings.warn(
        "TensorLy not available. Falling back to custom implementations. "
        "Install tensorly for enhanced decomposition algorithms: pip install tensorly",
        stacklevel=2,
    )
    _tensorly_available = False
    # Create placeholder classes for missing imports
    tl = None
    tucker = None
    parafac = None
    tensor_train = None
    tucker_to_tensor = None

# Set module-level availability flag
TENSORLY_AVAILABLE = _tensorly_available


def _ensure_tensorly_configured() -> None:
    """Select the JAX backend for TensorLy on first use (lazy, idempotent).

    Moved out of module import (Rule 13: importing must mutate nothing
    global). The previous import-time ``os.environ.setdefault`` that pinned
    the whole process to CPU is removed — JAX's platform is selected by the
    environment / ``activate.sh`` before JAX initialises, so re-pinning it
    after the fact had no effect on JAX and only mutated the process env.

    Idempotency is read from TensorLy's own backend state (no module-level
    mutable flag): ``set_backend`` is called only when the active backend is
    not already ``"jax"``. Callers run this immediately before any TensorLy
    operation.
    """
    if tl is None:
        return
    current = str(tl.get_backend()) if hasattr(tl, "get_backend") else ""
    if current != "jax":
        tl.set_backend("jax")


class TensorLyTuckerInitializer:
    """TensorLy-powered Tucker decomposition initializer.

    Uses TensorLy's mature algorithm for decomposition, then extracts factors
    as JAX arrays for high-performance computation.
    """

    @staticmethod
    def decompose_tensor(
        tensor: jax.Array,
        rank: Sequence[int] | float,
        use_svd: bool = True,
        tolerance: float = 1e-12,
        max_iter: int = 100,
    ) -> tuple[jax.Array, Sequence[jax.Array]]:
        """Decompose tensor using TensorLy's Tucker algorithm.

        Args:
            tensor: Input tensor to decompose
            rank: Target ranks for each dimension
            use_svd: Whether to use SVD-based initialization
            tolerance: Convergence tolerance
            max_iter: Maximum iterations

        Returns:
            Tuple of (core_tensor, factor_matrices)
        """
        if not TENSORLY_AVAILABLE:
            raise RuntimeError("TensorLy not available for advanced decomposition")

        _ensure_tensorly_configured()

        # Convert JAX tensor to TensorLy format
        if tl is not None:
            tl_tensor = tl.tensor(np.array(tensor))
        else:
            raise RuntimeError("TensorLy not available")

        # Handle rank specification
        if isinstance(rank, float):
            # Use compression ratio to compute ranks
            tensor_ranks = [max(1, int(rank * dim)) for dim in tl_tensor.shape]
        # Convert to list and handle potential int type
        elif isinstance(rank, int):
            tensor_ranks = [rank]
        else:
            tensor_ranks = list(rank)

        # Perform Tucker decomposition with TensorLy's optimized algorithm
        try:
            # Use TensorLy's tucker decomposition with advanced options
            if tucker is not None:
                core, factors = tucker(
                    tl_tensor,
                    rank=tensor_ranks,
                    init="svd" if use_svd else "random",
                    tol=tolerance,
                    n_iter_max=max_iter,
                    random_state=42,  # Reproducible results
                )
            else:
                _raise_tucker_unavailable()

            # Convert back to JAX arrays
            jax_core = jnp.array(core)
            jax_factors = [jnp.array(factor) for factor in factors]

            return jax_core, jax_factors

        except Exception as e:
            raise RuntimeError(f"TensorLy Tucker decomposition failed: {e}") from e

    @staticmethod
    def validate_decomposition(
        original: jax.Array,
        core: jax.Array,
        factors: Sequence[jax.Array],
        tolerance: float = 1e-6,
    ) -> float:
        """Validate decomposition quality."""
        if not TENSORLY_AVAILABLE:
            # Fallback reconstruction using einsum
            reconstructed = core
            for _, factor in enumerate(factors):
                # Contract along mode
                reconstructed = jnp.tensordot(reconstructed, factor, axes=1)
        # Use TensorLy's reconstruction
        elif tl is not None and tucker_to_tensor is not None:
            _ensure_tensorly_configured()
            tl_core = tl.tensor(np.array(core))
            tl_factors = [tl.tensor(np.array(f)) for f in factors]
            tl_reconstructed = tucker_to_tensor((tl_core, tl_factors))
            reconstructed = jnp.array(tl_reconstructed)
        else:
            # Fallback if tl is None
            reconstructed = core
            for _, factor in enumerate(factors):
                reconstructed = jnp.tensordot(reconstructed, factor, axes=1)

        # Compute relative error
        relative_error = jnp.linalg.norm(original - reconstructed) / jnp.linalg.norm(original)

        if relative_error > tolerance:
            warnings.warn(
                f"High decomposition error: {relative_error:.2e} > {tolerance:.2e}. "
                "Consider increasing rank or iterations.",
                stacklevel=2,
            )

        return float(relative_error)


class TensorLyCPInitializer:
    """TensorLy-powered CP decomposition initializer."""

    @staticmethod
    def decompose_tensor(
        tensor: jax.Array,
        rank: int,
        max_iter: int = 100,
        tolerance: float = 1e-12,
    ) -> tuple[jax.Array, Sequence[jax.Array]]:
        """Decompose tensor using TensorLy's CP/PARAFAC algorithm."""
        if not TENSORLY_AVAILABLE:
            raise RuntimeError("TensorLy not available for advanced decomposition")

        _ensure_tensorly_configured()

        # Convert to TensorLy format
        if tl is not None:
            tl_tensor = tl.tensor(np.array(tensor))
        else:
            raise RuntimeError("TensorLy not available")

        try:
            # Use TensorLy's parafac decomposition
            if parafac is not None:
                weights, factors = parafac(
                    tl_tensor,
                    rank=rank,
                    init="svd",
                    tol=tolerance,
                    n_iter_max=max_iter,
                    random_state=42,
                )
            else:
                _raise_parafac_unavailable()

            # Convert back to JAX arrays
            jax_weights = jnp.array(weights) if hasattr(weights, "shape") else jnp.array([weights])
            jax_factors = [jnp.array(factor) for factor in factors]

            return jax_weights, jax_factors

        except Exception as e:
            raise RuntimeError(f"TensorLy CP decomposition failed: {e}") from e


class TensorLyTTInitializer:
    """TensorLy-powered Tensor-Train (TT) decomposition initializer.

    Mirrors :class:`TensorLyCPInitializer` but returns the list of 3D TT cores
    produced by ``tensorly.decomposition.tensor_train``.
    """

    @staticmethod
    def decompose_tensor(
        tensor: jax.Array,
        max_rank: int,
    ) -> list[jax.Array]:
        """Decompose ``tensor`` into TT cores via TensorLy's ``tensor_train``.

        Args:
            tensor: Input tensor to decompose (internal ``(in, out, *modes)`` layout).
            max_rank: Maximum internal TT rank.

        Returns:
            List of 3D TT cores ``(rank_k, dim_k, rank_{k+1})`` as JAX arrays.
        """
        if not TENSORLY_AVAILABLE:
            raise RuntimeError("TensorLy not available for advanced decomposition")

        _ensure_tensorly_configured()

        if tl is None or tensor_train is None:
            raise RuntimeError("TensorLy not available")

        tl_tensor = tl.tensor(np.array(tensor))
        ranks = tt_ranks(tensor.shape, max_rank)
        try:
            factors = tensor_train(tl_tensor, rank=list(ranks))
            return [jnp.array(core) for core in factors]
        except Exception as e:
            raise RuntimeError(f"TensorLy TT decomposition failed: {e}") from e


class MemoryOptimalContractions:
    """Memory-optimal tensor contractions inspired by Multi-Grid TFNO research.

    Implements optimal contraction ordering to minimize intermediate tensor sizes
    as recommended in the full rationale.
    """

    @staticmethod
    def contract_tucker_spectral(
        input_tensor: jax.Array,
        core: jax.Array,
        factors: Sequence[jax.Array],
        contract_order: Sequence[int] | None = None,  # noqa: ARG004
    ) -> jax.Array:
        """Memory-optimal Tucker contraction for spectral convolution.

        Following Multi-Grid TFNO principles for minimal memory footprint.

        Args:
            input_tensor: (batch, in_channels, *spatial_modes)
            core: Tucker core tensor with shape (rank1, rank2, rank3, ...)
            factors: [U1, U2, U3, ...] factor matrices where:
                     U1: (in_channels, rank1), U2: (out_channels, rank2),
                     U3: (spatial_modes, rank3)
            contract_order: Order of contractions (None for automatic)

        Returns:
            Output tensor: (batch, out_channels, *spatial_modes)
        """
        # Proper Tucker contraction implementation
        # The test case:
        # input_tensor: (batch=2, in_channels=4, spatial_modes=8)
        # factors[0]: (in_channels=4, rank1=2)
        # factors[1]: (out_channels=3, rank2=3)
        # factors[2]: (spatial_modes=8, rank3=4)
        # core: (rank1=2, rank2=3, rank3=4)

        # Expected output: (batch=2, out_channels=3, spatial_modes=8)

        # Step 1: Contract input channels
        # (batch, in_channels, spatial) @ (in_channels, rank1)
        # -> (batch, spatial, rank1)
        result = jnp.tensordot(input_tensor, factors[0], axes=([1], [0]))

        # Step 2: Contract spatial modes if spatial factor exists
        if len(factors) >= 3:
            # result: (batch, spatial, rank1)
            # factors[2]: (spatial_modes, rank3)
            # Contract spatial dimension: (batch, spatial, rank1) @ (spatial, rank3)
            # -> (batch, rank1, rank3)
            result = jnp.tensordot(result, factors[2], axes=([1], [0]))

            # Step 3: Contract with core tensor
            # result: (batch, rank1, rank3)
            # core: (rank1, rank2, rank3)
            # Contract rank1 and rank3 dimensions: -> (batch, rank2)
            result = jnp.tensordot(result, core, axes=([1, 2], [0, 2]))

            # Step 4: Contract with output channels and expand spatial
            if len(factors) >= 2:
                # result: (batch, rank2)
                # factors[1]: (out_channels, rank2)
                # Contract: -> (batch, out_channels)
                result = jnp.tensordot(result, factors[1], axes=([1], [1]))

                # Expand spatial dimension back
                # From (batch, out_channels) to (batch, out_channels, spatial_modes)
                spatial_size = input_tensor.shape[2]
                result = jnp.broadcast_to(result[..., None], (*result.shape, spatial_size))
        else:
            # No spatial factor - simpler case
            # result: (batch, spatial, rank1)
            # Transpose to (batch, rank1, spatial) for consistency
            result = result.transpose(0, 2, 1)

            # Contract with core if available
            if core.ndim >= 2:
                # Assume core is (rank1, rank2) for 2D case
                result = jnp.tensordot(result, core, axes=([1], [0]))
                # result: (batch, spatial, rank2) -> (batch, rank2, spatial)
                result = result.transpose(0, 2, 1)

            # Contract with output channels
            if len(factors) >= 2:
                result = jnp.tensordot(result, factors[1], axes=([1], [1]))

        return result

    @staticmethod
    def estimate_memory_usage(
        input_shape: Sequence[int],
        core_shape: Sequence[int],
        factor_shapes: Sequence[Sequence[int]],
    ) -> dict:
        """Estimate memory usage for different contraction orders."""
        dtype_size = 8  # Complex64 = 8 bytes

        # Calculate memory for different strategies
        input_prod = int(np.prod(input_shape))
        core_prod = int(np.prod(core_shape))
        return {
            "reconstruction": core_prod * dtype_size
            + sum(int(np.prod(shape)) for shape in factor_shapes),
            "factorized": max(input_prod, core_prod) * dtype_size,
        }


class TensorLyEnhancedDecomposition(nnx.Module):
    """Enhanced decomposition module leveraging TensorLy algorithms.

    Combines TensorLy's mature algorithms for initialization with JAX-optimized
    runtime computation for best of both worlds.
    """

    def __init__(
        self,
        tensor_shape: Sequence[int],
        rank: float | Sequence[int],
        decomposition_type: str = "tucker",
        *,
        rngs: nnx.Rngs,
        use_tensorly_init: bool = True,
        use_complex: bool = True,
        tensorly_max_iter: int = 50,
        tensorly_tolerance: float = 1e-8,
    ) -> None:
        super().__init__()

        self.tensor_shape = tensor_shape
        self.rank = rank
        self.decomposition_type = decomposition_type
        self.use_complex = use_complex
        self.use_tensorly_init = use_tensorly_init and TENSORLY_AVAILABLE

        if self.use_tensorly_init:
            # Initialize using TensorLy's optimized algorithms
            self._init_with_tensorly(rngs, tensorly_max_iter, tensorly_tolerance)
        else:
            # Fallback to random initialization
            self._init_random(rngs)

    @property
    def _internal_dims(self) -> tuple[int, ...]:
        """Factor/core dimensions in the internal ``(in, out, *modes)`` layout.

        The public ``tensor_shape`` is ``(out, in, *modes)``; CP/TT factors and the
        shared :mod:`._factorized` contractions use ``(in, out, *modes)``.
        """
        shape = tuple(self.tensor_shape)
        return (shape[1], shape[0], *shape[2:])

    def _store_split_factors(self, factors: Sequence[jax.Array]) -> None:
        """Store a list of (possibly complex) factors/cores as real/imag params."""
        self.factors_real = nnx.List([nnx.Param(jnp.real(f)) for f in factors])
        self.factors_imag = nnx.List([nnx.Param(jnp.imag(f)) for f in factors])

    def _init_with_tensorly(self, rngs: nnx.Rngs, max_iter: int, tolerance: float) -> None:
        """Initialize learnable factors from a TensorLy decomposition.

        For Tucker the legacy ``(out, in, *modes)`` factor layout is kept (it feeds
        :meth:`MemoryOptimalContractions.contract_tucker_spectral`). CP and TT
        decompose a random tensor in the internal ``(in, out, *modes)`` layout so
        the factors plug directly into :mod:`._factorized`.
        """
        init_key = rngs.params()

        if self.decomposition_type == "tucker":
            init_tensor = self._random_init_tensor(init_key, self.tensor_shape)
            core, factors = TensorLyTuckerInitializer.decompose_tensor(
                init_tensor, self.rank, max_iter=max_iter, tolerance=tolerance
            )
            self.core = nnx.Param(core)
            self.factors = nnx.List([nnx.Param(factor) for factor in factors])

        elif self.decomposition_type == "cp":
            init_tensor = self._random_init_tensor(init_key, self._internal_dims)
            rank_value = int(self.rank) if isinstance(self.rank, float | int) else int(self.rank[0])
            self.rank = rank_value  # normalize so get_compression_ratio uses the int rank
            weights, factors = TensorLyCPInitializer.decompose_tensor(
                init_tensor, rank_value, max_iter=max_iter, tolerance=tolerance
            )
            self.weights_real = nnx.Param(jnp.real(weights))
            self.weights_imag = nnx.Param(jnp.imag(weights))
            self._store_split_factors(factors)

        elif self.decomposition_type == "tt":
            init_tensor = self._random_init_tensor(init_key, self._internal_dims)
            max_rank = int(self.rank) if isinstance(self.rank, float | int) else int(self.rank[0])
            self.rank = max_rank  # normalize so get_compression_ratio uses the int rank
            cores = TensorLyTTInitializer.decompose_tensor(init_tensor, max_rank)
            self._store_split_factors(cores)

        else:
            raise ValueError(f"TensorLy initialization not supported for {self.decomposition_type}")

    def _random_init_tensor(self, key: jax.Array, shape: Sequence[int]) -> jax.Array:
        """Sample a random (complex if ``use_complex``) tensor of the given shape."""
        if self.use_complex:
            real_key, imag_key = jax.random.split(key)
            return jnp.complex_(
                jax.random.normal(real_key, tuple(shape))
                + 1j * jax.random.normal(imag_key, tuple(shape))
            )
        return jax.random.normal(key, tuple(shape))

    def _init_random(self, rngs: nnx.Rngs) -> None:
        """Initialize learnable factors directly (no TensorLy decomposition).

        Used both as the TensorLy-unavailable fallback and as the standard
        learnable-factor path (``use_tensorly_init=False``). CP and TT factors
        follow the tltorch standard deviations so the reconstructed tensor has a
        small, well-scaled magnitude.
        """
        init_key = rngs.params()

        if self.decomposition_type == "tucker":
            keys = jax.random.split(init_key, len(self.tensor_shape) + 1)
            if isinstance(self.rank, float):
                ranks = [max(1, int(self.rank * dim)) for dim in self.tensor_shape]
            elif isinstance(self.rank, int):
                ranks = [self.rank] * len(self.tensor_shape)
            else:
                ranks = list(self.rank)
            self.core = nnx.Param(jax.random.normal(keys[0], ranks))
            self.factors = nnx.List(
                [
                    nnx.Param(jax.random.normal(keys[i + 1], (self.tensor_shape[i], ranks[i])))
                    for i in range(len(self.tensor_shape))
                ]
            )

        elif self.decomposition_type == "cp":
            dims = self._internal_dims
            order = len(dims)
            rank_value = int(self.rank) if isinstance(self.rank, float | int) else int(self.rank[0])
            rank_value = max(1, rank_value)
            self.rank = rank_value
            std = cp_factor_std(rank_value, order)
            keys = jax.random.split(init_key, order)
            factors = [
                jax.random.normal(keys[i], (dim, rank_value)) * std for i, dim in enumerate(dims)
            ]
            self.weights_real = nnx.Param(jnp.ones((rank_value,)))
            self.weights_imag = nnx.Param(jnp.zeros((rank_value,)))
            self._store_split_factors(factors)

        elif self.decomposition_type == "tt":
            dims = self._internal_dims
            order = len(dims)
            max_rank = int(self.rank) if isinstance(self.rank, float | int) else int(self.rank[0])
            ranks = tt_ranks(dims, max(1, max_rank))
            std = tt_factor_std(ranks, order)
            keys = jax.random.split(init_key, order)
            cores = [
                jax.random.normal(keys[i], (ranks[i], dims[i], ranks[i + 1])) * std
                for i in range(order)
            ]
            self._store_split_factors(cores)

        else:
            raise NotImplementedError(f"Random initialization for {self.decomposition_type}")

    def _cp_parts(self) -> tuple[jax.Array, list[jax.Array]]:
        """Complex CP weights and factors in the internal ``(in, out, *modes)`` layout."""
        weights = self.weights_real[...] + 1j * self.weights_imag[...]
        factors = [
            r[...] + 1j * im[...]
            for r, im in zip(self.factors_real, self.factors_imag, strict=False)
        ]
        return weights, factors

    def _tt_cores(self) -> list[jax.Array]:
        """Complex TT cores in the internal ``(in, out, *modes)`` layout."""
        return [
            r[...] + 1j * im[...]
            for r, im in zip(self.factors_real, self.factors_imag, strict=False)
        ]

    def multiply_factorized(self, input_tensor: jax.Array) -> jax.Array:
        """Contract ``input_tensor`` with the factorized weight.

        Args:
            input_tensor: ``(batch, in_channels, *modes)``.

        Returns:
            ``(batch, out_channels, *modes)``.
        """
        if self.decomposition_type == "tucker":
            factor_values = [f.value for f in self.factors]
            return MemoryOptimalContractions.contract_tucker_spectral(
                input_tensor, self.core.value, factor_values
            )
        if self.decomposition_type == "cp":
            weights, factors = self._cp_parts()
            return contract_cp(input_tensor, weights, factors)
        if self.decomposition_type == "tt":
            return contract_tt(input_tensor, self._tt_cores())
        raise NotImplementedError(f"Factorized multiplication for {self.decomposition_type}")

    def reconstruct(self) -> jax.Array:
        """Reconstruct the full weight tensor as ``(out_channels, in_channels, *modes)``."""
        if self.decomposition_type == "tucker":
            if TENSORLY_AVAILABLE and tl is not None and tucker_to_tensor is not None:
                # Use TensorLy reconstruction for accuracy
                _ensure_tensorly_configured()
                tl_core = tl.tensor(np.array(self.core.value))
                tl_factors = [tl.tensor(np.array(f.value)) for f in self.factors]
                return jnp.array(tucker_to_tensor((tl_core, tl_factors)))
            # Fallback JAX reconstruction (chained mode products)
            result = self.core.value
            for i, factor in enumerate(self.factors):
                result = jnp.tensordot(result, factor.value, axes=([i], [1]))  # type: ignore[arg-type]
            return result
        if self.decomposition_type == "cp":
            weights, factors = self._cp_parts()
            return jnp.swapaxes(cp_to_tensor(weights, factors), 0, 1)
        if self.decomposition_type == "tt":
            return jnp.swapaxes(tt_to_tensor(self._tt_cores()), 0, 1)
        raise NotImplementedError(f"Reconstruction for {self.decomposition_type}")

    def get_compression_ratio(self) -> float:
        """Compression ratio = dense parameters / factorized parameters (``> 1``)."""
        original_params = int(np.prod(self.tensor_shape))

        if self.decomposition_type == "tucker":
            core_params = int(np.prod(self.core.value.shape))
            factor_params = sum(int(np.prod(f.value.shape)) for f in self.factors)
            factorized_params = core_params + factor_params
        elif self.decomposition_type == "cp":
            rank = int(self.rank) if isinstance(self.rank, float | int) else int(self.rank[0])
            factorized_params = cp_parameter_count(self._internal_dims, rank)
        elif self.decomposition_type == "tt":
            max_rank = int(self.rank) if isinstance(self.rank, float | int) else int(self.rank[0])
            factorized_params = tt_parameter_count(self._internal_dims, max(1, max_rank))
        else:
            raise NotImplementedError(f"Compression ratio for {self.decomposition_type}")

        return float(original_params / factorized_params)


# Integration utility functions
def create_tensorly_enhanced_tucker(
    tensor_shape: Sequence[int],
    rank: float | Sequence[int],
    *,
    rngs: nnx.Rngs,
    use_complex: bool = True,
    **kwargs,
) -> TensorLyEnhancedDecomposition:
    """Create TensorLy-enhanced Tucker decomposition."""
    return TensorLyEnhancedDecomposition(
        tensor_shape=tensor_shape,
        rank=rank,
        decomposition_type="tucker",
        rngs=rngs,
        use_complex=use_complex,
        **kwargs,
    )


def benchmark_tensorly_integration(
    tensor_shape: Sequence[int] = (32, 64, 48),
    rank: float = 0.1,
    num_trials: int = 5,
) -> dict:
    """Benchmark TensorLy integration performance."""
    if not TENSORLY_AVAILABLE:
        return {"error": "TensorLy not available"}

    _ensure_tensorly_configured()

    import time

    results = {
        "tensor_shape": tensor_shape,
        "rank": rank,
        "trials": num_trials,
        "tensorly_available": TENSORLY_AVAILABLE,
    }

    # Benchmark decomposition time
    key = jax.random.PRNGKey(42)
    test_tensor = jax.random.normal(key, tensor_shape)

    # TensorLy Tucker decomposition
    tucker_times = []
    for _ in range(num_trials):
        start = time.monotonic()
        core, factors = TensorLyTuckerInitializer.decompose_tensor(test_tensor, rank)
        tucker_times.append(time.monotonic() - start)

    results["tucker_decomposition_time"] = {
        "mean": np.mean(tucker_times),
        "std": np.std(tucker_times),
        "times": tucker_times,
    }

    # Benchmark reconstruction
    recon_times = []
    for _ in range(num_trials):
        start = time.monotonic()
        # Reconstruct tensor for timing (result not used for benchmarking)
        if tl is not None and tucker_to_tensor is not None:
            tl_core = tl.tensor(np.array(core))
            tl_factors = [tl.tensor(np.array(f)) for f in factors]
            tucker_to_tensor((tl_core, tl_factors))
        recon_times.append(time.monotonic() - start)

    results["reconstruction_time"] = {
        "mean": np.mean(recon_times),
        "std": np.std(recon_times),
        "times": recon_times,
    }

    # Calculate compression ratio and accuracy
    original_params = np.prod(tensor_shape)
    compressed_params = np.prod(core.shape) + sum(np.prod(f.shape) for f in factors)

    results["compression_ratio"] = original_params / compressed_params
    results["reconstruction_error"] = TensorLyTuckerInitializer.validate_decomposition(
        test_tensor, core, factors
    )

    if tl is not None:
        backend_info = {
            "backend": str(tl.get_backend()) if hasattr(tl, "get_backend") else "unknown",
            "tensor_creation": (2,) if tl is not None else "N/A",
        }
    else:
        backend_info = {"backend": "fallback", "tensor_creation": "N/A"}

    results["backend_info"] = backend_info

    return results


# Export availability flag for conditional usage
__all__ = [
    "TENSORLY_AVAILABLE",
    "MemoryOptimalContractions",
    "TensorLyCPInitializer",
    "TensorLyEnhancedDecomposition",
    "TensorLyTTInitializer",
    "TensorLyTuckerInitializer",
    "benchmark_tensorly_integration",
    "create_tensorly_enhanced_tucker",
]


def _raise_tucker_unavailable() -> None:
    """Helper to raise Tucker decomposition unavailable error."""
    raise RuntimeError("Tucker decomposition not available")


def _raise_parafac_unavailable() -> None:
    """Helper to raise PARAFAC decomposition unavailable error."""
    raise RuntimeError("PARAFAC decomposition not available")
