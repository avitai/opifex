"""
Quantum Operator Primitives

This module provides quantum operator building blocks for quantum mechanical
calculations, following FLAX NNX patterns and JAX compatibility requirements.
Core operators include Hamiltonians, density matrices, observables, and
efficient sparse matrix operations.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Array

from opifex.core.quantum.molecular_system import MolecularSystem


# pyright: reportArgumentType=false, reportOperatorIssue=false, reportReturnType=false, reportIncompatibleMethodOverride=false


class QuantumOperator(ABC):
    """
    Abstract base class for quantum operators.

    All quantum operators must implement the basic operator interface with
    JAX transformations support and adjoint operations.
    """

    def __init__(self, name: str = "operator"):
        """
        Initialize quantum operator.

        Args:
            name: Human-readable name for the operator
        """
        self.name = name

    @abstractmethod
    def __call__(self, state: Array, *args, **kwargs) -> Array:
        """
        Apply operator to quantum state.

        Args:
            state: Input quantum state
            *args: Additional arguments for operator application
            **kwargs: Additional keyword arguments

        Returns:
            Transformed quantum state
        """

    @abstractmethod
    def adjoint(self) -> "QuantumOperator":
        """
        Return adjoint (Hermitian conjugate) of operator.

        Returns:
            Adjoint operator
        """

    @property
    @abstractmethod
    def is_hermitian(self) -> bool:
        """Whether operator is Hermitian (self-adjoint)."""

    def expectation_value(self, state: Array, *args, **kwargs) -> jnp.complexfloating:
        """
        Compute expectation value <state|operator|state>.

        Args:
            state: Quantum state
            *args: Additional arguments for operator application
            **kwargs: Additional keyword arguments

        Returns:
            Expectation value
        """
        op_state = self(state, *args, **kwargs)
        result = jnp.vdot(state, op_state)
        return jnp.complex64(result)  # Ensure proper complex type


class HamiltonianOperator(QuantumOperator):
    """
    Hamiltonian operator for quantum systems.

    Implements kinetic + potential energy operators with various numerical
    methods for quantum mechanical calculations.
    """

    def __init__(
        self,
        molecular_system: MolecularSystem,
        kinetic_method: str = "finite_difference",
        potential_method: str = "coulomb",
        name: str = "hamiltonian",
    ):
        """
        Initialize Hamiltonian operator.

        Args:
            molecular_system: Molecular system definition
            kinetic_method: Method for kinetic energy ("finite_difference", "fft")
            potential_method: Method for potential energy ("coulomb", "neural")
            name: Operator name
        """
        super().__init__(name)
        self.molecular_system = molecular_system
        self.kinetic_method = kinetic_method
        self.potential_method = potential_method

    def __call__(self, state: Array, *args, **kwargs) -> Array:
        """
        Apply Hamiltonian to wavefunction: H|ψ⟩ = (T + V)|ψ⟩.

        Args:
            state: Input wavefunction
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            H|ψ⟩
        """
        # Kinetic energy contribution
        kinetic_part = self._apply_kinetic(state)

        # Potential energy contribution
        potential_part = self._apply_potential(state)

        return kinetic_part + potential_part

    def adjoint(self) -> "HamiltonianOperator":
        """Hamiltonian is Hermitian, so adjoint is itself."""
        return self

    @property
    def is_hermitian(self) -> bool:
        """Hamiltonian is always Hermitian."""
        return True

    def compute_energy(self, wavefunction: Array) -> Array:
        """
        Compute energy expectation value E = ⟨ψ|H|ψ⟩.

        Args:
            wavefunction: Normalized wavefunction

        Returns:
            Energy expectation value
        """
        h_psi = self(wavefunction)
        return jnp.real(jnp.vdot(wavefunction, h_psi))

    def _apply_kinetic(self, wavefunction: Array) -> Array:
        """Apply kinetic energy operator."""
        if self.kinetic_method == "finite_difference":
            # Simple finite difference approximation: -∇²/2
            # For 1D case: T = -d²/dx² / 2
            # Ensure wavefunction is Array type
            wf_array = jnp.asarray(wavefunction)
            grad1 = jnp.gradient(wf_array)
            if isinstance(grad1, list):
                grad1 = grad1[0]  # Take first component for 1D case
            grad2 = jnp.gradient(grad1)
            if isinstance(grad2, list):
                grad2 = grad2[0]  # Take first component for 1D case
            return -0.5 * grad2
        if self.kinetic_method == "spectral":
            # Spectral method using second derivative in frequency domain
            # T = -ℏ²∇²/(2m) with ℏ=m=1 for simplicity
            wf_array = jnp.asarray(wavefunction)
            n = len(wf_array)

            # Assume unit spacing for simplicity
            dx = 1.0

            # FFT of the wavefunction  handling
            fft_psi = jnp.fft.fft(wf_array)

            # Wave numbers for second derivative
            k = jnp.fft.fftfreq(n, dx) * 2 * jnp.pi

            # Second derivative in frequency domain: multiply by -k²
            # JAX-native precision handling - no explicit type casting
            fft_second_deriv = -(k**2) * fft_psi

            # Inverse FFT to get second derivative
            # JAX-native precision handling - no explicit type casting
            second_deriv = jnp.fft.ifft(fft_second_deriv)

            # Apply kinetic energy factor: T = -ℏ²∇²/(2m)
            kinetic_result = -0.5 * second_deriv

            # Return real part if input was real  handling
            if jnp.isrealobj(wf_array):
                return jnp.real(kinetic_result)
            return kinetic_result
        raise NotImplementedError(
            f"Kinetic method {self.kinetic_method} not implemented"
        )

    def _apply_potential(self, wavefunction: Array) -> Array:
        """Apply potential energy operator."""
        if self.potential_method == "coulomb":
            # For hydrogen-like atoms: V = -Z/r
            # Simplified 1D harmonic potential for testing
            x = jnp.linspace(-5, 5, len(wavefunction))
            potential = -1.0 / (jnp.abs(x) + 0.1)  # Avoid singularity
            return potential * wavefunction
        if self.potential_method == "harmonic":
            # Harmonic oscillator potential: V = (1/2) * k * x²
            # Using k = 1 for simplicity
            x = jnp.linspace(-5, 5, len(wavefunction))
            potential = 0.5 * x**2
            return potential * wavefunction
        raise NotImplementedError(
            f"Potential method {self.potential_method} not implemented"
        )


class DensityMatrix:
    """
    Density matrix representation for mixed quantum states.

    Provides density matrix operations including trace, expectation values,
    and validation of quantum mechanical constraints.
    """

    def __init__(self, matrix: Array):
        """
        Initialize density matrix.

        Args:
            matrix: Density matrix representation [Shape: (N, N)]
        """
        self.matrix = matrix

    def is_valid(self) -> bool:
        """
        Check if density matrix satisfies quantum mechanical constraints.

        Returns:
            True if valid density matrix
        """
        # Check trace = 1 using JAX-compatible operations
        trace_val = float(jnp.real(self.trace()))
        if not jnp.isclose(trace_val, 1.0, atol=1e-6):
            return False

        # Check positive semidefinite
        if not self.is_positive_semidefinite():
            return False

        # Check Hermitian using JAX-compatible operations
        hermitian_check = jnp.allclose(self.matrix, jnp.conj(self.matrix.T), atol=1e-8)
        return bool(hermitian_check)

    def trace(self) -> jnp.complexfloating:
        """Compute trace of density matrix."""
        result = jnp.trace(self.matrix)
        return jnp.complex64(result)

    def is_positive_semidefinite(self) -> bool:
        """Check if density matrix is positive semidefinite."""
        eigenvalues = jnp.linalg.eigvals(self.matrix)
        return bool(jnp.all(eigenvalues >= -1e-8))

    def expectation_value(self, observable: Array) -> jnp.complexfloating:
        """
        Compute expectation value of observable.

        Args:
            observable: Observable operator matrix

        Returns:
            Expectation value ⟨O⟩ = Tr(ρO)
        """
        result = jnp.trace(self.matrix @ observable)
        return jnp.complex64(result)


class Observable(QuantumOperator):
    """
    General observable operator for quantum measurements.

    Wraps arbitrary operator functions with observable interface and
    expectation value computation.
    """

    def __init__(
        self,
        operator_func: Callable,
        name: str = "observable",
        is_hermitian: bool = True,
    ):
        """
        Initialize observable operator.

        Args:
            operator_func: Function that applies operator to states
            name: Observable name
            is_hermitian: Whether operator is Hermitian
        """
        super().__init__(name)
        self.operator_func = operator_func
        self._is_hermitian = is_hermitian

    def __call__(self, state: Array, *args, **kwargs) -> Array:
        """Apply observable to quantum state."""
        return self.operator_func(state, *args, **kwargs)

    def adjoint(self) -> "Observable":
        """Return adjoint observable (simplified for testing)."""
        return self

    @property
    def is_hermitian(self) -> bool:
        """Return Hermitian property."""
        return self._is_hermitian

    def expectation_value(self, state: Array, *args, **kwargs) -> jnp.complexfloating:
        """
        Compute expectation value ⟨ψ|O|ψ⟩ with proper discrete integration.

        Args:
            state: Quantum state
            *args: Additional arguments for operator application
            **kwargs: Additional keyword arguments

        Returns:
            Expectation value
        """
        op_state = self.operator_func(state, *args, **kwargs)

        # For discrete integration, we need to include the differential element
        # If the first argument is a grid (array), compute dx from it
        if args and len(args) > 0 and isinstance(args[0], jax.Array):
            grid = args[0]
            if len(grid) > 1:
                # Calculate grid spacing for uniform grids
                dx = grid[1] - grid[0]
                # Discrete integration: ∫ ψ*(x) O ψ(x) dx ≈ Σ ψ*[i] O ψ[i] dx
                result = jnp.vdot(state, op_state) * dx
            else:
                result = jnp.vdot(state, op_state)
        else:
            # No grid provided, assume unit spacing or continuous case
            result = jnp.vdot(state, op_state)

        return jnp.complex64(result)


class MomentumOperator(QuantumOperator):
    """
    Momentum operator implementation.

    Provides momentum operator with multiple numerical methods including
    finite difference and spectral (FFT) approaches for quantum mechanical calculations.
    """

    def __init__(
        self,
        method: str = "finite_difference",
        order: int = 2,
        hbar: float = 1.0,
        name: str = "momentum",
    ):
        """
        Initialize momentum operator.

        Args:
            method: Numerical method ("finite_difference", "spectral")
            order: Order of finite difference approximation
            hbar: Reduced Planck constant
            name: Operator name
        """
        super().__init__(name)
        self.method = method
        self.order = order
        self.hbar = hbar

    def __call__(self, state: Array, *args, **kwargs) -> Array:
        """
        Apply momentum operator: p = -iℏ d/dx.

        Args:
            state: Input wavefunction
            *args: Additional arguments (dx: grid spacing)
            **kwargs: Additional keyword arguments

        Returns:
            p|ψ⟩
        """
        # Extract dx from args, default to 1.0
        dx = args[0] if args else kwargs.get("dx", 1.0)

        if self.method == "finite_difference":
            # p = -iℏ ∇ with improved boundary handling
            gradient = self._finite_difference_gradient(state, dx)
            return -1j * self.hbar * gradient
        if self.method == "spectral":
            # Spectral method using FFT for periodic boundaries
            gradient = self._spectral_gradient(state, dx)
            return -1j * self.hbar * gradient
        raise NotImplementedError(f"Method {self.method} not implemented")

    def _finite_difference_gradient(self, wavefunction: Array, dx: float) -> Array:
        """
        Compute gradient using finite differences with boundary handling.

        Args:
            wavefunction: Input wavefunction
            dx: Grid spacing

        Returns:
            Gradient array
        """
        # Ensure wavefunction is Array type for gradient computation
        wf_array = jnp.asarray(wavefunction)
        # Use central differences for interior points
        gradient = jnp.gradient(wf_array, dx)

        # Handle jnp.gradient return type (can be Array or list[Array])
        if isinstance(gradient, list):
            gradient = gradient[0]  # Take first component for 1D case

        # For better accuracy, use higher-order stencils for interior points
        if len(wf_array) >= 5:
            # 4th order central difference for interior points using vectorized operations
            n = len(wf_array)
            grad_improved = jnp.zeros_like(gradient)

            # Vectorized 4th order central difference for interior points
            # Create index arrays for vectorized computation
            interior_indices = jnp.arange(2, n - 2)

            # Vectorized computation using JAX array indexing
            interior_grad = (
                -wf_array[interior_indices + 2]
                + 8 * wf_array[interior_indices + 1]
                - 8 * wf_array[interior_indices - 1]
                + wf_array[interior_indices - 2]
            ) / (12 * dx)

            # Update interior points vectorized
            grad_improved = grad_improved.at[interior_indices].set(interior_grad)

            # Use standard gradient for boundary points
            grad_improved = grad_improved.at[:2].set(gradient[:2])
            return grad_improved.at[-2:].set(gradient[-2:])

        return gradient

    def _spectral_gradient(self, wavefunction: Array, dx: float) -> Array:
        """
        Compute gradient using spectral (FFT) method with JAX-native precision.

        Uses JAX-native precision handling instead of explicit casting.
        """
        try:
            # Ensure input is JAX array  handling
            wf_array = jnp.asarray(wavefunction)
            n = len(wf_array)

            # For very small arrays, use finite difference instead
            if n < 4:
                return self._finite_difference_gradient(wf_array, dx)

            # JAX-native precision handling - no explicit type casting
            # Convert to complex if needed, let JAX X64 mode handle precision
            if jnp.isrealobj(wf_array):
                wf_complex = jnp.asarray(wf_array) + 0j
                was_real = True
            else:
                wf_complex = jnp.asarray(wf_array)
                was_real = False

            # FFT of the wavefunction  handling
            fft_psi = jnp.fft.fft(wf_complex)

            # Wave numbers  handling
            k = jnp.fft.fftfreq(n, dx) * 2 * jnp.pi

            # Multiply by ik in frequency domain  handling
            fft_grad = 1j * k * fft_psi

            # Inverse FFT to get gradient  handling
            gradient = jnp.fft.ifft(fft_grad)

            # Return real part if input was real  handling
            if was_real:
                return jnp.real(gradient)
            return gradient

        except Exception as e:
            # If all spectral methods fail, fallback to finite difference
            import warnings

            warnings.warn(
                f"Spectral gradient failed ({type(e).__name__}: {e}), "
                f"falling back to finite difference method",
                RuntimeWarning,
            )
            return self._finite_difference_gradient(wavefunction, dx)

    def adjoint(self) -> "MomentumOperator":
        """Momentum operator is Hermitian."""
        return self

    @property
    def is_hermitian(self) -> bool:
        """Momentum operator is anti-Hermitian."""
        return False

    def expectation_value(self, state: Array, dx: float = 1.0) -> jnp.complexfloating:
        """
        Compute momentum expectation value with proper discrete integration.

        Args:
            state: Quantum state
            dx: Grid spacing

        Returns:
            Momentum expectation value
        """
        momentum_state = self(state, dx)
        # Include dx factor for discrete integration:
        # ∫ ψ*(x) p ψ(x) dx ≈ Σ ψ*[i] p ψ[i] dx
        result = jnp.vdot(state, momentum_state) * dx
        return jnp.complex64(result)


class KineticEnergyOperator(QuantumOperator):
    """
    Kinetic energy operator T = p²/(2m).

    Implements kinetic energy with finite difference methods for
    quantum mechanical calculations.
    """

    def __init__(
        self,
        mass: float = 1.0,
        hbar: float = 1.0,
        method: str = "finite_difference",
        name: str = "kinetic_energy",
    ):
        """
        Initialize kinetic energy operator.

        Args:
            mass: Particle mass
            hbar: Reduced Planck constant
            method: Numerical method
            name: Operator name
        """
        super().__init__(name)
        self.mass = mass
        self.hbar = hbar
        self.method = method

    def __call__(self, state: Array, *args, **kwargs) -> Array:
        """
        Apply kinetic energy operator: T|ψ⟩ = -ℏ²∇²|ψ⟩/(2m).

        Args:
            state: Input wavefunction
            *args: Additional arguments (dx: grid spacing)
            **kwargs: Additional keyword arguments

        Returns:
            T|ψ⟩
        """
        # Extract dx from args, default to 1.0
        dx = args[0] if args else kwargs.get("dx", 1.0)

        # Ensure wavefunction is Array type
        wf_array = jnp.asarray(state)

        if self.method == "finite_difference":
            # Second derivative using finite differences: -ℏ²∇²/(2m)
            grad1 = jnp.gradient(wf_array, dx)
            if isinstance(grad1, list):
                grad1 = grad1[0]  # Take first component for 1D case
            grad2 = jnp.gradient(grad1, dx)
            if isinstance(grad2, list):
                grad2 = grad2[0]  # Take first component for 1D case
            return -(self.hbar**2) / (2 * self.mass) * grad2
        if self.method == "spectral":
            # Spectral method using FFT for second derivative with JAX-native precision
            n = len(wf_array)

            # JAX-native precision handling - no explicit type casting
            # Convert to complex if needed, let JAX X64 mode handle precision
            if jnp.isrealobj(wf_array):
                wf_complex = jnp.asarray(wf_array) + 0j
                was_real = True
            else:
                wf_complex = jnp.asarray(wf_array)
                was_real = False

            # FFT of the wavefunction  handling
            fft_psi = jnp.fft.fft(wf_complex)

            # Wave numbers for second derivative  handling
            k = jnp.fft.fftfreq(n, dx) * 2 * jnp.pi

            # Second derivative in frequency domain: multiply by -k²
            # JAX-native precision handling - no explicit type casting
            fft_second_deriv = -(k**2) * fft_psi

            # Inverse FFT to get second derivative  handling
            second_deriv = jnp.fft.ifft(fft_second_deriv)

            # Apply kinetic energy factor: T = -ℏ²∇²/(2m)
            kinetic_result = -(self.hbar**2) / (2 * self.mass) * second_deriv

            # Return real part if input was real  handling
            if was_real:
                return jnp.real(kinetic_result)
            return kinetic_result
        raise NotImplementedError(f"Method {self.method} not implemented")

    def adjoint(self) -> "KineticEnergyOperator":
        """Kinetic energy operator is Hermitian."""
        return self

    @property
    def is_hermitian(self) -> bool:
        """Kinetic energy operator is Hermitian."""
        return True

    def expectation_value(self, state: Array, dx: float = 1.0) -> jnp.complexfloating:
        """
        Compute kinetic energy expectation value with proper discrete integration.

        Args:
            state: Quantum state
            dx: Grid spacing

        Returns:
            Kinetic energy expectation value
        """
        kinetic_state = self(state, dx)
        # Include dx factor for discrete integration:
        # ∫ ψ*(x) T ψ(x) dx ≈ Σ ψ*[i] T ψ[i] dx
        result = jnp.real(jnp.vdot(state, kinetic_state)) * dx
        return jnp.complex_(result)


class PotentialEnergyOperator(QuantumOperator):
    """
    Potential energy operator V(x).

    Multiplication operator for potential energy with arbitrary
    potential functions.
    """

    def __init__(
        self, potential_func: Callable[[Array], Array], name: str = "potential_energy"
    ):
        """
        Initialize potential energy operator.

        Args:
            potential_func: Function that computes V(x)
            name: Operator name
        """
        super().__init__(name)
        self.potential_func = potential_func

    def __call__(self, state: Array, *args, **kwargs) -> Array:
        """
        Apply potential energy operator: V|ψ⟩ = V(x)|ψ⟩.

        Args:
            state: Input wavefunction
            *args: Additional arguments (x_grid: position grid)
            **kwargs: Additional keyword arguments

        Returns:
            V|ψ⟩
        """
        # Extract x_grid from args or kwargs
        if args:
            x_grid = args[0]
        elif "x_grid" in kwargs:
            x_grid = kwargs["x_grid"]
        else:
            raise ValueError("x_grid must be provided as argument or keyword argument")

        potential_values = self.potential_func(x_grid)
        return potential_values * state

    def adjoint(self) -> "PotentialEnergyOperator":
        """Potential energy operator is Hermitian for real potentials."""
        return self

    @property
    def is_hermitian(self) -> bool:
        """Potential energy operator is Hermitian for real potentials."""
        return True

    def expectation_value(self, state: Array, x_grid: Array) -> jnp.complexfloating:
        """
        Compute potential energy expectation value with proper discrete integration.

        Args:
            state: Quantum state
            x_grid: Position grid

        Returns:
            Potential energy expectation value
        """
        v_state = self(state, x_grid)
        # Include dx factor for discrete integration:
        # ∫ ψ*(x) V ψ(x) dx ≈ Σ ψ*[i] V ψ[i] dx
        if len(x_grid) > 1:
            dx = x_grid[1] - x_grid[0]
            result = jnp.real(jnp.vdot(state, v_state)) * dx
        else:
            result = jnp.real(jnp.vdot(state, v_state))
        return jnp.complex_(result)


class SparseOperator(QuantumOperator):
    """
    Sparse matrix operator for efficient large-scale calculations.

    Provides sparse matrix-vector multiplication with JAX compatibility
    for quantum operators with sparse structure.
    """

    def __init__(
        self,
        indices: tuple[Array, Array],
        values: Array,
        shape: tuple[int, int],
        name: str = "sparse_operator",
    ):
        """
        Initialize sparse operator.

        Args:
            indices: (row_indices, col_indices) for non-zero elements
            values: Values of non-zero elements
            shape: Matrix shape (rows, cols)
            name: Operator name
        """
        super().__init__(name)
        self.indices = indices
        self.values = values
        self.shape = shape
        self.nnz = len(values)
        self.density = self.nnz / (shape[0] * shape[1])

    def __call__(self, state: Array, *args, **kwargs) -> Array:
        """
        Apply sparse operator to vector.

        Args:
            state: Input vector
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Result of sparse matrix-vector multiplication
        """
        # Sparse matrix-vector multiplication using scatter-add
        row_indices, col_indices = self.indices

        # Gather values from input vector at column indices
        gathered_values = state[col_indices]

        # Multiply with sparse matrix values
        scattered_values = self.values * gathered_values

        # Initialize result with proper dtype to handle complex values
        # Use the dtype of scattered_values to ensure complex support
        result = jnp.zeros(self.shape[0], dtype=scattered_values.dtype)

        # Use scatter-add to accumulate values
        return result.at[row_indices].add(scattered_values)

    def adjoint(self) -> "SparseOperator":
        """Return adjoint sparse operator."""
        row_indices, col_indices = self.indices
        # Transpose by swapping indices and conjugating values
        adj_indices = (col_indices, row_indices)
        adj_values = jnp.conj(self.values)
        adj_shape = (self.shape[1], self.shape[0])

        return SparseOperator(adj_indices, adj_values, adj_shape, f"{self.name}_adj")

    @property
    def is_hermitian(self) -> bool:
        """Simplified check - assume non-Hermitian unless proven otherwise."""
        return False


class OperatorComposition(QuantumOperator):
    """
    Composition of multiple quantum operators.

    Implements operator composition (A ∘ B)(x) = A(B(x)) with
    proper adjoint and Hermitian property handling.
    """

    def __init__(self, operators: list[QuantumOperator], name: str = "composition"):
        """
        Initialize operator composition.

        Args:
            operators: List of operators to compose (applied left to right)
            name: Composition name
        """
        super().__init__(name)
        self.operators = operators

    def __call__(self, state: Array, *args, **kwargs) -> Array:
        """
        Apply composition of operators.

        Args:
            state: Input state
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Composed operator applied to state
        """
        result = state
        for operator in self.operators:
            result = operator(result, *args, **kwargs)
        return result

    def adjoint(self) -> "OperatorComposition":
        """
        Return adjoint of composition: (A ∘ B)† = B† ∘ A†.

        Returns:
            Adjoint composition
        """
        adj_operators = [op.adjoint() for op in reversed(self.operators)]
        return OperatorComposition(adj_operators, f"{self.name}_adj")

    @property
    def is_hermitian(self) -> bool:
        """
        Check if composition is Hermitian.

        Returns:
            True if all operators are Hermitian and composition is self-adjoint
        """
        # Simplified: only true if single Hermitian operator
        return len(self.operators) == 1 and self.operators[0].is_hermitian
