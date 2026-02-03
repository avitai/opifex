"""
Boundary and Initial Conditions for Opifex Framework.

This module provides comprehensive boundary condition and initial condition
specifications for PDEs, ODEs, and quantum mechanical problems, with support
for time-dependent conditions, symbolic constraints, and quantum-specific
boundary conditions following physics principles.

Key Features:
- Classical boundary conditions (Dirichlet, Neumann, Robin)
- Time-dependent boundary conditions
- Initial conditions for time-dependent problems
- Quantum mechanical boundary conditions (wavefunction normalization)
- Electronic density constraints and conservation laws
- Molecular symmetry constraint handling
- Symbolic constraint expressions
- Integration with problem definitions

Boundary Condition Application:
- OOP boundary conditions can apply themselves using `.apply()` methods
- Internally uses optimized functional implementations from
  `opifex.core.physics.boundaries`
- Supports boundary-specific application (left, right, or all)
- Collections can apply multiple BCs with `.apply_all()`
- Full support for time-dependent and spatially-varying conditions

Neural DFT Integration:
- Quantum mechanical boundary conditions for electronic structure
- Electronic density constraints and particle number conservation
- Molecular symmetry preservation for quantum calculations
- Physics constraint enforcement in neural DFT training

Examples:
    >>> # Create and apply a Dirichlet boundary condition
    >>> bc = DirichletBC(boundary="left", value=0.0)
    >>> params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> constrained = bc.apply(params)  # Only left boundary modified

    >>> # Apply multiple boundary conditions
    >>> bcs = BoundaryConditionCollection([
    ...     DirichletBC(boundary="left", value=0.0),
    ...     DirichletBC(boundary="right", value=10.0),
    ... ])
    >>> constrained = bcs.apply_all(params)  # Both boundaries applied
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

import jax
import jax.numpy as jnp

from opifex.core.physics.boundaries import (
    apply_dirichlet,
    apply_neumann,
    apply_robin,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


__all__ = [
    # Base classes
    "BoundaryCondition",
    # Collections
    "BoundaryConditionCollection",
    "Constraint",
    "DensityConstraint",
    # Classical boundary conditions
    "DirichletBC",
    "InitialCondition",
    "NeumannBC",
    "PhysicsConstraint",
    "QuantumConstraint",
    # Initial conditions
    "QuantumInitialCondition",
    "RobinBC",
    # Symbolic constraints
    "SymbolicConstraint",
    "SymmetryConstraint",
    # Quantum boundary conditions
    "WavefunctionBC",
]


# ============================================================================
# Base Classes
# ============================================================================


class BoundaryCondition(ABC):
    """Abstract base class for boundary conditions."""

    def __init__(
        self,
        boundary: str,
        time_dependent: bool = False,
        spatial_dependent: bool = True,
    ):
        """Initialize boundary condition.

        Args:
            boundary: Boundary identifier (e.g., "left", "right", "top", "bottom")
            time_dependent: Whether condition varies with time
            spatial_dependent: Whether condition varies with spatial position
        """
        # Validate boundary identifier
        valid_boundaries = {
            "left",
            "right",
            "top",
            "bottom",
            "front",
            "back",
            "inlet",
            "outlet",
            "wall",
            "symmetry",
            "infinity",
            "all",
        }
        if boundary not in valid_boundaries:
            raise ValueError(
                f"Invalid boundary: {boundary}. Must be one of {valid_boundaries}"
            )

        self.boundary = boundary
        self.time_dependent = time_dependent
        self.spatial_dependent = spatial_dependent

    @abstractmethod
    def validate(self) -> bool:
        """Validate boundary condition specification."""

    @abstractmethod
    def evaluate(self, x: jax.Array, t: float = 0.0) -> jax.Array:
        """Evaluate boundary condition at given position and time."""


class InitialCondition:
    """Initial condition specification for time-dependent problems."""

    def __init__(
        self,
        value: float | Callable[[jax.Array], jax.Array],
        dimension: int = 1,
        derivative_order: int = 0,
        name: str | None = None,
    ):
        """Initialize initial condition.

        Args:
            value: Constant value or function defining initial condition
            dimension: Dimension of the solution field
            derivative_order: Order of time derivative (0=position, 1=velocity, etc.)
            name: Optional name for the initial condition
        """
        if dimension <= 0:
            raise ValueError("Dimension must be positive")

        self.value = value
        self.dimension = dimension
        self.derivative_order = derivative_order
        self.name = name or f"ic_order_{derivative_order}"

    def validate(self) -> bool:
        """Validate initial condition specification."""
        try:
            # Check if value is callable or numeric
            if not (callable(self.value) or isinstance(self.value, (int, float))):
                return False

            # Check non-negative derivative order
            return not self.derivative_order < 0
        except Exception:
            return False

    def evaluate(self, x: jax.Array) -> jax.Array:
        """Evaluate initial condition at given position."""
        if callable(self.value):
            return self.value(x)
        if self.dimension == 1:
            return jnp.full_like(x[..., 0], self.value)
        return jnp.full((*x.shape[:-1], self.dimension), self.value)


class Constraint(ABC):
    """Abstract base class for constraints."""

    def __init__(self, constraint_type: str, tolerance: float = 1e-8):
        """Initialize constraint.

        Args:
            constraint_type: Type of constraint
            tolerance: Tolerance for constraint satisfaction
        """
        self.constraint_type = constraint_type
        self.tolerance = tolerance

    @abstractmethod
    def validate(self) -> bool:
        """Validate constraint specification."""


# ============================================================================
# Classical Boundary Conditions
# ============================================================================


class DirichletBC(BoundaryCondition):
    """Dirichlet boundary condition: u = g on boundary."""

    def __init__(
        self,
        boundary: str,
        value: float | Callable[..., jax.Array],
        time_dependent: bool = False,
    ):
        """Initialize Dirichlet boundary condition.

        Args:
            boundary: Boundary identifier
            value: Constant value or function g(x) or g(x, t) for boundary
            time_dependent: Whether condition varies with time
        """
        super().__init__(boundary, time_dependent, spatial_dependent=True)
        self.value = value
        self.condition_type = "dirichlet"

    def validate(self) -> bool:
        """Validate Dirichlet boundary condition."""
        try:
            # Check if value is callable or numeric
            return callable(self.value) or isinstance(self.value, (int, float))
        except Exception:
            return False

    def evaluate(self, x: jax.Array, t: float = 0.0) -> jax.Array:
        """Evaluate Dirichlet condition at given position and time."""
        if callable(self.value):
            if self.time_dependent:
                return self.value(x, t)
            return self.value(x)
        return jnp.full_like(x[..., 0], self.value)

    def _evaluate_boundary_value(
        self, x: jax.Array | None = None, t: float = 0.0
    ) -> float:
        """Evaluate boundary value from specification.

        Helper method to reduce branches in apply().
        """
        if not callable(self.value):
            return self.value

        # Handle callable value
        if x is not None:
            evaluated = self.value(x, t) if self.time_dependent else self.value(x)
        else:
            dummy_x = jnp.array([0.0])
            evaluated = (
                self.value(dummy_x, t) if self.time_dependent else self.value(dummy_x)
            )

        # Extract scalar from evaluated result
        if hasattr(evaluated, "__getitem__"):
            return float(evaluated[0])
        return float(evaluated)

    def apply(
        self,
        params: jax.Array,
        x: jax.Array | None = None,
        t: float = 0.0,
        weight: float = 1.0,
        **kwargs: Any,
    ) -> jax.Array:
        """Apply Dirichlet boundary condition to parameters.

        This method bridges the OOP boundary specification with the functional
        boundary application system, allowing BC objects to apply themselves.

        Args:
            params: Parameter array to constrain
            x: Optional spatial coordinates for function evaluation
            t: Time for time-dependent conditions
            weight: Constraint weight (0-1, default 1.0)
            **kwargs: Additional arguments (left_boundary, right_boundary)

        Returns:
            Parameters with Dirichlet boundary condition applied

        Examples:
            >>> bc = DirichletBC(boundary="left", value=0.0)
            >>> params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> constrained = bc.apply(params)
            >>> # Only left boundary modified: constrained[0] == 0.0
        """
        boundary_value = self._evaluate_boundary_value(x, t)

        # Check if user explicitly provided left_boundary or right_boundary
        if "left_boundary" in kwargs or "right_boundary" in kwargs:
            return apply_dirichlet(
                params, boundary_value=boundary_value, weight=weight, **kwargs
            )

        # Respect the boundary attribute - only apply to specified side
        if self.boundary == "left":
            right_val = float(params[..., -1].item()) if params.size > 0 else None
            return apply_dirichlet(
                params,
                left_boundary=boundary_value,
                right_boundary=right_val,
                weight=weight,
            )
        if self.boundary == "right":
            left_val = float(params[..., 0].item()) if params.size > 0 else None
            return apply_dirichlet(
                params,
                left_boundary=left_val,
                right_boundary=boundary_value,
                weight=weight,
            )
        # For "all" or other boundaries, apply to both
        return apply_dirichlet(params, boundary_value=boundary_value, weight=weight)


class NeumannBC(BoundaryCondition):
    """Neumann boundary condition: du/dn = g on boundary."""

    def __init__(
        self,
        boundary: str,
        value: float | Callable[..., jax.Array],
        time_dependent: bool = False,
    ):
        """Initialize Neumann boundary condition.

        Args:
            boundary: Boundary identifier
            value: Constant value or function g(x) or g(x, t) for normal derivative
            time_dependent: Whether condition varies with time
        """
        super().__init__(boundary, time_dependent, spatial_dependent=True)
        self.value = value
        self.condition_type = "neumann"

    def validate(self) -> bool:
        """Validate Neumann boundary condition."""
        try:
            # Check if value is callable or numeric
            return callable(self.value) or isinstance(self.value, (int, float))
        except Exception:
            return False

    def evaluate(self, x: jax.Array, t: float = 0.0) -> jax.Array:
        """Evaluate Neumann condition at given position and time."""
        if callable(self.value):
            if self.time_dependent:
                return self.value(x, t)
            return self.value(x)
        return jnp.full_like(x[..., 0], self.value)

    def apply(
        self,
        params: jax.Array,
        x: jax.Array | None = None,
        t: float = 0.0,
        weight: float = 1.0,
    ) -> jax.Array:
        """Apply Neumann boundary condition to parameters.

        This method bridges the OOP boundary specification with the functional
        boundary application system, allowing BC objects to apply themselves.

        Args:
            params: Parameter array to constrain
            x: Optional spatial coordinates (unused for Neumann, kept for
                interface consistency)
            t: Time for time-dependent conditions (unused for Neumann, kept
                for interface consistency)
            weight: Constraint weight (0-1, default 1.0)

        Returns:
            Parameters with Neumann boundary condition applied (zero derivative)

        Examples:
            >>> bc = NeumannBC(boundary="wall", value=0.0)
            >>> params = jnp.array([10.0, 2.0, 3.0, 4.0, 20.0])
            >>> constrained = bc.apply(params)
            >>> # constrained[0] == params[1], constrained[-1] == params[-2]
        """
        # Apply using functional implementation (zero derivative)
        # Note: x and t are ignored for Neumann BC but accepted for consistent interface
        return apply_neumann(params, weight=weight)


class RobinBC(BoundaryCondition):
    """Robin (mixed) boundary condition: alpha*u + beta*du/dn = gamma on boundary."""

    def __init__(
        self,
        boundary: str,
        alpha: float | Callable[..., float],
        beta: float | Callable[..., float],
        gamma: float | Callable[..., jax.Array],
        time_dependent: bool = False,
    ):
        """Initialize Robin boundary condition.

        Args:
            boundary: Boundary identifier
            alpha: Coefficient for u term
            beta: Coefficient for du/dn term
            gamma: Right-hand side function
            time_dependent: Whether condition varies with time
        """
        try:
            # Validate alpha parameter
            if callable(alpha):
                try:
                    # Try array input first, fall back to scalar if needed
                    test_val = jnp.array([1.0])
                    alpha(test_val)
                except (TypeError, IndexError):
                    try:
                        alpha(1.0)  # Fall back to scalar test
                    except Exception as e:
                        raise ValueError(
                            "alpha function must accept array or scalar input"
                        ) from e
            elif not isinstance(alpha, (int, float)):
                raise TypeError("alpha must be callable or numeric")  # noqa: TRY301

            if callable(beta):
                try:
                    # Try array input first, fall back to scalar if needed
                    test_val = jnp.array([1.0])
                    beta(test_val)
                except (TypeError, IndexError):
                    try:
                        beta(1.0)  # Fall back to scalar test
                    except Exception as e:
                        raise ValueError(
                            "beta function must accept array or scalar input"
                        ) from e
            elif not isinstance(beta, (int, float)):
                raise TypeError("beta must be callable or numeric")  # noqa: TRY301

            # Only validate zero constraint for constant values
            if (
                not callable(alpha)
                and not callable(beta)
                and alpha == 0.0
                and beta == 0.0
            ):
                raise ValueError("Both alpha and beta cannot be zero")  # noqa: TRY301

        except ValueError:
            raise  # Re-raise ValueError messages
        except Exception as e:
            raise ValueError(f"Invalid alpha or beta function: {e}") from e

        super().__init__(boundary, time_dependent, spatial_dependent=True)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.condition_type = "robin"

    def validate(self) -> bool:
        """Validate Robin boundary condition."""
        try:
            # Validate alpha parameter
            if callable(self.alpha):
                try:
                    test_val = jnp.array([1.0])
                    self.alpha(test_val)
                except (TypeError, IndexError):
                    try:
                        self.alpha(1.0)
                    except Exception:
                        return False
            elif not isinstance(self.alpha, (int, float)):
                return False

            # Validate beta parameter
            if callable(self.beta):
                try:
                    test_val = jnp.array([1.0])
                    self.beta(test_val)
                except (TypeError, IndexError):
                    try:
                        self.beta(1.0)
                    except Exception:
                        return False
            elif not isinstance(self.beta, (int, float)):
                return False

            # Check zero constraint for constant values - return negated condition
            return not (
                not callable(self.alpha)
                and not callable(self.beta)
                and self.alpha == 0.0
                and self.beta == 0.0
            )

        except Exception:
            return False

    def evaluate(self, x: jax.Array, t: float = 0.0) -> jax.Array:
        """Evaluate Robin condition coefficients and RHS at given position and time."""
        if callable(self.alpha):
            alpha_val = self.alpha(x, t) if self.time_dependent else self.alpha(x)
        else:
            alpha_val = self.alpha

        if callable(self.beta):
            beta_val = self.beta(x, t) if self.time_dependent else self.beta(x)
        else:
            beta_val = self.beta

        if callable(self.gamma):
            gamma_val = self.gamma(x, t) if self.time_dependent else self.gamma(x)
        else:
            gamma_val = jnp.full_like(x[..., 0], self.gamma)

        # Return array with Robin condition coefficients
        return jnp.array(
            [alpha_val, beta_val, gamma_val[0] if gamma_val.ndim > 0 else gamma_val]
        )

    def _evaluate_coefficient(
        self, coeff, x: jax.Array | None = None, t: float = 0.0
    ) -> float:
        """Evaluate coefficient from specification.

        Helper method to reduce branches in apply().
        """
        if not callable(coeff):
            return coeff

        # Handle callable coefficient
        if x is not None:
            evaluated = coeff(x, t) if self.time_dependent else coeff(x)
        else:
            dummy_x = jnp.array([0.0])
            evaluated = coeff(dummy_x, t) if self.time_dependent else coeff(dummy_x)

        # Extract scalar from evaluated result
        if isinstance(evaluated, (jnp.ndarray, list, tuple)):
            return float(evaluated[0])  # type: ignore[index]
        return float(evaluated)  # pyright: ignore[reportArgumentType]

    def apply(
        self,
        params: jax.Array,
        x: jax.Array | None = None,
        t: float = 0.0,
        weight: float = 1.0,
    ) -> jax.Array:
        """Apply Robin boundary condition to parameters.

        This method bridges the OOP boundary specification with the functional
        boundary application system, allowing BC objects to apply themselves.

        Robin condition: alpha*u + beta*du/dn = gamma on boundary

        Args:
            params: Parameter array to constrain
            x: Optional spatial coordinates for function evaluation
            t: Time for time-dependent conditions
            weight: Constraint weight (0-1, default 1.0)

        Returns:
            Parameters with Robin boundary condition applied

        Examples:
            >>> bc = RobinBC(boundary="left", alpha=1.0, beta=0.0, gamma=0.0)
            >>> params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> constrained = bc.apply(params)
            >>> # Behaves like Dirichlet when beta=0
        """
        alpha_val = self._evaluate_coefficient(self.alpha, x, t)
        beta_val = self._evaluate_coefficient(self.beta, x, t)
        gamma_val = self._evaluate_coefficient(self.gamma, x, t)

        # Respect the boundary attribute - only apply to specified side
        # Since apply_robin doesn't have left/right parameters, we need to
        # do this manually
        if self.boundary == "left":
            # Only modify left boundary, preserve right
            result = apply_robin(
                params, alpha=alpha_val, beta=beta_val, gamma=gamma_val, weight=weight
            )
            return result.at[..., -1].set(params[..., -1])
        if self.boundary == "right":
            # Only modify right boundary, preserve left
            result = apply_robin(
                params, alpha=alpha_val, beta=beta_val, gamma=gamma_val, weight=weight
            )
            return result.at[..., 0].set(params[..., 0])
        # For "all" or other boundaries, apply to both
        return apply_robin(
            params, alpha=alpha_val, beta=beta_val, gamma=gamma_val, weight=weight
        )


# ============================================================================
# Quantum Boundary Conditions
# ============================================================================


class WavefunctionBC(BoundaryCondition):
    """Quantum mechanical wavefunction boundary conditions."""

    # Add proper type annotations
    condition_type: str
    value: complex | float | None = None
    norm_value: float | None

    def __init__(
        self,
        condition_type: str,
        boundary: str = "all",
        value: complex | None = None,
        norm_value: float | None = None,
    ):
        """Initialize wavefunction boundary condition.

        Args:
            condition_type: Type of wavefunction condition
                ("vanishing", "normalization", "periodic", "boundary")
            boundary: Boundary identifier for spatial boundaries
            value: Value for boundary conditions (real values auto-converted to complex)
            norm_value: Normalization value for wavefunction
        """
        # Validate condition type immediately
        valid_types = {"vanishing", "normalization", "periodic", "boundary"}
        if condition_type not in valid_types:
            raise ValueError(
                f"Invalid condition_type: {condition_type}. "
                f"Must be one of {valid_types}"
            )

        super().__init__(boundary, time_dependent=False, spatial_dependent=True)
        self.condition_type = condition_type
        # Convert real values to complex for quantum boundary conditions
        if value is not None and isinstance(value, (int, float)):
            self.value = complex(value)
        else:
            self.value = value
        self.norm_value = norm_value

    def validate(self) -> bool:
        """Validate wavefunction boundary condition."""
        valid_types = {"vanishing", "normalization", "periodic", "boundary"}
        if self.condition_type not in valid_types:
            return False

        if self.condition_type == "normalization":
            return self.norm_value is not None and self.norm_value > 0

        if self.condition_type == "periodic":
            return self.value is not None

        if self.condition_type == "boundary":
            return True  # Boundary conditions are always valid

        # For vanishing conditions, no additional requirements
        return True

    def evaluate(self, x: jax.Array, t: float = 0.0) -> jax.Array:
        """Evaluate wavefunction boundary condition."""
        if self.condition_type == "vanishing":
            return jnp.zeros_like(x)
        if self.condition_type == "normalization":
            return jnp.full_like(x[..., 0], self.norm_value or 1.0)
        if self.condition_type == "periodic" and self.value is not None:
            return jnp.full_like(x, self.value)
        return jnp.zeros_like(x)


class DensityConstraint(Constraint):
    """Electronic density constraints for quantum systems."""

    def __init__(
        self,
        constraint_type: str,
        n_electrons: int | None = None,
        tolerance: float = 1e-8,
        enforcement_method: str = "lagrange",
    ):
        """Initialize density constraint.

        Args:
            constraint_type: Type of constraint
                ("conservation", "positivity", "particle_number")
            n_electrons: Number of electrons for particle number conservation
            tolerance: Tolerance for constraint satisfaction
            enforcement_method: Method for constraint enforcement
        """
        # Validate constraint type immediately
        valid_types = {"conservation", "positivity", "particle_number"}
        if constraint_type not in valid_types:
            raise ValueError(
                f"Invalid constraint_type: {constraint_type}. "
                f"Must be one of {valid_types}"
            )

        # Validate enforcement method immediately
        valid_methods = {"lagrange", "penalty", "projection"}
        if enforcement_method not in valid_methods:
            raise ValueError(
                f"Invalid enforcement_method: {enforcement_method}. "
                f"Must be one of {valid_methods}"
            )

        super().__init__(constraint_type, tolerance)
        self.n_electrons = n_electrons
        self.enforcement_method = enforcement_method

    def validate(self) -> bool:
        """Validate density constraint."""
        valid_types = {"conservation", "positivity", "particle_number"}
        if self.constraint_type not in valid_types:
            return False

        # Both conservation and particle_number require n_electrons
        if (
            self.constraint_type in ("conservation", "particle_number")
            and self.n_electrons is None
        ):
            return False

        valid_methods = {"lagrange", "penalty", "projection"}
        return self.enforcement_method in valid_methods


class SymmetryConstraint(Constraint):
    """Molecular symmetry constraints for quantum systems."""

    def __init__(
        self,
        point_group: str | None = None,
        operations: Sequence[str] | None = None,
        symmetry_type: str = "point_group",
        lattice_vectors: jax.Array | None = None,
        enforce_in_loss: bool = True,
    ):
        """Initialize symmetry constraint.

        Args:
            point_group: Point group symmetry (e.g., "C2v", "D2h")
            operations: List of symmetry operations
            symmetry_type: Type of symmetry ("point_group", "translational")
            lattice_vectors: Lattice vectors for periodic systems
            enforce_in_loss: Whether to enforce symmetry in loss function
        """
        # Validate symmetry type immediately
        valid_types = {"point_group", "translational", "rotational", "lattice"}
        if symmetry_type not in valid_types:
            raise ValueError(
                f"Invalid symmetry_type: {symmetry_type}. Must be one of {valid_types}"
            )

        super().__init__(symmetry_type, tolerance=1e-8)
        self.point_group = point_group
        # Keep operations as None if not provided, don't default to empty list
        self.operations = operations
        self.symmetry_type = symmetry_type
        self.lattice_vectors = lattice_vectors
        self.enforce_in_loss = enforce_in_loss

    def validate(self) -> bool:
        """Validate symmetry constraint."""
        valid_types = {"point_group", "translational", "rotational", "lattice"}
        if self.symmetry_type not in valid_types:
            return False

        # For point_group type, require either point_group OR operations
        if (
            self.symmetry_type == "point_group"
            and not self.point_group
            and not self.operations
        ):
            return False

        # For lattice type, require lattice_vectors
        if self.symmetry_type == "lattice" and self.lattice_vectors is None:
            return False

        # For translational type, require lattice_vectors
        return not (
            self.symmetry_type == "translational" and self.lattice_vectors is None
        )


# ============================================================================
# Quantum Initial Conditions
# ============================================================================


class QuantumInitialCondition(InitialCondition):
    """Quantum mechanical initial conditions."""

    def __init__(
        self,
        condition_type: str,
        value: Callable[[jax.Array], jax.Array],
        normalization: float = 1.0,
        n_electrons: int | None = None,
    ):
        """Initialize quantum initial condition.

        Args:
            condition_type: Type of condition
                ("ground_state", "excited_state", "custom", "density", "wavefunction")
            value: Function defining initial quantum state
            normalization: Normalization constant
            n_electrons: Number of electrons (for density conditions)
        """
        # Validate condition type immediately
        valid_types = {
            "ground_state",
            "excited_state",
            "custom",
            "density",
            "wavefunction",
        }
        if condition_type not in valid_types:
            raise ValueError(
                f"Invalid condition_type: {condition_type}. "
                f"Must be one of {valid_types}"
            )

        super().__init__(value, dimension=1)
        self.condition_type = condition_type
        self.normalization = normalization
        self.n_electrons = n_electrons

    def validate(self) -> bool:
        """Validate quantum initial condition."""
        valid_types = {
            "ground_state",
            "excited_state",
            "custom",
            "density",
            "wavefunction",
        }
        if self.condition_type not in valid_types:
            return False

        if not callable(self.value):
            return False

        return self.normalization > 0


# ============================================================================
# Symbolic Constraints
# ============================================================================


class SymbolicConstraint(Constraint):
    """Symbolic constraint expressions."""

    # Add proper type annotations
    expression: str
    variables: Sequence[str]
    parameters: dict[str, Any]
    physics_law: str | None

    def __init__(
        self,
        expression: str,
        variables: Sequence[str],
        parameters: dict[str, Any] | None = None,
        constraint_type: str = "general",
        tolerance: float = 1e-8,
    ):
        """Initialize symbolic constraint.

        Args:
            expression: Mathematical expression as string
            variables: List of variable names in expression
            parameters: Optional parameters for expression evaluation
            constraint_type: Type of symbolic constraint
            tolerance: Tolerance for constraint satisfaction
        """
        super().__init__(constraint_type, tolerance)
        self.expression = expression
        self.variables = variables
        self.parameters = parameters or {}
        self.physics_law = None

    def validate(self) -> bool:
        """Validate symbolic constraint."""
        try:
            # Basic validation - check non-empty expression and variables
            return not (not self.expression or not self.variables)
        except Exception:
            return False


class PhysicsConstraint(SymbolicConstraint):
    """Physics-based symbolic constraints."""

    def __init__(
        self,
        constraint_type: str,
        expression: str,
        variables: Sequence[str],
        physics_law: str | None = None,
        parameters: dict[str, Any] | None = None,
        tolerance: float = 1e-6,
    ):
        """Initialize physics constraint.

        Args:
            constraint_type: Type of physics constraint
            expression: Mathematical expression of the constraint
            variables: Variables involved in the constraint
            physics_law: Name of the physics law being enforced
            parameters: Parameters in the constraint
            tolerance: Tolerance for constraint satisfaction
        """
        super().__init__(expression, variables, parameters, constraint_type, tolerance)
        self.physics_law = physics_law
        self.tolerance = tolerance

    def validate(self) -> bool:
        """Validate physics constraint."""
        valid_types = {
            "mass_conservation",
            "energy_conservation",
            "momentum_conservation",
            "momentum",
            "charge_conservation",
            "continuity_equation",
            # Add quantum constraint types for inheritance
            "particle_number",
            "density_positivity",
            "wavefunction_normalization",
            "normalization",  # Standard normalization constraint
            "hermiticity",
            "unitarity",  # Add unitarity for QuantumConstraint inheritance
            "time_reversal_symmetry",
        }
        if self.constraint_type not in valid_types:
            return False
        return super().validate()


class QuantumConstraint(PhysicsConstraint):
    """Quantum mechanical physics constraints."""

    def __init__(
        self,
        constraint_type: str,
        expression: str,
        variables: Sequence[str],
        parameters: dict[str, Any] | None = None,
        tolerance: float = 1e-8,
        enforcement_method: str = "penalty",
    ):
        """Initialize quantum constraint.

        Args:
            constraint_type: Type of quantum constraint
            expression: Mathematical expression
            variables: Variables in the constraint
            parameters: Constraint parameters
            tolerance: Tolerance for quantum accuracy
            enforcement_method: Method for enforcing constraint
        """
        # Validate enforcement method immediately
        valid_methods = {"penalty", "lagrange", "projection", "soft"}
        if enforcement_method not in valid_methods:
            raise ValueError(
                f"Invalid enforcement_method: {enforcement_method}. "
                f"Must be one of {valid_methods}"
            )

        super().__init__(
            constraint_type,
            expression,
            variables,
            physics_law="quantum_mechanics",
            parameters=parameters,
            tolerance=tolerance,
        )
        self.enforcement_method = enforcement_method

    def validate(self) -> bool:
        """Validate quantum constraint."""
        valid_types = {
            "particle_number",
            "density_positivity",
            "wavefunction_normalization",
            "normalization",  # Standard normalization constraint
            "hermiticity",
            "unitarity",  # Add unitarity as valid quantum constraint type
            "time_reversal_symmetry",
        }
        if self.constraint_type not in valid_types:
            return False

        valid_methods = {"penalty", "lagrange", "projection", "soft"}
        if self.enforcement_method not in valid_methods:
            return False

        return super().validate()


# ============================================================================
# Boundary Condition Collections
# ============================================================================


class BoundaryConditionCollection:
    """Collection and management of boundary conditions."""

    def __init__(self, boundary_conditions: Sequence[BoundaryCondition]):
        """Initialize boundary condition collection.

        Args:
            boundary_conditions: List of boundary conditions
        """
        self.conditions = list(boundary_conditions)
        self._boundary_map = {bc.boundary: bc for bc in self.conditions}

    def __len__(self) -> int:
        """Return number of boundary conditions."""
        return len(self.conditions)

    def validate(self) -> bool:
        """Validate all boundary conditions in collection."""
        try:
            return all(bc.validate() for bc in self.conditions)
        except Exception:
            return False

    def get_boundary_condition(self, boundary: str) -> BoundaryCondition | None:
        """Get boundary condition for specific boundary."""
        return self._boundary_map.get(boundary)

    def get_by_type(self, condition_type: str) -> list[BoundaryCondition]:
        """Get all boundary conditions of specific type."""
        results = []

        # Pre-compute condition type comparisons for efficiency
        condition_type_lower = condition_type.lower()

        # Use JAX-compatible operations where possible
        for bc in self.conditions:
            # Check both class name and condition_type attribute for compatibility
            class_name = type(bc).__name__
            bc_condition_type = getattr(bc, "condition_type", None)

            # Direct class name match (case-insensitive)
            class_name_lower = class_name.lower()

            # Optimized condition checking using JAX-compatible logic
            is_match = (
                class_name_lower == condition_type_lower
                or (
                    bc_condition_type is not None
                    and bc_condition_type == condition_type
                )
                or (
                    condition_type_lower == "dirichlet"
                    and "dirichlet" in class_name_lower
                )
                or (condition_type_lower == "neumann" and "neumann" in class_name_lower)
                or (condition_type_lower == "robin" and "robin" in class_name_lower)
            )

            if is_match:
                results.append(bc)

        return results

    def add_condition(self, condition: BoundaryCondition) -> None:
        """Add a new boundary condition."""
        self.conditions.append(condition)
        self._boundary_map[condition.boundary] = condition

    def remove_condition(self, boundary: str) -> bool:
        """Remove boundary condition for specific boundary."""
        if boundary in self._boundary_map:
            condition = self._boundary_map.pop(boundary)
            self.conditions.remove(condition)
            return True
        return False

    def apply_all(
        self,
        params: jax.Array,
        x: jax.Array | None = None,
        t: float = 0.0,
        weight: float = 1.0,
    ) -> jax.Array:
        """Apply all boundary conditions in collection to parameters.

        This method sequentially applies each boundary condition in the collection,
        allowing multiple BCs to be enforced together.

        Args:
            params: Parameter array to constrain
            x: Optional spatial coordinates for function evaluation
            t: Time for time-dependent conditions
            weight: Global constraint weight applied to all BCs (0-1, default 1.0)

        Returns:
            Parameters with all boundary conditions applied

        Examples:
            >>> bc1 = DirichletBC(boundary="left", value=0.0)
            >>> bc2 = NeumannBC(boundary="right", value=0.0)
            >>> collection = BoundaryConditionCollection([bc1, bc2])
            >>> params = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
            >>> constrained = collection.apply_all(params)
        """
        result = params
        for bc in self.conditions:
            # Check if BC has apply method (should be DirichletBC, NeumannBC, RobinBC)
            if hasattr(bc, "apply"):
                result = bc.apply(result, x=x, t=t, weight=weight)  # type: ignore[attr-defined]
        return result
