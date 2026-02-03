"""Group theory for Lie groups in geometry.

This module implements common Lie groups used in geometric deep learning,
including SO(3) and SE(3) for rotations and rigid transformations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class SO3Group:
    """Special Orthogonal Group SO(3) - 3D rotations.

    Represents rotation matrices in 3D space with determinant +1.
    """

    def __init__(self):
        """Initialize SO(3) group."""
        self.dimension = 3  # Manifold dimension
        self.matrix_size = 3  # 3x3 rotation matrices

    def identity(self) -> jax.Array:
        """Identity element of SO(3)."""
        return jnp.eye(3)

    def random_element(self, key: jax.Array) -> jax.Array:
        """Generate random rotation matrix.

        Uses QR decomposition of random matrix to ensure uniform
        distribution on SO(3).

        Args:
            key: JAX random key

        Returns:
            Random 3x3 rotation matrix
        """
        # Generate random 3x3 matrix
        random_matrix = jax.random.normal(key, (3, 3))

        # QR decomposition
        Q, _ = jnp.linalg.qr(random_matrix)

        # Ensure determinant is +1
        det_Q = jnp.linalg.det(Q)
        return Q * det_Q

    def quaternion_to_matrix(self, q: jax.Array) -> jax.Array:
        """Convert unit quaternion to rotation matrix.

        Args:
            q: Unit quaternion [w, x, y, z]

        Returns:
            3x3 rotation matrix
        """
        w, x, y, z = q[0], q[1], q[2], q[3]

        # Normalize quaternion
        norm = jnp.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w / norm, x / norm, y / norm, z / norm

        # Convert to rotation matrix
        return jnp.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)],
            ]
        )

    def matrix_to_axis_angle(self, R: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Convert rotation matrix to axis-angle representation.

        Args:
            R: Rotation matrix

        Returns:
            Tuple of (axis, angle) where angle is a scalar Array
        """
        # Extract angle
        trace = jnp.trace(R)
        angle = jnp.arccos(jnp.clip((trace - 1) / 2, -1.0, 1.0))

        # Extract axis (handle special cases)
        axis_vec = jnp.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        axis_norm = jnp.linalg.norm(axis_vec)

        # Handle near-zero rotation
        normalized_axis = axis_vec / jnp.maximum(axis_norm, 1e-8)
        default_axis = jnp.array([1.0, 0.0, 0.0])

        # Ensure both branches return same type
        axis = jnp.where(axis_norm > 1e-6, normalized_axis, default_axis)

        return axis, angle

    def compose(self, g1: jax.Array, g2: jax.Array) -> jax.Array:
        """Group composition (matrix multiplication).

        Args:
            g1: First rotation matrix
            g2: Second rotation matrix

        Returns:
            Composed rotation g1 @ g2
        """
        return g1 @ g2

    def inverse(self, g: jax.Array) -> jax.Array:
        """Group inverse (transpose for orthogonal matrices).

        Args:
            g: Rotation matrix

        Returns:
            Inverse rotation (transpose)
        """
        return g.T

    def act_on_vector(self, g: jax.Array, v: jax.Array) -> jax.Array:
        """Group action on 3D vectors.

        Args:
            g: Rotation matrix
            v: 3D vector

        Returns:
            Rotated vector g @ v
        """
        return g @ v

    def exp_map(self, tangent: jax.Array) -> jax.Array:
        """Exponential map from so(3) to SO(3).

        Args:
            tangent: Element of so(3) (3D vector representing rotation)

        Returns:
            Rotation matrix via Rodrigues' formula
        """
        angle = jnp.linalg.norm(tangent)

        # Handle zero rotation - ensure consistent typing
        default_axis = jnp.array([1.0, 0.0, 0.0])
        normalized_tangent = tangent / jnp.maximum(angle, 1e-8)
        axis = jnp.where(angle > 1e-8, normalized_tangent, default_axis)
        # Ensure axis is properly typed as jax.Array
        axis = jnp.asarray(axis)

        # Rodrigues' formula
        K = self._skew_symmetric(axis)
        R = jnp.eye(3) + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * K @ K

        return jnp.where(angle > 1e-8, R, jnp.eye(3))

    def log_map(self, R: jax.Array) -> jax.Array:
        """Logarithmic map from SO(3) to so(3).

        Args:
            R: Rotation matrix

        Returns:
            Element of so(3) (3D rotation vector)
        """
        axis, angle = self.matrix_to_axis_angle(R)
        return angle * axis

    def _skew_symmetric(self, v: jax.Array) -> jax.Array:
        """Convert 3D vector to skew-symmetric matrix.

        Args:
            v: 3D vector [x, y, z]

        Returns:
            Skew-symmetric matrix for cross product
        """
        return jnp.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


class SE3Group:
    """Special Euclidean Group SE(3) - 3D rotations and translations.

    Represents rigid body transformations in 3D space.
    """

    def __init__(self):
        """Initialize SE(3) group."""
        self.so3 = SO3Group()
        self.dimension = 6  # 3 for rotation + 3 for translation
        self.matrix_size = 4  # 4x4 homogeneous matrices

    def identity(self) -> jax.Array:
        """Identity element of SE(3)."""
        return jnp.eye(4)

    def from_rotation_translation(self, R: jax.Array, t: jax.Array) -> jax.Array:
        """Construct SE(3) element from rotation and translation.

        Args:
            R: Rotation matrix
            t: Translation vector

        Returns:
            4x4 homogeneous transformation matrix
        """
        transform = jnp.zeros((4, 4))
        transform = transform.at[:3, :3].set(R)
        transform = transform.at[:3, 3].set(t)
        return transform.at[3, 3].set(1.0)

    def to_rotation_translation(self, T: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Extract rotation and translation from SE(3) element.

        Args:
            T: 4x4 transformation matrix

        Returns:
            Tuple of (rotation_matrix, translation_vector)
        """
        R = T[:3, :3]
        t = T[:3, 3]
        return R, t

    def compose(self, g1: jax.Array, g2: jax.Array) -> jax.Array:
        """Group composition for SE(3).

        Args:
            g1: First transformation matrix
            g2: Second transformation matrix

        Returns:
            Composed transformation g1 @ g2
        """
        return g1 @ g2

    def inverse(self, g: jax.Array) -> jax.Array:
        """Group inverse for SE(3).

        Args:
            g: Transformation matrix

        Returns:
            Inverse transformation
        """
        R, t = self.to_rotation_translation(g)
        R_inv = R.T
        t_inv = -R_inv @ t
        return self.from_rotation_translation(R_inv, t_inv)

    def act_on_point(self, g: jax.Array, p: jax.Array) -> jax.Array:
        """Group action on 3D points.

        Args:
            g: Transformation matrix
            p: 3D point

        Returns:
            Transformed point
        """
        # Convert to homogeneous coordinates
        p_homo = jnp.concatenate([p, jnp.array([1.0])])

        # Apply transformation
        p_transformed = g @ p_homo

        # Convert back to 3D coordinates
        return p_transformed[:3]

    def adjoint_action(self, g: jax.Array, xi: jax.Array) -> jax.Array:
        """Adjoint action of SE(3) on its Lie algebra se(3).

        Args:
            g: SE(3) group element
            xi: se(3) algebra element (6D vector: [rho, phi])

        Returns:
            Transformed algebra element
        """
        R, t = self.to_rotation_translation(g)

        # Split se(3) element into translation and rotation parts
        rho = xi[:3]  # Translation part
        phi = xi[3:]  # Rotation part

        # Compute adjoint action
        adj_rho = R @ rho + jnp.cross(t, R @ phi)
        adj_phi = R @ phi

        return jnp.concatenate([adj_rho, adj_phi])


# JAX pytree registration
def _so3_tree_flatten(group):
    return (), None


def _so3_tree_unflatten(aux_data, children):
    return SO3Group()


def _se3_tree_flatten(group):
    return (), None


def _se3_tree_unflatten(aux_data, children):
    return SE3Group()


# Register groups as JAX pytrees
jax.tree_util.register_pytree_node(SO3Group, _so3_tree_flatten, _so3_tree_unflatten)
jax.tree_util.register_pytree_node(SE3Group, _se3_tree_flatten, _se3_tree_unflatten)
