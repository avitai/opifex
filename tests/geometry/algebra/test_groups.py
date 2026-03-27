"""Tests for Lie group implementations (SO3, SE3)."""

import jax
import jax.numpy as jnp

from opifex.geometry.algebra.groups import SE3Group, SO3Group


class TestSO3Group:
    """Tests for SO(3) rotation group."""

    def test_identity_is_eye(self):
        """Identity element is the 3x3 identity matrix."""
        so3 = SO3Group()
        assert jnp.allclose(so3.identity(), jnp.eye(3))

    def test_random_element_is_rotation(self):
        """Random element has determinant +1 and is orthogonal."""
        so3 = SO3Group()
        R = so3.random_element(jax.random.PRNGKey(0))
        assert R.shape == (3, 3)
        assert jnp.allclose(jnp.linalg.det(R), 1.0, atol=1e-5)
        assert jnp.allclose(R @ R.T, jnp.eye(3), atol=1e-5)

    def test_compose_is_matrix_multiply(self):
        """Composition of rotations is matrix multiplication."""
        so3 = SO3Group()
        R1 = so3.random_element(jax.random.PRNGKey(0))
        R2 = so3.random_element(jax.random.PRNGKey(1))
        composed = so3.compose(R1, R2)
        assert jnp.allclose(composed, R1 @ R2, atol=1e-5)

    def test_inverse_gives_transpose(self):
        """Inverse of rotation is its transpose."""
        so3 = SO3Group()
        R = so3.random_element(jax.random.PRNGKey(0))
        R_inv = so3.inverse(R)
        assert jnp.allclose(R @ R_inv, jnp.eye(3), atol=1e-5)

    def test_act_on_vector_preserves_norm(self):
        """Rotation preserves vector norm."""
        so3 = SO3Group()
        R = so3.random_element(jax.random.PRNGKey(0))
        v = jnp.array([1.0, 2.0, 3.0])
        Rv = so3.act_on_vector(R, v)
        assert jnp.allclose(jnp.linalg.norm(v), jnp.linalg.norm(Rv), atol=1e-5)

    def test_quaternion_to_matrix_identity(self):
        """Identity quaternion [1,0,0,0] gives identity matrix."""
        so3 = SO3Group()
        q = jnp.array([1.0, 0.0, 0.0, 0.0])
        R = so3.quaternion_to_matrix(q)
        assert jnp.allclose(R, jnp.eye(3), atol=1e-5)

    def test_exp_log_roundtrip(self):
        """exp(log(R)) recovers original rotation (near identity)."""
        so3 = SO3Group()
        # Small rotation angle to stay near identity
        tangent = jnp.array([0.1, 0.2, -0.1])
        R = so3.exp_map(tangent)
        recovered = so3.log_map(R)
        assert jnp.allclose(tangent, recovered, atol=1e-4)


class TestSE3Group:
    """Tests for SE(3) rigid transformation group."""

    def test_identity_is_4x4_eye(self):
        """Identity element is the 4x4 identity matrix."""
        se3 = SE3Group()
        assert jnp.allclose(se3.identity(), jnp.eye(4))

    def test_from_rotation_translation(self):
        """Constructs valid 4x4 homogeneous matrix."""
        se3 = SE3Group()
        R = jnp.eye(3)
        t = jnp.array([1.0, 2.0, 3.0])
        T = se3.from_rotation_translation(R, t)
        assert T.shape == (4, 4)
        assert jnp.allclose(T[:3, :3], R)
        assert jnp.allclose(T[:3, 3], t)
        assert jnp.allclose(T[3], jnp.array([0, 0, 0, 1]))

    def test_to_rotation_translation_roundtrip(self):
        """Decomposition recovers original R and t."""
        se3 = SE3Group()
        R = jnp.eye(3)
        t = jnp.array([1.0, 2.0, 3.0])
        T = se3.from_rotation_translation(R, t)
        R_out, t_out = se3.to_rotation_translation(T)
        assert jnp.allclose(R_out, R)
        assert jnp.allclose(t_out, t)

    def test_compose_identity(self):
        """Composing with identity gives original."""
        se3 = SE3Group()
        T = se3.from_rotation_translation(jnp.eye(3), jnp.array([1.0, 0.0, 0.0]))
        result = se3.compose(T, se3.identity())
        assert jnp.allclose(result, T, atol=1e-5)

    def test_inverse_composition_gives_identity(self):
        """T * T^{-1} = I."""
        se3 = SE3Group()
        T = se3.from_rotation_translation(jnp.eye(3), jnp.array([5.0, -3.0, 1.0]))
        T_inv = se3.inverse(T)
        result = se3.compose(T, T_inv)
        assert jnp.allclose(result, jnp.eye(4), atol=1e-5)

    def test_act_on_point_translation(self):
        """Pure translation moves point correctly."""
        se3 = SE3Group()
        T = se3.from_rotation_translation(jnp.eye(3), jnp.array([1.0, 2.0, 3.0]))
        p = jnp.array([0.0, 0.0, 0.0])
        result = se3.act_on_point(T, p)
        assert jnp.allclose(result, jnp.array([1.0, 2.0, 3.0]))
