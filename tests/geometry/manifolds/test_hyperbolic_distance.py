"""The Poincare-disk geodesic distance of ``HyperbolicManifold``, against a float64 reference."""

import jax
import jax.numpy as jnp
import numpy as np

from opifex.geometry.manifolds.hyperbolic import HyperbolicManifold


FLOAT32_EPS = float(np.finfo(np.float32).eps)

# The distance of two float32 points of the unit disk is 2 asinh(sqrt(t)) with
# t = |x - y|^2 / ((1 - |x|^2)(1 - |y|^2)). Evaluated in float32: the difference of a close
# pair is exact (Sterbenz's lemma); the squares, sums, the product of the two gaps, the two
# square roots and the quotient are about eight correctly rounded operations of at most half
# an ulp each, the gaps' own rounding is damped by |x|^2 / (1 - |x|^2) < 1 for the points
# used here, and two ulp are allotted to asinh. To first order the relative error is below
# 6 ulp; 8 eps bounds it. The float64 reference, computed from the same float32 inputs, is
# exact at this scale.
DISTANCE_RTOL = 8 * FLOAT32_EPS


def _reference_distance(point1: np.ndarray, point2: np.ndarray) -> float:
    """The unit-disk distance in float64, ``2 asinh(sqrt(t))``, from the float32 inputs."""
    x = np.asarray(point1, dtype=np.float64)
    y = np.asarray(point2, dtype=np.float64)
    t = np.sum((x - y) ** 2) / ((1.0 - np.sum(x**2)) * (1.0 - np.sum(y**2)))
    return float(2.0 * np.arcsinh(np.sqrt(t)))


class TestHyperbolicGeodesicDistance:
    """``geodesic_distance`` is the Poincare-disk distance of each pair of points."""

    def setup_method(self) -> None:
        """A unit-curvature disk and a point away from the origin and the boundary."""
        self.manifold = HyperbolicManifold(curvature=-1.0, dimension=2)
        self.point = jnp.array([0.1, 0.2], dtype=jnp.float32)

    def test_a_close_pair_keeps_its_distance(self) -> None:
        """A pair 1e-5 apart is that far, not 0: ``1 + 2t`` would round ``t`` away."""
        other = self.point + jnp.array([1e-5, 0.0], dtype=jnp.float32)

        distance = float(self.manifold.geodesic_distance(self.point, other))

        expected = _reference_distance(np.asarray(self.point), np.asarray(other))
        np.testing.assert_allclose(distance, expected, rtol=DISTANCE_RTOL, atol=0.0)

    def test_a_far_pair_matches_the_reference(self) -> None:
        """A pair across the disk, where ``t`` is large, matches the float64 distance."""
        other = jnp.array([-0.6, 0.5], dtype=jnp.float32)

        distance = float(self.manifold.geodesic_distance(self.point, other))

        expected = _reference_distance(np.asarray(self.point), np.asarray(other))
        np.testing.assert_allclose(distance, expected, rtol=DISTANCE_RTOL, atol=0.0)

    def test_identical_points_are_exactly_zero_with_a_zero_gradient(self) -> None:
        """``d(x, x)`` is 0 and its gradient is 0, not NaN, in both arguments."""
        distance, gradients = jax.value_and_grad(self.manifold.geodesic_distance, argnums=(0, 1))(
            self.point, self.point
        )

        assert float(distance) == 0.0
        for gradient in gradients:
            np.testing.assert_array_equal(np.asarray(gradient), np.zeros(2, dtype=np.float32))

    def test_leading_axes_give_one_distance_per_pair(self) -> None:
        """Points of shape ``(2, 3, 2)`` give distances of shape ``(2, 3)``, one per pair."""
        key1, key2 = jax.random.split(jax.random.key(0))
        points1 = 0.8 * self.manifold.random_point(key1, shape=(2, 3))
        points2 = 0.8 * self.manifold.random_point(key2, shape=(2, 3))

        distances = self.manifold.geodesic_distance(points1, points2)

        assert distances.shape == (2, 3)
        for index in np.ndindex(2, 3):
            expected = _reference_distance(np.asarray(points1[index]), np.asarray(points2[index]))
            np.testing.assert_allclose(
                float(distances[index]), expected, rtol=DISTANCE_RTOL, atol=0.0
            )

    def test_the_distance_scales_with_the_radius_of_curvature(self) -> None:
        """At curvature -4 the radius is 1/2, and every distance is half the unit-disk one."""
        other = jnp.array([-0.3, 0.4], dtype=jnp.float32)
        curved = HyperbolicManifold(curvature=-4.0, dimension=2)

        unit_distance = self.manifold.geodesic_distance(self.point, other)
        curved_distance = curved.geodesic_distance(self.point, other)

        # The radius sqrt(1/4) and the product with it are exact in float32.
        assert float(curved_distance) == 0.5 * float(unit_distance)
