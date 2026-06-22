"""JAX/Flax-NNX transform-compatibility tests for the refined compute paths.

Every transformable array computation introduced for the scientific-integrity
compute paths must remain ``jit``/``grad``/``vmap`` clean and agree with eager
execution. Each test below pins one such computation against its eager result.

These cover the residual scientific-integrity compute paths outside the L2O
subsystem (``scientific_integration.translation_invariance_error`` and
``solvers.hybrid.relative_field_discrepancy``). The former L2O stub paths
(multi-objective feedback, keep-best refinement, the RL optimization step)
were removed with the fabricated L2O modules in the SOTA rebuild.
"""

import jax
import jax.numpy as jnp
import pytest

from opifex.optimization.scientific_integration import translation_invariance_error
from opifex.solvers.hybrid import relative_field_discrepancy


class TestTranslationInvarianceTransforms:
    """Transform compatibility for the translation-invariance check."""

    def test_translation_invariance_is_jit_vmap_compatible(self) -> None:
        """``jit`` and ``vmap`` over a field batch match eager evaluation."""
        fields = jnp.stack(
            [
                jnp.array([1.0, 1.0, 1.0, 1.0]),
                jnp.array([0.0, 1.0, 2.0, 3.0]),
            ]
        )
        shifts = (1,)
        axes = (0,)

        eager = jnp.stack([translation_invariance_error(f, shifts, axes) for f in fields])

        jitted = jax.jit(translation_invariance_error, static_argnums=(1, 2))
        jit_eager = jnp.stack([jitted(f, shifts, axes) for f in fields])

        batched = jax.vmap(lambda f: translation_invariance_error(f, shifts, axes))(fields)

        assert jnp.isfinite(batched).all()
        assert batched.shape == (2,)
        assert jnp.allclose(jit_eager, eager)
        assert jnp.allclose(batched, eager)
        # Constant field is translation invariant; structured field is not.
        assert float(eager[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(eager[1]) > 0.0

    def test_translation_invariance_is_grad_clean(self) -> None:
        """The discrepancy is differentiable w.r.t. the field."""
        field = jnp.array([0.0, 1.0, 2.0, 3.0])
        grad = jax.grad(lambda f: translation_invariance_error(f, (1,), (0,)))(field)
        assert grad.shape == field.shape
        assert jnp.isfinite(grad).all()


class TestHybridDiscrepancyTransforms:
    """Transform compatibility for the hybrid classical/neural discrepancy."""

    def test_hybrid_error_is_jit_grad_clean(self) -> None:
        """``jit`` and ``grad`` of the discrepancy match eager evaluation."""
        classical = jnp.array([1.0, 2.0, 3.0, 4.0])
        neural = jnp.array([1.1, 1.9, 3.2, 3.8])

        eager = relative_field_discrepancy(classical, neural)
        jitted = jax.jit(relative_field_discrepancy)(classical, neural)
        assert jnp.allclose(jitted, eager)
        assert jnp.isfinite(eager)

        grad = jax.grad(relative_field_discrepancy, argnums=1)(classical, neural)
        assert grad.shape == neural.shape
        assert jnp.isfinite(grad).all()

    def test_hybrid_error_is_vmap_compatible(self) -> None:
        """``vmap`` over a batch of field pairs matches eager evaluation."""
        classical = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        neural = jnp.array([[1.1, 1.9], [2.8, 4.2], [5.0, 6.0]])

        eager = jnp.stack(
            [relative_field_discrepancy(c, n) for c, n in zip(classical, neural, strict=True)]
        )
        batched = jax.vmap(relative_field_discrepancy)(classical, neural)
        assert batched.shape == (3,)
        assert jnp.allclose(batched, eager)
