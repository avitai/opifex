"""Tests for ``bounded_while_loop`` and ``unvmap_any`` in ``opifex.core.solver``."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from opifex.core.solver import bounded_while_loop, unvmap_any


class TestBoundedWhileLoop:
    """Verify exact step caps, vmapped early exit, and forward/reverse AD."""

    def test_validation_and_non_power_of_base_step_caps(self) -> None:
        """Rejects invalid max_steps/base and caps steps at exact max_steps < base."""
        with pytest.raises(ValueError, match="max_steps"):
            bounded_while_loop(
                lambda x: x < 10.0, lambda x: x + 1.0, jnp.float32(0.0), max_steps=-1
            )
        with pytest.raises(ValueError, match="base"):
            bounded_while_loop(
                lambda x: x < 10.0,
                lambda x: x + 1.0,
                jnp.float32(0.0),
                max_steps=4,
                base=1,
            )

        out0 = bounded_while_loop(
            lambda x: x < 10.0, lambda x: x + 1.0, jnp.float32(3.0), max_steps=0
        )
        assert float(out0) == 3.0

        out1 = bounded_while_loop(
            lambda x: x < 10.0, lambda x: x + 1.0, jnp.float32(3.0), max_steps=1
        )
        assert float(out1) == 4.0

        # Non-power-of-base max_steps=5 with base=16 must stop at 5, not 16.
        out5 = bounded_while_loop(
            lambda x: x < 100.0,
            lambda x: x + 1.0,
            jnp.float32(0.0),
            max_steps=5,
            base=16,
        )
        assert float(out5) == 5.0

    def test_vmap_early_exit_and_forward_reverse_higher_order_ad(self) -> None:
        """Exits early on a batch and differentiates under jvp, grad, nested vmap, and jvp(grad)."""
        calls = [0]

        def body(carry: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
            x, step = carry
            active = step < 3
            jax.debug.callback(lambda: calls.__setitem__(0, calls[0] + 1))
            return jnp.where(active, x * x, x), step + active.astype(jnp.int32)

        batch_x = jnp.array([1.1, 1.2, 1.3, 1.4], dtype=jnp.float32)
        res_x, res_step = jax.jit(
            jax.vmap(
                lambda x0: bounded_while_loop(
                    lambda c: c[1] < 3,
                    body,
                    (x0, jnp.int32(0)),
                    max_steps=256,
                    base=4,
                )
            )
        )(batch_x)
        jax.block_until_ready(res_x)
        assert calls[0] < 16
        np.testing.assert_array_equal(np.asarray(res_step), np.full(4, 3, dtype=np.int32))
        np.testing.assert_allclose(np.asarray(res_x), np.asarray(batch_x) ** 8, rtol=1e-5)

        def power_eight(x0: jax.Array) -> jax.Array:
            final_x, _ = bounded_while_loop(
                lambda c: c[1] < 3,
                lambda c: (
                    jnp.where(c[1] < 3, c[0] * c[0], c[0]),
                    c[1] + (c[1] < 3).astype(jnp.int32),
                ),
                (x0, jnp.int32(0)),
                max_steps=64,
                base=4,
            )
            return final_x

        with jax.enable_x64(True):
            grid = jnp.array([[1.1, 1.2], [0.9, 1.3]], dtype=jnp.float64)
            assert bool(unvmap_any(jnp.array([False, True])))
            nested_grad = jax.jit(jax.vmap(jax.vmap(jax.grad(power_eight))))(grid)
            np.testing.assert_allclose(
                np.asarray(nested_grad), 8.0 * (np.asarray(grid) ** 7), rtol=1e-12
            )
            x_pt = jnp.float64(1.25)
            _, jvp_d2 = jax.jvp(jax.grad(power_eight), (x_pt,), (jnp.float64(1.0),))
            assert float(jvp_d2) == pytest.approx(56.0 * (1.25**6), rel=1e-12)
