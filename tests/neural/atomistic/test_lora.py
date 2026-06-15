r"""Tests for the equivariant-safe LoRA adapter on ``nnx.Linear`` layers.

:class:`~opifex.neural.atomistic.lora.LoRALinear` wraps an ``nnx.Linear`` with a
low-rank correction ``W_eff = W + (alpha / rank) * A @ B`` where ``A`` is a small
random ``(in, rank)`` matrix and ``B`` is a ``(rank, out)`` matrix **initialised
to zero**, so the adapter is the identity (reproduces the base layer exactly) at
initialisation -- the LoRA contract of Hu et al. 2021 (LoRA, arXiv:2106.09685)
and the ``../mace`` ``mace/modules/lora.py`` ``LoRAFCLayer`` (which uses the same
``delta = A @ B`` low-rank, ``A`` random-small / ``B`` zero, e3nn ``(in, out)``
weight layout).

Load-bearing checks:

* at init (``B = 0``) the wrapped layer's output equals the base linear's output;
* after perturbing ``B`` the output equals a hand-computed
  ``x @ (W + (alpha/rank) * A @ B) + b``;
* the adapter is per-channel (operates on each output feature independently, so
  it is safe to apply per-irrep on an equivariant linear);
* a ``jit`` / ``grad`` / ``vmap`` smoke test on the forward (REQUIRED).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.neural.atomistic.lora import LoRALinear


def _base_linear(in_features: int = 4, out_features: int = 3) -> nnx.Linear:
    """A small deterministic ``nnx.Linear`` for adapter tests."""
    return nnx.Linear(in_features, out_features, rngs=nnx.Rngs(0))


class TestLoRAInitIsIdentity:
    def test_output_equals_base_at_init(self) -> None:
        """With ``B = 0`` the LoRA forward reproduces the base linear exactly."""
        base = _base_linear()
        adapter = LoRALinear(base, rank=2, alpha=1.0)
        x = jax.random.normal(jax.random.PRNGKey(1), (5, base.in_features))
        assert jnp.allclose(adapter(x), base(x), atol=1e-6)

    def test_lora_b_is_zero_at_init(self) -> None:
        """``B`` is zero-initialised so the correction starts at exactly zero."""
        adapter = LoRALinear(_base_linear(), rank=2, alpha=1.0)
        assert jnp.all(adapter.lora_b.value == 0.0)

    def test_lora_a_is_nonzero_at_init(self) -> None:
        """``A`` is random-small (nonzero) so gradients can flow once ``B`` moves."""
        adapter = LoRALinear(_base_linear(), rank=2, alpha=1.0)
        assert jnp.any(adapter.lora_a.value != 0.0)

    def test_effective_kernel_equals_base_at_init(self) -> None:
        """The fused effective kernel equals the base kernel at init."""
        base = _base_linear()
        adapter = LoRALinear(base, rank=2, alpha=1.0)
        assert jnp.allclose(adapter.effective_kernel(), base.kernel.value, atol=1e-6)


class TestLoRAPerturbedMatchesFormula:
    def test_output_matches_hand_computed_w_eff(self) -> None:
        """After setting ``B``, output equals ``x @ (W + (alpha/r) A@B) + b``."""
        base = _base_linear(in_features=4, out_features=3)
        rank, alpha = 2, 4.0
        adapter = LoRALinear(base, rank=rank, alpha=alpha)
        # Perturb B to a known nonzero value.
        new_b = jnp.arange(rank * base.out_features, dtype=jnp.float32).reshape(
            rank, base.out_features
        )
        adapter.lora_b.value = new_b
        x = jax.random.normal(jax.random.PRNGKey(2), (6, base.in_features))

        scaling = alpha / rank
        w_eff = base.kernel.value + scaling * (adapter.lora_a.value @ new_b)
        assert base.bias is not None  # narrow Param | None for the readout
        expected = x @ w_eff + base.bias.value
        assert jnp.allclose(adapter(x), expected, atol=1e-5)

    def test_effective_kernel_matches_formula(self) -> None:
        """``effective_kernel`` equals ``W + (alpha/rank) * A @ B`` exactly."""
        base = _base_linear()
        rank, alpha = 3, 2.0
        adapter = LoRALinear(base, rank=rank, alpha=alpha)
        adapter.lora_b.value = jnp.ones_like(adapter.lora_b.value)
        scaling = alpha / rank
        expected = base.kernel.value + scaling * (adapter.lora_a.value @ adapter.lora_b.value)
        assert jnp.allclose(adapter.effective_kernel(), expected, atol=1e-6)


class TestLoRAEquivariantSafety:
    def test_per_output_channel_independence(self) -> None:
        """A correction on output channel ``j`` only affects channel ``j``.

        Per-channel independence is what lets the same adapter wrap an
        equivariant linear (one block per irrep) without mixing irreps.
        """
        base = _base_linear(in_features=4, out_features=3)
        adapter = LoRALinear(base, rank=2, alpha=1.0)
        # Set B to affect only output channel 0.
        b = jnp.zeros_like(adapter.lora_b.value)
        b = b.at[:, 0].set(1.0)
        adapter.lora_b.value = b
        x = jax.random.normal(jax.random.PRNGKey(3), (5, base.in_features))
        delta = adapter(x) - base(x)
        # Only channel 0 changed.
        assert jnp.any(jnp.abs(delta[:, 0]) > 0)
        assert jnp.allclose(delta[:, 1:], 0.0, atol=1e-6)


class TestLoRATransforms:
    def test_jit_grad_vmap_smoke(self) -> None:
        """The LoRA forward is ``jit`` / ``grad`` / ``vmap`` clean (REQUIRED)."""
        base = _base_linear(in_features=4, out_features=3)
        adapter = LoRALinear(base, rank=2, alpha=1.0)
        adapter.lora_b.value = 0.1 * jnp.ones_like(adapter.lora_b.value)
        graphdef, state = nnx.split(adapter)

        def forward(state: nnx.State, x: jax.Array) -> jax.Array:
            model = nnx.merge(graphdef, state)
            return jnp.sum(model(x))

        x = jax.random.normal(jax.random.PRNGKey(4), (5, base.in_features))

        # jit
        jitted = jax.jit(forward)
        assert jnp.isfinite(jitted(state, x))

        # grad wrt the LoRA state -> finite, and the B grad is nonzero.
        grads = jax.grad(forward)(state, x)
        flat = dict(nnx.to_flat_state(grads))
        assert jnp.all(jnp.isfinite(flat[("lora_b",)].value))
        assert jnp.any(flat[("lora_b",)].value != 0.0)

        # vmap over a batch of inputs.
        xs = jax.random.normal(jax.random.PRNGKey(5), (8, 5, base.in_features))
        out = jax.vmap(lambda single: forward(state, single))(xs)
        assert out.shape == (8,)
        assert jnp.all(jnp.isfinite(out))
