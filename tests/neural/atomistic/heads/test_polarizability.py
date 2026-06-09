r"""Tests for the :class:`PolarizabilityHead` molecular-polarizability readout.

Load-bearing physics contracts:

* the head emits ``{"polarizability"}`` -- a symmetric ``3x3`` Cartesian tensor;
* the polarizability decomposes into an isotropic (:math:`l = 0`) scalar part
  :math:`\alpha_\mathrm{iso}\,\mathbf{I}` plus a symmetric-traceless
  (:math:`l = 2`) anisotropic part built from charge-weighted position outer
  products :math:`q_i\,(3\,\mathbf{r}_i\mathbf{r}_i^\top - |\mathbf{r}_i|^2\mathbf{I})`
  (Schuett, Unke & Gastegger 2021, PaiNN tensorial readout; MACE polarizability);
* the tensor is **rotationally equivariant**: rotating the positions by ``R``
  conjugates the polarizability, :math:`\alpha(R\mathbf{r}) = R\,\alpha(\mathbf{r})\,R^\top`;
* an isotropic-only configuration yields a multiple of the identity;
* the forward pass is ``jit``/``grad``/``vmap`` clean.

Tight numerical assertions run under ``jax.enable_x64(True)`` (the conftest
defaults ``jax_enable_x64`` to ``False``).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array  # noqa: TC002

from opifex.core.quantum.molecular_system import MolecularSystem
from opifex.core.quantum.registry import PropertyHeadRegistry
from opifex.geometry.algebra import SO3Group
from opifex.neural.atomistic.heads.polarizability import PolarizabilityHead


_FEATURE_DIM = 8
_GRAPH: tuple[Array, Array] = (jnp.asarray([0, 1, 2]), jnp.asarray([1, 2, 0]))
_ROTATION_KEY = jax.random.PRNGKey(11)


def _positions() -> Array:
    """Return a non-symmetric 3-atom geometry (float64)."""
    return jnp.asarray([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]], dtype=jnp.float64)


def _system(positions: Array | None = None) -> MolecularSystem:
    """Return a 3-atom (water-like) system."""
    return MolecularSystem(
        atomic_numbers=jnp.asarray([8, 1, 1]),
        positions=_positions() if positions is None else positions,
    )


def _embeddings(seed: int = 0) -> dict[str, Array]:
    """Return per-atom invariant ``node_features`` for a 3-atom system."""
    key = jax.random.PRNGKey(seed)
    return {"node_features": jax.random.normal(key, (3, _FEATURE_DIM), dtype=jnp.float64)}


class TestPolarizabilityHead:
    def test_implemented_properties(self) -> None:
        head = PolarizabilityHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        assert head.implemented_properties == ("polarizability",)

    def test_polarizability_shape(self) -> None:
        head = PolarizabilityHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        alpha = head(_system(), _GRAPH, _embeddings())["polarizability"]
        assert alpha.shape == (3, 3)

    def test_polarizability_symmetric(self) -> None:
        with jax.enable_x64(True):
            head = PolarizabilityHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
            alpha = head(_system(), _GRAPH, _embeddings())["polarizability"]
            assert jnp.allclose(alpha, alpha.T, atol=1e-9)

    def test_polarizability_rotational_equivariance(self) -> None:
        with jax.enable_x64(True):
            head = PolarizabilityHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
            embeddings = _embeddings()
            rotation = SO3Group().random_element(_ROTATION_KEY).astype(jnp.float64)
            base = head(_system(), _GRAPH, embeddings)["polarizability"]
            rotated_positions = _positions() @ rotation.T
            moved = head(_system(positions=rotated_positions), _GRAPH, embeddings)["polarizability"]
            assert jnp.allclose(moved, rotation @ base @ rotation.T, atol=1e-5)

    def test_isotropic_only_is_multiple_of_identity(self) -> None:
        with jax.enable_x64(True):
            head = PolarizabilityHead(
                feature_dim=_FEATURE_DIM, isotropic_only=True, rngs=nnx.Rngs(0)
            )
            alpha = head(_system(), _GRAPH, _embeddings())["polarizability"]
            scalar = alpha[0, 0]
            assert jnp.allclose(alpha, scalar * jnp.eye(3), atol=1e-9)

    def test_registered_under_name(self) -> None:
        assert PropertyHeadRegistry().require("polarizability") is PolarizabilityHead

    def test_jit_grad_vmap_smoke(self) -> None:
        head = PolarizabilityHead(feature_dim=_FEATURE_DIM, rngs=nnx.Rngs(0))
        graphdef, state = nnx.split(head)
        node_features = _embeddings()["node_features"]

        def alpha_for(positions: Array) -> Array:
            rebuilt = nnx.merge(graphdef, state)
            system = MolecularSystem(atomic_numbers=jnp.asarray([8, 1, 1]), positions=positions)
            return rebuilt(system, _GRAPH, {"node_features": node_features})["polarizability"]

        jitted = jax.jit(alpha_for)
        alpha = jitted(_positions())
        assert alpha.shape == (3, 3)
        assert bool(jnp.all(jnp.isfinite(alpha)))

        gradient = jax.grad(lambda p: jnp.sum(alpha_for(p) ** 2))(_positions())
        assert gradient.shape == _positions().shape
        assert bool(jnp.all(jnp.isfinite(gradient)))

        batch = jnp.stack([_positions(), _positions() + 0.1])
        batched = jax.vmap(alpha_for)(batch)
        assert batched.shape == (2, 3, 3)
        assert bool(jnp.all(jnp.isfinite(batched)))
