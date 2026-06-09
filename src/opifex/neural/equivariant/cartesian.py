r"""Cartesian-tensor equivariant features (a CG-free alternative to irreps).

A native, ``jax``/``flax.nnx``-compatible implementation of the rank-2 *Cartesian
tensor* representation introduced by **TensorNet** (Simeon & de Fabritiis 2023,
arXiv:2306.06482) and generalised by **HotPP** (Wang et al. 2024,
arXiv:2402.15286). It is a drop-in *alternative* to the Clebsch-Gordan /
:class:`~opifex.neural.equivariant.IrrepsArray` path: instead of storing
spherical-harmonic irreps and coupling them with Clebsch-Gordan tensors, an
``l <= 2`` feature is carried as a single rank-2 Cartesian tensor
``X in R^{3x3}`` that transforms as

.. math::

    X \;\longrightarrow\; R\,X\,R^{\mathsf T}

under a rotation ``R in SO(3)``. TensorNet (Eq. 1-4) decomposes such a tensor
into three *irreducible Cartesian* parts that each carry a single ``O(3)`` irrep,

.. math::

    X = \underbrace{\tfrac{1}{3}\operatorname{tr}(X)\,I}_{\text{scalar }(l=0)}
      + \underbrace{\tfrac{1}{2}(X - X^{\mathsf T})}_{\text{antisymmetric }(l=1)}
      + \underbrace{\big[\tfrac{1}{2}(X + X^{\mathsf T})
          - \tfrac{1}{3}\operatorname{tr}(X)\,I\big]}_{\text{symmetric-traceless }(l=2)},

and exploits that the matrix product of two rank-2 tensors is equivariant,

.. math::

    (R\,X\,R^{\mathsf T})(R\,Y\,R^{\mathsf T}) = R\,(X\,Y)\,R^{\mathsf T},

so a *tensor product* needs no Clebsch-Gordan tensor -- it is a plain ``X @ Y``.
This keeps the building block runtime-competitive at ``l <= 2`` (three small
matrix ops instead of the CG einsum of
:class:`~opifex.neural.equivariant.FullyConnectedTensorProduct`).

Interoperability with the irreps kit is provided by :func:`to_irreps_array` /
:func:`from_irreps_array`, which convert a :class:`CartesianTensor` to and from
the ``1x0e + 1x1o + 1x2e`` :class:`~opifex.neural.equivariant.IrrepsArray`
layout. The change-of-basis between the symmetric-traceless ``3x3`` block and the
five ``l = 2`` spherical-harmonic components is *derived from* the existing
:func:`opifex.neural.equivariant.spherical_harmonics.spherical_harmonics` basis
(never hand-tabulated), so the conversion is exactly consistent with
:func:`opifex.geometry.algebra.wigner.wigner_d`: rotating the Cartesian tensor
rotates the ``l = 1`` / ``l = 2`` blocks by ``D^1(R)`` / ``D^2(R)``.

Scope: rank-0/1/2 only (``l <= 2``); higher-order Cartesian tensors (HotPP's
rank-``>= 3`` path) are intentionally out of scope here.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.neural.equivariant.irreps import Irreps, IrrepsArray
from opifex.neural.equivariant.spherical_harmonics import spherical_harmonics


#: The irreps layout a rank-2 Cartesian tensor decomposes into (``l <= 2``).
CARTESIAN_IRREPS = Irreps("1x0e+1x1o+1x2e")


@functools.cache
def _symmetric_to_l2_basis() -> np.ndarray:
    r"""Change-of-basis ``W`` mapping ``vec(S) -> Y_2``, derived from the SH basis.

    ``S`` is a symmetric-traceless ``3x3`` tensor (flattened row-major to a
    ``9``-vector) and ``Y_2`` is the five-component ``l = 2`` block in the e3nn
    real-spherical-harmonic basis used by
    :func:`opifex.neural.equivariant.spherical_harmonics.spherical_harmonics` and
    :func:`opifex.geometry.algebra.wigner.wigner_d`.

    The matrix is obtained by least-squares fitting ``Y_2(r)`` (with
    ``normalization="norm"``, so ``Y_2`` is a pure quadratic form in ``r``)
    against the symmetric-traceless part of the outer product ``r (x) r`` over
    random directions. Because both sides are exact quadratic forms the fit is
    exact (residual ``~1e-14``); deriving it from the canonical SH evaluator --
    rather than hand-tabulating the ``sqrt(3)`` coefficients -- guarantees the
    conversion is consistent with the rest of the equivariant kit.

    Returns:
        Real array ``W`` of shape ``(5, 9)``.
    """
    rng = np.random.default_rng(0)
    directions = rng.standard_normal((400, 3))
    symmetric_rows: list[np.ndarray] = []
    harmonic_rows: list[np.ndarray] = []
    for direction in directions:
        outer = np.outer(direction, direction)
        symmetric = outer - np.trace(outer) / 3.0 * np.eye(3)
        symmetric_rows.append(symmetric.reshape(9))
        harmonic = np.asarray(
            spherical_harmonics(
                2, jnp.asarray(direction), normalize=False, normalization="norm"
            ).array
        )[4:9]
        harmonic_rows.append(harmonic)
    design = np.asarray(symmetric_rows)
    targets = np.asarray(harmonic_rows)
    solution, *_ = np.linalg.lstsq(design, targets, rcond=None)
    weight = solution.T  # (5, 9): Y_2 = W @ vec(S)
    residual = float(np.abs(design @ weight.T - targets).max(initial=0.0))
    if residual > 1e-8:
        raise ValueError(
            f"Symmetric-traceless -> l=2 change of basis is inconsistent "
            f"(max residual {residual:.3e}); the SH basis may have changed."
        )
    return np.ascontiguousarray(weight)


@functools.cache
def _l2_to_symmetric_basis() -> np.ndarray:
    r"""Pseudo-inverse change-of-basis ``W^+`` mapping ``Y_2 -> vec(S)``.

    The right inverse of :func:`_symmetric_to_l2_basis` restricted to the
    symmetric-traceless subspace; shape ``(9, 5)``.

    Returns:
        Real array of shape ``(9, 5)``.
    """
    forward = _symmetric_to_l2_basis()  # (5, 9)
    return np.ascontiguousarray(np.linalg.pinv(forward))


def _antisymmetric_to_vector(tensor: Float[Array, "... 3 3"]) -> Float[Array, "... 3"]:
    r"""Dual axial vector of the antisymmetric part ``(X - X^T)/2``.

    Uses the convention ``v = (A[2,1], A[0,2], A[1,0])`` with
    ``A = (X - X^T)/2``; this vector transforms as ``v -> R v`` (``= D^1(R) v``
    in the e3nn ``l = 1`` basis, which here coincides with plain Cartesian
    ``(x, y, z)``) when ``X -> R X R^T`` and ``det R = +1``.
    """
    antisymmetric = 0.5 * (tensor - jnp.swapaxes(tensor, -1, -2))
    return jnp.stack(
        [antisymmetric[..., 2, 1], antisymmetric[..., 0, 2], antisymmetric[..., 1, 0]],
        axis=-1,
    )


def _vector_to_antisymmetric(vector: Float[Array, "... 3"]) -> Float[Array, "... 3 3"]:
    r"""Inverse of :func:`_antisymmetric_to_vector`: build ``A`` from its axial vector."""
    zero = jnp.zeros_like(vector[..., 0])
    component_x = vector[..., 0]
    component_y = vector[..., 1]
    component_z = vector[..., 2]
    row0 = jnp.stack([zero, -component_z, component_y], axis=-1)
    row1 = jnp.stack([component_z, zero, -component_x], axis=-1)
    row2 = jnp.stack([-component_y, component_x, zero], axis=-1)
    return jnp.stack([row0, row1, row2], axis=-2)


@jax.tree_util.register_pytree_node_class
class CartesianTensor:
    r"""A rank-2 Cartesian tensor feature transforming as ``X -> R X R^T``.

    The underlying array has shape ``(..., 3, 3)`` (arbitrary leading / channel
    dimensions). The object is a pytree whose single child is the array, so it
    flows through ``jit`` / ``grad`` / ``vmap`` unchanged. This is the
    TensorNet (arXiv:2306.06482) representation: an ``l <= 2`` equivariant
    feature stored as one Cartesian tensor rather than a spherical-harmonic
    :class:`~opifex.neural.equivariant.IrrepsArray`.
    """

    __slots__ = ("array",)

    array: Float[Array, "... 3 3"]

    def __init__(self, array: Float[Array, "... 3 3"]) -> None:
        """Wrap a ``(..., 3, 3)`` array, validating the trailing tensor shape."""
        if array.shape[-2:] != (3, 3):
            raise ValueError(
                f"CartesianTensor expects a (..., 3, 3) array, got shape {array.shape}"
            )
        self.array = array

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array."""
        return self.array.shape

    def rotate(self, rotation: Float[Array, "3 3"]) -> CartesianTensor:
        r"""Apply the equivariant action ``X -> R X R^T`` for a rotation ``R``."""
        rotated = jnp.einsum("ab,...bc,dc->...ad", rotation, self.array, rotation)
        return CartesianTensor(rotated)

    def tensor_product(self, other: CartesianTensor) -> CartesianTensor:
        r"""Equivariant tensor product via the matrix product ``X @ Y``.

        Equivariant because ``(R X R^T)(R Y R^T) = R (X Y) R^T`` (TensorNet,
        Eq. 5). No Clebsch-Gordan tensor is required.
        """
        return CartesianTensor(self.array @ other.array)

    def decompose(
        self,
    ) -> tuple[Float[Array, ""], Float[Array, "... 3"], Float[Array, "... 3 3"]]:
        r"""Decompose into irreducible Cartesian parts ``(scalar, vector, symmetric)``.

        Following TensorNet Eq. 1-4:

        * ``scalar`` -- the trace ``tr(X)`` (an ``l = 0`` invariant);
        * ``vector`` -- the axial vector dual to the antisymmetric part
          ``(X - X^T)/2`` (an ``l = 1`` feature, rotating by ``R``);
        * ``symmetric`` -- the symmetric-traceless part
          ``(X + X^T)/2 - tr(X)/3 * I`` (an ``l = 2`` feature, rotating by
          ``R . R^T``).

        Returns:
            A ``(scalar, vector, symmetric)`` triple with shapes
            ``(...,)``, ``(..., 3)`` and ``(..., 3, 3)``.
        """
        trace = jnp.trace(self.array, axis1=-2, axis2=-1)
        vector = _antisymmetric_to_vector(self.array)
        symmetric_full = 0.5 * (self.array + jnp.swapaxes(self.array, -1, -2))
        symmetric = symmetric_full - trace[..., None, None] / 3.0 * jnp.eye(
            3, dtype=self.array.dtype
        )
        return trace, vector, symmetric

    @classmethod
    def from_parts(
        cls,
        scalar: Float[Array, ""],
        vector: Float[Array, "... 3"],
        symmetric: Float[Array, "... 3 3"],
    ) -> CartesianTensor:
        r"""Assemble a Cartesian tensor from its irreducible parts.

        Inverse of :meth:`decompose`: ``X = scalar/3 * I + A(vector) + symmetric``
        where ``A(vector)`` is the antisymmetric matrix with axial vector
        ``vector`` and ``symmetric`` is the symmetric-traceless ``l = 2`` part.

        Args:
            scalar: The ``l = 0`` trace, shape ``(...,)``.
            vector: The ``l = 1`` axial vector, shape ``(..., 3)``.
            symmetric: The ``l = 2`` symmetric-traceless tensor, shape
                ``(..., 3, 3)``.

        Returns:
            The reconstructed :class:`CartesianTensor`.
        """
        identity = jnp.eye(3, dtype=symmetric.dtype)
        isotropic = scalar[..., None, None] / 3.0 * identity
        antisymmetric = _vector_to_antisymmetric(vector)
        return cls(isotropic + antisymmetric + symmetric)

    def tree_flatten(self) -> tuple[tuple[Float[Array, "... 3 3"]], None]:
        """Pytree flatten: the array is the single child, no aux-data."""
        return (self.array,), None

    @classmethod
    def tree_unflatten(
        cls, _aux_data: None, children: tuple[Float[Array, "... 3 3"]]
    ) -> CartesianTensor:
        """Pytree unflatten, bypassing shape re-validation for traced leaves."""
        instance = object.__new__(cls)
        instance.array = children[0]
        return instance

    def __repr__(self) -> str:
        """Return a short representation with the array shape."""
        return f"CartesianTensor(shape={self.array.shape})"


def to_irreps_array(tensor: CartesianTensor) -> IrrepsArray:
    r"""Convert a :class:`CartesianTensor` to an ``1x0e + 1x1o + 1x2e`` ``IrrepsArray``.

    The scalar (trace), vector (antisymmetric dual) and symmetric-traceless parts
    map to the ``l = 0`` / ``l = 1`` / ``l = 2`` blocks respectively. The ``l = 2``
    block uses the change-of-basis derived from the canonical spherical-harmonic
    evaluator (see :func:`_symmetric_to_l2_basis`), so the result is exactly
    consistent with :func:`opifex.geometry.algebra.wigner.wigner_d`: a rotation of
    ``tensor`` rotates the blocks by ``D^l(R)``.

    Args:
        tensor: A rank-2 Cartesian tensor feature with leading shape ``(...,)``.

    Returns:
        An :class:`~opifex.neural.equivariant.IrrepsArray` with layout
        ``CARTESIAN_IRREPS`` and array shape ``(..., 9)``.
    """
    scalar, vector, symmetric = tensor.decompose()
    weight = jnp.asarray(_symmetric_to_l2_basis(), dtype=tensor.array.dtype)
    leading = symmetric.shape[:-2]
    l2_block = jnp.einsum("ki,...i->...k", weight, symmetric.reshape(*leading, 9))
    flat = jnp.concatenate([scalar[..., None], vector, l2_block], axis=-1)
    return IrrepsArray(CARTESIAN_IRREPS, flat)


def from_irreps_array(features: IrrepsArray) -> CartesianTensor:
    r"""Convert an ``1x0e + 1x1o + 1x2e`` ``IrrepsArray`` back to a Cartesian tensor.

    Inverse of :func:`to_irreps_array`.

    Args:
        features: An :class:`~opifex.neural.equivariant.IrrepsArray` whose layout
            is exactly ``CARTESIAN_IRREPS``.

    Returns:
        The reconstructed :class:`CartesianTensor`.

    Raises:
        ValueError: If ``features.irreps`` is not ``CARTESIAN_IRREPS``.
    """
    if features.irreps != CARTESIAN_IRREPS:
        raise ValueError(
            f"from_irreps_array expects irreps {CARTESIAN_IRREPS!r}, got {features.irreps!r}"
        )
    array = features.array
    scalar = array[..., 0]
    vector = array[..., 1:4]
    l2_block = array[..., 4:9]
    inverse = jnp.asarray(_l2_to_symmetric_basis(), dtype=array.dtype)
    leading = array.shape[:-1]
    symmetric = jnp.einsum("ik,...k->...i", inverse, l2_block).reshape(*leading, 3, 3)
    return CartesianTensor.from_parts(scalar, vector, symmetric)


class CartesianLinear(nnx.Module):
    r"""Equivariant channel-mixing of stacked rank-2 Cartesian tensors.

    Given an input of shape ``(..., in_channels, 3, 3)`` the layer produces
    ``(..., out_channels, 3, 3)`` by mixing channels *separately* within each
    irreducible Cartesian part (scalar / vector / symmetric-traceless). Because
    each part carries a single ``O(3)`` irrep, an arbitrary learnable mixing over
    the channel axis -- applied independently per part -- commutes with the
    rotation action ``X -> R X R^T`` and is therefore equivariant (TensorNet uses
    the same per-irrep linear mixing, arXiv:2306.06482 Eq. 6).

    The three parts get independent ``(in_channels, out_channels)`` weight
    matrices, so the layer is the Cartesian analogue of
    :class:`~opifex.neural.equivariant.EquivariantLinear`.
    """

    def __init__(self, *, in_channels: int, out_channels: int, rngs: nnx.Rngs) -> None:
        """Build the per-part channel-mixing weights.

        Args:
            in_channels: Number of input Cartesian-tensor channels.
            out_channels: Number of output channels.
            rngs: Random number generators (keyword-only); ``rngs.params()``
                seeds the three weight matrices.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        key = rngs.params()
        scale = 1.0 / np.sqrt(in_channels)
        scalar_key, vector_key, symmetric_key = jax.random.split(key, 3)
        shape = (in_channels, out_channels)
        self.scalar_weight = nnx.Param(scale * jax.random.normal(scalar_key, shape))
        self.vector_weight = nnx.Param(scale * jax.random.normal(vector_key, shape))
        self.symmetric_weight = nnx.Param(scale * jax.random.normal(symmetric_key, shape))

    def __call__(self, tensors: CartesianTensor) -> CartesianTensor:
        """Mix channels equivariantly.

        Args:
            tensors: Input with array shape ``(..., in_channels, 3, 3)``.

        Returns:
            A :class:`CartesianTensor` with array shape ``(..., out_channels, 3, 3)``.

        Raises:
            ValueError: If the channel axis does not match ``in_channels``.
        """
        if tensors.array.shape[-3] != self.in_channels:
            raise ValueError(
                f"CartesianLinear expected {self.in_channels} input channels, "
                f"got {tensors.array.shape[-3]}"
            )
        scalar, vector, symmetric = tensors.decompose()
        dtype = tensors.array.dtype
        mixed_scalar = jnp.einsum("uw,...u->...w", self.scalar_weight[...].astype(dtype), scalar)
        mixed_vector = jnp.einsum("uw,...ui->...wi", self.vector_weight[...].astype(dtype), vector)
        mixed_symmetric = jnp.einsum(
            "uw,...uij->...wij", self.symmetric_weight[...].astype(dtype), symmetric
        )
        return CartesianTensor.from_parts(mixed_scalar, mixed_vector, mixed_symmetric)
