r"""Direct-evaluation GP kernels — RBF + Matern12/32/52 + ICM.

These kernels return the dense Gram matrix ``K(X_1, X_2)`` and are
consumed by the exact-GP fit/predict driver in
:mod:`opifex.uncertainty.gp.exact`. They are **distinct** from the
SDE-state-space matern kernels in
:mod:`opifex.uncertainty.statespace.kernels` which return the
``(F, L, Q_c, H, P_∞)`` quadruple required by the Kalman filter.

Kernel signatures
-----------------

All kernels follow the common signature

.. code-block:: python

    kernel(x1: jax.Array, x2: jax.Array, *, lengthscale: float,
           output_scale: float) -> jax.Array

returning a ``(n, m)`` Gram matrix for inputs ``(n, d)`` and
``(m, d)``. The ``kernel_fn`` parameter on
:func:`opifex.uncertainty.gp.fit_exact_gp` accepts any callable with
this shape; the ICM wrapper produced by
:func:`multi_output_icm_kernel` matches it exactly.

References
----------
* Rasmussen, C. E., Williams, C. K. I. 2006 — *Gaussian Processes for
  Machine Learning*, MIT Press; §4.2 (Matern family).
* Alvarez, M. A., Rosasco, L., Lawrence, N. D. 2012 — *Kernels for
  Vector-Valued Functions: A Review*, Foundations and Trends in
  Machine Learning vol. 4 no. 3, arXiv:1106.6251 (ICM).
"""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003 — kept eager for consistency

import jax
import jax.numpy as jnp


def _pairwise_distances(x1: jax.Array, x2: jax.Array) -> jax.Array:
    """Return the ``(n, m)`` Euclidean pairwise-distance matrix."""
    diff = x1[:, None, :] - x2[None, :, :]
    sq_distance = jnp.sum(diff**2, axis=-1)
    return jnp.sqrt(jnp.clip(sq_distance, a_min=0.0))


def _require_positive_kernel_hparams(*, lengthscale: float, output_scale: float) -> None:
    if lengthscale <= 0.0:
        raise ValueError(f"lengthscale must be strictly positive; got {lengthscale!r}.")
    if output_scale <= 0.0:
        raise ValueError(f"output_scale must be strictly positive; got {output_scale!r}.")


def matern12_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    lengthscale: float,
    output_scale: float,
) -> jax.Array:
    r"""Matern-1/2 (exponential / Ornstein-Uhlenbeck) kernel.

    .. math::

        k(r) = \sigma_{f}^{2}\,\exp\!\left(-\frac{r}{\ell}\right),
        \quad r = \lVert x_{1} - x_{2} \rVert_{2}.
    """
    _require_positive_kernel_hparams(lengthscale=lengthscale, output_scale=output_scale)
    r = _pairwise_distances(x1, x2)
    return (output_scale**2) * jnp.exp(-r / lengthscale)


def matern32_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    lengthscale: float,
    output_scale: float,
) -> jax.Array:
    r"""Matern-3/2 kernel.

    .. math::

        k(r) = \sigma_{f}^{2}\,
            \left(1 + \frac{\sqrt{3}\,r}{\ell}\right)
            \exp\!\left(-\frac{\sqrt{3}\,r}{\ell}\right).
    """
    _require_positive_kernel_hparams(lengthscale=lengthscale, output_scale=output_scale)
    sqrt3 = jnp.sqrt(jnp.asarray(3.0))
    scaled_r = sqrt3 * _pairwise_distances(x1, x2) / lengthscale
    return (output_scale**2) * (1.0 + scaled_r) * jnp.exp(-scaled_r)


def matern52_kernel(
    x1: jax.Array,
    x2: jax.Array,
    *,
    lengthscale: float,
    output_scale: float,
) -> jax.Array:
    r"""Matern-5/2 kernel.

    .. math::

        k(r) = \sigma_{f}^{2}\,
            \left(1 + \frac{\sqrt{5}\,r}{\ell}
                  + \frac{5\,r^{2}}{3\,\ell^{2}}\right)
            \exp\!\left(-\frac{\sqrt{5}\,r}{\ell}\right).
    """
    _require_positive_kernel_hparams(lengthscale=lengthscale, output_scale=output_scale)
    sqrt5 = jnp.sqrt(jnp.asarray(5.0))
    r = _pairwise_distances(x1, x2)
    scaled_r = sqrt5 * r / lengthscale
    poly = 1.0 + scaled_r + (5.0 / 3.0) * (r / lengthscale) ** 2
    return (output_scale**2) * poly * jnp.exp(-scaled_r)


def multi_output_icm_kernel(
    *,
    base_kernel_fn: Callable[..., jax.Array],
    coregionalisation: jax.Array,
) -> Callable[..., jax.Array]:
    r"""Intrinsic Coregionalisation Model (ICM) multi-output kernel.

    Wraps a scalar base kernel and a coregionalisation matrix
    ``B ∈ R^{T × T}`` into a multi-output kernel of the form
    ``k_ICM((x, i), (x', j)) = k_base(x, x') · B[i, j]`` (Alvarez,
    Rosasco, Lawrence 2012, §3.1).

    The inputs to the returned callable are arrays whose **last column**
    is an integer-valued task index in ``{0, …, T-1}`` and whose
    preceding columns are the spatial / feature coordinates that
    ``base_kernel_fn`` consumes. ``coregionalisation`` must be a
    square ``(T, T)`` array.

    Args:
        base_kernel_fn: Scalar kernel callable
            ``(x1_features, x2_features, *, lengthscale, output_scale) -> (n, m)``.
        coregionalisation: ``B`` matrix of shape ``(T, T)`` (typically
            ``B = W W^T + diag(κ)`` for low-rank plus diagonal
            structure).

    Returns:
        A callable with the same signature as ``base_kernel_fn`` but
        consuming task-indexed inputs.

    Raises:
        ValueError: If ``coregionalisation`` is not square.
    """
    if coregionalisation.ndim != 2 or coregionalisation.shape[0] != coregionalisation.shape[1]:
        raise ValueError(
            "coregionalisation must be a square (T, T) matrix; "
            f"got shape {coregionalisation.shape}."
        )

    def _icm(
        x1: jax.Array,
        x2: jax.Array,
        *,
        lengthscale: float,
        output_scale: float,
    ) -> jax.Array:
        features1, task1 = x1[:, :-1], x1[:, -1].astype(jnp.int32)
        features2, task2 = x2[:, :-1], x2[:, -1].astype(jnp.int32)
        base = base_kernel_fn(
            features1, features2, lengthscale=lengthscale, output_scale=output_scale
        )
        cor = coregionalisation[task1[:, None], task2[None, :]]
        return base * cor

    return _icm


def multi_output_lcm_kernel(
    *,
    components: tuple[tuple[Callable[..., jax.Array], jax.Array], ...],
) -> Callable[..., jax.Array]:
    r"""Linear Coregionalisation Model (LCM) multi-output kernel.

    Generalises ICM to a *sum* of base-kernel / coregionalisation pairs
    (Alvarez, Rosasco, Lawrence 2012 §3.2):

    .. math::

        k_{LCM}((x, i), (x', j)) = \sum_{q} k_{q}(x, x')\,B_{q}[i, j].

    Each ``B_q`` may be low-rank (rank-1 components reproduce the
    classical co-kriging linear model) or full-rank. With ``Q = 1`` the
    LCM collapses to a single ICM block.

    Args:
        components: A tuple of ``(base_kernel_fn, coregionalisation)``
            pairs. Each ``coregionalisation`` must be a square
            ``(T, T)`` array; all components must share the same
            ``T``.

    Returns:
        A callable matching the standard kernel signature
        ``(x1, x2, *, lengthscale, output_scale) -> Gram``.

    Raises:
        ValueError: If ``components`` is empty, if any
            ``coregionalisation`` is non-square, or if the components
            disagree on ``T``.
    """
    if len(components) == 0:
        raise ValueError("multi_output_lcm_kernel requires at least one component.")
    first_shape = components[0][1].shape
    if len(first_shape) != 2 or first_shape[0] != first_shape[1]:
        raise ValueError(
            f"Each LCM coregionalisation must be a square (T, T) matrix; got {first_shape}."
        )
    for _, cor in components[1:]:
        if cor.shape != first_shape:
            raise ValueError(
                "All LCM components must share the same number of tasks T; "
                f"got mismatched shapes {first_shape} and {cor.shape}."
            )

    def _lcm(
        x1: jax.Array,
        x2: jax.Array,
        *,
        lengthscale: float,
        output_scale: float,
    ) -> jax.Array:
        features1, task1 = x1[:, :-1], x1[:, -1].astype(jnp.int32)
        features2, task2 = x2[:, :-1], x2[:, -1].astype(jnp.int32)
        total = jnp.zeros((x1.shape[0], x2.shape[0]))
        for base_fn, cor in components:
            base = base_fn(features1, features2, lengthscale=lengthscale, output_scale=output_scale)
            total = total + base * cor[task1[:, None], task2[None, :]]
        return total

    return _lcm


def additive_kernel(
    *,
    component_kernel_fns: tuple[Callable[..., jax.Array], ...],
) -> Callable[..., jax.Array]:
    r"""Additive (OAK-base) kernel — sum of per-dimension univariate kernels.

    .. math::

        k_{\text{add}}(x, x') = \sum_{d=1}^{D} k_{d}(x_{d}, x'_{d}).

    Each ``component_kernel_fns[d]`` is evaluated on the ``d``-th input
    dimension only; the returned kernel has the standard
    ``(x1, x2, *, lengthscale, output_scale) -> Gram`` signature. This
    is the **first-order** (``max_order = 1``) case of the
    *Orthogonal Additive Kernel* (Lu, Boukouvalas, Hensman 2022 ICML —
    *Additive Gaussian Processes Revisited*). Higher-order interactions
    via Newton-Girard recursion + the Gaussian-measure orthogonality
    constraint are deferred to a follow-up slice; the first-order form
    already enables ANOVA-style interpretable GPs.

    Args:
        component_kernel_fns: One kernel callable per input dimension.
            All callables share the standard kernel signature.

    Returns:
        A callable matching the standard kernel signature.

    Raises:
        ValueError: If ``component_kernel_fns`` is empty.
    """
    if len(component_kernel_fns) == 0:
        raise ValueError("additive_kernel requires at least one component kernel.")

    def _additive(
        x1: jax.Array,
        x2: jax.Array,
        *,
        lengthscale: float,
        output_scale: float,
    ) -> jax.Array:
        if x1.shape[-1] != len(component_kernel_fns):
            raise ValueError(
                "Input dimensionality must equal the number of components; "
                f"got {x1.shape[-1]} dims but {len(component_kernel_fns)} components."
            )
        total = jnp.zeros((x1.shape[0], x2.shape[0]))
        for dim_index, component_fn in enumerate(component_kernel_fns):
            total = total + component_fn(
                x1[:, dim_index : dim_index + 1],
                x2[:, dim_index : dim_index + 1],
                lengthscale=lengthscale,
                output_scale=output_scale,
            )
        return total

    return _additive


def deep_kernel(
    *,
    feature_map: Callable[[jax.Array], jax.Array],
    base_kernel_fn: Callable[..., jax.Array],
) -> Callable[..., jax.Array]:
    r"""Deep-kernel composition ``k_deep(x, x') = k_base(φ(x), φ(x'))``.

    Wilson, Hu, Salakhutdinov, Xing 2016 (arXiv:1511.02222) introduced
    *Deep Kernel Learning*: combine a learnable neural feature map
    ``φ_θ`` with any base kernel ``k_base`` so the resulting kernel
    inherits the base hyperparameters while the NN provides the
    representation.

    The opifex composition is a thin wrapper over the existing
    ``kernel_fn`` API: any callable mapping ``(n, d) -> (n, d')``
    qualifies as the feature map — Python lambdas, ``flax.nnx``
    modules, ``nnx.Sequential`` extractors, etc. **No equinox
    dependency**: opifex uses ``flax.nnx`` for the NN component
    (matching the rest of the opifex neural backbone).

    Args:
        feature_map: Callable that lifts ``(n, d) -> (n, d')``.
        base_kernel_fn: Standard kernel callable
            ``(x1, x2, *, lengthscale, output_scale) -> (n, m)``.

    Returns:
        A callable matching the standard kernel signature.
    """

    def _deep_kernel(
        x1: jax.Array,
        x2: jax.Array,
        *,
        lengthscale: float,
        output_scale: float,
    ) -> jax.Array:
        return base_kernel_fn(
            feature_map(x1),
            feature_map(x2),
            lengthscale=lengthscale,
            output_scale=output_scale,
        )

    return _deep_kernel


__all__ = [
    "additive_kernel",
    "deep_kernel",
    "matern12_kernel",
    "matern32_kernel",
    "matern52_kernel",
    "multi_output_icm_kernel",
    "multi_output_lcm_kernel",
]
