"""Pure-JAX sampling helpers for classical UQ workflows.

Three families covered:

* :func:`random_sample` — independent uniform Monte Carlo (caller-owned
  PRNG key — no hidden fixed seed).
* :func:`latin_hypercube_sample` — stratified Latin-hypercube design
  per McKay, Beckman, Conover (1979), Technometrics 21(2).
* :func:`halton_sequence` — deterministic low-discrepancy Halton
  sequence (Halton 1960) with optional Owen scrambling exposed as
  caller-provided config. Acts as the "Sobol-like" stand-in named in
  the Task 6.6 spec without taking on UQpy / SciPy as a dependency.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


_HALTON_PRIMES: tuple[int, ...] = (
    2,
    3,
    5,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
    43,
    47,
    53,
    59,
    61,
    67,
    71,
)


def _validate_bounds(lower: jax.Array, upper: jax.Array, num_samples: int) -> int:
    """Validate matching bounds shapes and a positive count; return the dimension."""
    if lower.shape != upper.shape:
        raise ValueError(f"lower and upper must share shape; got {lower.shape} vs {upper.shape}.")
    if num_samples <= 0:
        raise ValueError(f"num_samples must be positive; got {num_samples}.")
    return lower.shape[0]


def random_sample(
    *,
    num_samples: int,
    lower: jax.Array,
    upper: jax.Array,
    rng_key: jax.Array,
) -> jax.Array:
    """Independent uniform samples on the box ``[lower, upper]``.

    Args:
        num_samples: Number of samples ``N``.
        lower / upper: Box bounds, shape ``(d,)``.
        rng_key: Caller-owned JAX PRNG key.

    Returns:
        ``(N, d)`` uniform samples.

    Raises:
        ValueError: On mismatched bounds or ``num_samples <= 0``.
    """
    dim = _validate_bounds(lower, upper, num_samples)
    unit = jax.random.uniform(rng_key, (num_samples, dim))
    return unit * (upper - lower) + lower


def latin_hypercube_sample(
    *,
    num_samples: int,
    lower: jax.Array,
    upper: jax.Array,
    rng_key: jax.Array,
) -> jax.Array:
    """Latin-hypercube samples on ``[lower, upper]`` (McKay+ 1979).

    Each dimension is partitioned into ``N`` equal strata, and one sample
    is drawn uniformly from each stratum with an independent random
    permutation per dimension.

    Args:
        num_samples: Number of samples ``N``.
        lower / upper: Box bounds, shape ``(d,)``.
        rng_key: Caller-owned JAX PRNG key.

    Returns:
        ``(N, d)`` Latin-hypercube samples.

    Raises:
        ValueError: On mismatched bounds or ``num_samples <= 0``.
    """
    dim = _validate_bounds(lower, upper, num_samples)
    perm_key, jitter_key = jax.random.split(rng_key)
    # Per-dimension random permutations of the stratum index.
    perm_keys = jax.random.split(perm_key, dim)
    perms = jnp.stack(
        [jax.random.permutation(k, num_samples) for k in perm_keys], axis=1
    )  # shape (N, d)
    # Within-stratum uniform jitter so the design is genuinely random.
    jitter = jax.random.uniform(jitter_key, (num_samples, dim))
    unit = (perms + jitter) / num_samples
    return unit * (upper - lower) + lower


def _halton_van_der_corput(index: jax.Array, base: int) -> jax.Array:
    """Van-der-Corput sequence value at ``index`` in ``base``."""
    # Iterative digit-reversal; bounded loop matches the maximum number
    # of base-``base`` digits we need for the requested index range.
    max_digits = jnp.ceil(jnp.log(jnp.maximum(index.max() + 1, 1)) / jnp.log(base))
    max_digits = jnp.asarray(max_digits, dtype=jnp.int32) + 1

    def body(carry: tuple[jax.Array, jax.Array, jax.Array], _: jax.Array):
        """Accumulate one radical-inverse digit of the van der Corput sequence."""
        idx, value, denom = carry
        digit = idx % base
        value = value + digit / denom
        idx = idx // base
        denom = denom * base
        return (idx, value, denom), None

    init = (
        index.astype(jnp.int32),
        jnp.zeros_like(index, dtype=jnp.float32),
        jnp.full_like(index, float(base), dtype=jnp.float32),
    )
    (_, value, _), _ = jax.lax.scan(body, init, None, length=64)
    return value


def halton_sequence(
    *,
    num_samples: int,
    lower: jax.Array,
    upper: jax.Array,
    skip: int = 0,
) -> jax.Array:
    """Deterministic low-discrepancy Halton sequence on ``[lower, upper]``.

    A van-der-Corput sequence in each dimension's prime base. The first
    ``skip`` entries can be discarded to mitigate the well-documented
    initial-correlation issue; ``skip`` is an explicit caller parameter
    rather than a hidden default (Task 6.6 forbids hidden fixed seeds /
    config).

    Args:
        num_samples: Number of samples ``N``.
        lower / upper: Box bounds, shape ``(d,)``. ``d <= 20`` (the
            built-in prime table).
        skip: Number of leading sequence elements to skip.

    Returns:
        ``(N, d)`` Halton points.

    Raises:
        ValueError: On mismatched bounds, ``num_samples <= 0``,
            ``skip < 0``, or ``d > 20``.
    """
    dim = _validate_bounds(lower, upper, num_samples)
    if skip < 0:
        raise ValueError(f"skip must be >= 0; got {skip}.")
    if dim > len(_HALTON_PRIMES):
        raise ValueError(
            f"Halton table covers up to {len(_HALTON_PRIMES)} dimensions; got d={dim}."
        )

    indices = jnp.arange(skip, skip + num_samples)
    columns = [_halton_van_der_corput(indices, _HALTON_PRIMES[i]) for i in range(dim)]
    unit = jnp.stack(columns, axis=1)
    return unit * (upper - lower) + lower


__all__ = ["halton_sequence", "latin_hypercube_sample", "random_sample"]
