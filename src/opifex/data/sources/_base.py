"""Shared base class for Grain-compliant on-demand PDE data sources.

All Opifex PDE data sources generate samples lazily from an integer index and must
be deterministic: the same index always yields the same sample. They therefore share
an identical index-validation and deterministic-key-derivation step. This module
centralises that logic in :class:`GrainPDESource` so the concrete sources only
implement their physics-specific sample generation.
"""

from typing import SupportsIndex

import grain.python as grain
import jax


class GrainPDESource(grain.RandomAccessDataSource):
    """Base class for deterministic, lazily-generated Grain PDE data sources.

    Subclasses must set the ``n_samples`` and ``seed`` attributes in their
    ``__init__`` and implement ``__getitem__`` for their specific PDE. They should
    call :meth:`_resolve_key` at the top of ``__getitem__`` to validate the incoming
    index and obtain the per-sample PRNG key.

    Attributes:
        n_samples: Total number of samples the source can generate.
        seed: Base random seed; per-sample keys are derived as ``seed + index``.
    """

    n_samples: int
    seed: int

    def __len__(self) -> int:
        """Return the total number of samples in the data source."""
        return self.n_samples

    def _resolve_key(self, index: SupportsIndex | slice) -> tuple[int, jax.Array]:
        """Validate ``index`` and derive its deterministic PRNG key.

        Args:
            index: Sample index requested via ``__getitem__``.

        Returns:
            Tuple of the validated integer index and its deterministic
            ``jax.random.PRNGKey(self.seed + index)`` key.

        Raises:
            TypeError: If ``index`` is a slice or not an integer.
            IndexError: If ``index`` is out of bounds.
        """
        if isinstance(index, slice):
            raise TypeError("Slicing not supported, use integer index")

        if not isinstance(index, int):
            raise TypeError(f"Index must be an integer, got {type(index)}")

        if index < 0 or index >= self.n_samples:
            raise IndexError(
                f"Index {index} out of bounds for source with {self.n_samples} samples"
            )

        key = jax.random.PRNGKey(self.seed + index)
        return index, key
