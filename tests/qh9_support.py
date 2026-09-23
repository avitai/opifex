"""Shared access to the real QH9-Stable database for the tests that use it.

The database is large and local-only, so every test that reads it skips when it is absent;
the ones that compare against PySCF also skip when ``pyscf.dft`` is not installed.
"""

import importlib.util
from pathlib import Path

import pytest

from opifex.data.sources.qh9_source import QH9Example, read_qh9_sqlite


QH9_DB = Path("/mnt/ssd2/Data/qh9/raw/QH9Stable.db")
"""The QH9-Stable SQLite database (Yu et al. 2023) on the development machine."""

requires_qh9_db = pytest.mark.skipif(not QH9_DB.exists(), reason="QH9Stable.db not available")


def _has_pyscf_dft() -> bool:
    """Whether ``pyscf.dft`` is importable.

    ``find_spec`` raises rather than returning ``None`` for a submodule whose parent package
    is absent, so the lookup is guarded.
    """
    try:
        return importlib.util.find_spec("pyscf.dft") is not None
    except ModuleNotFoundError:
        return False


requires_pyscf_dft = pytest.mark.skipif(not _has_pyscf_dft(), reason="pyscf.dft not available")


def first_qh9_examples(n: int) -> list[QH9Example]:
    """Decode the first ``n`` QH9-Stable molecules, in id order.

    Args:
        n: How many molecules to decode.

    Returns:
        The decoded molecules.
    """
    return list(read_qh9_sqlite(QH9_DB, limit=n))
