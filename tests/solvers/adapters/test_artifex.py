"""Tests for :class:`ArtifexSolverAdapter`.

The Avitai Artifex generative backend exposes no stable ``DDPMModel`` with a
``sample(rngs, condition, num_samples)`` interface (the only ``sample`` methods
live on modality wrappers with an incompatible ``sample(n_samples, **kwargs)``
signature, and ``artifex.generative_models.models`` does not even import). The
adapter therefore follows the canonical "not wired" convention used elsewhere in
opifex (e.g. ``opifex.platform.registry.core`` and the inference-backend
sampler hooks) and raises :class:`NotImplementedError` from both entry points
rather than returning a placeholder ``Solution(converged=True)`` (a
Principle-of-Least-Astonishment trap) or an implicit ``None``.
"""

from unittest.mock import MagicMock

import pytest
from flax import nnx

from opifex.core.problems import create_optimization_problem
from opifex.solvers.adapters.artifex import ArtifexSolverAdapter


def test_artifex_adapter_initialization() -> None:
    """Initialisation stores the supplied generative model unchanged."""
    mock_model = MagicMock()
    adapter = ArtifexSolverAdapter(mock_model)
    assert adapter.model is mock_model


def test_artifex_adapter_solve_raises_not_implemented() -> None:
    """``solve`` raises ``NotImplementedError`` until the backend is wired.

    It must NOT return a dummy ``Solution(converged=True)``: a no-op that
    reports convergence is a silent-success trap.
    """
    adapter = ArtifexSolverAdapter(MagicMock())
    problem = create_optimization_problem(1, lambda x: x)

    with pytest.raises(NotImplementedError, match="artifex solver adapter not implemented"):
        adapter.solve(problem)


def test_artifex_adapter_sample_batch_raises_not_implemented() -> None:
    """``sample_batch`` raises ``NotImplementedError`` instead of returning ``None``."""
    adapter = ArtifexSolverAdapter(MagicMock())
    problem = create_optimization_problem(1, lambda x: x)

    with pytest.raises(NotImplementedError, match="artifex solver adapter not implemented"):
        adapter.sample_batch(problem, num_samples=4, rngs=nnx.Rngs(0))
