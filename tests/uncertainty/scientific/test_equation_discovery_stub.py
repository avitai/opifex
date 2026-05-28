"""Tests for the Step-10 equation-discovery stubs (Task 8.6)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.scientific.equation_discovery import (
    BayesianSINDyStub,
    CoefficientPosteriorIntervalStub,
    TermInclusionProbabilityStub,
)


_STEP_10_MESSAGE = r"Step 10 stub: see audit Migration Step 10"


def test_bayesian_sindy_stub_constructor_validates_arguments() -> None:
    library = (jnp.sin, jnp.cos)
    stub = BayesianSINDyStub(library=library, sparsity_threshold=0.1)
    assert stub.library == library
    assert stub.sparsity_threshold == 0.1
    assert stub.metadata == ()

    with pytest.raises(ValueError, match="library"):
        BayesianSINDyStub(library=(), sparsity_threshold=0.1)
    with pytest.raises(ValueError, match="sparsity_threshold"):
        BayesianSINDyStub(library=(jnp.sin,), sparsity_threshold=0.0)


def test_bayesian_sindy_stub_methods_raise_step10_canonical_message() -> None:
    stub = BayesianSINDyStub(library=(jnp.sin,), sparsity_threshold=0.1)
    with pytest.raises(NotImplementedError, match=_STEP_10_MESSAGE):
        stub.fit(jnp.zeros((4, 2)), jnp.zeros((4, 2)), rngs=nnx.Rngs(0))
    with pytest.raises(NotImplementedError, match=_STEP_10_MESSAGE):
        stub.term_inclusion_probabilities()


def test_term_inclusion_probability_stub_validates_term_names() -> None:
    stub = TermInclusionProbabilityStub(term_names=("x", "x_squared"))
    assert stub.term_names == ("x", "x_squared")
    with pytest.raises(ValueError, match="term_names"):
        TermInclusionProbabilityStub(term_names=())


def test_term_inclusion_probability_stub_call_raises_step10() -> None:
    stub = TermInclusionProbabilityStub(term_names=("x",))
    with pytest.raises(NotImplementedError, match=_STEP_10_MESSAGE):
        stub(jnp.zeros((4, 1)))


def test_coefficient_posterior_interval_stub_validates_alpha() -> None:
    stub = CoefficientPosteriorIntervalStub(alpha=0.05)
    assert stub.alpha == 0.05
    with pytest.raises(ValueError, match="alpha"):
        CoefficientPosteriorIntervalStub(alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        CoefficientPosteriorIntervalStub(alpha=1.0)


def test_coefficient_posterior_interval_stub_call_raises_step10() -> None:
    stub = CoefficientPosteriorIntervalStub(alpha=0.05)
    with pytest.raises(NotImplementedError, match=_STEP_10_MESSAGE):
        stub(jnp.zeros((8, 3)))


def test_equation_discovery_module_imports_cleanly() -> None:
    import opifex.uncertainty.scientific.equation_discovery as mod  # noqa: F401, PLC0415

    assert hasattr(mod, "BayesianSINDyStub")
    assert hasattr(mod, "TermInclusionProbabilityStub")
    assert hasattr(mod, "CoefficientPosteriorIntervalStub")
