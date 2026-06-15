"""Contract tests for :class:`SolutionDistribution`.

A :class:`SolutionDistribution` is the solver-side counterpart of
:class:`PredictiveDistribution` — it carries per-field arrays for a PDE
solution (``{"u": ..., "p": ...}``) plus solver-specific axis and
conservation metadata. The same eleven uncertainty leaves are exposed,
just keyed by field name. Round-tripping through
``as_predictive_distribution(field)`` must yield the canonical Phase 1
contract.
"""

from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.scientific.solutions import SolutionDistribution
from opifex.uncertainty.types import (
    _VARIANCE_ATOL,
    _VARIANCE_RTOL,
    PredictiveDistribution,
)


# ---------------------------------------------------------------------------
# Construction + validation
# ---------------------------------------------------------------------------


def _two_field_distribution(
    *,
    with_uncertainty: bool = False,
) -> SolutionDistribution:
    mean = {"u": jnp.zeros((4,)), "p": jnp.ones((4,))}
    samples = {"u": jnp.zeros((3, 4)), "p": jnp.ones((3, 4))}
    metadata: tuple[tuple[str, object], ...] = (
        ("uncertainty_sources", ("numerical",)),
        ("grid_axes", ()),
        ("time_axis", None),
        ("spatial_axes", (0,)),
        ("function_space_norm", "L2"),
        ("conservation_metrics", ()),
        ("covariance_form", "diag"),
    )
    if with_uncertainty:
        epistemic = {"u": jnp.full((4,), 0.1), "p": jnp.full((4,), 0.2)}
        aleatoric = {"u": jnp.full((4,), 0.3), "p": jnp.full((4,), 0.4)}
        total = {
            "u": jnp.full((4,), 0.4),  # epistemic + aleatoric
            "p": jnp.full((4,), 0.6),
        }
        variance = total
        return SolutionDistribution(
            mean=mean,
            samples=samples,
            variance=variance,
            epistemic=epistemic,
            aleatoric=aleatoric,
            total_uncertainty=total,
            metadata=metadata,
        )
    return SolutionDistribution(mean=mean, samples=samples, metadata=metadata)


def test_solution_distribution_constructs_with_minimal_fields() -> None:
    sd = _two_field_distribution()
    assert set(sd.mean) == {"u", "p"}
    assert sd.epistemic is None
    assert sd.metadata == (
        ("uncertainty_sources", ("numerical",)),
        ("grid_axes", ()),
        ("time_axis", None),
        ("spatial_axes", (0,)),
        ("function_space_norm", "L2"),
        ("conservation_metrics", ()),
        ("covariance_form", "diag"),
    )


def test_solution_distribution_is_frozen() -> None:
    sd = _two_field_distribution()
    with pytest.raises(dc.FrozenInstanceError):
        sd.mean = {"u": jnp.ones((4,))}  # type: ignore[misc]


def test_validate_rejects_missing_field_keys_in_uncertainty_leaves() -> None:
    """If ``epistemic`` is supplied it must cover every key in ``mean``."""
    mean = {"u": jnp.zeros((4,)), "p": jnp.ones((4,))}
    epistemic = {"u": jnp.full((4,), 0.1)}  # missing "p"
    aleatoric = {"u": jnp.full((4,), 0.3), "p": jnp.full((4,), 0.4)}
    sd = SolutionDistribution(
        mean=mean,
        epistemic=epistemic,
        aleatoric=aleatoric,
        metadata=(),
    )
    with pytest.raises(ValueError, match="must agree across"):
        sd.validate()


def test_validate_enforces_variance_additivity_per_field() -> None:
    """``total_uncertainty[k] == epistemic[k] + aleatoric[k]`` per field, within the
    canonical Phase 1 tolerances."""
    mean = {"u": jnp.zeros((4,))}
    epistemic = {"u": jnp.full((4,), 0.1)}
    aleatoric = {"u": jnp.full((4,), 0.3)}
    # Deliberately wrong: 0.5 instead of 0.4
    total = {"u": jnp.full((4,), 0.5)}
    sd = SolutionDistribution(
        mean=mean,
        epistemic=epistemic,
        aleatoric=aleatoric,
        total_uncertainty=total,
        metadata=(),
    )
    with pytest.raises(ValueError, match="variance-additivity"):
        sd.validate()


def test_validate_passes_when_variance_additivity_holds_within_tolerance() -> None:
    sd = _two_field_distribution(with_uncertainty=True)
    sd.validate()  # no raise


def test_validate_is_not_called_from_post_init() -> None:
    """Constructing with incomplete leaves must not raise — validate is
    a public preflight only (GUIDE_ALIGNMENT item 7)."""
    SolutionDistribution(
        mean={"u": jnp.zeros((4,))},
        epistemic={"u": jnp.full((4,), 0.1)},
        # aleatoric missing — additivity cannot hold — but no validate raised.
        metadata=(),
    )


# ---------------------------------------------------------------------------
# Variance-tolerance reuse from Phase 1 contract
# ---------------------------------------------------------------------------


def test_uses_canonical_variance_tolerances() -> None:
    """The tolerances must be the public Phase 1 constants — anything else
    flaps the round-trip variance test."""
    import opifex.uncertainty.scientific.solutions as sol_mod

    # Module either imports the canonical constants or assigns them.
    rtol = getattr(sol_mod, "_VARIANCE_RTOL", None)
    atol = getattr(sol_mod, "_VARIANCE_ATOL", None)
    assert rtol == _VARIANCE_RTOL, f"rtol must reuse canonical {_VARIANCE_RTOL}, got {rtol}"
    assert atol == _VARIANCE_ATOL, f"atol must reuse canonical {_VARIANCE_ATOL}, got {atol}"


# ---------------------------------------------------------------------------
# as_predictive_distribution projection
# ---------------------------------------------------------------------------


def test_as_predictive_distribution_returns_phase_1_contract() -> None:
    sd = _two_field_distribution(with_uncertainty=True)
    pd = sd.as_predictive_distribution("u")
    assert isinstance(pd, PredictiveDistribution)
    assert jnp.allclose(pd.mean, sd.mean["u"])
    assert sd.epistemic is not None
    assert pd.epistemic is not None
    assert jnp.allclose(pd.epistemic, sd.epistemic["u"])


def test_as_predictive_distribution_round_trips_through_validate() -> None:
    """The projected PredictiveDistribution must survive its own validate()
    — same tolerances apply, so no flap."""
    sd = _two_field_distribution(with_uncertainty=True)
    pd = sd.as_predictive_distribution("p")
    pd.validate()  # no raise


def test_as_predictive_distribution_raises_on_unknown_field() -> None:
    sd = _two_field_distribution()
    with pytest.raises(KeyError, match="unknown field"):
        sd.as_predictive_distribution("nonexistent")


# ---------------------------------------------------------------------------
# JAX PyTree behaviour
# ---------------------------------------------------------------------------


def test_solution_distribution_is_jax_pytree() -> None:
    sd = _two_field_distribution(with_uncertainty=True)
    leaves, treedef = jax.tree_util.tree_flatten(sd)
    assert len(leaves) > 0
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(rebuilt, SolutionDistribution)
    assert jnp.allclose(rebuilt.mean["u"], sd.mean["u"])


def test_solution_distribution_traces_under_jit() -> None:
    @jax.jit
    def double_mean(sd: SolutionDistribution) -> SolutionDistribution:
        # flax.struct.dataclass provides .replace() via dataclasses.replace;
        # pyright sees it as a dynamic method, hence the cast.
        return sd.replace(mean={k: 2.0 * v for k, v in sd.mean.items()})  # type: ignore[attr-defined]

    sd = _two_field_distribution()
    out = double_mean(sd)
    assert jnp.allclose(out.mean["u"], 2.0 * sd.mean["u"])


def test_solution_distribution_traces_under_vmap() -> None:
    """vmap over an axis added to every field array must not touch the
    static metadata."""

    def make_sd(scale: jax.Array) -> SolutionDistribution:
        return SolutionDistribution(
            mean={"u": scale * jnp.zeros((4,)), "p": scale * jnp.ones((4,))},
            metadata=(("uncertainty_sources", ("numerical",)),),
        )

    scales = jnp.arange(3.0)
    sds = jax.vmap(make_sd)(scales)
    assert sds.mean["u"].shape == (3, 4)
    assert sds.metadata == (("uncertainty_sources", ("numerical",)),)


def test_metadata_dict_accessor_round_trips() -> None:
    sd = _two_field_distribution()
    md = sd.metadata_dict()
    assert md["uncertainty_sources"] == ("numerical",)
    assert md["function_space_norm"] == "L2"


# ---------------------------------------------------------------------------
# No NNX leak
# ---------------------------------------------------------------------------


def test_no_nnx_imports_in_solutions_module() -> None:
    """``opifex.uncertainty.scientific.solutions`` is a pure-array value-object
    module — importing NNX would couple it to the trainable-state surface."""
    import inspect

    import opifex.uncertainty.scientific.solutions as sol_mod

    source = inspect.getsource(sol_mod)
    assert "from flax import nnx" not in source
    assert "import flax.nnx" not in source


# ---------------------------------------------------------------------------
# as_predictive_distribution under jit / vmap (Phase 6 Task 6.1 exit criterion)
# ---------------------------------------------------------------------------


def test_as_predictive_distribution_traces_under_jit() -> None:
    """Projecting a single field must trace cleanly — metadata stays static,
    only the per-field arrays become traced leaves."""

    @jax.jit
    def project_u(sd: SolutionDistribution) -> PredictiveDistribution:
        return sd.as_predictive_distribution("u")

    sd = _two_field_distribution(with_uncertainty=True)
    pd = project_u(sd)
    # Metadata round-trips as the original static tuple — no traced leaf escape.
    assert pd.metadata == sd.metadata
    assert jnp.allclose(pd.mean, sd.mean["u"])


def test_as_predictive_distribution_traces_under_vmap() -> None:
    """vmap over a batch axis added to every field array must keep metadata
    static and only batch the array leaves."""

    def make_then_project(scale: jax.Array) -> jax.Array:
        sd = SolutionDistribution(
            mean={"u": scale * jnp.ones((4,)), "p": scale * jnp.zeros((4,))},
            metadata=(("uncertainty_sources", ("numerical",)),),
        )
        pd = sd.as_predictive_distribution("u")
        return pd.mean

    scales = jnp.arange(3.0) + 1.0
    out = jax.vmap(make_then_project)(scales)
    assert out.shape == (3, 4)
    # Batched means scale linearly with the scan input.
    assert jnp.allclose(out, scales[:, None] * jnp.ones((1, 4)))


def test_projection_supports_grad() -> None:
    """Pure-array kernel path must be differentiable through the projection."""

    def loss(u_array: jax.Array) -> jax.Array:
        sd = SolutionDistribution(
            mean={"u": u_array, "p": jnp.zeros_like(u_array)},
            metadata=(("uncertainty_sources", ("numerical",)),),
        )
        pd = sd.as_predictive_distribution("u")
        return jnp.sum(pd.mean**2)

    u = jnp.arange(4.0)
    grad = jax.grad(loss)(u)
    assert jnp.allclose(grad, 2.0 * u)


# ---------------------------------------------------------------------------
# uncertainty_sources vocabulary
# ---------------------------------------------------------------------------


def test_uncertainty_sources_declares_six_canonical_values() -> None:
    """The audit-mandated set of source labels is exported as a module constant
    so callers and audits agree on the vocabulary."""
    from opifex.uncertainty.scientific.solutions import UNCERTAINTY_SOURCES

    assert set(UNCERTAINTY_SOURCES) == {
        "numerical",
        "parameter",
        "observation",
        "model_discrepancy",
        "ensemble",
        "calibration",
    }


# ---------------------------------------------------------------------------
# to_solution conversion (no in-place mutation)
# ---------------------------------------------------------------------------


def test_to_solution_keeps_mean_fields_and_stores_uq_in_auxiliary_data() -> None:
    from opifex.core.solver.interface import Solution

    sd = _two_field_distribution(with_uncertainty=True)
    solution = sd.to_solution(execution_time=1.23, converged=True)
    assert isinstance(solution, Solution)
    assert set(solution.fields) == {"u", "p"}
    assert jnp.allclose(solution.fields["u"], sd.mean["u"])
    assert solution.converged is True
    assert solution.execution_time == 1.23
    # UQ payload is recoverable from auxiliary_data — no backref needed.
    uq = solution.auxiliary_data["uq"]
    assert uq["epistemic"] is not None
    assert sd.epistemic is not None
    assert jnp.allclose(uq["epistemic"]["u"], sd.epistemic["u"])
    assert uq["metadata"] == sd.metadata


def test_to_solution_does_not_mutate_source_distribution() -> None:
    sd = _two_field_distribution()
    original_mean_u = sd.mean["u"]
    solution = sd.to_solution()
    # Mutate solution's view; source distribution stays intact.
    solution.fields["u"] = jnp.full_like(original_mean_u, 99.0)
    assert jnp.allclose(sd.mean["u"], original_mean_u)
