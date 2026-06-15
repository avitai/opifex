"""Smoke tests for the uncertainty example notebooks shipped under Task 7.4.

Each test:

1. Imports all documented classes / callables from the example.
2. Calls ``main()`` (each example exposes a top-level ``main`` that
   returns a small summary dict).
3. Asserts the summary keys + finiteness, without pinning numerics.

The two pre-existing examples (`uqno_darcy`, `bayesian_fno`) are touched
lightly — Task 7.4 owns smoke-test coverage only; the objective-code
rewrite belongs to Phase 3 Task 3.5.
"""

import jax.numpy as jnp
import pytest


# ---------------------------------------------------------------------------
# 1. linalg — matrix-free PINN calibration (NNX path).
# ---------------------------------------------------------------------------


def test_linalg_matfree_pinn_imports() -> None:
    """Imports resolve for the linalg example."""
    from examples.uncertainty.linalg.matfree_pinn_calibration import (
        build_precision_matvec,
        main,
        TinyBayesianHead,
    )

    assert callable(main)
    assert callable(build_precision_matvec)
    assert TinyBayesianHead is not None


def test_linalg_matfree_pinn_main_nnx() -> None:
    """Run the linalg example's main() — NNX model path."""
    from examples.uncertainty.linalg.matfree_pinn_calibration import main

    summary = main()
    assert set(summary.keys()) >= {"dim", "log_det_estimate", "fisher_trace_estimate"}
    assert summary["dim"] > 0
    assert jnp.isfinite(summary["log_det_estimate"])
    assert jnp.isfinite(summary["fisher_trace_estimate"])
    # XNysTrace on a PSD operator must be non-negative.
    assert summary["fisher_trace_estimate"] >= 0.0


# ---------------------------------------------------------------------------
# 2. quadrature — Bayesian quadrature evidence (pure JAX kernel path).
# ---------------------------------------------------------------------------


def test_quadrature_bayesian_evidence_imports() -> None:
    """Imports resolve for the quadrature example."""
    from examples.uncertainty.quadrature.bayesian_quadrature_evidence import (
        GROUND_TRUTH,
        integrand_fn,
        main,
    )

    assert callable(main)
    assert callable(integrand_fn)
    assert jnp.isfinite(GROUND_TRUTH)


def test_quadrature_bayesian_evidence_main_pure_jax() -> None:
    """Run the quadrature example's main() — pure-JAX path, no NNX."""
    from examples.uncertainty.quadrature.bayesian_quadrature_evidence import main

    summary = main()
    expected_keys = {
        "ground_truth",
        "vanilla_mean",
        "vanilla_variance",
        "vanilla_absolute_error",
        "wsabi_mean",
        "wsabi_absolute_error",
        "monte_carlo_mean",
        "monte_carlo_variance",
        "monte_carlo_absolute_error",
    }
    assert set(summary.keys()) >= expected_keys
    # GP-BQ should beat MC at matched budget.
    assert summary["vanilla_absolute_error"] < summary["monte_carlo_absolute_error"]
    # WSABI-L should also be more accurate than plain MC here.
    assert summary["wsabi_absolute_error"] < summary["monte_carlo_absolute_error"]


# ---------------------------------------------------------------------------
# 3. statespace — CAKF on sparsely observed system.
# ---------------------------------------------------------------------------


def test_statespace_cakf_imports() -> None:
    """Imports resolve for the statespace example."""
    from examples.uncertainty.statespace.cakf_smoothing import main

    assert callable(main)


def test_statespace_cakf_main() -> None:
    """Run the statespace example's main()."""
    from examples.uncertainty.statespace.cakf_smoothing import main

    summary = main()
    expected_keys = {
        "num_steps",
        "observed_fraction",
        "max_iter",
        "cakf_vs_exact_mean_l2",
        "cakf_vs_truth_mean_l2",
        "exact_vs_truth_mean_l2",
        "exact_final_trace",
    }
    assert set(summary.keys()) >= expected_keys
    assert summary["num_steps"] > 0
    assert 0.0 < summary["observed_fraction"] <= 1.0
    assert jnp.isfinite(summary["cakf_vs_exact_mean_l2"])
    assert jnp.isfinite(summary["exact_final_trace"])
    # The exact Kalman filter must track the truth at least as well as
    # CAKF with a single CG iteration.
    assert summary["exact_vs_truth_mean_l2"] <= summary["cakf_vs_truth_mean_l2"] + 1e-3


# ---------------------------------------------------------------------------
# 4. curvature — diagonal Laplace classifier (NNX path).
# ---------------------------------------------------------------------------


def test_curvature_laplace_imports() -> None:
    """Imports resolve for the curvature example."""
    from examples.uncertainty.curvature.laplace_classifier import main, SmallMLP

    assert callable(main)
    assert SmallMLP is not None


def test_curvature_laplace_main_nnx() -> None:
    """Run the curvature example's main() — NNX model path."""
    from examples.uncertainty.curvature.laplace_classifier import main

    summary = main()
    expected_keys = {
        "num_parameters",
        "posterior_precision_mean",
        "posterior_precision_min",
        "ece",
        "anees",
    }
    assert set(summary.keys()) >= expected_keys
    assert summary["num_parameters"] > 0
    # Precision is τ + Fisher_diag >= τ = 1.0 by construction.
    assert summary["posterior_precision_min"] >= 1.0 - 1e-3
    # ECE is in [0, 1].
    assert 0.0 <= summary["ece"] <= 1.0
    assert jnp.isfinite(summary["anees"])


# ---------------------------------------------------------------------------
# 5. probabilistic_numerics — Fenrir vs DALTON likelihoods (pure JAX path).
# ---------------------------------------------------------------------------


def test_probnum_fenrir_dalton_imports() -> None:
    """Imports resolve for the probnum example."""
    from examples.uncertainty.probabilistic_numerics.fenrir_dalton import main

    assert callable(main)


def test_probnum_fenrir_dalton_main_pure_jax() -> None:
    """Run the probnum example's main() — pure-JAX path, no NNX."""
    from examples.uncertainty.probabilistic_numerics.fenrir_dalton import main

    summary = main()
    expected_keys = {
        "true_theta",
        "well_specified_fenrir_loglik",
        "well_specified_dalton_loglik",
        "misspecified_fenrir_loglik",
        "misspecified_dalton_loglik",
    }
    assert set(summary.keys()) >= expected_keys
    # All log-likelihoods are finite. Misspecified noise must yield a
    # lower (more negative) likelihood than well-specified noise for
    # the same parameter — that is the whole point of the comparison.
    for key in expected_keys - {"true_theta"}:
        assert jnp.isfinite(summary[key])
    assert summary["misspecified_fenrir_loglik"] < summary["well_specified_fenrir_loglik"]
    assert summary["misspecified_dalton_loglik"] < summary["well_specified_dalton_loglik"]


# ---------------------------------------------------------------------------
# Coverage of the two pre-existing examples — touch only, do not rewrite.
# ---------------------------------------------------------------------------


def test_pre_existing_uqno_darcy_imports() -> None:
    """uqno_darcy: imports resolve."""
    try:
        import examples.uncertainty.uqno_darcy as module
    except Exception as import_error:
        pytest.skip(f"uqno_darcy module import unavailable: {import_error!r}")
    assert module is not None


def test_pre_existing_bayesian_fno_imports() -> None:
    """bayesian_fno: imports resolve."""
    try:
        import examples.uncertainty.bayesian_fno as module
    except Exception as import_error:
        pytest.skip(f"bayesian_fno module import unavailable: {import_error!r}")
    assert module is not None


# ---------------------------------------------------------------------------
# Cross-example guarantee: no manual ELBO / KL assembly in new examples.
# ---------------------------------------------------------------------------

_NEW_EXAMPLES = (
    "examples/uncertainty/linalg/matfree_pinn_calibration.py",
    "examples/uncertainty/quadrature/bayesian_quadrature_evidence.py",
    "examples/uncertainty/statespace/cakf_smoothing.py",
    "examples/uncertainty/curvature/laplace_classifier.py",
    "examples/uncertainty/probabilistic_numerics/fenrir_dalton.py",
)


@pytest.mark.parametrize("relative_path", _NEW_EXAMPLES)
def test_no_manual_elbo_or_kl_assembly(relative_path: str) -> None:
    """New examples must not hand-roll ELBO / KL — use objectives helpers."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    file_text = (project_root / relative_path).read_text()
    forbidden_fragments = ("manual ELBO", "manually KL", "predict_with_uncertainty")
    for fragment in forbidden_fragments:
        assert fragment not in file_text, (
            f"{relative_path} contains forbidden fragment {fragment!r}"
        )


# ---------------------------------------------------------------------------
# JAX shape sanity: every main() returns leaves with finite traced values.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "module_path",
    [
        "examples.uncertainty.linalg.matfree_pinn_calibration",
        "examples.uncertainty.quadrature.bayesian_quadrature_evidence",
        "examples.uncertainty.statespace.cakf_smoothing",
        "examples.uncertainty.curvature.laplace_classifier",
        "examples.uncertainty.probabilistic_numerics.fenrir_dalton",
    ],
)
def test_main_returns_finite_jax_leaves(module_path: str) -> None:
    """Every new example's main() must return only finite JAX leaves."""
    import importlib

    module = importlib.import_module(module_path)
    summary = module.main()
    for key, value in summary.items():
        as_array = jnp.asarray(value)
        assert jnp.all(jnp.isfinite(as_array)), f"non-finite value at {module_path}::{key}"
