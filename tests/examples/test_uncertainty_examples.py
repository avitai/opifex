"""The uncertainty examples' results, read from the shared example tests' run of each."""

import math
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest
from substrax.testing.example_suite import ExampleRunner


ROOT = Path(__file__).resolve().parents[2]
UNCERTAINTY = ROOT / "examples" / "uncertainty"

# ComputationAwareKalman.jl's own test (test/gp_regression.jl at fc6b4109) asserts CAKF equals the
# exact posterior with Julia's `≈`, whose relative tolerance is sqrt(eps) of the element type; the
# examples run in float32.
CAKF_EXACTNESS_RTOL = math.sqrt(float(np.finfo(np.float32).eps))


class MatfreeCalibration(TypedDict):
    """What ``linalg/matfree_pinn_calibration.py``'s ``main()`` returns."""

    dim: int
    log_det_estimate: float
    fisher_trace_estimate: float


class QuadratureEvidence(TypedDict):
    """What ``quadrature/bayesian_quadrature_evidence.py``'s ``main()`` returns."""

    ground_truth: float
    vanilla_mean: float
    vanilla_variance: float
    vanilla_absolute_error: float
    wsabi_mean: float
    wsabi_absolute_error: float
    monte_carlo_mean: float
    monte_carlo_variance: float
    monte_carlo_absolute_error: float


class CakfSmoothing(TypedDict):
    """What ``statespace/cakf_smoothing.py``'s ``main()`` returns."""

    num_steps: int
    observed_fraction: float
    max_iter: int
    cakf_vs_exact_mean_relative: float
    cakf_vs_exact_cov_relative: float
    cakf_vs_truth_mean_l2: float
    exact_vs_truth_mean_l2: float
    exact_final_trace: float


class LaplaceClassifier(TypedDict):
    """What ``curvature/laplace_classifier.py``'s ``main()`` returns."""

    num_parameters: int
    posterior_precision_mean: float
    posterior_precision_min: float
    ece: float
    anees: float


class FenrirDalton(TypedDict):
    """What ``probabilistic_numerics/fenrir_dalton.py``'s ``main()`` returns."""

    true_theta: float
    well_specified_fenrir_loglik: float
    well_specified_dalton_loglik: float
    misspecified_fenrir_loglik: float
    misspecified_dalton_loglik: float


def test_linalg_matfree_pinn_calibration(example_run: ExampleRunner) -> None:
    summary = example_run("uncertainty/linalg/matfree_pinn_calibration.py").returned_as(
        MatfreeCalibration
    )
    assert summary["dim"] > 0
    assert math.isfinite(summary["log_det_estimate"])
    assert math.isfinite(summary["fisher_trace_estimate"])
    # XNysTrace on a PSD operator must be non-negative.
    assert summary["fisher_trace_estimate"] >= 0.0


def test_quadrature_bayesian_evidence(example_run: ExampleRunner) -> None:
    summary = example_run("uncertainty/quadrature/bayesian_quadrature_evidence.py").returned_as(
        QuadratureEvidence
    )
    # GP-BQ beats MC at a matched budget, and so does WSABI-L.
    assert summary["vanilla_absolute_error"] < summary["monte_carlo_absolute_error"]
    assert summary["wsabi_absolute_error"] < summary["monte_carlo_absolute_error"]


def test_statespace_cakf_smoothing(example_run: ExampleRunner) -> None:
    summary = example_run("uncertainty/statespace/cakf_smoothing.py").returned_as(CakfSmoothing)
    assert summary["num_steps"] > 0
    assert 0.0 < summary["observed_fraction"] <= 1.0
    assert math.isfinite(summary["exact_final_trace"])
    # The observation is a scalar, so one CG iteration spans it and CAKF is the exact filter.
    assert summary["cakf_vs_exact_mean_relative"] <= CAKF_EXACTNESS_RTOL
    assert summary["cakf_vs_exact_cov_relative"] <= CAKF_EXACTNESS_RTOL


def test_curvature_laplace_classifier(example_run: ExampleRunner) -> None:
    summary = example_run("uncertainty/curvature/laplace_classifier.py").returned_as(
        LaplaceClassifier
    )
    assert summary["num_parameters"] > 0
    # Precision is tau + Fisher_diag >= tau = 1.0 by construction.
    assert summary["posterior_precision_min"] >= 1.0 - 1e-3
    assert 0.0 <= summary["ece"] <= 1.0
    assert math.isfinite(summary["anees"])


def test_probabilistic_numerics_fenrir_dalton(example_run: ExampleRunner) -> None:
    summary = example_run("uncertainty/probabilistic_numerics/fenrir_dalton.py").returned_as(
        FenrirDalton
    )
    assert math.isfinite(summary["well_specified_fenrir_loglik"])
    assert math.isfinite(summary["well_specified_dalton_loglik"])
    assert math.isfinite(summary["misspecified_fenrir_loglik"])
    assert math.isfinite(summary["misspecified_dalton_loglik"])
    # Misspecified noise yields a lower likelihood than well-specified noise for the
    # same parameter, which is the point of the comparison.
    assert summary["misspecified_fenrir_loglik"] < summary["well_specified_fenrir_loglik"]
    assert summary["misspecified_dalton_loglik"] < summary["well_specified_dalton_loglik"]


_NEW_EXAMPLES = (
    "linalg/matfree_pinn_calibration.py",
    "quadrature/bayesian_quadrature_evidence.py",
    "statespace/cakf_smoothing.py",
    "curvature/laplace_classifier.py",
    "probabilistic_numerics/fenrir_dalton.py",
)


@pytest.mark.parametrize("relative", _NEW_EXAMPLES)
def test_no_manual_elbo_or_kl_assembly(relative: str) -> None:
    """The examples use the objectives helpers rather than hand-rolling ELBO / KL."""
    text = (UNCERTAINTY / relative).read_text()
    for fragment in ("manual ELBO", "manually KL", "predict_with_uncertainty"):
        assert fragment not in text, f"{relative} contains forbidden fragment {fragment!r}"
