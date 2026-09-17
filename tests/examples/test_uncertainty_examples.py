"""The uncertainty examples' results, checked from a run of each in its own interpreter.

Each example's ``main()`` returns a summary dict; the run goes through
``substrax.testing.run_example`` on the CPU backend, and the assertions read the
decoded summary. Numerics are bounded, not pinned.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from substrax.testing import run_example


ROOT = Path(__file__).resolve().parents[2]
UNCERTAINTY = ROOT / "examples" / "uncertainty"


@pytest.fixture
def output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "outputs"
    path.mkdir()
    monkeypatch.setenv("AVITAI_OUTPUT_DIR", str(path))
    return path


def _summary(
    relative: str, output_dir: Path, request: pytest.FixtureRequest
) -> dict[str, float | int]:
    budget = float(request.config.getoption("timeout"))
    run = run_example(
        UNCERTAINTY / relative,
        repo_root=ROOT,
        output_dir=output_dir,
        timeout=budget,
        call_main=True,
    )
    assert run.result.check().returncode == 0
    assert isinstance(run.summary, dict)
    return run.summary


@pytest.mark.timeout(0)
def test_linalg_matfree_pinn_calibration(output_dir: Path, request: pytest.FixtureRequest) -> None:
    summary = _summary("linalg/matfree_pinn_calibration.py", output_dir, request)
    assert set(summary) >= {"dim", "log_det_estimate", "fisher_trace_estimate"}
    assert summary["dim"] > 0
    assert math.isfinite(summary["log_det_estimate"])
    assert math.isfinite(summary["fisher_trace_estimate"])
    # XNysTrace on a PSD operator must be non-negative.
    assert summary["fisher_trace_estimate"] >= 0.0


@pytest.mark.timeout(0)
def test_quadrature_bayesian_evidence(output_dir: Path, request: pytest.FixtureRequest) -> None:
    summary = _summary("quadrature/bayesian_quadrature_evidence.py", output_dir, request)
    assert set(summary) >= {
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
    # GP-BQ beats MC at a matched budget, and so does WSABI-L.
    assert summary["vanilla_absolute_error"] < summary["monte_carlo_absolute_error"]
    assert summary["wsabi_absolute_error"] < summary["monte_carlo_absolute_error"]


@pytest.mark.timeout(0)
def test_statespace_cakf_smoothing(output_dir: Path, request: pytest.FixtureRequest) -> None:
    summary = _summary("statespace/cakf_smoothing.py", output_dir, request)
    assert set(summary) >= {
        "num_steps",
        "observed_fraction",
        "max_iter",
        "cakf_vs_exact_mean_l2",
        "cakf_vs_truth_mean_l2",
        "exact_vs_truth_mean_l2",
        "exact_final_trace",
    }
    assert summary["num_steps"] > 0
    assert 0.0 < summary["observed_fraction"] <= 1.0
    assert math.isfinite(summary["cakf_vs_exact_mean_l2"])
    assert math.isfinite(summary["exact_final_trace"])
    # The exact Kalman filter tracks the truth at least as well as CAKF with one CG
    # iteration.
    assert summary["exact_vs_truth_mean_l2"] <= summary["cakf_vs_truth_mean_l2"] + 1e-3


@pytest.mark.timeout(0)
def test_curvature_laplace_classifier(output_dir: Path, request: pytest.FixtureRequest) -> None:
    summary = _summary("curvature/laplace_classifier.py", output_dir, request)
    assert set(summary) >= {
        "num_parameters",
        "posterior_precision_mean",
        "posterior_precision_min",
        "ece",
        "anees",
    }
    assert summary["num_parameters"] > 0
    # Precision is tau + Fisher_diag >= tau = 1.0 by construction.
    assert summary["posterior_precision_min"] >= 1.0 - 1e-3
    assert 0.0 <= summary["ece"] <= 1.0
    assert math.isfinite(summary["anees"])


@pytest.mark.timeout(0)
def test_probabilistic_numerics_fenrir_dalton(
    output_dir: Path, request: pytest.FixtureRequest
) -> None:
    summary = _summary("probabilistic_numerics/fenrir_dalton.py", output_dir, request)
    keys = {
        "true_theta",
        "well_specified_fenrir_loglik",
        "well_specified_dalton_loglik",
        "misspecified_fenrir_loglik",
        "misspecified_dalton_loglik",
    }
    assert set(summary) >= keys
    # Misspecified noise yields a lower likelihood than well-specified noise for the
    # same parameter, which is the point of the comparison.
    for key in keys - {"true_theta"}:
        assert math.isfinite(summary[key])
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
