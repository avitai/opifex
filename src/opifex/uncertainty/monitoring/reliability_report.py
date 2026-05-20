"""Monitoring inputs + `UQReliabilityReport` builder."""

from __future__ import annotations

import dataclasses as dc

import jax  # noqa: TC002 — kept eager for consistency with the rest of opifex.uncertainty

from opifex.uncertainty.reports import UQReliabilityReport


@dc.dataclass(frozen=True, slots=True, kw_only=True)
class MonitoringInputs:
    """Static provenance for a monitoring run.

    Pattern A frozen dataclass — scalar / string / tuple fields only.
    """

    model_name: str
    method: str
    source_package: str
    dataset: str
    split: str
    run_id: str
    assumption_warnings: tuple[str, ...] = ()
    assumption_status: str = "ok"


def build_reliability_report(
    *,
    inputs: MonitoringInputs,
    calibration_ece: jax.Array | None = None,
    calibration_nll: jax.Array | None = None,
    brier_score: jax.Array | None = None,
    empirical_coverage: jax.Array | None = None,
    mean_interval_width: jax.Array | None = None,
    interval_score: jax.Array | None = None,
    crps: jax.Array | None = None,
    energy_score: jax.Array | None = None,
    spread_skill_ratio: jax.Array | None = None,
    ood_auroc: jax.Array | None = None,
    ood_auprc: jax.Array | None = None,
    ood_fpr95: jax.Array | None = None,
    aurc: jax.Array | None = None,
) -> UQReliabilityReport:
    """Assemble a :class:`UQReliabilityReport` from already-computed metrics.

    Performs ``report.validate()`` before returning so half-populated
    reports surface immediately instead of being silently propagated.

    Args:
        inputs: :class:`MonitoringInputs` carrying provenance fields.
        calibration_ece, calibration_nll, brier_score, empirical_coverage,
        mean_interval_width, interval_score, crps, energy_score,
        spread_skill_ratio, ood_auroc, ood_auprc, ood_fpr95, aurc:
            Optional per-subsystem metric values.

    Returns:
        Validated :class:`UQReliabilityReport`.

    Raises:
        ValueError: If no metric value is supplied (delegated to
            :meth:`UQReliabilityReport.validate`).

    """
    metadata = (
        ("model_name", inputs.model_name),
        ("method", inputs.method),
        ("source_package", inputs.source_package),
        ("dataset", inputs.dataset),
        ("split", inputs.split),
        ("run_id", inputs.run_id),
        ("assumption_warnings", inputs.assumption_warnings),
        ("assumption_status", inputs.assumption_status),
    )
    report = UQReliabilityReport(
        calibration_ece=calibration_ece,
        calibration_nll=calibration_nll,
        brier_score=brier_score,
        empirical_coverage=empirical_coverage,
        mean_interval_width=mean_interval_width,
        interval_score=interval_score,
        crps=crps,
        energy_score=energy_score,
        spread_skill_ratio=spread_skill_ratio,
        ood_auroc=ood_auroc,
        ood_auprc=ood_auprc,
        ood_fpr95=ood_fpr95,
        aurc=aurc,
        metadata=metadata,
    )
    report.validate()
    return report
