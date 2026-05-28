"""Diagnostics for SBI estimators — SBC and posterior contraction.

* :func:`simulation_based_calibration` implements the Talts+ 2018 (SBC)
  procedure: for each of ``num_runs`` ground-truth draws ``theta_star ~ p(theta)``,
  observe ``x_star ~ p(x | theta_star)``, then count where ``theta_star`` falls
  among ``L`` posterior samples. The resulting rank statistics are uniform on
  ``{0, ..., L}`` for a well-specified simulator + posterior; deviation from
  uniformity is a calibration error.

* :func:`expected_posterior_contraction` measures the expected fraction
  ``1 - var_post / var_prior`` across observations. Informative likelihoods
  produce contraction > 0; uninformative observations produce ~0.

References
----------
* Talts, Betancourt, Simpson, Vehtari, Gelman (2018) — ``arXiv:1804.06788``.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from artifex.generative_models.core.rng import extract_rng_key
from flax import nnx, struct
from scipy import stats as scipy_stats

from opifex.uncertainty.sbi.posterior_estimation import NeuralPosteriorEstimator  # noqa: TC001
from opifex.uncertainty.sbi.simulators import Simulator  # noqa: TC001
from opifex.uncertainty.types import (
    metadata_to_dict,
    MetadataItems,
)


_SIMULATE_STREAMS: tuple[str, ...] = ("sbi_simulate", "sample", "default")
_SAMPLE_STREAMS: tuple[str, ...] = ("sbi_sample", "sample", "default")


@struct.dataclass(slots=True, kw_only=True)
class SBCResult:
    """Typed result of a Simulation-Based Calibration run (pattern (B)).

    Fields:

    * ``ranks`` — ``(num_runs, theta_dim)`` per-dimension rank of the
      ground-truth ``theta_star`` among the posterior samples.
    * ``ks_statistic`` — ``(theta_dim,)`` per-dimension Kolmogorov-Smirnov
      statistic against ``Uniform(0, 1)`` on the normalised ranks.
    * ``ks_pvalue`` — ``(theta_dim,)`` per-dimension KS p-value (large
      values support uniformity, i.e., calibration).

    """

    ranks: jax.Array
    ks_statistic: jax.Array
    ks_pvalue: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate field-shape invariants. Not called from pytree path.

        Raises:
            ValueError: When ranks/statistic/p-value shapes are inconsistent.

        """
        if self.ranks.ndim != 2:
            raise ValueError(
                f"ranks must be 2-d (num_runs, theta_dim); got shape={self.ranks.shape}."
            )
        theta_dim = self.ranks.shape[1]
        if self.ks_statistic.shape != (theta_dim,):
            raise ValueError(
                f"ks_statistic shape {self.ks_statistic.shape} must equal ({theta_dim},) "
                "to match ranks.shape[1]."
            )
        if self.ks_pvalue.shape != (theta_dim,):
            raise ValueError(
                f"ks_pvalue shape {self.ks_pvalue.shape} must equal ({theta_dim},) "
                "to match ranks.shape[1]."
            )


@struct.dataclass(slots=True, kw_only=True)
class PosteriorContractionResult:
    """Typed result of an :func:`expected_posterior_contraction` run.

    Fields:

    * ``contraction`` — scalar mean contraction across observations and
      parameter dimensions (``1 - var_post / var_prior`` averaged).
    * ``per_dim`` — ``(theta_dim,)`` per-dimension mean contraction.

    """

    contraction: jax.Array
    per_dim: jax.Array
    metadata: MetadataItems = struct.field(pytree_node=False, default=())

    def metadata_dict(self) -> dict[str, Any]:
        """Return a fresh ``dict`` view of the immutable metadata."""
        return metadata_to_dict(self.metadata)

    def validate(self) -> None:
        """Eager-validate shape invariants."""
        if self.per_dim.ndim != 1:
            raise ValueError(f"per_dim must be 1-d; got shape={self.per_dim.shape}.")
        if self.contraction.ndim != 0:
            raise ValueError(
                f"contraction must be a 0-d scalar; got shape={self.contraction.shape}."
            )


def simulation_based_calibration(
    estimator: NeuralPosteriorEstimator,
    simulator: Simulator,
    *,
    rngs: nnx.Rngs,
    num_runs: int,
    num_posterior_samples: int = 100,
) -> SBCResult:
    """Run SBC on ``estimator`` using fresh draws from ``simulator``.

    Args:
        estimator: Fitted NPE.
        simulator: The same simulator used to generate the fit data.
        rngs: Caller-owned :class:`nnx.Rngs` carrying ``sbi_simulate``
            and ``sbi_sample`` streams.
        num_runs: Number of ground-truth draws used to populate ranks.
        num_posterior_samples: Posterior samples drawn per run for the rank
            computation.

    Returns:
        :class:`SBCResult` with rank statistics + per-dim KS statistics.

    """
    sim_key = extract_rng_key(
        rngs, streams=_SIMULATE_STREAMS, context="simulation_based_calibration.simulate"
    )
    sample_key = extract_rng_key(
        rngs, streams=_SAMPLE_STREAMS, context="simulation_based_calibration.sample"
    )

    prior_key, simulate_key = jax.random.split(sim_key)
    theta_star = simulator.prior_sampler(prior_key, num_runs)
    x_star = simulator.simulate_fn(simulate_key, theta_star)
    if simulator.summary_fn is not None:
        x_star = simulator.summary_fn(x_star)

    per_run_keys = jax.random.split(sample_key, num_runs)
    rank_rows: list[jax.Array] = []
    for run_index in range(num_runs):
        pred = estimator.predict_distribution(
            x_star[run_index],
            rngs=nnx.Rngs(sbi_sample=per_run_keys[run_index]),
            num_samples=num_posterior_samples,
        )
        samples = pred.samples
        if samples is None:
            raise RuntimeError(
                "estimator.predict_distribution did not produce samples; SBC requires samples."
            )
        # Rank: count of posterior samples below the ground truth, per dim.
        rank = jnp.sum(samples < theta_star[run_index], axis=0)
        rank_rows.append(rank)

    ranks = jnp.stack(rank_rows, axis=0)
    # Normalise to (0, 1) for KS against Uniform.
    normalised = ranks.astype(jnp.float32) / float(num_posterior_samples)
    ks_stats: list[float] = []
    ks_ps: list[float] = []
    theta_dim = ranks.shape[1]
    for dim_idx in range(theta_dim):
        result = scipy_stats.kstest(normalised[:, dim_idx], "uniform")
        ks_stats.append(float(result.statistic))
        ks_ps.append(float(result.pvalue))

    return SBCResult(
        ranks=ranks.astype(jnp.int32),
        ks_statistic=jnp.asarray(ks_stats),
        ks_pvalue=jnp.asarray(ks_ps),
        metadata=(
            ("method", "simulation_based_calibration"),
            ("num_runs", num_runs),
            ("num_posterior_samples", num_posterior_samples),
        ),
    )


def expected_posterior_contraction(
    estimator: NeuralPosteriorEstimator,
    simulator: Simulator,
    *,
    rngs: nnx.Rngs,
    num_observations: int = 16,
    num_posterior_samples: int = 200,
) -> PosteriorContractionResult:
    """Compute mean ``1 - var_post / var_prior`` across observations.

    Positive on informative observations, ~0 on uninformative ones, and
    can be slightly negative under finite-sample noise (no clipping here
    — leave the raw signal for the caller).
    """
    sim_key = extract_rng_key(
        rngs, streams=_SIMULATE_STREAMS, context="expected_posterior_contraction.simulate"
    )
    sample_key = extract_rng_key(
        rngs, streams=_SAMPLE_STREAMS, context="expected_posterior_contraction.sample"
    )

    prior_key, simulate_key, prior_var_key = jax.random.split(sim_key, 3)
    theta_star = simulator.prior_sampler(prior_key, num_observations)
    x_star = simulator.simulate_fn(simulate_key, theta_star)
    if simulator.summary_fn is not None:
        x_star = simulator.summary_fn(x_star)

    # Estimate the prior marginal variance with a fresh batch.
    prior_samples = simulator.prior_sampler(prior_var_key, max(num_observations * 8, 256))
    prior_variance = jnp.var(prior_samples, axis=0)

    per_run_keys = jax.random.split(sample_key, num_observations)
    contraction_rows: list[jax.Array] = []
    for run_index in range(num_observations):
        pred = estimator.predict_distribution(
            x_star[run_index],
            rngs=nnx.Rngs(sbi_sample=per_run_keys[run_index]),
            num_samples=num_posterior_samples,
        )
        posterior_variance = pred.variance
        if posterior_variance is None:
            raise RuntimeError(
                "estimator.predict_distribution did not populate variance; "
                "posterior contraction requires marginal posterior variance."
            )
        # ``posterior_variance`` is the marginal posterior variance per-dim.
        contraction = 1.0 - posterior_variance / jnp.where(prior_variance > 0, prior_variance, 1.0)
        contraction_rows.append(contraction)

    stacked = jnp.stack(contraction_rows, axis=0)
    per_dim = jnp.mean(stacked, axis=0)
    overall = jnp.mean(per_dim)
    return PosteriorContractionResult(
        contraction=overall,
        per_dim=per_dim,
        metadata=(
            ("method", "expected_posterior_contraction"),
            ("num_observations", num_observations),
            ("num_posterior_samples", num_posterior_samples),
        ),
    )


__all__ = [
    "PosteriorContractionResult",
    "SBCResult",
    "expected_posterior_contraction",
    "simulation_based_calibration",
]
