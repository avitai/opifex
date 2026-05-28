r"""Bayesian spectral-convolution surface for the FNO family.

Houses the FNO-flavoured Bayesian spectral-convolution variants:

* :class:`BayesianSpectralConvolution` — re-exported from the canonical
  :mod:`opifex.uncertainty.layers.bayesian` (variational
  diagonal-Gaussian posterior over the complex spectral weights;
  Li et al. 2021 spectral block with a Phase-3-compliant
  ``deterministic`` flag + per-call ``rngs`` pattern).
* :class:`ComputationAwareSpectralConvolution` — Task 10.4 sibling
  whose uncertainty over the spectral weights is maintained as a
  *low-rank CAKF posterior* (Pförtner+ 2024 arXiv:2405.08971;
  Wenger+ 2023 CAGP precursor, arXiv:2306.07879) rather than a
  diagonal-Gaussian one. The CAKF representation is *implicit* —
  ``posterior_cov = prior_cov - factor @ factor^T`` — so a single
  rank-``r`` matrix captures all of the posterior shrinkage gained
  by ``r`` CG iterations against caller-supplied observations of the
  flattened spectral-weight vector.

The forward pass of CASpec uses the CAKF posterior **mean** as the
spectral weights and emits a deterministic output of the same shape as
:class:`BayesianSpectralConvolution`. ``cakf_refine`` returns a new
CASpec instance whose CAKF posterior has been advanced by one
``cakf_update`` call. The new instance shares the same configuration
(``in_channels``, ``out_channels``, ``modes``, …) but carries a
refined ``(cakf_mean, cakf_factor)`` pair.

References
----------
* Pförtner, M., Wenger, J., Cockayne, J., Hennig, P. 2024 —
  *Computation-Aware Kalman Filtering and Smoothing*,
  arXiv:2405.08971 (PRIMARY — the CAKF algorithm).
* Wenger, J., Pleiss, G., Pförtner, M., Hennig, P., Cunningham, J.
  2023 — *Posterior and Computational Uncertainty in Gaussian
  Processes*, arXiv:2306.07879 (CAGP precursor).
* Li, Z. et al. 2021 — *Fourier Neural Operator for Parametric
  Partial Differential Equations*, arXiv:2010.08895 (the spectral
  block that BSC + CASpec parameterise).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from flax import nnx

from opifex.uncertainty.layers.bayesian import (
    BayesianSpectralConvolution as _BayesianSpectralConvolution,
)
from opifex.uncertainty.statespace.cakf import cakf_update


# Re-export under the FNO-Bayesian namespace; consumers should be able to
# import ``BayesianSpectralConvolution`` from either home.
BayesianSpectralConvolution = _BayesianSpectralConvolution


@dataclass(frozen=True, slots=True, kw_only=True)
class _CAKFSpectralRefinement:
    """Output of :meth:`ComputationAwareSpectralConvolution.cakf_refine`.

    Attributes:
        cakf_mean: Refined posterior-mean vector for the flattened
            spectral weights (shape ``(flat_parameter_dim,)``).
        cakf_factor: Refined low-rank CAKF factor (shape
            ``(flat_parameter_dim, max_iter)``).
    """

    cakf_mean: jax.Array
    cakf_factor: jax.Array


class ComputationAwareSpectralConvolution(nnx.Module):
    """Spectral conv with CAKF low-rank posterior over flattened weights.

    The constructor signature mirrors
    :class:`BayesianSpectralConvolution`. Internally the module owns a
    BSC instance for the spectral kernel shape + initialisation, plus
    two ``nnx.Variable`` slots that hold the CAKF posterior mean and
    low-rank factor over the **flattened real-valued weight vector**
    (real + imaginary parts of every spectral weight tensor are
    concatenated). The forward pass overlays the current CAKF mean onto
    the BSC mean parameters and runs BSC in deterministic mode.

    Args:
        in_channels: Number of input feature channels.
        out_channels: Number of output feature channels.
        modes: Number of retained Fourier modes per axis. 1D or 2D
            tuples (see :class:`BayesianSpectralConvolution`).
        prior_std: Diagonal-Gaussian prior std-dev used by the
            underlying BSC's variational kernel and adopted here as
            the prior marginal-covariance scale ``prior_std² · I`` for
            CAKF.
        deterministic: Forwarded to the underlying BSC; defaults to
            ``False`` so the sampling toggle is consistent with BSC.
        rngs: Caller-owned ``nnx.Rngs`` for parameter init.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes: tuple[int, ...],
        prior_std: float = 1.0,
        deterministic: bool = False,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        super().__init__()
        self._bsc = _BayesianSpectralConvolution(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            prior_std=prior_std,
            deterministic=deterministic,
            rngs=rngs,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.prior_std = prior_std
        self.deterministic = deterministic
        self.flat_parameter_dim = int(self._count_flat_parameters())
        # Initialise the CAKF posterior at the prior: zero mean + empty factor.
        self.cakf_mean = nnx.Variable(jnp.zeros(self.flat_parameter_dim))
        self.cakf_factor = nnx.Variable(jnp.zeros((self.flat_parameter_dim, 0)))

    def __call__(self, x: jax.Array) -> jax.Array:
        """Deterministic forward using the underlying BSC posterior mean."""
        return self._bsc(x, deterministic=True)

    def cakf_refine(
        self,
        *,
        observation: jax.Array,
        observation_matrix: jax.Array,
        observation_cov: jax.Array,
        max_iter: int,
    ) -> _CAKFSpectralRefinement:
        """Advance the CAKF posterior by one ``cakf_update`` call.

        Args:
            observation: ``y`` — observed values for the flattened
                spectral weight vector (shape ``(k,)``).
            observation_matrix: ``H`` — observation matrix mapping the
                flat weight vector to observations (shape
                ``(k, flat_parameter_dim)``).
            observation_cov: ``Λ`` — observation noise covariance
                (shape ``(k, k)``).
            max_iter: Maximum CG iterations for ``cakf_update``;
                determines the rank gained per call. Must be a static
                Python ``int`` for tracing.

        Returns:
            A :class:`_CAKFSpectralRefinement` carrying the new
            ``(cakf_mean, cakf_factor)`` pair. The caller is responsible
            for storing the result back on the module via
            ``module.cakf_mean.value = result.cakf_mean`` if persistent
            refinement is desired.
        """
        prior_cov = self.prior_std**2 * jnp.eye(self.flat_parameter_dim)
        new_mean, new_factor = cakf_update(
            mean=self.cakf_mean.value,
            prior_cov=prior_cov,
            factor=self.cakf_factor.value,
            observation=observation,
            observation_matrix=observation_matrix,
            observation_cov=observation_cov,
            max_iter=max_iter,
        )
        return _CAKFSpectralRefinement(cakf_mean=new_mean, cakf_factor=new_factor)

    def _count_flat_parameters(self) -> int:
        """Return the size of the flattened real-valued spectral weight vector."""
        size = int(self._bsc.weight_mean[...].size + self._bsc.weight_imag_mean[...].size)
        if len(self.modes) == 2:
            size += int(
                self._bsc.weight_neg_h_mean[...].size + self._bsc.weight_neg_h_imag_mean[...].size
            )
        return size


__all__ = [
    "BayesianSpectralConvolution",
    "ComputationAwareSpectralConvolution",
]
