"""Every Student-t predictor records the same likelihood label.

The dense Laplace GP and the Markov Laplace, variational, power-EP and posterior-linearisation paths
each return a :class:`opifex.uncertainty.types.PredictiveDistribution` whose ``likelihood`` metadata
names the response distribution. Code that groups or filters predictions by that key needs one
spelling per likelihood. The Bernoulli, Poisson, Gaussian and Beta wrappers already agree across
paths, and the Student-t label is ``"students_t"``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import pytest

from opifex.uncertainty.gp.laplace_likelihoods import (
    fit_studentst_laplace_gp,
    predict_studentst_laplace_gp,
)
from opifex.uncertainty.markov import (
    fit_studentst_markov_laplace_gp,
    fit_studentst_markov_pep_gp,
    fit_studentst_markov_pl_gp,
    fit_studentst_markov_vi_gp,
    predict_studentst_markov_laplace_gp,
    predict_studentst_markov_pep_gp,
    predict_studentst_markov_pl_gp,
    predict_studentst_markov_vi_gp,
)
from opifex.uncertainty.statespace import matern32_kernel


if TYPE_CHECKING:
    from collections.abc import Callable

    from opifex.uncertainty.types import PredictiveDistribution


def _dense_laplace_predictive() -> PredictiveDistribution:
    """Fit the dense Student-t Laplace GP on toy data and predict at two inputs."""
    x_train = jnp.linspace(-1.5, 1.5, 20)[:, None]
    noise = 0.1 * jax.random.normal(jax.random.PRNGKey(0), (20,))
    state = fit_studentst_laplace_gp(
        x_train=x_train,
        y_train=jnp.sin(2.0 * x_train[:, 0]) + noise,
        lengthscale=0.5,
        output_scale=1.0,
        num_newton_iterations=10,
    )
    return predict_studentst_laplace_gp(state=state, x_test=jnp.zeros((2, 1)))


def _markov_predictive(
    fit: Callable[..., object], predict: Callable[..., PredictiveDistribution]
) -> PredictiveDistribution:
    """Fit a Markov Student-t GP on toy data and predict at three times."""
    times = jnp.linspace(0.0, 4.0, 18)
    noise = 0.1 * jax.random.normal(jax.random.PRNGKey(1), (18,))
    state = fit(
        times=times,
        observations=jnp.sin(2.0 * times) + noise,
        state_space_kernel=matern32_kernel(variance=1.0, lengthscale=0.5),
        num_iterations=5,
    )
    return predict(state=state, times_test=jnp.linspace(0.5, 3.5, 3))


_PREDICTIVES: dict[str, Callable[[], PredictiveDistribution]] = {
    "gp_laplace": _dense_laplace_predictive,
    "markov_laplace": lambda: _markov_predictive(
        fit_studentst_markov_laplace_gp, predict_studentst_markov_laplace_gp
    ),
    "markov_vi": lambda: _markov_predictive(
        fit_studentst_markov_vi_gp, predict_studentst_markov_vi_gp
    ),
    "markov_pep": lambda: _markov_predictive(
        fit_studentst_markov_pep_gp, predict_studentst_markov_pep_gp
    ),
    "markov_pl": lambda: _markov_predictive(
        fit_studentst_markov_pl_gp, predict_studentst_markov_pl_gp
    ),
}


@pytest.mark.parametrize("name", sorted(_PREDICTIVES))
def test_studentst_predictor_records_students_t(name: str) -> None:
    """The predictive metadata names the Student-t likelihood ``"students_t"``."""
    metadata = _PREDICTIVES[name]().metadata_dict()
    assert metadata["likelihood"] == "students_t"
    assert metadata["link"] == "identity"
