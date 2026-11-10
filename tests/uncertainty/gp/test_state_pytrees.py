"""Every GP state is a pytree, and survives JAX and NNX transformations.

A fitted state that cannot cross a ``jit`` boundary cannot be cached, carried through
``scan``, or returned from a compiled fit --- and before these containers were registered,
returning one from a jitted function raised ``TypeError``.

**The static split is the part worth testing, not the registration.** Registration is one
decorator; getting ``data`` and ``meta`` wrong is silent. A callable marked as data breaks
flattening, and an array marked as metadata is baked into the treedef, so the value is frozen
at trace time and every change silently recompiles instead of flowing through. Both are
asserted per class below.

**Which compatibility is required here, and which is interop.** ``opifex.uncertainty.gp`` is
JAX code: it defines no ``nnx.Module``, and its only contact with NNX is ``rff.py`` accepting
an ``nnx.Rngs`` for key material. So ``jit``, ``grad`` and ``vmap`` are the *requirement* and
are tested as such. NNX transforms are tested as *interop*: no NNX module consumes a GP state
today, but 82 files under ``neural`` are NNX-based and these containers are plain pytrees, so
the boundary is one a caller will cross. The distinction is kept because a test that claims a
requirement it does not have is as misleading as a missing one.

**Registration is what makes that boundary legal, not a convenience.** Flax's own
``docs_nnx/guides/jax_and_nnx_transforms.rst`` states the condition: NNX and JAX transforms
mix "so long as the JAX-transformed function in your code is pure and has valid argument
types that are recognized by JAX". An unregistered container is not such a type --- which is
why, before this change, ``jax.jit`` and ``nnx.jit`` rejected a fitted state alike.
"""

from __future__ import annotations

import dataclasses
import functools
import typing as tp

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from opifex.uncertainty.gp import (
    ExactGPState,
    fit_exact_gp,
    LaplaceGPState,
    predict_exact_gp,
    QuasisepCarmaGPState,
    QuasisepGPState,
    RFFGPState,
    SpatioTemporalGPState,
    StochasticSVGPState,
    SVGPState,
)
from tests.uncertainty.gp._accuracy import assert_reduction_agreement


#: The fitted problem the transform-agreement tests below are posed on.
TRAIN_SIZE = 6
LENGTHSCALE = 0.4
OUTPUT_SCALE = 1.0
NOISE_STD = 0.1


#: Every GP state container, with the fields that must be treedef metadata rather than
#: leaves. The rule behind the table is one line --- callables and plain integer counts are
#: metadata, everything else is data --- and the table exists so a future field has to be
#: classified deliberately rather than by whichever decorator was copied.
STATE_METADATA: tp.Final = {
    ExactGPState: {"kernel_fn"},
    LaplaceGPState: {"kernel_fn"},
    RFFGPState: set(),
    QuasisepGPState: set(),
    QuasisepCarmaGPState: set(),
    SpatioTemporalGPState: {"num_space"},
    SVGPState: {"kernel_fn", "cached_n"},
    StochasticSVGPState: {"kernel_fn", "log_likelihood_fn"},
}

STATE_CLASSES: tp.Final = tuple(STATE_METADATA)


def _dummy(state_class: type) -> object:
    """Build an instance with placeholder values, without fitting anything.

    Fitting eight different GP families to test a container property would measure the fits
    rather than the containers. The fields are read from the dataclass itself, so a field
    added later is covered without editing this helper.

    Args:
        state_class: The container to instantiate.

    Returns:
        An instance whose fields carry type-appropriate placeholders.
    """
    values: dict[str, object] = {}
    for info in dataclasses.fields(state_class):
        if info.default is not dataclasses.MISSING or (
            info.default_factory is not dataclasses.MISSING
        ):
            continue
        annotation = str(info.type)
        if "int" in annotation and "Array" not in annotation:
            values[info.name] = 2
        elif "Callable" in annotation:
            values[info.name] = jnp.add
        elif "float" in annotation and "Array" not in annotation:
            values[info.name] = 0.5
        else:
            values[info.name] = jnp.ones((2, 2))
    return state_class(**values)


class TestEveryStateIsAPytree:
    """Registration, and the data/metadata split it depends on."""

    @pytest.mark.parametrize("state_class", STATE_CLASSES, ids=lambda c: c.__name__)
    def test_it_flattens_to_its_array_fields(self, state_class: type) -> None:
        """An unregistered container flattens to one opaque leaf: itself."""
        leaves = jax.tree.leaves(_dummy(state_class))

        assert leaves, f"{state_class.__name__} produced no leaves"
        assert not any(isinstance(leaf, state_class) for leaf in leaves), (
            f"{state_class.__name__} is still an opaque leaf, so it is not registered"
        )

    @pytest.mark.parametrize("state_class", STATE_CLASSES, ids=lambda c: c.__name__)
    def test_the_metadata_fields_are_exactly_the_declared_ones(self, state_class: type) -> None:
        """The silent half. A misclassified field changes nothing visible until it does.

        An array marked metadata is baked into the treedef, so it is frozen at trace time and
        each new value recompiles rather than flowing through; a callable marked data breaks
        flattening outright.
        """
        marked = {
            info.name
            for info in dataclasses.fields(state_class)
            if info.metadata.get("pytree_node") is False
        }

        assert marked == STATE_METADATA[state_class]

    @pytest.mark.parametrize("state_class", STATE_CLASSES, ids=lambda c: c.__name__)
    def test_it_round_trips_through_flatten_and_unflatten(self, state_class: type) -> None:
        """What `jit` does at every boundary."""
        state = _dummy(state_class)
        leaves, treedef = jax.tree.flatten(state)
        rebuilt = jax.tree.unflatten(treedef, leaves)

        assert type(rebuilt) is state_class
        for info in dataclasses.fields(state_class):
            original, restored = getattr(state, info.name), getattr(rebuilt, info.name)
            if isinstance(original, jax.Array):
                assert jnp.array_equal(original, restored)
            else:
                assert original == restored

    @pytest.mark.parametrize("state_class", STATE_CLASSES, ids=lambda c: c.__name__)
    def test_unflattening_tolerates_placeholder_values(self, state_class: type) -> None:
        """JAX rebuilds containers with placeholders while tracing.

        ``docs/101/pytrees.md`` warns that transformations "build instances of your type with
        placeholder objects standing in for the real contents", so ``__init__`` must not
        validate or convert. None of these defines ``__post_init__``; this is what makes that
        a property rather than a note.
        """
        _, treedef = jax.tree.flatten(_dummy(state_class))
        placeholders = [object() for _ in range(treedef.num_leaves)]

        jax.tree.unflatten(treedef, placeholders)


class TestTheStateCrossesTransformBoundaries:
    """The capability the registration exists for, on the one state with a real fit path."""

    @staticmethod
    def _fitted() -> tuple[ExactGPState, jax.Array]:
        """A small fitted GP.

        Returns:
            The fitted state and the inputs it was fitted on.
        """
        x = jnp.linspace(-1.0, 1.0, TRAIN_SIZE).reshape(-1, 1)
        y = jnp.sin(3.0 * x[:, 0])
        return fit_exact_gp(
            x_train=x,
            y_train=y,
            lengthscale=LENGTHSCALE,
            output_scale=OUTPUT_SCALE,
            noise_std=NOISE_STD,
        ), x

    def test_a_jitted_fit_can_return_the_state(self) -> None:
        """This raised ``TypeError`` before registration.

        ``isfinite`` was the whole numeric assertion here, which a fit that solved against
        the wrong right-hand side passes. The budget is torch-harmonics' reduction tolerance
        over the ``n``-term triangular solves; see :mod:`tests.uncertainty.gp._accuracy`.
        """
        x = jnp.linspace(-1.0, 1.0, TRAIN_SIZE).reshape(-1, 1)
        y = jnp.sin(3.0 * x[:, 0])

        state = jax.jit(
            lambda a, b: fit_exact_gp(
                x_train=a,
                y_train=b,
                lengthscale=LENGTHSCALE,
                output_scale=OUTPUT_SCALE,
                noise_std=NOISE_STD,
            )
        )(x, y)

        assert isinstance(state, ExactGPState)
        eager, _ = self._fitted()
        assert_reduction_agreement(
            state.cholesky, eager.cholesky, terms=TRAIN_SIZE, what="cholesky, compiled fit"
        )
        assert_reduction_agreement(
            state.alpha, eager.alpha, terms=TRAIN_SIZE, what="alpha, compiled fit"
        )

    def test_a_jitted_function_can_take_the_state(self) -> None:
        """The other direction, which is how a cached fit gets reused.

        The variance passes an explicit ``scale``. It is ``k_diag - sum(v * v)``, and at a
        training point the GP nearly interpolates, so the result is about 7.7e-03 while both
        operands are about 1. The rounding happens at the operands' scale --- measured at
        exactly 0.5 to 1.0 ULP of ``k_diag`` on both backends in both precisions --- so
        scaling the budget by the cancelled result instead violates it at 1.63x.
        """
        state, x = self._fitted()

        predicted = jax.jit(lambda s, q: predict_exact_gp(state=s, x_test=q))(state, x)
        eager = predict_exact_gp(state=state, x_test=x)

        assert predicted.mean is not None, "the predictive distribution carries no mean"
        assert predicted.mean.shape == (TRAIN_SIZE,)
        assert_reduction_agreement(
            jnp.asarray(predicted.mean),
            jnp.asarray(eager.mean),
            terms=TRAIN_SIZE,
            what="mean, compiled predict",
        )
        assert_reduction_agreement(
            jnp.asarray(predicted.variance),
            jnp.asarray(eager.variance),
            terms=TRAIN_SIZE,
            what="variance, compiled predict",
            scale=OUTPUT_SCALE**2,
        )

    def test_vmap_over_fits_produces_one_batched_state(self) -> None:
        """Fitting a batch of targets at once, which is how a curve over seeds is built.

        An unregistered container cannot be a ``vmap`` output at all, and a misclassified
        array field would be batched into the treedef rather than the leaves. Batching must
        also not change the fit: measured, ``alpha`` is bit-identical to the loop on CPU and
        differs by 2064 ULP on CUDA float32, so bit-equality is not the assertable property
        and the published reduction budget is.
        """
        x = jnp.linspace(-1.0, 1.0, TRAIN_SIZE).reshape(-1, 1)
        targets = jnp.stack([jnp.sin(3.0 * x[:, 0]), jnp.cos(3.0 * x[:, 0])])

        states = jax.vmap(
            lambda y: fit_exact_gp(
                x_train=x,
                y_train=y,
                lengthscale=LENGTHSCALE,
                output_scale=OUTPUT_SCALE,
                noise_std=NOISE_STD,
            )
        )(targets)

        assert isinstance(states, ExactGPState)
        assert states.alpha.shape == (2, TRAIN_SIZE)
        looped = jnp.stack(
            [
                fit_exact_gp(
                    x_train=x,
                    y_train=target,
                    lengthscale=LENGTHSCALE,
                    output_scale=OUTPUT_SCALE,
                    noise_std=NOISE_STD,
                ).alpha
                for target in targets
            ]
        )
        assert_reduction_agreement(
            states.alpha, looped, terms=TRAIN_SIZE, what="alpha, batched versus looped fit"
        )

    def test_vmap_predicts_from_a_batch_of_states(self) -> None:
        """The other half: a batched state consumed by a vmapped prediction."""
        x = jnp.linspace(-1.0, 1.0, TRAIN_SIZE).reshape(-1, 1)
        targets = jnp.stack([jnp.sin(3.0 * x[:, 0]), jnp.cos(3.0 * x[:, 0])])
        fit = functools.partial(
            fit_exact_gp,
            x_train=x,
            lengthscale=LENGTHSCALE,
            output_scale=OUTPUT_SCALE,
            noise_std=NOISE_STD,
        )
        states = jax.vmap(lambda y: fit(y_train=y))(targets)

        means = jax.vmap(lambda s: predict_exact_gp(state=s, x_test=x).mean)(states)

        assert means.shape == (2, TRAIN_SIZE)
        looped = jnp.stack(
            [predict_exact_gp(state=fit(y_train=target), x_test=x).mean for target in targets]
        )
        assert_reduction_agreement(
            means, looped, terms=TRAIN_SIZE, what="mean, batched versus looped predict"
        )

    def test_gradients_flow_through_the_state(self) -> None:
        """Every data field is differentiable; the metadata field is not a leaf.

        ``isfinite`` on the leaves was the whole numeric assertion, and a ``stop_gradient``
        on any one field passes it, because the remaining fields still carry signal. The
        gradient with respect to ``alpha`` is checked against the value the chain rule fixes
        exactly: ``sum(mean)`` is linear in ``alpha`` with coefficient ``sum(k_cross, axis=1)``,
        so no tolerance is needed beyond the reduction budget for that one sum.
        """
        state, x = self._fitted()

        grads = jax.grad(lambda s: jnp.sum(predict_exact_gp(state=s, x_test=x).mean))(state)

        assert len(jax.tree.leaves(grads)) == len(jax.tree.leaves(state))
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(grads))
        expected = jnp.sum(
            state.kernel_fn(
                state.x_train,
                x,
                lengthscale=state.lengthscale,
                output_scale=state.output_scale,
            ),
            axis=1,
        )
        assert_reduction_agreement(
            grads.alpha, expected, terms=TRAIN_SIZE, what="d sum(mean) / d alpha"
        )
        assert float(jnp.max(jnp.abs(grads.alpha))) > 0.0


class TestTheStateWorksUnderNnxTransforms:
    """Interop, not a requirement of this module, and the difference is deliberate.

    This module defines no ``nnx.Module``; its required compatibility is with JAX's own
    transforms, above. NNX consumes standard JAX pytrees, so these pass once the class is
    registered --- and they are worth asserting because a container that silently became a
    graph node, or whose float hyperparameters were inferred static, would fail exactly here
    and nowhere else.
    """

    @staticmethod
    def _problem() -> tuple[jax.Array, jax.Array]:
        """A small regression problem.

        Returns:
            Inputs and targets.
        """
        x = jnp.linspace(-1.0, 1.0, TRAIN_SIZE).reshape(-1, 1)
        return x, jnp.sin(3.0 * x[:, 0])

    def test_nnx_jit_can_return_the_state(self) -> None:
        """``nnx.jit`` has its own boundary handling; a plain pytree must pass through it."""
        x, y = self._problem()

        @nnx.jit
        def fit(a: jax.Array, b: jax.Array) -> ExactGPState:
            return fit_exact_gp(
                x_train=a,
                y_train=b,
                lengthscale=LENGTHSCALE,
                output_scale=OUTPUT_SCALE,
                noise_std=NOISE_STD,
            )

        assert isinstance(fit(x, y), ExactGPState)
        assert_reduction_agreement(
            fit(x, y).alpha,
            fit_exact_gp(
                x_train=x,
                y_train=y,
                lengthscale=LENGTHSCALE,
                output_scale=OUTPUT_SCALE,
                noise_std=NOISE_STD,
            ).alpha,
            terms=TRAIN_SIZE,
            what="alpha from a fit compiled by nnx.jit",
        )

    def test_nnx_jit_can_take_the_state(self) -> None:
        """A fitted state passed into a compiled NNX function, giving the same numbers.

        Crossing NNX's boundary must be a no-op numerically. Shape alone would pass for a
        state whose float hyperparameters NNX had inferred static and frozen at the value of
        the first call --- which is the failure mode the class below this one guards against.
        """
        x, y = self._problem()
        state = fit_exact_gp(
            x_train=x,
            y_train=y,
            lengthscale=LENGTHSCALE,
            output_scale=OUTPUT_SCALE,
            noise_std=NOISE_STD,
        )

        @nnx.jit
        def score(s: ExactGPState, q: jax.Array) -> jax.Array:
            return predict_exact_gp(state=s, x_test=q).mean

        produced = score(state, x)

        assert produced.shape == (TRAIN_SIZE,)
        assert_reduction_agreement(
            produced,
            predict_exact_gp(state=state, x_test=x).mean,
            terms=TRAIN_SIZE,
            what="mean under nnx.jit versus eager",
        )

    def test_nnx_grad_differentiates_through_the_state(self) -> None:
        """Gradients reach every data field under NNX's own transform."""
        x, y = self._problem()
        state = fit_exact_gp(
            x_train=x,
            y_train=y,
            lengthscale=LENGTHSCALE,
            output_scale=OUTPUT_SCALE,
            noise_std=NOISE_STD,
        )

        grads = nnx.grad(lambda s: jnp.sum(predict_exact_gp(state=s, x_test=x).mean))(state)

        assert len(jax.tree.leaves(grads)) == len(jax.tree.leaves(state))
        jax_grads = jax.grad(lambda s: jnp.sum(predict_exact_gp(state=s, x_test=x).mean))(state)
        assert_reduction_agreement(
            grads.alpha, jax_grads.alpha, terms=TRAIN_SIZE, what="nnx.grad versus jax.grad"
        )

    def test_a_float_hyperparameter_is_data_and_not_metadata(self) -> None:
        """The reason these are ``flax.struct`` containers rather than ``nnx.Pytree``.

        Measured on flax 0.12.9: ``nnx.is_data(0.5)`` is ``False``, so NNX's own inference
        classifies a Python float as static. A state whose ``lengthscale`` were static could
        not be differentiated with respect to it, which is the capability
        ``exact_gp_loocv_log_predictive`` exists to serve --- so this asserts the opposite
        classification holds here.
        """
        assert not nnx.is_data(0.5), "flax changed its inference; the rationale needs re-reading"

        state, _ = TestTheStateCrossesTransformBoundaries._fitted()
        metadata = {
            info.name
            for info in dataclasses.fields(ExactGPState)
            if info.metadata.get("pytree_node") is False
        }

        assert "lengthscale" not in metadata
        assert len(jax.tree.leaves(state)) == len(dataclasses.fields(ExactGPState)) - len(metadata)
