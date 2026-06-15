r"""Variational Monte Carlo energy-minimisation driver.

Minimises the variational energy :math:`E[\theta] = \langle E_{\mathrm{loc}}
\rangle_{|\psi_\theta|^2}` of a neural wavefunction. Each iteration:

#. samples walkers from :math:`|\psi_\theta|^2` with the injected
   :class:`~opifex.neural.quantum.vmc.protocols.Sampler`;
#. evaluates the per-walker local energy
   (:mod:`~opifex.neural.quantum.vmc.hamiltonian`);
#. forms a parameter update, using either

   * the **FermiNet custom-gradient estimator** -- the score-function gradient
     :math:`\nabla_\theta E = 2\,\langle (E_{\mathrm{loc}} - \langle
     E_{\mathrm{loc}}\rangle)\,\nabla_\theta \log|\psi|\rangle` with
     :math:`E_{\mathrm{loc}}` as a stop-gradient baseline -- fed to an
     ``optax`` optimiser (the Adam bootstrap), or
   * **MinSR / SPRING** natural-gradient preconditioning of that same signal
     (:mod:`~opifex.neural.quantum.vmc.optimizers`).

The whole step is ``jit``-compiled. The reported energy is the mean local energy
over the final batch with its Monte Carlo standard error
:math:`\sigma / \sqrt{N}`.
"""

from __future__ import annotations

import math
from collections.abc import Callable  # noqa: TC003
from dataclasses import dataclass, field
from typing import Literal

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float  # noqa: TC002

from opifex.neural.quantum.vmc.hamiltonian import KineticMethod, local_energy
from opifex.neural.quantum.vmc.optimizers import (
    minsr_update,
    spring_update,
    SpringState,
)
from opifex.neural.quantum.vmc.protocols import Sampler  # noqa: TC001
from opifex.neural.quantum.vmc.wavefunctions import FermiNet  # noqa: TC001


OptimizerName = Literal["adam", "minsr", "spring"]


@dataclass(frozen=True, slots=True, kw_only=True)
class VMCConfig:
    """Configuration for a VMC optimisation run.

    Args:
        batch_size: Number of MCMC walkers.
        iterations: Number of optimisation steps.
        optimizer: ``"adam"``, ``"minsr"`` or ``"spring"``.
        learning_rate: Step size applied to the parameter update.
        equilibration_steps: Burn-in MCMC sweeps before optimisation begins.
        diag_shift: Tikhonov shift for the MinSR/SPRING Gram solve.
        momentum: SPRING momentum coefficient.
        proj_reg: SPRING projection regulariser.
        clip_local_energy: Median-absolute-deviation clipping window for the
            local energy (FermiNet variance reduction); ``0`` disables it.
        kinetic_method: Laplacian method for the kinetic energy.
    """

    batch_size: int = 1024
    iterations: int = 500
    optimizer: OptimizerName = "spring"
    learning_rate: float = 0.05
    equilibration_steps: int = 200
    diag_shift: float = 1e-3
    momentum: float = 0.99
    proj_reg: float = 1e-3
    clip_local_energy: float = 5.0
    kinetic_method: KineticMethod = "forward"


@dataclass(frozen=True, slots=True)
class VMCResult:
    """Outcome of a VMC run.

    Args:
        energy: Mean local energy over the final batch (Hartree).
        energy_error: Monte Carlo standard error of the mean energy.
        energy_history: Per-iteration mean energy.
        walkers: Final walker positions.
    """

    energy: Array
    energy_error: Array
    energy_history: Float[Array, " iterations"]
    walkers: Float[Array, "batch nelectron ndim"]


def _clip_local_energy(energies: Array, window: float) -> Array:
    """Median-absolute-deviation clip the local energy (variance reduction).

    Non-finite local energies -- which arise when a walker wanders onto a
    nuclear cusp and the kinetic energy diverges -- are first replaced by the
    finite median so a single bad walker cannot poison the batch gradient
    (the FermiNet / DeepQMC outlier-robust estimator). The remaining energies
    are then clipped to a median-absolute-deviation window.
    """
    finite = jnp.isfinite(energies)
    safe = jnp.where(finite, energies, 0.0)
    # Median over the finite subset (infs masked to the median itself).
    centre = jnp.median(jnp.where(finite, safe, jnp.median(safe)))
    energies = jnp.where(finite, energies, centre)
    if window <= 0.0:
        return energies
    deviation = jnp.mean(jnp.abs(energies - centre))
    lower = centre - window * deviation
    upper = centre + window * deviation
    return jnp.clip(energies, lower, upper)


@dataclass(slots=True)
class VMCDriver:
    """Drives VMC energy minimisation of a :class:`FermiNet` ansatz.

    Args:
        ansatz: The wavefunction module.
        sampler: The Born-density sampler.
        config: Optimisation configuration.
    """

    ansatz: FermiNet
    sampler: Sampler
    config: VMCConfig
    _graphdef: nnx.GraphDef = field(init=False)

    def __post_init__(self) -> None:
        """Capture the static graph definition of the ansatz."""
        self._graphdef, _ = nnx.split(self.ansatz)

    def _log_abs_of(self, state: nnx.State) -> Callable[[Array], Array]:
        """Build a ``positions -> log|psi|`` closure for the given parameters."""

        def log_abs(positions: Array) -> Array:
            model = nnx.merge(self._graphdef, state)
            return model(positions)[1]

        return log_abs

    def _local_energies(self, state: nnx.State, walkers: Array) -> Array:
        """Vectorised local energy of every walker for the given parameters."""
        energy_fn = local_energy(
            self._log_abs_of(state),
            atoms=self.ansatz.atoms,
            charges=self.ansatz.charges,
            method=self.config.kinetic_method,
        )
        return jax.vmap(energy_fn)(walkers)

    def _per_sample_jacobian(self, state: nnx.State, walkers: Array) -> tuple[Array, tuple]:
        """Per-sample Jacobian of ``log|psi|`` flattened over parameters.

        Returns ``(jacobian, structure)`` where ``jacobian`` has shape
        ``(batch, n_params)`` and ``structure`` is the ``(treedef, shapes)``
        needed by :meth:`_unflatten_update` to rebuild the parameter ``State``.
        """
        leaves, treedef = jax.tree.flatten(state)

        def single_jacobian(positions: Array) -> list:
            def log_abs_from_leaves(leaf_list: list) -> Array:
                rebuilt = jax.tree.unflatten(treedef, leaf_list)
                return nnx.merge(self._graphdef, rebuilt)(positions)[1]

            return jax.grad(log_abs_from_leaves)(leaves)

        per_sample = jax.vmap(single_jacobian)(walkers)
        flat = jnp.concatenate(
            [jnp.reshape(leaf, (walkers.shape[0], -1)) for leaf in per_sample],
            axis=1,
        )
        return flat, (treedef, [leaf.shape for leaf in leaves])

    def _unflatten_update(self, update: Array, structure: tuple) -> nnx.State:
        """Reshape a flat update vector back into a parameter ``State`` pytree."""
        treedef, shapes = structure
        leaves = []
        offset = 0
        for shape in shapes:
            size = math.prod(shape) if shape else 1
            leaves.append(jnp.reshape(update[offset : offset + size], shape))
            offset += size
        return jax.tree.unflatten(treedef, leaves)

    def _score_gradient(self, state: nnx.State, walkers: Array) -> tuple[Array, Array]:
        r"""Compute the score-function energy gradient and mean local energy.

        Returns ``(grad_state_pytree, mean_energy)`` where the gradient is the
        FermiNet :math:`2\langle (E_{loc}-\bar E)\nabla\log|\psi|\rangle`
        estimator (the local energy enters as a stop-gradient baseline).
        """
        energies = self._local_energies(state, walkers)
        clipped = _clip_local_energy(energies, self.config.clip_local_energy)
        centred = clipped - jnp.mean(clipped)

        def weighted_log_abs(params_state: nnx.State) -> Array:
            log_abs = self._log_abs_of(params_state)
            per_walker = jax.vmap(log_abs)(walkers)  # type: ignore[arg-type]
            return 2.0 * jnp.mean(jax.lax.stop_gradient(centred) * per_walker)

        grads = jax.grad(weighted_log_abs)(state)
        return grads, jnp.mean(energies)

    def run(self, key: Array) -> VMCResult:
        """Run the full VMC optimisation.

        Args:
            key: PRNG key.

        Returns:
            A :class:`VMCResult` with the final energy, its standard error, the
            energy history and the final walkers.
        """
        config = self.config
        _, state = nnx.split(self.ansatz)
        key, init_key, equil_key = jax.random.split(key, 3)

        nelectron = sum(self.ansatz.nspins)
        ndim = self.ansatz.ndim
        centre = jnp.mean(self.ansatz.atoms, axis=0)
        walkers = centre + jax.random.normal(
            init_key, (config.batch_size, nelectron, ndim), dtype=self.ansatz.atoms.dtype
        )

        # Equilibrate the walkers under the (random) initial wavefunction.
        log_abs = self._log_abs_of(state)
        equil = MetropolisEquilibrator(self.sampler, config.equilibration_steps)
        walkers, _ = equil.run(log_abs, walkers, equil_key)

        optax_opt = optax.adam(config.learning_rate)
        # optax annotates ``Params`` as an array pytree; an ``nnx.State`` is a
        # valid pytree of arrays and works at runtime (the standard nnx+optax
        # functional pattern), but the static types do not line up.
        opt_state: optax.OptState = optax_opt.init(state)  # type: ignore[arg-type]
        spring_state: SpringState | None = None

        @jax.jit
        def adam_step(
            state: nnx.State, opt_state: optax.OptState, walkers: Array, key: Array
        ) -> tuple[nnx.State, optax.OptState, Array, Array]:
            walkers, _ = self.sampler.sample(self._log_abs_of(state), walkers, key)
            grads, energy = self._score_gradient(state, walkers)
            updates, opt_state = optax_opt.update(grads, opt_state)
            state = optax.apply_updates(state, updates)  # type: ignore[assignment]
            return state, opt_state, walkers, energy

        @jax.jit
        def ngd_step(
            state: nnx.State,
            spring_carry: SpringState,
            walkers: Array,
            key: Array,
        ) -> tuple[nnx.State, SpringState, Array, Array]:
            walkers, _ = self.sampler.sample(self._log_abs_of(state), walkers, key)
            energies = self._local_energies(state, walkers)
            clipped = _clip_local_energy(energies, config.clip_local_energy)
            jacobian, structure = self._per_sample_jacobian(state, walkers)
            # A walker on a nuclear cusp can yield a non-finite Jacobian row;
            # zero it so it contributes nothing to the natural-gradient solve.
            jacobian = jnp.where(jnp.isfinite(jacobian), jacobian, 0.0)
            if config.optimizer == "spring":
                update, spring_carry = spring_update(
                    jacobian,
                    clipped,
                    spring_carry,
                    diag_shift=config.diag_shift,
                    momentum=config.momentum,
                    proj_reg=config.proj_reg,
                )
            else:
                update = minsr_update(jacobian, clipped, diag_shift=config.diag_shift)
            update_state = self._unflatten_update(config.learning_rate * update, structure)
            state = jax.tree.map(lambda p, u: p - u, state, update_state)
            return state, spring_carry, walkers, jnp.mean(clipped)

        history = []
        if config.optimizer != "adam":
            n_params = self._per_sample_jacobian(state, walkers[:2])[0].shape[1]
            spring_state = SpringState(old_updates=jnp.zeros(n_params))

        for _ in range(config.iterations):
            key, step_key = jax.random.split(key)
            if config.optimizer == "adam":
                state, opt_state, walkers, energy = adam_step(state, opt_state, walkers, step_key)
            else:
                assert spring_state is not None
                state, spring_state, walkers, energy = ngd_step(
                    state, spring_state, walkers, step_key
                )
            history.append(energy)

        final_energies = self._local_energies(state, walkers)
        clipped = _clip_local_energy(final_energies, config.clip_local_energy)
        mean = jnp.mean(clipped)
        error = jnp.std(clipped) / jnp.sqrt(clipped.shape[0])
        # Persist the optimised parameters back into the module.
        nnx.update(self.ansatz, state)
        return VMCResult(
            energy=mean,
            energy_error=error,
            energy_history=jnp.asarray(history),
            walkers=walkers,
        )


@dataclass(frozen=True, slots=True)
class MetropolisEquilibrator:
    """Run burn-in MCMC sweeps before optimisation (no parameter updates)."""

    sampler: Sampler
    steps: int

    def run(
        self, log_abs: Callable[[Array], Array], walkers: Array, key: Array
    ) -> tuple[Array, Array]:
        """Advance ``walkers`` by repeated sampler calls and report acceptance."""
        acceptance = jnp.array(0.0)
        n = max(self.steps // max(self.sampler.steps, 1), 1)
        for _ in range(n):
            key, subkey = jax.random.split(key)
            walkers, acceptance = self.sampler.sample(log_abs, walkers, subkey)
        return walkers, acceptance


__all__ = [
    "MetropolisEquilibrator",
    "OptimizerName",
    "VMCConfig",
    "VMCDriver",
    "VMCResult",
]
