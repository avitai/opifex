"""Generate the extended-precision process-noise references used by the state-space tests.

Writes ``tests/uncertainty/statespace/_process_noise_references.py``. For each linear
time-invariant SDE ``dx = F x dt + L dW`` with diffusion ``Q_c`` it records the transition
``A = exp(F dt)`` and the process noise ``Q = int_0^dt exp(F s) L Q_c L^T exp(F^T s) ds`` at steps
from 1e-4 to 1e4, computed with mpmath and rounded once to float64:

* Matern-1/2 to Matern-7/2 in companion form with unit variance and lengthscale, as built by
  ``opifex.uncertainty.statespace``. ``F + lambda I`` is nilpotent, so the truncated series for
  ``A`` is exact. ``P_inf`` solves the Lyapunov equation and ``Q = P_inf - A P_inf A^T`` is
  evaluated at 160 digits, far above the cancellation at small steps.
* Integrated Wiener processes of order 1 to 4, from the closed form of Kramer & Hennig
  (arXiv:2012.10106, section 3.2).
* Integrated Ornstein-Uhlenbeck processes of order 2 and 3 with rate -1, from Van Loan's block
  exponential (Van Loan 1978, theorem 1) at ``70 + 2 dt / ln 10`` digits, enough for the
  ``exp(dt)`` block to cancel exactly.

Usage::

    uv run --no-project --with mpmath python scripts/generate_process_noise_references.py
    uv run ruff format tests/uncertainty/statespace/_process_noise_references.py
"""

from __future__ import annotations

import math
from pathlib import Path

import mpmath as mp


OUTPUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "uncertainty"
    / "statespace"
    / "_process_noise_references.py"
)
STEP_SCALES = (1e-4, 1e-2, 1.0, 1e2, 1e4)


def _lyapunov(drift: mp.matrix, noise: mp.matrix, size: int) -> mp.matrix:
    """Solve ``F P + P F^T + noise = 0`` by vectorisation."""
    system = mp.zeros(size * size)
    rhs = mp.zeros(size * size, 1)
    for row in range(size):
        for column in range(size):
            index = row * size + column
            for k in range(size):
                system[index, k * size + column] += drift[row, k]
                system[index, row * size + k] += drift[column, k]
            rhs[index] = -noise[row, column]
    solution = mp.lu_solve(system, rhs)
    stationary = mp.zeros(size)
    for row in range(size):
        for column in range(size):
            stationary[row, column] = solution[row * size + column]
    return stationary


def _matern(order: int) -> dict[str, object]:
    """Return the Matern-(order + 1/2) SDE and its discretisation at every step scale."""
    mp.mp.dps = 160
    size = order + 1
    rate = mp.sqrt(2 * (mp.mpf(order) + mp.mpf(1) / 2))
    drift = mp.zeros(size)
    for row in range(size - 1):
        drift[row, row + 1] = 1
    for column in range(size):
        drift[size - 1, column] = -mp.binomial(size, column) * rate ** (size - column)
    diffusion = (
        2
        * mp.sqrt(mp.pi)
        * rate ** (2 * order + 1)
        * mp.gamma(order + 1)
        / mp.gamma(order + mp.mpf(1) / 2)
    )
    dispersion = mp.zeros(size, 1)
    dispersion[size - 1, 0] = 1
    stationary = _lyapunov(drift, dispersion * diffusion * dispersion.T, size)
    if abs(stationary[0, 0] - 1) > mp.mpf(10) ** -100:
        raise ValueError(f"Matern-{order}/2 stationary variance is {stationary[0, 0]}, not 1")
    nilpotent = drift + rate * mp.eye(size)
    steps = []
    for scale in STEP_SCALES:
        dt = mp.mpf(scale) / rate
        series = mp.eye(size)
        term = mp.eye(size)
        for power in range(1, size):
            term = term * nilpotent * dt / power
            series += term
        transition = mp.exp(-rate * dt) * series
        noise = stationary - transition * stationary * transition.T
        steps.append({"scale": scale, "dt": dt, "transition": transition, "process_noise": noise})
    return {
        "name": f"matern{2 * order + 1}2",
        "drift": drift,
        "dispersion": dispersion,
        "diffusion": mp.matrix([[diffusion]]),
        "stationary_cov": stationary,
        "steps": steps,
    }


def _integrated_wiener(order: int) -> dict[str, object]:
    """Return the order-``order`` integrated Wiener process and its closed-form discretisation."""
    mp.mp.dps = 60
    size = order + 1
    drift = mp.zeros(size)
    for row in range(size - 1):
        drift[row, row + 1] = 1
    dispersion = mp.zeros(size, 1)
    dispersion[size - 1, 0] = 1
    steps = []
    for scale in STEP_SCALES:
        dt = mp.mpf(scale)
        transition = mp.zeros(size)
        noise = mp.zeros(size)
        for row in range(size):
            for column in range(size):
                if column >= row:
                    transition[row, column] = dt ** (column - row) / mp.factorial(column - row)
                power = 2 * order + 1 - row - column
                noise[row, column] = dt**power / (
                    power * mp.factorial(order - row) * mp.factorial(order - column)
                )
        steps.append({"scale": scale, "dt": dt, "transition": transition, "process_noise": noise})
    return {
        "name": f"iwp{order}",
        "drift": drift,
        "dispersion": dispersion,
        "diffusion": mp.matrix([[1]]),
        "stationary_cov": None,
        "steps": steps,
    }


def _integrated_ornstein_uhlenbeck(order: int) -> dict[str, object]:
    """Return the order-``order`` IOUP with rate -1 and its Van Loan discretisation."""
    size = order + 1
    steps = []
    drift = None
    dispersion = None
    for scale in STEP_SCALES:
        mp.mp.dps = 70 + int(2 * scale / math.log(10))
        drift = mp.zeros(size)
        for row in range(size - 1):
            drift[row, row + 1] = 1
        drift[size - 1, size - 1] = -1
        dispersion = mp.zeros(size, 1)
        dispersion[size - 1, 0] = 1
        dt = mp.mpf(scale)
        block = mp.zeros(2 * size)
        noise_rate = dispersion * dispersion.T
        for row in range(size):
            for column in range(size):
                block[row, column] = drift[row, column] * dt
                block[row, size + column] = noise_rate[row, column] * dt
                block[size + row, size + column] = -drift[column, row] * dt
        exponential = mp.expm(block)
        transition = mp.zeros(size)
        upper_right = mp.zeros(size)
        for row in range(size):
            for column in range(size):
                transition[row, column] = exponential[row, column]
                upper_right[row, column] = exponential[row, size + column]
        noise = upper_right * transition.T
        noise = (noise + noise.T) / 2
        steps.append({"scale": scale, "dt": dt, "transition": transition, "process_noise": noise})
    return {
        "name": f"ioup{order}",
        "drift": drift,
        "dispersion": dispersion,
        "diffusion": mp.matrix([[1]]),
        "stationary_cov": None,
        "steps": steps,
    }


def _matrix_literal(matrix: mp.matrix) -> str:
    """Return a nested-tuple literal of the matrix rounded to float64."""
    rows = []
    for row in range(matrix.rows):
        values = ", ".join(repr(float(matrix[row, column])) for column in range(matrix.cols))
        rows.append(f"({values},)")
    return "(" + ", ".join(rows) + ",)"


def _sde_literal(sde: dict[str, object]) -> str:
    """Return the ``SdeReference(...)`` literal of one generated SDE."""
    steps = []
    for step in sde["steps"]:
        steps.append(
            "StepReference("
            f"scale={step['scale']!r}, dt={float(step['dt'])!r}, "
            f"transition={_matrix_literal(step['transition'])}, "
            f"process_noise={_matrix_literal(step['process_noise'])})"
        )
    stationary = sde["stationary_cov"]
    stationary_literal = "None" if stationary is None else _matrix_literal(stationary)
    return (
        f"SdeReference(name={sde['name']!r}, "
        f"drift={_matrix_literal(sde['drift'])}, "
        f"dispersion={_matrix_literal(sde['dispersion'])}, "
        f"diffusion={_matrix_literal(sde['diffusion'])}, "
        f"stationary_cov={stationary_literal}, "
        f"steps=({', '.join(steps)},))"
    )


HEADER = '''"""Extended-precision process-noise references for the state-space tests.

Generated by ``scripts/generate_process_noise_references.py`` with mpmath {version}; do not edit by
hand. Each ``SdeReference`` holds a linear time-invariant SDE ``dx = F x dt + L dW`` with diffusion
``Q_c`` and, per step, the transition ``exp(F dt)`` and the process noise
``int_0^dt exp(F s) L Q_c L^T exp(F^T s) ds``, rounded once to float64. Matern SDEs use unit
variance and lengthscale and step scales ``lambda dt``; integrated processes use ``dt`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass


type Matrix = tuple[tuple[float, ...], ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class StepReference:
    """Transition and process noise of one step."""

    scale: float
    dt: float
    transition: Matrix
    process_noise: Matrix


@dataclass(frozen=True, slots=True, kw_only=True)
class SdeReference:
    """A linear time-invariant SDE and its reference discretisation at every step scale."""

    name: str
    drift: Matrix
    dispersion: Matrix
    diffusion: Matrix
    stationary_cov: Matrix | None
    steps: tuple[StepReference, ...]

'''


def main() -> None:
    """Generate every reference and write the test module."""
    sdes = [_matern(order) for order in range(4)]
    sdes += [_integrated_wiener(order) for order in (1, 2, 3, 4)]
    sdes += [_integrated_ornstein_uhlenbeck(order) for order in (2, 3)]
    body = ",\n".join(f"    {sde['name']!r}: {_sde_literal(sde)}" for sde in sdes)
    text = (
        HEADER.format(version=mp.__version__)
        + "\nREFERENCES: dict[str, SdeReference] = {\n"
        + body
        + ",\n}\n"
    )
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT} ({len(sdes)} SDEs)")


if __name__ == "__main__":
    main()
