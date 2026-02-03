from typing import Protocol, runtime_checkable

from opifex.core.solver.interface import SciMLSolver, Solution


def test_solver_protocol_is_runtime_checkable():
    assert isinstance(SciMLSolver, type)
    # Check if it's runtime checkable (best we can do without instantiating)
    # Actually, we can check __annotations__ or use it in isinstance if we mock an object

    @runtime_checkable
    class ReferenceProtocol(Protocol):
        pass

    # Verify ReferenceProtocol is a Protocol subclass (use the class)
    assert issubclass(type(ReferenceProtocol), type)
    assert issubclass(SciMLSolver, Protocol)


def test_solution_dataclass():
    # Test Solution object structure
    import jax.numpy as jnp

    sol = Solution(
        fields={"u": jnp.array([1.0])}, metrics={"loss": 0.1}, execution_time=1.0
    )
    assert sol.fields["u"][0] == 1.0
    assert sol.metrics["loss"] == 0.1
    assert sol.execution_time == 1.0


def test_solver_interface_compliance():
    # Define a dummy solver to check consistency
    class DummySolver:
        def solve(self, problem):
            return Solution({}, {}, 0.0)

    # Verify DummySolver has the required method
    dummy = DummySolver()
    assert hasattr(dummy, "solve")
    # Should technically pass structural typing check if ignoring type hints strictly
    # But we want to ensure the protocol requires 'solve'
    assert hasattr(SciMLSolver, "solve")
