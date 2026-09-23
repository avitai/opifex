"""Problems, the methods that solve them, and ``solve``.

A problem states *what* to solve and a method states *how*, as two separate objects.
This is the separation of Rackauckas and Nie (2017, J. Open Res. Softw. 5(1), 15), in
which ``solve(prob, alg)`` selects its implementation by the type of the algorithm: here
the method owns ``solve``, and the free :func:`solve` checks that the method was given a
problem of the family it solves before calling it.

Both are frozen pytrees. A problem's instance data -- coefficients, initial and boundary
data, the time interval -- are leaves, so a sweep over them compiles once and
differentiates; what sets an array shape or selects a code path is static metadata. A
method holds its configuration as static metadata and anything it learns as leaves, so
one compiled program serves every problem instance and a new program is compiled only
for a new method configuration.

The family a method solves is its generic argument, ``class M(Method[SomeProblem])``,
stated once. A type checker then rejects a mismatched pairing before anything runs, and
:func:`solve` rejects it at run time for callers the checker does not see.
"""

import abc
import typing
from typing import Any, ClassVar

from flax import struct

from opifex.core.solver.solution import Solution


class Problem(struct.PyTreeNode):
    """The base of every problem family.

    Subclasses declare their fields as a dataclass does. Instance data are pytree leaves;
    a field that sets a shape or a code path is declared ``struct.field(pytree_node=False)``.
    Validation belongs in a ``create`` classmethod rather than ``__post_init__``, which a
    struct dataclass runs again on every unflatten, including inside ``jit``.
    """


class IncompatibleProblemError(TypeError):
    """A method was given a problem outside the family it solves."""


class Method[P: Problem](struct.PyTreeNode, abc.ABC):
    """A way of solving one family of problems.

    Attributes:
        problem_type: The family this method solves, taken from the generic argument of
            the class that first specialises ``Method``; ``None`` while the class is
            still generic in its problem.
    """

    problem_type: ClassVar[type[Problem] | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Record the problem family named by this class's ``Method[...]`` base."""
        super().__init_subclass__(**kwargs)
        for base in cls.__dict__.get("__orig_bases__", ()):
            if typing.get_origin(base) is not Method:
                continue
            (argument,) = typing.get_args(base)
            if isinstance(argument, type) and issubclass(argument, Problem):
                cls.problem_type = argument

    @abc.abstractmethod
    def solve(self, problem: P) -> Solution:
        """Solve one problem of this method's family.

        Args:
            problem: The problem instance.

        Returns:
            The solution.
        """


def solve[P: Problem](problem: P, method: Method[P]) -> Solution:
    """Solve ``problem`` with ``method``, after checking that they belong together.

    Args:
        problem: The problem instance.
        method: A method whose family includes the problem's type.

    Returns:
        The solution the method produces.

    Raises:
        IncompatibleProblemError: If the method declares no family, or the problem is
            not a member of it.
    """
    family = method.problem_type
    method_name = type(method).__name__
    if family is None:
        msg = (
            f"{method_name} does not declare the problem family it solves; "
            f"specialise it as Method[SomeProblem]"
        )
        raise IncompatibleProblemError(msg)
    if not isinstance(problem, family):
        msg = f"{method_name} solves {family.__name__} problems, not {type(problem).__name__}"
        raise IncompatibleProblemError(msg)
    return method.solve(problem)


__all__ = ["IncompatibleProblemError", "Method", "Problem", "solve"]
