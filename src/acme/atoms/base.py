"""The :class:`Atom` base class shared by every nonlinear atom."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from acme.expressions.affine import AffineExpr
from acme.expressions.expression import (
    Curvature,
    Expression,
    unique_variables,
    unvec,
)
from acme.expressions.expression import vec as _vec

if TYPE_CHECKING:
    from acme.canon import CanonContext
    from acme.expressions.variable import Variable


class Atom(Expression):
    """Base class of nonlinear atoms with affine arguments."""

    _curvature: Curvature = Curvature.CONVEX
    #: Quadratic atoms are kept in the objective's ``P`` instead of being lifted to cones.
    is_quadratic = False

    def __init__(self, *args: AffineExpr, shape: tuple[int, ...]) -> None:
        """Store the affine arguments and the result shape."""
        self.args = list(args)
        self.shape = tuple(shape)

    @property
    def curvature(self) -> Curvature:
        """Curvature of the atom."""
        return self._curvature

    @property
    def value(self) -> np.ndarray | float | None:
        """Numeric value of the atom at the current variable values."""
        values = [a.value for a in self.args]
        if any(v is None for v in values):
            return None
        return unvec(_vec(self.evaluate([np.asarray(v, dtype=float) for v in values])), self.shape)

    def variables(self) -> list[Variable]:
        """Variables of the arguments."""
        return unique_variables(v for a in self.args for v in a.variables())

    def evaluate(self, values: list[np.ndarray]) -> np.ndarray | float:
        """Evaluate the atom on numeric argument values (one array per argument)."""
        return self.numeric(values[0])

    def numeric(self, a: np.ndarray) -> np.ndarray | float:
        """Evaluate a single-argument atom on a numeric value."""
        raise NotImplementedError

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Return an affine epigraph (convex) / hypograph (concave) bound, adding cones to ``ctx``."""
        raise NotImplementedError

    def add_quadratic(self, ctx: CanonContext, weights: np.ndarray) -> None:
        """Add ``weights . self`` to the quadratic part of the objective (quadratic atoms only)."""
        raise NotImplementedError
