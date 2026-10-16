"""Power-cone atoms: elementwise powers, square root and inverse."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from akme.atoms.base import Atom
from akme.atoms.quadratic import Square
from akme.expressions.affine import AffineExpr, Constant
from akme.expressions.expression import (
    Curvature,
    Expression,
    require_affine,
)

if TYPE_CHECKING:
    from akme.canon import CanonContext


class Power(Atom):
    """Elementwise power ``a^p`` for ``p`` not in {0, 1, 2}.

    ``p > 1``: convex, evaluates to ``|a|^p``; ``0 < p < 1``: concave (``a >= 0``);
    ``p < 0``: convex (``a > 0``).
    """

    def __init__(self, a: AffineExpr, p: float) -> None:
        """Create ``a^p``."""
        super().__init__(a, shape=a.shape)
        self.p = float(p)
        self._curvature = Curvature.CONCAVE if 0 < self.p < 1 else Curvature.CONVEX

    def numeric(self, a: np.ndarray) -> np.ndarray:
        """a^p."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.power(np.abs(a), self.p) if self.p > 1 else np.power(a, self.p)

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Power-cone representation depending on the range of ``p``."""
        (a,) = self.args
        t = ctx.new_var(self.shape, source=self)
        one = Constant(1.0)
        if self.p > 1:
            ctx.pow3(t, one, a, 1.0 / self.p)  # t^(1/p) >= |a|
        elif self.p > 0:
            ctx.pow3(a, one, t, self.p)  # a^p >= |t|
        else:
            ctx.pow3(t, a, one, 1.0 / (1.0 - self.p))  # t a^(-p) >= 1
        return t


def power(x: Any, p: float) -> Expression:
    """Elementwise power (see :class:`Power` for curvature and domain)."""
    a = require_affine(x, "power")
    p = float(p)
    if p == 1:
        return a
    if p == 0:
        return Constant(np.ones(a.shape))
    if p == 2:
        return Square(a, shape=a.shape)
    return Power(a, p)


def sqrt(x: Any) -> Expression:
    """Elementwise square root (concave)."""
    return power(x, 0.5)


def inv_pos(x: Any) -> Expression:
    """Elementwise ``1 / x`` for ``x > 0`` (convex)."""
    return power(x, -1)
