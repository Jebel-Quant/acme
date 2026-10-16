"""Norm atoms: 1-, 2-, infinity- and general p-norms of the flattened argument."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from acme.atoms.affine import sum as _sum
from acme.atoms.base import Atom
from acme.atoms.piecewise import abs as _abs
from acme.expressions.affine import AffineExpr
from acme.expressions.expression import (
    DCPError,
    Expression,
    require_affine,
)
from acme.expressions.expression import vec as _vec

if TYPE_CHECKING:
    from acme.canon import CanonContext


class NormInf(Atom):
    """Infinity norm (of the flattened argument)."""

    def numeric(self, a: np.ndarray) -> float:
        """Largest absolute entry."""
        return float(np.max(np.abs(a)))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Epigraph: t >= a_i, t >= -a_i."""
        (a,) = self.args
        t = ctx.new_var((), source=self)
        ctx.nonneg(t - a)
        ctx.nonneg(t + a)
        return t


class Norm2(Atom):
    """Euclidean (Frobenius) norm of the flattened argument."""

    def numeric(self, a: np.ndarray) -> float:
        """||a||_2."""
        return float(np.linalg.norm(_vec(a)))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """(t, a) in SOC."""
        (a,) = self.args
        t = ctx.new_var((), source=self)
        ctx.soc(t, a.flatten())
        return t


class PNorm(Atom):
    """p-norm (1 < p < inf, p != 2) of the flattened argument via power cones."""

    def __init__(self, a: AffineExpr, p: float) -> None:
        """Create ``||a||_p``."""
        super().__init__(a, shape=())
        self.p = float(p)

    def numeric(self, a: np.ndarray) -> float:
        """||a||_p."""
        return float(np.linalg.norm(_vec(a), self.p))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Epigraph: sum r = t, (r_i, t, a_i) in Pow(1/p)."""
        (a,) = self.args
        t = ctx.new_var((), source=self)
        r = ctx.new_var((a.size,))
        ctx.zero(t - _sum(r))
        ctx.pow3(r, t, a.flatten(), 1.0 / self.p)
        return t


def norm(x: Any, p: float | str = 2) -> Expression:
    """Vector p-norm of the flattened argument (``p`` in [1, inf], ``"fro"`` = 2).

    Example:
        >>> import numpy as np
        >>> import acme as ak
        >>> x = ak.Variable(2)
        >>> x.value = np.array([3.0, -4.0])
        >>> float(ak.norm(x).value), float(ak.norm(x, 1).value), float(ak.norm(x, "inf").value)
        (5.0, 7.0, 4.0)
        >>> ak.norm(x).curvature
        <Curvature.CONVEX: 'convex'>
    """
    a = require_affine(x, "norm")
    if p in ("fro", 2):
        return Norm2(a, shape=())
    if p == 1:
        return _sum(_abs(a))
    if p in ("inf", np.inf):
        return NormInf(a.flatten(), shape=())
    p = float(p)
    if p < 1:
        msg = "norm requires p >= 1"
        raise DCPError(msg)
    return PNorm(a, p)


def norm1(x: Any) -> Expression:
    """1-norm."""
    return norm(x, 1)


def norm2(x: Any) -> Expression:
    """Euclidean norm."""
    return norm(x, 2)


def norm_inf(x: Any) -> Expression:
    """Infinity norm."""
    return norm(x, np.inf)


def pnorm(x: Any, p: float = 2) -> Expression:
    """p-norm (alias of :func:`norm`)."""
    return norm(x, p)
