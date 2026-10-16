"""Exponential-cone atoms: exp, log, entropy and log-sum-exp."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from akme.atoms.affine import sum as _sum
from akme.atoms.base import Atom
from akme.expressions.affine import AffineExpr, Constant
from akme.expressions.expression import (
    Curvature,
    Expression,
    require_affine,
)
from akme.expressions.expression import vec as _vec

if TYPE_CHECKING:
    from akme.canon import CanonContext


class Exp(Atom):
    """Elementwise exponential."""

    def numeric(self, a: np.ndarray) -> np.ndarray:
        """exp(a)."""
        return np.exp(a)

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """(a, 1, t) in K_exp."""
        (a,) = self.args
        t = ctx.new_var(self.shape, source=self)
        ctx.exp3(a, Constant(1.0), t)
        return t


class Log(Atom):
    """Elementwise natural logarithm (concave)."""

    _curvature = Curvature.CONCAVE

    def numeric(self, a: np.ndarray) -> np.ndarray:
        """log(a)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(a)

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """(t, 1, a) in K_exp."""
        (a,) = self.args
        t = ctx.new_var(self.shape, source=self)
        ctx.exp3(t, Constant(1.0), a)
        return t


class Entr(Atom):
    """Elementwise entropy ``-a log a`` (concave)."""

    _curvature = Curvature.CONCAVE

    def numeric(self, a: np.ndarray) -> np.ndarray:
        """-a log a (0 at a = 0)."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(a == 0, 0.0, -a * np.log(a))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """(t, a, 1) in K_exp."""
        (a,) = self.args
        t = ctx.new_var(self.shape, source=self)
        ctx.exp3(t, a, Constant(1.0))
        return t


class LogSumExp(Atom):
    """``log(sum(exp(a)))`` of the flattened argument."""

    def numeric(self, a: np.ndarray) -> float:
        """Stable log-sum-exp."""
        a = _vec(a)
        m = np.max(a)
        return float(m + np.log(np.sum(np.exp(a - m))))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """(a_i - t, 1, u_i) in K_exp, sum u <= 1."""
        (a,) = self.args
        t = ctx.new_var((), source=self)
        u = ctx.new_var((a.size,))
        ctx.exp3(a.flatten() - t, Constant(1.0), u)
        ctx.nonneg(1 - _sum(u))
        return t


def exp(x: Any) -> Expression:
    """Elementwise exponential."""
    a = require_affine(x, "exp")
    return Exp(a, shape=a.shape)


def log(x: Any) -> Expression:
    """Elementwise natural logarithm."""
    a = require_affine(x, "log")
    return Log(a, shape=a.shape)


def entr(x: Any) -> Expression:
    """Elementwise entropy ``-x log x``."""
    a = require_affine(x, "entr")
    return Entr(a, shape=a.shape)


def log_sum_exp(x: Any) -> Expression:
    """``log(sum(exp(x)))``.

    Example:
        >>> import numpy as np
        >>> import akme as ak
        >>> x = ak.Variable(3)
        >>> x.value = np.zeros(3)
        >>> round(float(ak.log_sum_exp(x).value), 6) == round(float(np.log(3)), 6)
        True
        >>> ak.log_sum_exp(x).curvature
        <Curvature.CONVEX: 'convex'>
    """
    return LogSumExp(require_affine(x, "log_sum_exp"), shape=())
