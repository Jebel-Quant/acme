"""Quadratic atoms, kept in the objective's ``P`` when possible: sum of squares, square, quadratic form."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from acme.atoms.affine import stack_rows
from acme.atoms.base import Atom
from acme.expressions.affine import AffineExpr, Constant
from acme.expressions.expression import (
    Curvature,
    DCPError,
    Expression,
    as_expr,
    require_affine,
)
from acme.expressions.expression import vec as _vec

if TYPE_CHECKING:
    from acme.canon import CanonContext


class SumSquares(Atom):
    """Sum of squared entries."""

    is_quadratic = True

    def numeric(self, a: np.ndarray) -> float:
        """Sum of squares."""
        return float(np.sum(np.square(a)))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """||a||^2 <= t  <=>  ||(t - 1, 2 a)|| <= t + 1."""
        (a,) = self.args
        t = ctx.new_var((), source=self)
        ctx.soc(t + 1, stack_rows([t - 1, 2 * a.flatten()]))
        return t

    def add_quadratic(self, ctx: CanonContext, weights: np.ndarray) -> None:
        """Add w ||a||^2."""
        (a,) = self.args
        ctx.quad(a.flatten(), float(weights[0]) * sp.eye_array(a.size, format="csr"))


class Square(Atom):
    """Elementwise square."""

    is_quadratic = True

    def numeric(self, a: np.ndarray) -> np.ndarray:
        """a^2."""
        return np.square(a)

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """a_i^2 <= t_i via one 3-dimensional SOC per entry."""
        (a,) = self.args
        t = ctx.new_var(self.shape, source=self)
        ctx.soc3(t + 1, t - 1, 2 * a)
        return t

    def add_quadratic(self, ctx: CanonContext, weights: np.ndarray) -> None:
        """sum_i w_i a_i^2."""
        (a,) = self.args
        ctx.quad(a.flatten(), sp.diags_array(weights).tocsr())


class QuadForm(Atom):
    """Quadratic form ``a^T S a`` for a constant symmetric ``S`` (PSD: convex, NSD: concave)."""

    is_quadratic = True

    def __init__(self, a: AffineExpr, s: np.ndarray) -> None:
        """Create ``a^T S a``; ``S`` must be semidefinite."""
        super().__init__(a, shape=())
        self.s = (s + s.T) / 2.0
        eigvals, eigvecs = np.linalg.eigh(self.s)
        tol = 1e-10 * np.max(np.abs(eigvals), initial=1.0)
        if np.all(eigvals >= -tol):
            self._curvature = Curvature.CONVEX
            keep = eigvals > tol
            self._factor = eigvecs[:, keep] * np.sqrt(eigvals[keep])
        elif np.all(eigvals <= tol):
            self._curvature = Curvature.CONCAVE
            keep = eigvals < -tol
            self._factor = eigvecs[:, keep] * np.sqrt(-eigvals[keep])
        else:
            msg = "quad_form requires a positive or negative semidefinite matrix"
            raise DCPError(msg)

    def numeric(self, a: np.ndarray) -> float:
        """a^T S a."""
        a = _vec(a)
        return float(a @ self.s @ a)

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Factor ``+-S = L L^T`` and bound ``||L^T a||^2`` with a rotated SOC."""
        (a,) = self.args
        if self._factor.shape[1] == 0:
            return Constant(0.0)
        bound = SumSquares(self._factor.T @ a.flatten(), shape=()).canon(ctx)
        return bound if self._curvature is Curvature.CONVEX else -bound

    def add_quadratic(self, ctx: CanonContext, weights: np.ndarray) -> None:
        """Add w a^T S a."""
        (a,) = self.args
        ctx.quad(a.flatten(), float(weights[0]) * self.s)


def sum_squares(x: Any) -> Expression:
    """Sum of squared entries."""
    return SumSquares(require_affine(x, "sum_squares"), shape=())


def square(x: Any) -> Expression:
    """Elementwise square."""
    a = require_affine(x, "square")
    return Square(a, shape=a.shape)


def quad_form(x: Any, s: Any) -> Expression:
    """Quadratic form ``x^T S x`` with constant semidefinite ``S``.

    Example:
        >>> import numpy as np
        >>> import acme as ak
        >>> x = ak.Variable(2)
        >>> x.value = np.array([3.0, -4.0])
        >>> float(ak.quad_form(x, np.diag([1.0, 2.0])).value)
        41.0
        >>> ak.quad_form(x, np.eye(2)).curvature
        <Curvature.CONVEX: 'convex'>
    """
    a = require_affine(x, "quad_form")
    s = np.asarray(as_expr(s).value, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] != a.size:
        msg = f"quad_form: matrix of shape {s.shape} does not match vector of size {a.size}"
        raise ValueError(msg)
    return QuadForm(a, s)
