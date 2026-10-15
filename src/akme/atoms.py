"""Atoms (nonlinear functions) and affine helper functions.

Every nonlinear atom takes *affine* arguments only. This keeps canonicalization flat:
each atom is replaced by an epigraph (convex) or hypograph (concave) variable plus a few
cone constraints, and the light DCP check only has to track signs of linear maps.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from akme.expressions.affine import AffineExpr, Constant
from akme.expressions.expression import (
    Curvature,
    DCPError,
    Expression,
    add,
    as_expr,
    broadcast,
    flat_index,
    multiply,
    require_affine,
    selection_matrix,
    unique_variables,
    unvec,
)
from akme.expressions.expression import vec as _vec

if TYPE_CHECKING:
    from akme.canon import CanonContext
    from akme.expressions.variable import Variable

__all__ = [
    "abs",
    "diag",
    "entr",
    "exp",
    "hstack",
    "inv_pos",
    "lambda_max",
    "lambda_min",
    "log",
    "log_sum_exp",
    "max",
    "maximum",
    "min",
    "minimum",
    "multiply",
    "norm",
    "norm1",
    "norm2",
    "norm_inf",
    "pnorm",
    "pos",
    "power",
    "quad_form",
    "reshape",
    "sqrt",
    "square",
    "sum",
    "sum_squares",
    "trace",
    "vec",
    "vstack",
]


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
        return unvec(_vec(self.numeric(*[np.asarray(v, dtype=float) for v in values])), self.shape)

    def variables(self) -> list[Variable]:
        """Variables of the arguments."""
        return unique_variables(v for a in self.args for v in a.variables())

    def numeric(self, *values: np.ndarray) -> np.ndarray | float:
        """Evaluate the atom on numeric argument values."""
        raise NotImplementedError

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Return an affine epigraph (convex) / hypograph (concave) bound, adding cones to ``ctx``."""
        raise NotImplementedError

    def add_quadratic(self, ctx: CanonContext, weights: np.ndarray) -> None:
        """Add ``weights . self`` to the quadratic part of the objective (quadratic atoms only)."""
        raise NotImplementedError


# ------------------------------------------------------------- piecewise linear
class Abs(Atom):
    """Elementwise absolute value."""

    def numeric(self, a: np.ndarray) -> np.ndarray:
        """|a|."""
        return np.abs(a)

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Epigraph: t >= a, t >= -a."""
        (a,) = self.args
        t = ctx.new_var(self.shape, source=self)
        ctx.nonneg(t - a)
        ctx.nonneg(t + a)
        return t


class Pos(Atom):
    """Elementwise positive part ``max(a, 0)``."""

    def numeric(self, a: np.ndarray) -> np.ndarray:
        """max(a, 0)."""
        return np.maximum(a, 0.0)

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Epigraph: t >= a, t >= 0."""
        (a,) = self.args
        t = ctx.new_var(self.shape, source=self)
        ctx.nonneg(t - a)
        ctx.nonneg(t)
        return t


class Maximum(Atom):
    """Elementwise maximum of several expressions."""

    def numeric(self, *values: np.ndarray) -> np.ndarray:
        """Elementwise max."""
        return np.maximum.reduce(np.broadcast_arrays(*values))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Epigraph: t >= a_i for all i."""
        t = ctx.new_var(self.shape, source=self)
        for a in self.args:
            ctx.nonneg(t - a)
        return t


class Minimum(Atom):
    """Elementwise minimum of several expressions."""

    _curvature = Curvature.CONCAVE

    def numeric(self, *values: np.ndarray) -> np.ndarray:
        """Elementwise min."""
        return np.minimum.reduce(np.broadcast_arrays(*values))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Hypograph: t <= a_i for all i."""
        t = ctx.new_var(self.shape, source=self)
        for a in self.args:
            ctx.nonneg(a - t)
        return t


class Max(Atom):
    """Largest entry."""

    def numeric(self, a: np.ndarray) -> float:
        """max(a)."""
        return float(np.max(a))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Epigraph: t >= a_i."""
        (a,) = self.args
        t = ctx.new_var((), source=self)
        ctx.nonneg(t - a)
        return t


class Min(Atom):
    """Smallest entry."""

    _curvature = Curvature.CONCAVE

    def numeric(self, a: np.ndarray) -> float:
        """min(a)."""
        return float(np.min(a))

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Hypograph: t <= a_i."""
        (a,) = self.args
        t = ctx.new_var((), source=self)
        ctx.nonneg(a - t)
        return t


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


# --------------------------------------------------------------- second order
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
        ctx.soc(t + 1, _stack([t - 1, 2 * a.flatten()]))
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
        ctx.zero(t - sum(r))
        ctx.pow3(r, t, a.flatten(), 1.0 / self.p)
        return t


# ---------------------------------------------------------------- exponential
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
        ctx.nonneg(1 - sum(u))
        return t


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


# ----------------------------------------------------------------- spectral
class LambdaMax(Atom):
    """Largest eigenvalue of a symmetric matrix."""

    def numeric(self, a: np.ndarray) -> float:
        """lambda_max((A + A^T) / 2)."""
        return float(np.linalg.eigvalsh((a + a.T) / 2)[-1])

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """Epigraph: t I - A >> 0."""
        (a,) = self.args
        t = ctx.new_var((), source=self)
        ctx.psd(_scaled_identity(t, a.shape[0]) - a)
        return t


class LambdaMin(Atom):
    """Smallest eigenvalue of a symmetric matrix (concave)."""

    _curvature = Curvature.CONCAVE

    def numeric(self, a: np.ndarray) -> float:
        """lambda_min((A + A^T) / 2)."""
        return float(np.linalg.eigvalsh((a + a.T) / 2)[0])

    def canon(self, ctx: CanonContext) -> AffineExpr:
        """A - t I >> 0."""
        (a,) = self.args
        t = ctx.new_var((), source=self)
        ctx.psd(a - _scaled_identity(t, a.shape[0]))
        return t


# ============================================================= public functions
def abs(x: Any) -> Expression:  # noqa: A001
    """Elementwise absolute value."""
    a = require_affine(x, "abs")
    return Abs(a, shape=a.shape)


def pos(x: Any) -> Expression:
    """Elementwise ``max(x, 0)``."""
    a = require_affine(x, "pos")
    return Pos(a, shape=a.shape)


def maximum(*args: Any) -> Expression:
    """Elementwise maximum of expressions (scalars broadcast)."""
    exprs = broadcast(*(require_affine(a, "maximum") for a in args))
    return Maximum(*exprs, shape=exprs[0].shape)  # type: ignore[arg-type]


def minimum(*args: Any) -> Expression:
    """Elementwise minimum of expressions (scalars broadcast)."""
    exprs = broadcast(*(require_affine(a, "minimum") for a in args))
    return Minimum(*exprs, shape=exprs[0].shape)  # type: ignore[arg-type]


def max(x: Any) -> Expression:  # noqa: A001
    """Largest entry."""
    return Max(require_affine(x, "max").flatten(), shape=())  # type: ignore[arg-type]


def min(x: Any) -> Expression:  # noqa: A001
    """Smallest entry."""
    return Min(require_affine(x, "min").flatten(), shape=())  # type: ignore[arg-type]


def norm(x: Any, p: float | str = 2) -> Expression:
    """Vector p-norm of the flattened argument (``p`` in [1, inf], ``"fro"`` = 2)."""
    a = require_affine(x, "norm")
    if p in ("fro", 2):
        return Norm2(a, shape=())
    if p == 1:
        return sum(abs(a))
    if p in ("inf", np.inf):
        return NormInf(a.flatten(), shape=())  # type: ignore[arg-type]
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


def sum_squares(x: Any) -> Expression:
    """Sum of squared entries."""
    return SumSquares(require_affine(x, "sum_squares"), shape=())


def square(x: Any) -> Expression:
    """Elementwise square."""
    a = require_affine(x, "square")
    return Square(a, shape=a.shape)


def quad_form(x: Any, s: Any) -> Expression:
    """Quadratic form ``x^T S x`` with constant semidefinite ``S``."""
    a = require_affine(x, "quad_form")
    s = np.asarray(as_expr(s).value, dtype=float)
    if s.ndim != 2 or s.shape[0] != s.shape[1] or s.shape[0] != a.size:
        msg = f"quad_form: matrix of shape {s.shape} does not match vector of size {a.size}"
        raise ValueError(msg)
    return QuadForm(a, s)


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
    """``log(sum(exp(x)))``."""
    return LogSumExp(require_affine(x, "log_sum_exp"), shape=())


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


def lambda_max(x: Any) -> Expression:
    """Largest eigenvalue of a symmetric matrix expression."""
    return LambdaMax(_square_matrix(x, "lambda_max"), shape=())


def lambda_min(x: Any) -> Expression:
    """Smallest eigenvalue of a symmetric matrix expression."""
    return LambdaMin(_square_matrix(x, "lambda_min"), shape=())


# ----------------------------------------------------------------- affine ops
def sum(x: Any, axis: int | None = None) -> Expression:  # noqa: A001
    """Sum of entries, or along ``axis`` for matrices."""
    expr = as_expr(x)
    if axis is None:
        return expr.apply_linear(sp.csr_array(np.ones((1, expr.size))), ())
    if expr.ndim != 2 or axis not in (0, 1, -1, -2):
        msg = "axis sums are only supported for matrices"
        raise ValueError(msg)
    axis = axis % 2
    out_shape = (expr.shape[1 - axis],)
    rows = np.broadcast_to(np.expand_dims(flat_index(out_shape), axis), expr.shape).flatten(order="F")
    matrix = sp.csr_array((np.ones(expr.size), (rows, np.arange(expr.size))), shape=(out_shape[0], expr.size))
    return expr.apply_linear(matrix, out_shape)


def trace(x: Any) -> Expression:
    """Trace of a square matrix."""
    expr = as_expr(x)
    if expr.ndim != 2 or expr.shape[0] != expr.shape[1]:
        msg = "trace requires a square matrix"
        raise ValueError(msg)
    n = expr.shape[0]
    return sum(expr[np.arange(n), np.arange(n)])


def diag(x: Any) -> Expression:
    """Diagonal of a square matrix, or a diagonal matrix from a vector."""
    expr = as_expr(x)
    if expr.ndim == 2:
        n = expr.shape[0]
        if expr.shape != (n, n):
            msg = "diag requires a square matrix or a vector"
            raise ValueError(msg)
        return expr[np.arange(n), np.arange(n)]
    n = expr.size
    matrix = sp.csr_array((np.ones(n), (np.arange(n) * (n + 1), np.arange(n))), shape=(n * n, n))
    return expr.apply_linear(matrix, (n, n))


def reshape(x: Any, shape: int | tuple[int, ...]) -> Expression:
    """Column-major reshape."""
    return as_expr(x).reshape(shape)


def vec(x: Any) -> Expression:
    """Column-major flattening."""
    return as_expr(x).flatten()


def hstack(exprs: list[Any]) -> Expression:
    """Horizontal concatenation (numpy semantics)."""
    return _stack_like(exprs, np.hstack)


def vstack(exprs: list[Any]) -> Expression:
    """Vertical concatenation (numpy semantics)."""
    return _stack_like(exprs, np.vstack)


# -------------------------------------------------------------------- helpers
def _stack(exprs: list[Expression]) -> Expression:
    """Concatenate flattened expressions into one vector."""
    sizes = [e.size for e in exprs]
    total = int(np.sum(sizes))
    pieces, start = [], 0
    for e, size in zip(exprs, sizes, strict=True):
        embed = sp.csr_array((np.ones(size), (np.arange(start, start + size), np.arange(size))), shape=(total, size))
        pieces.append(e.apply_linear(embed, (total,)))
        start += size
    return add(*pieces)


def _stack_like(exprs: list[Any], fn: Any) -> Expression:
    """Concatenate expressions laid out like ``fn`` (np.hstack / np.vstack) does with arrays."""
    exprs = [as_expr(e) for e in exprs]
    offsets = np.cumsum([0] + [e.size for e in exprs])
    layout = fn([flat_index(e.shape) + off for e, off in zip(exprs, offsets, strict=False)])
    stacked = _stack(exprs)
    return stacked.apply_linear(selection_matrix(layout.flatten(order="F"), stacked.size), layout.shape)


def _scaled_identity(t: AffineExpr, n: int) -> AffineExpr:
    """``t * I_n`` for a scalar affine ``t``."""
    return t.apply_linear(sp.csr_array(np.eye(n).reshape(-1, 1, order="F")), (n, n))


def _square_matrix(x: Any, what: str) -> AffineExpr:
    """Require an affine square matrix argument."""
    a = require_affine(x, what)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        msg = f"{what} requires a square matrix"
        raise ValueError(msg)
    return a
