"""Expression base class, curvature bookkeeping and generic (lazy) linear maps.

All expressions are flattened in column-major (Fortran) order, as in cvxpy. Affine
expressions are evaluated eagerly into :class:`~acme.expressions.affine.AffineExpr`;
anything containing a nonlinear atom is kept as a small tree of :class:`AddExpr` and
:class:`LinMapExpr` nodes whose leaves are atoms or affine expressions.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from acme.expressions._ops import ops

if TYPE_CHECKING:
    from acme.constraints import Constraint
    from acme.expressions.affine import AffineExpr
    from acme.expressions.variable import Variable


class DCPError(ValueError):
    """Raised when an expression or problem violates the (light) DCP rules of acme."""


class Curvature(enum.Enum):
    """Curvature of an expression."""

    CONSTANT = "constant"
    AFFINE = "affine"
    CONVEX = "convex"
    CONCAVE = "concave"
    UNKNOWN = "unknown"

    def negate(self) -> Curvature:
        """Curvature of the negated expression."""
        if self is Curvature.CONVEX:
            return Curvature.CONCAVE
        if self is Curvature.CONCAVE:
            return Curvature.CONVEX
        return self

    @property
    def is_affine(self) -> bool:
        """True for constant and affine curvature."""
        return self in (Curvature.CONSTANT, Curvature.AFFINE)

    @property
    def is_convex(self) -> bool:
        """True if the expression is convex (affine included)."""
        return self.is_affine or self is Curvature.CONVEX

    @property
    def is_concave(self) -> bool:
        """True if the expression is concave (affine included)."""
        return self.is_affine or self is Curvature.CONCAVE

    @staticmethod
    def add(curvatures: Iterable[Curvature]) -> Curvature:
        """Curvature of a sum of expressions with the given curvatures."""
        curvatures = list(curvatures)
        if Curvature.UNKNOWN in curvatures:
            return Curvature.UNKNOWN
        nonlinear = {c for c in curvatures if not c.is_affine}
        if not nonlinear:
            if all(c is Curvature.CONSTANT for c in curvatures):
                return Curvature.CONSTANT
            return Curvature.AFFINE
        if len(nonlinear) == 1:
            return nonlinear.pop()
        return Curvature.UNKNOWN


def vec(value: Any) -> np.ndarray:
    """Flatten a numeric value in column-major order into a float vector."""
    return np.asarray(value, dtype=float).flatten(order="F")


def unvec(values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray | float:
    """Reshape a flat column-major vector back to ``shape`` (scalars become floats)."""
    if shape == ():
        return float(values[0])
    return np.asarray(values, dtype=float).reshape(shape, order="F")


def selection_matrix(indices: np.ndarray, in_size: int) -> sp.csr_array:
    """Sparse matrix picking entries ``indices`` (flat, column-major) from a vector of length ``in_size``."""
    indices = np.asarray(indices, dtype=int).ravel()
    out_size = indices.size
    return sp.csr_array((np.ones(out_size), (np.arange(out_size), indices)), shape=(out_size, in_size))


def flat_index(shape: tuple[int, ...]) -> np.ndarray:
    """Array of shape ``shape`` holding the column-major flat index of each entry."""
    size = int(np.prod(shape, dtype=int))
    return np.arange(size).reshape(shape, order="F")


class Expression:
    """Base class of all acme expressions.

    Supports ``+ - * / @ **``, unary minus, indexing, ``.T`` and the comparisons
    ``==``, ``<=``, ``>=``, ``>>`` and ``<<`` which build constraints.
    """

    # Make numpy defer to our reflected operators (``ndarray @ expr`` etc.).
    __array_ufunc__ = None
    __array_priority__ = 100

    shape: tuple[int, ...]

    # ------------------------------------------------------------------ basics
    @property
    def size(self) -> int:
        """Number of entries."""
        return int(np.prod(self.shape, dtype=int))

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return len(self.shape)

    @property
    def curvature(self) -> Curvature:
        """Curvature of the expression."""
        raise NotImplementedError

    @property
    def value(self) -> np.ndarray | float | None:
        """Numeric value given the current variable values (``None`` if any is unset)."""
        raise NotImplementedError

    def variables(self) -> list[Variable]:
        """Variables appearing in the expression, in order of first appearance."""
        raise NotImplementedError

    def is_affine(self) -> bool:
        """True if the expression is affine."""
        return self.curvature.is_affine

    def is_convex(self) -> bool:
        """True if the expression is convex."""
        return self.curvature.is_convex

    def is_concave(self) -> bool:
        """True if the expression is concave."""
        return self.curvature.is_concave

    def is_constant(self) -> bool:
        """True if the expression contains no variables."""
        return self.curvature is Curvature.CONSTANT

    def __repr__(self) -> str:
        """Short description including shape and curvature."""
        return f"{type(self).__name__}(shape={self.shape}, curvature={self.curvature.value})"

    __hash__ = object.__hash__

    # ------------------------------------------------------- linear operations
    def apply_linear(self, matrix: sp.sparray | np.ndarray, shape: tuple[int, ...]) -> Expression:
        """Return ``matrix @ vec(self)`` reshaped to ``shape``."""
        return LinMapExpr(self, sp.csr_array(matrix), shape)

    def _index_map(self, fn: Callable[[np.ndarray], np.ndarray]) -> Expression:
        """Apply a numpy re-indexing function (transpose, slicing, ...) to the entries."""
        selected = np.asarray(fn(flat_index(self.shape)))
        return self.apply_linear(selection_matrix(selected.flatten(order="F"), self.size), selected.shape)

    def __getitem__(self, key: Any) -> Expression:
        """Numpy-style indexing."""
        return self._index_map(lambda idx: idx[key])

    @property
    def T(self) -> Expression:  # noqa: N802 - numpy naming
        """Transpose."""
        return self._index_map(lambda idx: idx.T)

    def flatten(self) -> Expression:
        """Column-major flattening into a vector."""
        return self.apply_linear(sp.eye_array(self.size, format="csr"), (self.size,))

    def reshape(self, shape: int | tuple[int, ...]) -> Expression:
        """Column-major reshape."""
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if int(np.prod(shape, dtype=int)) != self.size:
            msg = f"cannot reshape expression of size {self.size} into shape {shape}"
            raise ValueError(msg)
        return self.apply_linear(sp.eye_array(self.size, format="csr"), shape)

    def sum(self, axis: int | None = None) -> Expression:
        """Sum of entries, optionally along ``axis``."""
        return ops.sum(self, axis=axis)

    # ---------------------------------------------------------------- algebra
    def __add__(self, other: Any) -> Expression:
        """Addition (scalars broadcast)."""
        return add(self, other)

    def __radd__(self, other: Any) -> Expression:
        """Reflected addition."""
        return add(other, self)

    def __sub__(self, other: Any) -> Expression:
        """Subtraction."""
        return add(self, -as_expr(other))

    def __rsub__(self, other: Any) -> Expression:
        """Reflected subtraction."""
        return add(other, -self)

    def __neg__(self) -> Expression:
        """Negation."""
        return self.apply_linear(-sp.eye_array(self.size, format="csr"), self.shape)

    def __pos__(self) -> Expression:
        """Unary plus."""
        return self

    def __mul__(self, other: Any) -> Expression:
        """Elementwise multiplication by a constant."""
        return multiply(self, other)

    def __rmul__(self, other: Any) -> Expression:
        """Reflected elementwise multiplication."""
        return multiply(other, self)

    def __truediv__(self, other: Any) -> Expression:
        """Division by a constant."""
        other = as_expr(other)
        if not other.is_constant():
            msg = "can only divide by a constant"
            raise DCPError(msg)
        return multiply(self, 1.0 / np.asarray(other.value, dtype=float))

    def __matmul__(self, other: Any) -> Expression:
        """Matrix product with a constant."""
        return matmul(self, other)

    def __rmatmul__(self, other: Any) -> Expression:
        """Reflected matrix product."""
        return matmul(other, self)

    def __pow__(self, p: float) -> Expression:
        """Elementwise power (see :func:`acme.power`)."""
        return ops.power(self, p)

    # ------------------------------------------------------------ constraints
    def __eq__(self, other: object) -> Constraint:  # ty: ignore[invalid-method-override] - builds a constraint
        """Equality constraint."""
        return ops.equality(self, other)

    def __le__(self, other: Any) -> Constraint:
        """Inequality constraint ``self <= other``."""
        return ops.inequality(self, other)

    def __lt__(self, other: Any) -> Constraint:
        """Strict inequalities are not supported."""
        msg = "strict inequalities are not supported; use <="
        raise TypeError(msg)

    def __gt__(self, other: Any) -> Constraint:
        """Strict inequalities are not supported."""
        msg = "strict inequalities are not supported; use >="
        raise TypeError(msg)

    def __ge__(self, other: Any) -> Constraint:
        """Inequality constraint ``self >= other``."""
        return ops.inequality(other, self)

    def __rshift__(self, other: Any) -> Constraint:
        """Semidefinite constraint ``self - other >> 0``."""
        return ops.psd(self - as_expr(other))

    def __rrshift__(self, other: Any) -> Constraint:
        """Reflected semidefinite constraint."""
        return ops.psd(as_expr(other) - self)

    def __lshift__(self, other: Any) -> Constraint:
        """Semidefinite constraint ``other - self >> 0``."""
        return ops.psd(as_expr(other) - self)

    def __rlshift__(self, other: Any) -> Constraint:
        """Reflected semidefinite constraint."""
        return ops.psd(self - as_expr(other))


class LinMapExpr(Expression):
    """A constant sparse linear map applied to a (non-affine) expression."""

    def __init__(self, child: Expression, matrix: sp.csr_array, shape: tuple[int, ...]) -> None:
        """Create ``matrix @ vec(child)`` with result shape ``shape``."""
        if matrix.shape != (int(np.prod(shape, dtype=int)), child.size):
            msg = f"linear map of shape {matrix.shape} does not fit {child.shape} -> {shape}"
            raise ValueError(msg)
        self.child = child
        self.matrix = matrix
        self.shape = tuple(shape)

    @property
    def curvature(self) -> Curvature:
        """Curvature: preserved by nonnegative maps, flipped by nonpositive ones."""
        child = self.child.curvature
        if child.is_affine:
            return child
        data = self.matrix.data
        if np.all(data >= 0):
            return child
        if np.all(data <= 0):
            return child.negate()
        return Curvature.UNKNOWN

    @property
    def value(self) -> np.ndarray | float | None:
        """Numeric value."""
        inner = self.child.value
        if inner is None:
            return None
        return unvec(self.matrix @ vec(inner), self.shape)

    def variables(self) -> list[Variable]:
        """Variables of the child expression."""
        return self.child.variables()

    def apply_linear(self, matrix: sp.sparray | np.ndarray, shape: tuple[int, ...]) -> Expression:
        """Compose linear maps instead of nesting them."""
        return LinMapExpr(self.child, sp.csr_array(sp.csr_array(matrix) @ self.matrix), shape)


class AddExpr(Expression):
    """Sum of expressions of identical shape."""

    def __init__(self, args: list[Expression]) -> None:
        """Create the sum of ``args`` (which must share one shape)."""
        self.args = args
        self.shape = args[0].shape

    @property
    def curvature(self) -> Curvature:
        """Curvature of the sum."""
        return Curvature.add(a.curvature for a in self.args)

    @property
    def value(self) -> np.ndarray | float | None:
        """Numeric value."""
        values = [a.value for a in self.args]
        if any(v is None for v in values):
            return None
        return unvec(np.sum([vec(v) for v in values], axis=0), self.shape)

    def variables(self) -> list[Variable]:
        """Variables of all summands."""
        return unique_variables(v for a in self.args for v in a.variables())


# ---------------------------------------------------------------------- helpers
def unique_variables(variables: Iterable[Variable]) -> list[Variable]:
    """Deduplicate variables by identity, keeping the first appearance order."""
    seen: dict[int, Variable] = {}
    for v in variables:
        seen.setdefault(v.id, v)
    return list(seen.values())


def as_expr(value: Any) -> Expression:
    """Convert numbers / arrays to constant expressions; expressions pass through."""
    if isinstance(value, Expression):
        return value
    return ops.constant(value)


def constant_value(expr: Expression) -> np.ndarray:
    """Numeric value of a constant expression as an ndarray."""
    return np.asarray(expr.value, dtype=float)


def broadcast_to(expr: Expression, shape: tuple[int, ...]) -> Expression:
    """Broadcast a scalar expression to ``shape`` (other shapes must already match)."""
    if expr.shape == shape:
        return expr
    if expr.size == 1:
        size = int(np.prod(shape, dtype=int))
        return expr.apply_linear(sp.csr_array(np.ones((size, 1))), shape)
    msg = f"cannot broadcast shape {expr.shape} to {shape}"
    raise ValueError(msg)


def broadcast(*exprs: Expression) -> list[Expression]:
    """Broadcast expressions to a common shape (only scalar broadcasting is supported)."""
    shapes = {e.shape for e in exprs if e.size != 1} or {max((e.shape for e in exprs), key=len)}
    if len(shapes) > 1:
        msg = f"incompatible shapes {sorted(shapes)}"
        raise ValueError(msg)
    shape = shapes.pop()
    return [broadcast_to(e, shape) for e in exprs]


def add(*args: Any) -> Expression:
    """Sum of expressions (scalars broadcast)."""
    exprs = broadcast(*(as_expr(a) for a in args))
    affine = [e for e in exprs if isinstance(e, ops.affine)]
    if len(affine) == len(exprs):
        return ops.affine.add_all(affine)
    flat: list[Expression] = []
    for e in exprs:
        flat.extend(e.args if isinstance(e, AddExpr) else [e])
    return AddExpr(flat)


def multiply(a: Any, b: Any) -> Expression:
    """Elementwise product; at least one factor must be constant."""
    a, b = as_expr(a), as_expr(b)
    if not a.is_constant():
        a, b = b, a
    if not a.is_constant():
        msg = "cannot multiply two non-constant expressions"
        raise DCPError(msg)
    a, b = broadcast(a, b)
    return b.apply_linear(sp.diags_array(vec(a.value)).tocsr(), b.shape)


def matmul(a: Any, b: Any) -> Expression:
    """Matrix product; exactly the constant side may be any numpy-compatible shape."""
    a, b = as_expr(a), as_expr(b)
    if a.ndim == 0 or b.ndim == 0:
        msg = "matmul does not accept scalars; use * instead"
        raise ValueError(msg)
    shape = np.matmul(np.zeros(a.shape), np.zeros(b.shape)).shape
    if a.is_constant():
        # vec(C X) = (I_k kron C) vec(X), treating 1-d operands as row / column
        c = constant_value(a).reshape(1, -1) if a.ndim == 1 else constant_value(a)
        k = 1 if b.ndim == 1 else b.shape[1]
        return b.apply_linear(sp.kron(sp.eye_array(k), sp.csr_array(c), format="csr"), shape)
    if b.is_constant():
        # vec(X D) = (D^T kron I_m) vec(X)
        d = constant_value(b).reshape(-1, 1) if b.ndim == 1 else constant_value(b)
        m = 1 if a.ndim == 1 else a.shape[0]
        return a.apply_linear(sp.kron(sp.csr_array(d.T), sp.eye_array(m), format="csr"), shape)
    msg = "cannot multiply two non-constant expressions; use quad_form or sum_squares"
    raise DCPError(msg)


def require_affine(value: Any, what: str) -> AffineExpr:
    """Convert to an expression and require it to be affine (atom arguments)."""
    expr = as_expr(value)
    if not isinstance(expr, ops.affine):
        msg = f"argument of {what} must be affine, got {expr.curvature.value} expression"
        raise DCPError(msg)
    return expr
