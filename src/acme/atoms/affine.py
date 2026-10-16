"""Affine helper functions: sums, trace, diagonals, reshaping and stacking."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from acme.expressions.expression import (
    Expression,
    add,
    as_expr,
    flat_index,
    selection_matrix,
)

if TYPE_CHECKING:
    pass


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


def stack_rows(exprs: list[Expression]) -> Expression:
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
    stacked = stack_rows(exprs)
    return stacked.apply_linear(selection_matrix(layout.flatten(order="F"), stacked.size), layout.shape)
