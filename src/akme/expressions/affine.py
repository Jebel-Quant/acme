"""Eagerly evaluated affine expressions ``sum_v C_v vec(v) + d``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from akme.expressions.expression import Curvature, Expression, unvec, vec

if TYPE_CHECKING:
    from akme.expressions.variable import Variable


class AffineExpr(Expression):
    """Affine expression stored as sparse coefficient blocks per variable plus a constant.

    ``terms`` maps ``variable.id -> (variable, C)`` with ``C`` of shape ``(size, variable.size)``;
    the constant ``const`` is a flat vector of length ``size``.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        terms: dict[int, tuple[Variable, sp.csr_array]] | None = None,
        const: np.ndarray | None = None,
    ) -> None:
        """Create an affine expression from coefficient blocks and a constant."""
        self.shape = tuple(shape)
        self.terms = terms or {}
        self.const = np.zeros(self.size) if const is None else np.asarray(const, dtype=float).ravel()

    @property
    def curvature(self) -> Curvature:
        """Constant if there are no variable terms, else affine."""
        return Curvature.AFFINE if self.terms else Curvature.CONSTANT

    @property
    def value(self) -> np.ndarray | float | None:
        """Numeric value given the current variable values."""
        total = self.const.copy()
        for var, coeff in self.terms.values():
            if var.value is None:
                return None
            total += coeff @ vec(var.value)
        return unvec(total, self.shape)

    def variables(self) -> list[Variable]:
        """Variables with a (structurally) nonzero coefficient."""
        return [var for var, _ in self.terms.values()]

    def apply_linear(self, matrix: sp.sparray | np.ndarray, shape: tuple[int, ...]) -> AffineExpr:
        """Eagerly apply a constant linear map."""
        matrix = sp.csr_array(matrix)
        terms = {k: (var, sp.csr_array(matrix @ coeff)) for k, (var, coeff) in self.terms.items()}
        return AffineExpr(shape, terms, matrix @ self.const)

    @staticmethod
    def add_all(exprs: list[AffineExpr]) -> AffineExpr:
        """Sum of affine expressions of equal shape."""
        terms: dict[int, tuple[Variable, sp.csr_array]] = {}
        const = np.zeros(exprs[0].size)
        for e in exprs:
            const = const + e.const
            for k, (var, coeff) in e.terms.items():
                terms[k] = (var, coeff + terms[k][1]) if k in terms else (var, coeff)
        return AffineExpr(exprs[0].shape, terms, const)

    def coefficient_matrix(self, offsets: dict[int, int], n: int) -> sp.csr_array:
        """Coefficient matrix with respect to the stacked variable vector of length ``n``."""
        rows, cols, data = [], [], []
        for k, (_, coeff) in self.terms.items():
            c = sp.coo_array(coeff)
            rows.append(c.row)
            cols.append(c.col + offsets[k])
            data.append(c.data)
        if not data:
            return sp.csr_array((self.size, n))
        return sp.csr_array(
            (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
            shape=(self.size, n),
        )


class Constant(AffineExpr):
    """A constant (number or array)."""

    def __init__(self, value: Any) -> None:
        """Wrap a number or array."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim > 2:
            msg = "akme supports at most 2-dimensional expressions"
            raise ValueError(msg)
        super().__init__(arr.shape, None, vec(arr))


def vstack_affine(exprs: list[AffineExpr]) -> AffineExpr:
    """Stack flattened affine expressions into one vector."""
    sizes = [e.size for e in exprs]
    total = sum(sizes)
    pieces = []
    start = 0
    for e, size in zip(exprs, sizes, strict=True):
        embed = sp.csr_array((np.ones(size), (np.arange(start, start + size), np.arange(size))), shape=(total, size))
        pieces.append(e.apply_linear(embed, (total,)))
        start += size
    return AffineExpr.add_all(pieces)
