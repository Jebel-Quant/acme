"""Optimisation variables."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import scipy.sparse as sp

from akme.expressions.affine import AffineExpr
from akme.expressions.expression import unvec, vec

_ids = itertools.count()


class Variable(AffineExpr):
    """An optimisation variable of shape ``()``, ``(n,)`` or ``(m, n)``.

    Args:
        shape: Shape as int or tuple; ``()`` (default) is a scalar.
        name: Optional display name.
        nonneg: Add the implicit constraint ``x >= 0``.
        nonpos: Add the implicit constraint ``x <= 0``.
        symmetric: Square matrix variable constrained to be symmetric.
        PSD: Square matrix variable constrained to be symmetric positive semidefinite.
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] = (),
        name: str | None = None,
        *,
        nonneg: bool = False,
        nonpos: bool = False,
        symmetric: bool = False,
        PSD: bool = False,  # noqa: N803 - cvxpy naming
    ) -> None:
        """Create a variable."""
        shape = (shape,) if isinstance(shape, int) else tuple(shape)
        if len(shape) > 2:
            msg = "akme supports at most 2-dimensional variables"
            raise ValueError(msg)
        if (symmetric or PSD) and (len(shape) != 2 or shape[0] != shape[1]):
            msg = "symmetric / PSD variables must be square matrices"
            raise ValueError(msg)
        self.id = next(_ids)
        self.name = name or f"var{self.id}"
        self.attributes = {"nonneg": nonneg, "nonpos": nonpos, "symmetric": symmetric or PSD, "PSD": PSD}
        self._value: np.ndarray | None = None
        size = int(np.prod(shape, dtype=int))
        super().__init__(shape, {self.id: (self, sp.eye_array(size, format="csr"))}, np.zeros(size))

    def __repr__(self) -> str:
        """Short description."""
        return f"Variable(shape={self.shape}, name={self.name!r})"

    @property
    def value(self) -> np.ndarray | float | None:
        """Value from the last solve (or set by the user for warm starts)."""
        if self._value is None:
            return None
        return unvec(self._value, self.shape)

    @value.setter
    def value(self, value: Any) -> None:
        """Set (or clear with ``None``) the variable's value."""
        if value is None:
            self._value = None
            return
        flat = vec(np.broadcast_to(np.asarray(value, dtype=float), self.shape))
        self._value = flat

    def implicit_constraints(self) -> list[Any]:
        """Constraints implied by the variable's attributes."""
        cons: list[Any] = []
        if self.attributes["nonneg"]:
            cons.append(self >= 0)
        if self.attributes["nonpos"]:
            cons.append(self <= 0)
        if self.attributes["PSD"]:
            cons.append(self >> 0)
        if self.attributes["symmetric"]:
            n = self.shape[0]
            iu = np.triu_indices(n, 1)
            if iu[0].size:
                cons.append(self[iu] == self.T[iu])
        return cons
