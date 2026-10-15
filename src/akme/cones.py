"""Cone descriptors of the conic standard form ``A x + s = b, s in K``."""

from __future__ import annotations

import enum
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


class ConeKind(enum.Enum):
    """Supported cones (Clarabel conventions)."""

    ZERO = "zero"  # s = 0
    NONNEG = "nonneg"  # s >= 0
    SOC = "soc"  # s = (t, x), ||x||_2 <= t
    PSD = "psd"  # s = svec(X) of the upper triangle, off-diagonals scaled by sqrt(2), X >> 0
    EXP = "exp"  # s = (x, y, z), y exp(x / y) <= z, y > 0
    POW = "pow"  # s = (x, y, z), x^a y^(1-a) >= |z|, x, y >= 0

    @property
    def is_polyhedral(self) -> bool:
        """True for the zero and nonnegative cones."""
        return self in (ConeKind.ZERO, ConeKind.NONNEG)


#: Row order of cone blocks in the standard form (polyhedral rows first).
ORDER = [ConeKind.ZERO, ConeKind.NONNEG, ConeKind.SOC, ConeKind.PSD, ConeKind.EXP, ConeKind.POW]


@dataclass(frozen=True)
class ConeSpec:
    """A single cone: kind, number of rows and (for power cones) the exponent."""

    kind: ConeKind
    dim: int
    alpha: float | None = None

    @property
    def psd_order(self) -> int:
        """Matrix order ``n`` of a PSD cone with ``n (n + 1) / 2`` rows."""
        return round((np.sqrt(8 * self.dim + 1) - 1) / 2)


def svec_matrix(n: int) -> sp.csr_array:
    """Map ``vec(X)`` (column-major, n x n) to the scaled upper triangle of ``(X + X^T) / 2``."""
    rows, cols, data = [], [], []
    r = 0
    for j in range(n):
        for i in range(j + 1):
            if i == j:
                rows.append(r)
                cols.append(i + j * n)
                data.append(1.0)
            else:
                w = np.sqrt(2.0) / 2.0
                rows += [r, r]
                cols += [i + j * n, j + i * n]
                data += [w, w]
            r += 1
    return sp.csr_array((data, (rows, cols)), shape=(r, n * n))


def smat(values: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`svec_matrix` for symmetric matrices."""
    out = np.zeros((n, n))
    r = 0
    for j in range(n):
        for i in range(j + 1):
            if i == j:
                out[i, i] = values[r]
            else:
                out[i, j] = out[j, i] = values[r] / np.sqrt(2.0)
            r += 1
    return out
