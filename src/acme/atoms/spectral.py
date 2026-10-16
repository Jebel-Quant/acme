"""Spectral atoms: largest and smallest eigenvalue of a symmetric matrix."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from acme.atoms.base import Atom
from acme.expressions.affine import AffineExpr
from acme.expressions.expression import (
    Curvature,
    Expression,
    require_affine,
)

if TYPE_CHECKING:
    from acme.canon import CanonContext


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


def lambda_max(x: Any) -> Expression:
    """Largest eigenvalue of a symmetric matrix expression."""
    return LambdaMax(_square_matrix(x, "lambda_max"), shape=())


def lambda_min(x: Any) -> Expression:
    """Smallest eigenvalue of a symmetric matrix expression."""
    return LambdaMin(_square_matrix(x, "lambda_min"), shape=())


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
