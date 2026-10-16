"""Piecewise-linear atoms: absolute value, positive part, elementwise and entrywise max/min."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from akme.atoms.base import Atom
from akme.expressions.affine import AffineExpr
from akme.expressions.expression import (
    Curvature,
    Expression,
    broadcast,
    require_affine,
)

if TYPE_CHECKING:
    from akme.canon import CanonContext


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

    def evaluate(self, values: list[np.ndarray]) -> np.ndarray:
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

    def evaluate(self, values: list[np.ndarray]) -> np.ndarray:
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
    exprs = [require_affine(e, "maximum") for e in broadcast(*(require_affine(a, "maximum") for a in args))]
    return Maximum(*exprs, shape=exprs[0].shape)


def minimum(*args: Any) -> Expression:
    """Elementwise minimum of expressions (scalars broadcast)."""
    exprs = [require_affine(e, "minimum") for e in broadcast(*(require_affine(a, "minimum") for a in args))]
    return Minimum(*exprs, shape=exprs[0].shape)


def max(x: Any) -> Expression:  # noqa: A001
    """Largest entry."""
    return Max(require_affine(x, "max").flatten(), shape=())


def min(x: Any) -> Expression:  # noqa: A001
    """Smallest entry."""
    return Min(require_affine(x, "min").flatten(), shape=())
