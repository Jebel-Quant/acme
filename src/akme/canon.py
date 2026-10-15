"""Canonicalization of a problem into the conic standard form.

    minimize    1/2 x^T P x + q^T x + offset
    subject to  A x + s = b,  s in K,

with ``K`` a product of the cones in :mod:`akme.cones` (polyhedral rows first).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.sparse as sp

from akme.atoms import Atom
from akme.cones import ORDER, ConeKind, ConeSpec, svec_matrix
from akme.expressions.affine import AffineExpr, Constant, vstack_affine
from akme.expressions.expression import AddExpr, Expression, LinMapExpr, as_expr, broadcast, require_affine, vec
from akme.expressions.variable import Variable

if TYPE_CHECKING:
    from akme.constraints import Constraint


@dataclass
class ConeBlock:
    """Rows ``s(x)`` (an affine vector) constrained to a product of cones of one kind."""

    kind: ConeKind
    expr: AffineExpr
    cones: list[ConeSpec]
    owner: Constraint | None = None


@dataclass
class CanonContext:
    """Collects auxiliary variables, cone blocks and quadratic objective terms."""

    aux: list[Variable] = field(default_factory=list)
    aux_source: dict[int, Expression] = field(default_factory=dict)
    blocks: list[ConeBlock] = field(default_factory=list)
    quad_terms: list[tuple[AffineExpr, sp.sparray | np.ndarray]] = field(default_factory=list)

    # ---------------------------------------------------------------- variables
    def new_var(self, shape: tuple[int, ...], source: Expression | None = None) -> Variable:
        """Create an auxiliary variable; ``source`` is the expression it bounds (for warm starts)."""
        var = Variable(shape, name=f"aux{len(self.aux)}")
        self.aux.append(var)
        if source is not None:
            self.aux_source[var.id] = source
        return var

    # -------------------------------------------------------------------- cones
    def _block(self, kind: ConeKind, expr: Any, cones: list[ConeSpec], owner: Constraint | None) -> None:
        """Append the flattened affine rows ``expr`` as a block of ``cones``."""
        expr = as_expr(expr)
        if not isinstance(expr, AffineExpr):
            msg = "cone rows must be affine"
            raise TypeError(msg)
        self.blocks.append(ConeBlock(kind, expr.flatten(), cones, owner))

    def zero(self, expr: Any, owner: Constraint | None = None) -> None:
        """``expr == 0``."""
        expr = as_expr(expr)
        self._block(ConeKind.ZERO, expr, [ConeSpec(ConeKind.ZERO, expr.size)], owner)

    def nonneg(self, expr: Any, owner: Constraint | None = None) -> None:
        """``expr >= 0``."""
        expr = as_expr(expr)
        self._block(ConeKind.NONNEG, expr, [ConeSpec(ConeKind.NONNEG, expr.size)], owner)

    def soc(self, t: Any, x: Any, owner: Constraint | None = None) -> None:
        """``||x||_2 <= t`` for scalar ``t``."""
        rows = vstack_affine([require_affine(t, "SOC").flatten(), require_affine(x, "SOC").flatten()])
        self._block(ConeKind.SOC, rows, [ConeSpec(ConeKind.SOC, rows.size)], owner)

    def soc3(self, t: Any, u: Any, v: Any, owner: Constraint | None = None) -> None:
        """Elementwise ``|(u_i, v_i)| <= t_i``."""
        rows, k = _interleave(t, u, v)
        self._block(ConeKind.SOC, rows, [ConeSpec(ConeKind.SOC, 3)] * k, owner)

    def exp3(self, x: Any, y: Any, z: Any, owner: Constraint | None = None) -> None:
        """Elementwise ``(x_i, y_i, z_i)`` in the exponential cone."""
        rows, k = _interleave(x, y, z)
        self._block(ConeKind.EXP, rows, [ConeSpec(ConeKind.EXP, 3)] * k, owner)

    def pow3(self, x: Any, y: Any, z: Any, alpha: float, owner: Constraint | None = None) -> None:
        """Elementwise ``(x_i, y_i, z_i)`` in the power cone with exponent ``alpha``."""
        rows, k = _interleave(x, y, z)
        self._block(ConeKind.POW, rows, [ConeSpec(ConeKind.POW, 3, alpha)] * k, owner)

    def psd(self, matrix: Any, owner: Constraint | None = None) -> None:
        """Symmetric part of the square matrix expression is PSD."""
        matrix = as_expr(matrix)
        n = matrix.shape[0]
        rows = matrix.apply_linear(svec_matrix(n), (n * (n + 1) // 2,))
        self._block(ConeKind.PSD, rows, [ConeSpec(ConeKind.PSD, rows.size)], owner)

    # ---------------------------------------------------------------- objective
    def quad(self, expr: AffineExpr, matrix: sp.sparray | np.ndarray) -> None:
        """Add ``expr^T M expr`` to the objective (``M`` symmetric PSD)."""
        self.quad_terms.append((expr, matrix))


def _interleave(*parts: Any) -> tuple[AffineExpr, int]:
    """Broadcast parts to one size ``k`` and interleave them as ``(p0_0, p1_0, p2_0, p0_1, ...)``."""
    exprs = [require_affine(e, "cone").flatten() for e in broadcast(*(as_expr(p).flatten() for p in parts))]
    k, m = exprs[0].size, len(exprs)
    stacked = vstack_affine(exprs)
    perm = (np.arange(m)[None, :] * k + np.arange(k)[:, None]).ravel()
    return stacked.apply_linear(sp.csr_array((np.ones(k * m), (np.arange(k * m), perm))), (k * m,)), k


# ------------------------------------------------------------- canonicalization
def canon_upper(expr: Expression, ctx: CanonContext) -> AffineExpr:
    """Affine ``a`` with ``a >= expr`` attainable with equality (``expr`` must be convex)."""
    if isinstance(expr, AffineExpr):
        return expr
    if isinstance(expr, AddExpr):
        return AffineExpr.add_all([canon_upper(a, ctx) for a in expr.args])
    if isinstance(expr, LinMapExpr):
        return canon_upper(expr.child, ctx).apply_linear(expr.matrix, expr.shape)
    if isinstance(expr, Atom):
        return expr.canon(ctx)
    msg = f"cannot canonicalize {type(expr).__name__}"
    raise TypeError(msg)


def canon_objective(expr: Expression, weights: np.ndarray, ctx: CanonContext) -> AffineExpr:
    """Canonicalize the scalar ``weights . vec(expr)``; quadratic atoms go to ``ctx.quad_terms``."""
    if isinstance(expr, AffineExpr):
        return expr.apply_linear(weights.reshape(1, -1), ())
    if isinstance(expr, AddExpr):
        return AffineExpr.add_all([canon_objective(a, weights, ctx) for a in expr.args])
    if isinstance(expr, LinMapExpr):
        return canon_objective(expr.child, expr.matrix.T @ weights, ctx)
    if isinstance(expr, Atom) and expr.is_quadratic:
        expr.add_quadratic(ctx, weights)
        return Constant(0.0)
    return canon_upper(expr, ctx).apply_linear(weights.reshape(1, -1), ())


# ---------------------------------------------------------------- standard form
@dataclass
class StandardForm:
    """``min 1/2 x'Px + q'x + offset  s.t.  Ax + s = b, s in K`` plus the maps back to the model."""

    p: sp.csc_array
    q: np.ndarray
    offset: float
    a: sp.csc_array
    b: np.ndarray
    cones: list[ConeSpec]
    variables: list[Variable]
    offsets: dict[int, int]
    blocks: list[tuple[ConeBlock, slice]]
    aux_source: dict[int, Expression]
    n_user: int

    @property
    def n(self) -> int:
        """Number of (user and auxiliary) variables."""
        return self.q.size

    @property
    def is_polyhedral(self) -> bool:
        """True if all cones are zero / nonnegative cones."""
        return all(c.kind.is_polyhedral for c in self.cones)

    @property
    def n_eq(self) -> int:
        """Number of zero-cone rows (they come first)."""
        return int(np.sum([c.dim for c in self.cones if c.kind is ConeKind.ZERO]))

    @property
    def n_ineq(self) -> int:
        """Number of nonnegative-cone rows (they follow the zero rows)."""
        return int(np.sum([c.dim for c in self.cones if c.kind is ConeKind.NONNEG]))

    def unpack(self, x: np.ndarray, z: np.ndarray | None) -> None:
        """Write primal values to the variables and duals to the constraints."""
        for var in self.variables:
            start = self.offsets[var.id]
            var.value = x[start : start + var.size].reshape(var.shape, order="F")
        if z is None:
            return
        for block, rows in self.blocks:
            if block.owner is not None:
                block.owner.set_dual(z[rows])

    def initial_point(self) -> np.ndarray | None:
        """Stacked warm-start vector from user variable values (aux from their atoms), if available."""
        x = np.zeros(self.n)
        for i, var in enumerate(self.variables):
            value = var.value
            if i < self.n_user and value is None:
                return None
            if i >= self.n_user:
                source = self.aux_source.get(var.id)
                value = None if source is None else source.value
            if value is not None:
                start = self.offsets[var.id]
                x[start : start + var.size] = vec(value)
        return x


def build_standard_form(objective: Expression, constraints: list[Constraint]) -> StandardForm:
    """Canonicalize a (minimization) objective and constraints."""
    ctx = CanonContext()
    linear = canon_objective(objective, np.ones(1), ctx)
    for con in constraints:
        con.canon(ctx)

    # variable ordering: user variables by first appearance, then auxiliaries
    user: dict[int, Variable] = {}
    for v in objective.variables():
        user.setdefault(v.id, v)
    for con in constraints:
        for v in con.variables():
            user.setdefault(v.id, v)
    aux_ids = {v.id for v in ctx.aux}
    user_vars = [v for v in user.values() if v.id not in aux_ids]
    variables = user_vars + ctx.aux
    offsets, n = {}, 0
    for v in variables:
        offsets[v.id] = n
        n += v.size

    # objective
    q = linear.coefficient_matrix(offsets, n).toarray().ravel()
    offset = float(linear.const[0]) if linear.const.size else 0.0
    p = sp.csc_array((n, n))
    for expr, matrix in ctx.quad_terms:
        c = expr.coefficient_matrix(offsets, n)
        m = sp.csr_array(matrix)
        p = p + 2.0 * (c.T @ m @ c)
        q = q + 2.0 * (c.T @ (m @ expr.const))
        offset += float(expr.const @ (m @ expr.const))
    p = sp.csc_array((p + p.T) / 2.0)

    # constraints, ordered by cone kind
    blocks = sorted(ctx.blocks, key=lambda blk: ORDER.index(blk.kind))
    a_parts, b_parts, cones, placed = [], [], [], []
    row = 0
    for blk in blocks:
        a_parts.append(-blk.expr.coefficient_matrix(offsets, n))
        b_parts.append(blk.expr.const)
        placed.append((blk, slice(row, row + blk.expr.size)))
        row += blk.expr.size
        for cone in blk.cones:
            last = cones[-1] if cones else None
            if last is not None and cone.kind.is_polyhedral and last.kind is cone.kind:
                cones[-1] = ConeSpec(cone.kind, last.dim + cone.dim)
            else:
                cones.append(cone)
    a = sp.csc_array(sp.vstack(a_parts)) if a_parts else sp.csc_array((0, n))
    b = np.concatenate(b_parts) if b_parts else np.zeros(0)
    return StandardForm(p, q, offset, a, b, cones, variables, offsets, placed, ctx.aux_source, len(user_vars))
