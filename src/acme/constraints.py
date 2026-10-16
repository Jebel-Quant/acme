"""Constraints: ``==``, ``<=`` / ``>=``, ``>>`` and explicit cone memberships."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Any

import numpy as np

from acme.canon import canon_upper
from acme.cones import smat
from acme.expressions._ops import ops
from acme.expressions.expression import Expression, add, as_expr, require_affine, unique_variables, unvec

if TYPE_CHECKING:
    from acme.canon import CanonContext
    from acme.expressions.variable import Variable

_ids = itertools.count()


class Constraint:
    """Base class of constraints; ``dual_value`` is set after a successful solve."""

    def __init__(self, *args: Expression) -> None:
        """Store the constraint's expressions."""
        self.id = next(_ids)
        self.args = list(args)
        self._dual: np.ndarray | float | None = None

    def __repr__(self) -> str:
        """Short description."""
        return f"{type(self).__name__}(shape={self.args[0].shape})"

    @property
    def dual_value(self) -> np.ndarray | float | None:
        """Lagrange multiplier from the last solve."""
        return self._dual

    def variables(self) -> list[Variable]:
        """Variables appearing in the constraint."""
        return unique_variables(v for a in self.args for v in a.variables())

    def is_dcp(self) -> bool:
        """True if the constraint satisfies the DCP rules."""
        return all(a.is_affine() for a in self.args)

    def canon(self, ctx: CanonContext) -> None:
        """Add the constraint's cones to ``ctx``."""
        raise NotImplementedError

    def set_dual(self, values: np.ndarray) -> None:
        """Store the dual from the raw cone multipliers of this constraint's rows."""
        self._dual = unvec(values, self.args[0].shape)


class Equality(Constraint):
    """``lhs == rhs`` (affine)."""

    def __init__(self, lhs: Any, rhs: Any) -> None:
        """Create ``lhs == rhs``."""
        super().__init__(add(as_expr(lhs), -as_expr(rhs)))

    def canon(self, ctx: CanonContext) -> None:
        """Require lhs - rhs in the zero cone."""
        ctx.zero(require_affine(self.args[0], "=="), owner=self)

    def set_dual(self, values: np.ndarray) -> None:
        """Multiplier ``nu`` of ``lhs - rhs = 0`` in ``L = f + nu (lhs - rhs)``."""
        self._dual = unvec(-values, self.args[0].shape)


class Inequality(Constraint):
    """``lhs <= rhs`` with ``lhs - rhs`` convex."""

    def __init__(self, lhs: Any, rhs: Any) -> None:
        """Create ``lhs <= rhs``."""
        super().__init__(add(as_expr(lhs), -as_expr(rhs)))

    def is_dcp(self) -> bool:
        """``lhs - rhs`` must be convex."""
        return self.args[0].is_convex()

    def canon(self, ctx: CanonContext) -> None:
        """Epigraph-canonicalize ``lhs - rhs`` and require ``rhs - lhs >= 0``."""
        ctx.nonneg(-canon_upper(self.args[0], ctx), owner=self)


class PSD(Constraint):
    """``A >> 0`` for a square affine matrix expression (its symmetric part is used)."""

    def __init__(self, expr: Any) -> None:
        """Create ``expr >> 0``."""
        expr = as_expr(expr)
        if expr.ndim != 2 or expr.shape[0] != expr.shape[1]:
            msg = "PSD constraints require a square matrix expression"
            raise ValueError(msg)
        super().__init__(expr)

    def canon(self, ctx: CanonContext) -> None:
        """Symmetric part in the PSD cone."""
        ctx.psd(require_affine(self.args[0], ">>"), owner=self)

    def set_dual(self, values: np.ndarray) -> None:
        """Dual as a symmetric matrix."""
        self._dual = smat(values, self.args[0].shape[0])


class SOC(Constraint):
    """Second-order cone ``||x||_2 <= t`` (``x`` flattened, ``t`` scalar)."""

    def __init__(self, t: Any, x: Any) -> None:
        """Create ``||x||_2 <= t``."""
        t = as_expr(t)
        if t.size != 1:
            msg = "SOC: t must be a scalar"
            raise ValueError(msg)
        super().__init__(t, as_expr(x))

    def canon(self, ctx: CanonContext) -> None:
        """(t, x) in SOC."""
        t, x = (require_affine(a, "SOC") for a in self.args)
        ctx.soc(t.flatten(), x.flatten(), owner=self)

    def set_dual(self, values: np.ndarray) -> None:
        """Dual as the raw cone vector ``(u, v)``."""
        self._dual = np.asarray(values, dtype=float)


class ExpCone(Constraint):
    """Elementwise exponential cone ``y exp(x / y) <= z``."""

    def __init__(self, x: Any, y: Any, z: Any) -> None:
        """Create ``(x, y, z) in K_exp``."""
        super().__init__(as_expr(x), as_expr(y), as_expr(z))

    def canon(self, ctx: CanonContext) -> None:
        """Interleaved 3-dimensional exponential cones."""
        x, y, z = (require_affine(a, "ExpCone") for a in self.args)
        ctx.exp3(x, y, z, owner=self)

    def set_dual(self, values: np.ndarray) -> None:
        """Dual as an ``(k, 3)`` array, one row per cone."""
        self._dual = np.asarray(values, dtype=float).reshape(-1, 3)


class PowCone3D(Constraint):
    """Elementwise power cone ``x^alpha y^(1 - alpha) >= |z|``."""

    def __init__(self, x: Any, y: Any, z: Any, alpha: float) -> None:
        """Create ``(x, y, z) in K_pow(alpha)``."""
        if not 0 < alpha < 1:
            msg = "alpha must lie in (0, 1)"
            raise ValueError(msg)
        super().__init__(as_expr(x), as_expr(y), as_expr(z))
        self.alpha = float(alpha)

    def canon(self, ctx: CanonContext) -> None:
        """Interleaved 3-dimensional power cones."""
        x, y, z = (require_affine(a, "PowCone3D") for a in self.args)
        ctx.pow3(x, y, z, self.alpha, owner=self)

    def set_dual(self, values: np.ndarray) -> None:
        """Dual as an ``(k, 3)`` array, one row per cone."""
        self._dual = np.asarray(values, dtype=float).reshape(-1, 3)


ops.equality = Equality
ops.inequality = Inequality
ops.psd = PSD
