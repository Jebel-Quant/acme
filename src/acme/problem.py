"""cvxpy-style problems: ``Problem(Minimize(...), constraints).solve()``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from acme import status as st
from acme.canon import StandardForm, build_standard_form
from acme.constraints import Constraint
from acme.expressions.expression import DCPError, Expression, as_expr, unique_variables
from acme.expressions.variable import Variable
from acme.solvers import dispatch


class Minimize:
    """Minimization objective (must be convex)."""

    sense = 1.0

    def __init__(self, expr: Any) -> None:
        """Wrap a scalar expression."""
        self.expr = as_expr(expr)
        if self.expr.size != 1:
            msg = "the objective must be a scalar"
            raise ValueError(msg)

    def is_dcp(self) -> bool:
        """True if the objective is convex."""
        return self.expr.is_convex()

    @property
    def value(self) -> float | None:
        """Objective value at the current variable values."""
        v = self.expr.value
        return None if v is None else float(np.asarray(v).ravel()[0])


class Maximize(Minimize):
    """Maximization objective (must be concave)."""

    sense = -1.0

    def is_dcp(self) -> bool:
        """True if the objective is concave."""
        return self.expr.is_concave()


@dataclass
class SolverStats:
    """Information about the last solve."""

    solver_name: str
    solve_time: float
    num_iters: int


class Problem:
    """An optimisation problem.

    ``solve()`` routes polyhedral problems (affine constraints, piecewise-linear atoms,
    quadratic objective) to acme's active-set solver and everything else to Clarabel.
    """

    def __init__(self, objective: Minimize, constraints: list[Constraint] | None = None) -> None:
        """Create a problem from an objective and a list of constraints."""
        self.objective = objective
        self.constraints = list(constraints or [])
        self.status: str | None = None
        self.value: float | None = None
        self.solver_stats: SolverStats | None = None

    def variables(self) -> list[Variable]:
        """User variables of the problem."""
        found = list(self.objective.expr.variables())
        for con in self.constraints:
            found.extend(con.variables())
        return unique_variables(found)

    def is_dcp(self) -> bool:
        """True if objective and all constraints follow the DCP rules."""
        return self.objective.is_dcp() and all(c.is_dcp() for c in self.constraints)

    def get_problem_data(self) -> StandardForm:
        """Compile to the conic standard form (see :class:`acme.canon.StandardForm`)."""
        if not self.objective.is_dcp():
            kind = "convex" if self.objective.sense > 0 else "concave"
            msg = f"objective is {self.objective.expr.curvature.value}, expected {kind}"
            raise DCPError(msg)
        for con in self.constraints:
            if not con.is_dcp():
                msg = f"constraint {con!r} does not follow the DCP rules"
                raise DCPError(msg)
        implicit = [c for v in self.variables() for c in v.implicit_constraints()]
        expr: Expression = self.objective.expr if self.objective.sense > 0 else -self.objective.expr
        return build_standard_form(expr, self.constraints + implicit)

    def is_polyhedral(self) -> bool:
        """True if the feasible set is polyhedral (solved by the active-set method)."""
        return self.get_problem_data().is_polyhedral

    def solve(
        self, solver: str | None = None, *, warm_start: bool = False, verbose: bool = False, **solver_opts: Any
    ) -> float | None:
        """Solve the problem and return the optimal value (``None`` without a solution or certificate).

        Args:
            solver: ``None`` (automatic), ``"ACTIVE_SET"`` or ``"CLARABEL"``.
            warm_start: Start the active-set solver from the variables' current values.
            verbose: Print solver output.
            **solver_opts: Backend options (Clarabel settings, or ``max_iter`` / ``tol`` for active-set).

        Example:
            >>> import acme as ak
            >>> x = ak.Variable(2)
            >>> prob = ak.Problem(ak.Maximize(ak.sum(x)), [ak.norm2(x) <= 1])
            >>> round(prob.solve(), 6)
            1.414214
            >>> prob.status, prob.solver_stats.solver_name
            ('optimal', 'CLARABEL')
            >>> x.value.round(6).tolist()
            [0.707107, 0.707107]
        """
        form = self.get_problem_data()
        raw = dispatch.solve(form, solver, warm_start=warm_start, verbose=verbose, **solver_opts)
        self.status = raw.status
        self.solver_stats = SolverStats(raw.solver_name, raw.solve_time, raw.iterations)
        sense = self.objective.sense
        if raw.status in st.SOLUTION_PRESENT and raw.x is not None:
            form.unpack(raw.x, raw.z)
            value = 0.5 * raw.x @ (form.p @ raw.x) + form.q @ raw.x + form.offset
            self.value = float(sense * value)
        else:
            for var in form.variables:
                var.value = None
            if raw.status in (st.INFEASIBLE, st.INFEASIBLE_INACCURATE):
                self.value = float(sense * np.inf)
            elif raw.status in (st.UNBOUNDED, st.UNBOUNDED_INACCURATE):
                self.value = float(-sense * np.inf)
            else:
                self.value = None
        return self.value
