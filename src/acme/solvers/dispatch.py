"""Pick a backend: active-set for polyhedral problems, Clarabel otherwise."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from acme import status as st
from acme.solvers import active_set, clarabel_

if TYPE_CHECKING:
    from acme.canon import StandardForm
    from acme.solvers.base import RawResult


def choose_solver(form: StandardForm, solver: str | None = None) -> str:
    """Name of the backend used for ``form`` (``solver`` overrides the automatic choice).

    Example:
        >>> import acme as ak
        >>> from acme.solvers import choose_solver
        >>> x = ak.Variable(2)
        >>> qp = ak.Problem(ak.Minimize(ak.sum_squares(x)), [x >= 1])
        >>> socp = ak.Problem(ak.Minimize(ak.sum(x)), [ak.norm2(x) <= 1])
        >>> choose_solver(qp.get_problem_data()), choose_solver(socp.get_problem_data())
        ('ACTIVE_SET', 'CLARABEL')
        >>> choose_solver(qp.get_problem_data(), "clarabel")
        'CLARABEL'
    """
    if solver is None:
        return st.ACTIVE_SET if form.is_polyhedral else st.CLARABEL
    solver = solver.upper()
    if solver not in (st.ACTIVE_SET, st.CLARABEL):
        msg = f"unknown solver {solver!r}; choose {st.ACTIVE_SET!r} or {st.CLARABEL!r}"
        raise ValueError(msg)
    if solver == st.ACTIVE_SET and not form.is_polyhedral:
        msg = "ACTIVE_SET requires a polyhedral feasible set; this problem has non-polyhedral cones"
        raise ValueError(msg)
    return solver


def solve(
    form: StandardForm, solver: str | None = None, *, warm_start: bool = False, verbose: bool = False, **options: Any
) -> RawResult:
    """Solve the standard form with the chosen backend."""
    if choose_solver(form, solver) == st.ACTIVE_SET:
        return active_set.solve(form, warm_start=warm_start, verbose=verbose, **options)
    return clarabel_.solve(form, verbose=verbose, **options)
