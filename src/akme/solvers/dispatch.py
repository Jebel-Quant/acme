"""Pick a backend: active-set for polyhedral problems, Clarabel otherwise."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from akme import status as st
from akme.solvers import active_set, clarabel_

if TYPE_CHECKING:
    from akme.canon import StandardForm
    from akme.solvers.base import RawResult


def choose_solver(form: StandardForm, solver: str | None = None) -> str:
    """Name of the backend used for ``form`` (``solver`` overrides the automatic choice)."""
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
