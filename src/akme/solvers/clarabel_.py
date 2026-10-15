"""Interior-point backend: Clarabel."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import clarabel
import numpy as np
import scipy.sparse as sp

from akme import status as st
from akme.cones import ConeKind, ConeSpec
from akme.solvers.base import RawResult

if TYPE_CHECKING:
    from akme.canon import StandardForm

_STATUS = {
    "Solved": st.OPTIMAL,
    "AlmostSolved": st.OPTIMAL_INACCURATE,
    "PrimalInfeasible": st.INFEASIBLE,
    "AlmostPrimalInfeasible": st.INFEASIBLE_INACCURATE,
    "DualInfeasible": st.UNBOUNDED,
    "AlmostDualInfeasible": st.UNBOUNDED_INACCURATE,
    "MaxIterations": st.USER_LIMIT,
    "MaxTime": st.USER_LIMIT,
}


def _cone(spec: ConeSpec) -> Any:
    """Clarabel cone object for a cone spec."""
    if spec.kind is ConeKind.ZERO:
        return clarabel.ZeroConeT(spec.dim)
    if spec.kind is ConeKind.NONNEG:
        return clarabel.NonnegativeConeT(spec.dim)
    if spec.kind is ConeKind.SOC:
        return clarabel.SecondOrderConeT(spec.dim)
    if spec.kind is ConeKind.EXP:
        return clarabel.ExponentialConeT()
    if spec.kind is ConeKind.POW:
        return clarabel.PowerConeT(spec.alpha)
    return clarabel.PSDTriangleConeT(spec.psd_order)


def solve(form: StandardForm, *, verbose: bool = False, **options: Any) -> RawResult:
    """Solve the standard form with Clarabel; ``options`` are Clarabel settings."""
    settings = clarabel.DefaultSettings()
    settings.verbose = verbose
    for key, value in options.items():
        if not hasattr(settings, key):
            msg = f"unknown Clarabel setting {key!r}"
            raise ValueError(msg)
        setattr(settings, key, value)

    start = time.perf_counter()
    solver = clarabel.DefaultSolver(
        sp.csc_matrix(sp.triu(form.p)),
        form.q,
        sp.csc_matrix(form.a),
        form.b,
        [_cone(c) for c in form.cones],
        settings,
    )
    sol = solver.solve()
    elapsed = time.perf_counter() - start

    status = _STATUS.get(str(sol.status).split(".")[-1], st.SOLVER_ERROR)
    x = np.asarray(sol.x, dtype=float) if status in st.SOLUTION_PRESENT else None
    z = np.asarray(sol.z, dtype=float) if status in st.SOLUTION_PRESENT else None
    return RawResult(status, x, z, int(sol.iterations), elapsed, st.CLARABEL)
