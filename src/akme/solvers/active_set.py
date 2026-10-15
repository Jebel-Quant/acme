"""Primal active-set method for convex quadratic programs.

Solves

    minimize    1/2 x^T P x + q^T x
    subject to  E x = f,  G x <= h

with ``P`` positive semidefinite (Nocedal & Wright, Algorithm 16.3). Each iteration
works in the null space ``Z`` of the working-set rows: on a positive definite reduced
Hessian it takes a (damped) Newton step; along directions of zero curvature it moves
on a descent ray up to the first blocking constraint (no blocker means unbounded).
A feasible starting point is found by a phase-1 problem ``min tau s.t. Gx - tau <= h``
solved with the same routine. Dense numpy linear algebra -- aimed at small and medium
problems, with warm starts from a previous point and working set.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.linalg as la

from akme import status as st
from akme.solvers.base import RawResult

if TYPE_CHECKING:
    from akme.canon import StandardForm


@dataclass
class QPResult:
    """Result of :func:`solve_qp`.

    Multipliers satisfy ``P x + q + E^T y_eq + G^T y_ineq = 0`` with ``y_ineq >= 0``.
    """

    status: str
    x: np.ndarray | None
    y_eq: np.ndarray | None
    y_ineq: np.ndarray | None
    iterations: int
    working_set: list[int] = field(default_factory=list)


def solve_qp(
    p: np.ndarray,
    q: np.ndarray,
    e: np.ndarray | None = None,
    f: np.ndarray | None = None,
    g: np.ndarray | None = None,
    h: np.ndarray | None = None,
    *,
    x0: np.ndarray | None = None,
    working_set: list[int] | None = None,
    max_iter: int | None = None,
    tol: float = 1e-9,
) -> QPResult:
    """Solve a convex QP with the primal active-set method.

    Args:
        p: Positive semidefinite Hessian ``(n, n)``.
        q: Linear term ``(n,)``.
        e: Equality matrix ``(me, n)``.
        f: Equality right-hand side ``(me,)``.
        g: Inequality matrix ``(mi, n)``.
        h: Inequality right-hand side ``(mi,)``.
        x0: Optional starting point (need not be feasible).
        working_set: Optional initial guess of active inequality indices.
        max_iter: Iteration limit (default ``10 (n + me + mi) + 50``).
        tol: Relative tolerance for feasibility, optimality and rank decisions.
    """
    q = np.asarray(q, dtype=float).ravel()
    n = q.size
    p = np.asarray(p, dtype=float).reshape(n, n)
    e = np.zeros((0, n)) if e is None else np.asarray(e, dtype=float).reshape(-1, n)
    f = np.zeros(0) if f is None else np.asarray(f, dtype=float).ravel()
    g = np.zeros((0, n)) if g is None else np.asarray(g, dtype=float).reshape(-1, n)
    h = np.zeros(0) if h is None else np.asarray(h, dtype=float).ravel()
    me, mi = e.shape[0], g.shape[0]
    max_iter = max_iter or 10 * (n + me + mi) + 50
    feas_tol = tol * max(1.0, np.max(np.abs(h), initial=0.0), np.max(np.abs(f), initial=0.0))

    # independent equality rows; inconsistent systems are infeasible
    keep = _independent_rows(e, tol)
    ek, fk = e[keep], f[keep]
    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float).copy()
    if ek.shape[0]:
        x = x + np.linalg.lstsq(ek, fk - ek @ x, rcond=None)[0]
        if np.max(np.abs(e @ x - f), initial=0.0) > 1e3 * feas_tol:
            return QPResult(st.INFEASIBLE, None, None, None, 0)

    iterations = 0
    candidates = list(working_set or [])
    if mi and np.max(g @ x - h) > feas_tol:
        status, x, w1, iterations = _phase_one(ek, g, h, x, max_iter, tol, feas_tol)
        if status != st.OPTIMAL:
            return QPResult(status, None, None, None, iterations)
        candidates = candidates or w1
    elif working_set is None and x0 is not None:
        candidates = list(np.flatnonzero(np.abs(h - g @ x) <= feas_tol))
    w = _independent_working_set(ek, g, h, x, candidates, feas_tol, tol)

    status, x, w, y, it = _active_set(p, q, ek, g, h, x, w, max_iter - iterations, tol)
    iterations += it
    if status != st.OPTIMAL:
        return QPResult(status, None, None, None, iterations, w)
    y_eq = np.zeros(me)
    y_eq[keep] = y[: len(keep)]
    y_ineq = np.zeros(mi)
    y_ineq[w] = np.maximum(y[len(keep) :], 0.0)
    return QPResult(st.OPTIMAL, x, y_eq, y_ineq, iterations, w)


def solve(form: StandardForm, *, warm_start: bool = False, verbose: bool = False, **options: Any) -> RawResult:
    """Solve a polyhedral standard form; ``options`` are passed to :func:`solve_qp`."""
    if not form.is_polyhedral:
        msg = "the active-set solver only handles polyhedral problems (zero / nonnegative cones)"
        raise ValueError(msg)
    a = form.a.toarray()
    meq = form.n_eq
    x0 = form.initial_point() if warm_start else None
    start = time.perf_counter()
    res = solve_qp(form.p.toarray(), form.q, a[:meq], form.b[:meq], a[meq:], form.b[meq:], x0=x0, **options)
    elapsed = time.perf_counter() - start
    if verbose:
        print(f"akme active-set: status={res.status}, iterations={res.iterations}, time={elapsed:.3g}s")
    z = None if res.y_eq is None else np.concatenate([res.y_eq, res.y_ineq])  # type: ignore[list-item]
    return RawResult(res.status, res.x, z, res.iterations, elapsed, st.ACTIVE_SET)


# ------------------------------------------------------------------- internals
def _independent_rows(e: np.ndarray, tol: float) -> np.ndarray:
    """Indices of a maximal set of linearly independent rows (QR with column pivoting)."""
    if e.shape[0] == 0:
        return np.zeros(0, dtype=int)
    _, r, piv = la.qr(e.T, mode="economic", pivoting=True)
    d = np.abs(np.diag(r))
    rank = int(np.sum(d > tol * max(e.shape) * (d[0] if d.size else 1.0)))
    return np.sort(piv[:rank])


def _independent_working_set(
    e: np.ndarray, g: np.ndarray, h: np.ndarray, x: np.ndarray, candidates: list[int], feas_tol: float, tol: float
) -> list[int]:
    """Greedily keep active candidate rows that are independent of the equalities and each other."""
    w: list[int] = []
    rows = e
    for i in dict.fromkeys(int(c) for c in candidates):
        if i >= g.shape[0] or abs(h[i] - g[i] @ x) > feas_tol:
            continue
        trial = np.vstack([rows, g[i]])
        if (
            trial.shape[0] <= trial.shape[1]
            and np.linalg.matrix_rank(trial, tol=tol * max(trial.shape)) == trial.shape[0]
        ):
            rows = trial
            w.append(i)
    return w


def _phase_one(
    e: np.ndarray, g: np.ndarray, h: np.ndarray, x: np.ndarray, max_iter: int, tol: float, feas_tol: float
) -> tuple[str, np.ndarray, list[int], int]:
    """Find a feasible point via ``min tau s.t. E x = f, G x - tau <= h, tau >= 0``."""
    n, mi = x.size, g.shape[0]
    p1 = np.zeros((n + 1, n + 1))
    q1 = np.zeros(n + 1)
    q1[-1] = 1.0
    e1 = np.hstack([e, np.zeros((e.shape[0], 1))])
    g1 = np.vstack([np.hstack([g, -np.ones((mi, 1))]), np.eye(1, n + 1, n) * -1.0])
    h1 = np.concatenate([h, [0.0]])
    x1 = np.concatenate([x, [np.max(g @ x - h)]])
    status, xs, w, _, it = _active_set(p1, q1, e1, g1, h1, x1, [], max_iter, tol)
    if status != st.OPTIMAL:
        return status, x, [], it
    if xs[-1] > feas_tol:
        return st.INFEASIBLE, x, [], it
    return st.OPTIMAL, xs[:n], [i for i in w if i < mi], it


def _direction(
    p: np.ndarray, grad: np.ndarray, z: np.ndarray, x: np.ndarray, tol: float
) -> tuple[np.ndarray | None, bool]:
    """Step in the null space ``z``: Newton step, descent ray (zero curvature) or ``None`` (stationary)."""
    if z.shape[1] == 0:
        return None, False
    gz = z.T @ grad
    gscale = max(1.0, float(np.linalg.norm(grad)))
    if np.linalg.norm(gz) <= tol * gscale:
        return None, False
    evals, evecs = np.linalg.eigh(z.T @ p @ z)
    curved = evals > 1e-10 * max(1.0, float(np.max(np.abs(evals))))
    coef = evecs.T @ gz
    if np.linalg.norm(coef[~curved]) > tol * gscale:
        d = -(z @ (evecs[:, ~curved] @ coef[~curved]))
        return d / np.linalg.norm(d), True
    d = -(z @ (evecs[:, curved] @ (coef[curved] / evals[curved])))
    if np.linalg.norm(d) <= tol * max(1.0, float(np.linalg.norm(x))):
        return None, False
    return d, False


def _active_set(
    p: np.ndarray,
    q: np.ndarray,
    e: np.ndarray,
    g: np.ndarray,
    h: np.ndarray,
    x: np.ndarray,
    w: list[int],
    max_iter: int,
    tol: float,
) -> tuple[str, np.ndarray, list[int], np.ndarray, int]:
    """Primal active-set iterations from a feasible ``x`` with independent working set ``w``."""
    n, me = q.size, e.shape[0]
    row_norms = np.linalg.norm(g, axis=1)
    w = list(w)
    for it in range(1, max_iter + 1):
        grad = p @ x + q
        aw = np.vstack([e, g[w]]) if w else e
        z = la.null_space(aw, rcond=tol) if aw.shape[0] else np.eye(n)
        d, ray = _direction(p, grad, z, x, tol)

        if d is None:
            y = np.linalg.lstsq(aw.T, -grad, rcond=None)[0] if aw.shape[0] else np.zeros(0)
            mu = y[me:]
            if mu.size == 0 or np.min(mu) >= -tol * max(1.0, float(np.linalg.norm(grad))):
                return st.OPTIMAL, x, w, y, it
            w.pop(int(np.argmin(mu)))
            continue

        # ratio test over inactive constraints moving towards their bound
        gd = g @ d
        moving = gd > tol * row_norms * np.linalg.norm(d)
        moving[w] = False
        alpha, block = (np.inf if ray else 1.0), None
        if np.any(moving):
            idx = np.flatnonzero(moving)
            ratios = np.maximum(h[idx] - g[idx] @ x, 0.0) / gd[idx]
            k = int(np.argmin(ratios))
            if ratios[k] <= alpha:
                alpha, block = float(ratios[k]), int(idx[k])
        if block is None and ray:
            return st.UNBOUNDED, x, w, np.zeros(0), it
        x = x + alpha * d
        if block is not None:
            w.append(block)
    return st.USER_LIMIT, x, w, np.zeros(0), max_iter
