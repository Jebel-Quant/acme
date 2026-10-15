"""The primal active-set QP solver, checked against Clarabel."""

import clarabel
import numpy as np
import pytest
import scipy.sparse as sp

import akme as ak
from akme.solvers.active_set import solve_qp


def _clarabel(p, q, e, f, g, h):
    """Solve the same QP with Clarabel; return status, solution and objective."""
    a = sp.csc_matrix(np.vstack([e, g]))
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    cones = [clarabel.ZeroConeT(e.shape[0]), clarabel.NonnegativeConeT(g.shape[0])]
    sol = clarabel.DefaultSolver(sp.csc_matrix(np.triu(p)), q, a, np.concatenate([f, h]), cones, settings).solve()
    return str(sol.status), np.asarray(sol.x), sol.obj_val


def _random_qp(rng, n, me, mi, rank=None):
    """Random feasible convex QP; ``rank`` < n gives a semidefinite P (0: an LP)."""
    f_ = rng.standard_normal((n if rank is None else rank, n))
    p = f_.T @ f_
    q = rng.standard_normal(n)
    x_feas = rng.standard_normal(n)
    e = rng.standard_normal((me, n))
    g = rng.standard_normal((mi, n))
    return p, q, e, e @ x_feas, g, g @ x_feas + rng.uniform(0.0, 1.0, mi)


def _check_kkt(res, p, q, e, f, g, h, tol=1e-7):
    """Assert primal feasibility, dual feasibility, stationarity and complementarity."""
    x = res.x
    assert np.abs(e @ x - f).max(initial=0) < tol
    assert (g @ x - h).max(initial=-1) < tol
    assert res.y_ineq.min(initial=0) >= -tol
    np.testing.assert_allclose(p @ x + q + e.T @ res.y_eq + g.T @ res.y_ineq, 0, atol=1e-6)
    np.testing.assert_allclose(res.y_ineq * (g @ x - h), 0, atol=1e-6)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize(("n", "me", "mi", "rank"), [(5, 0, 8, None), (10, 2, 25, None), (12, 3, 30, 4), (8, 0, 20, 0)])
def test_random_qps_match_clarabel(seed, n, me, mi, rank):
    """Random qps match clarabel."""
    rng = np.random.default_rng(seed)
    p, q, e, f, g, h = _random_qp(rng, n, me, mi, rank)
    if rank is not None and rank < n:
        g = np.vstack([g, np.eye(n), -np.eye(n)])  # keep semidefinite problems bounded
        h = np.concatenate([h, np.full(2 * n, 10.0)])
    res = solve_qp(p, q, e, f, g, h)
    status, _, obj_ref = _clarabel(p, q, e, f, g, h)
    assert status == "Solved"
    assert res.status == ak.OPTIMAL
    obj = 0.5 * res.x @ p @ res.x + q @ res.x
    assert obj == pytest.approx(obj_ref, abs=1e-6, rel=1e-6)
    _check_kkt(res, p, q, e, f, g, h)


def test_unconstrained_and_equality_only():
    """Unconstrained and equality only."""
    p, q = np.diag([2.0, 4.0]), np.array([-2.0, -4.0])
    np.testing.assert_allclose(solve_qp(p, q).x, [1.0, 1.0])
    res = solve_qp(p, q, e=np.array([[1.0, 1.0]]), f=np.array([1.0]))
    np.testing.assert_allclose(res.x, [1 / 3, 2 / 3])


def test_redundant_equalities():
    """Redundant equalities."""
    rng = np.random.default_rng(3)
    p, q, e, f, g, h = _random_qp(rng, 6, 2, 10)
    e2, f2 = np.vstack([e, e[0] + e[1]]), np.concatenate([f, [f[0] + f[1]]])
    res = solve_qp(p, q, e2, f2, g, h)
    ref = solve_qp(p, q, e, f, g, h)
    np.testing.assert_allclose(res.x, ref.x, atol=1e-8)
    _check_kkt(res, p, q, e2, f2, g, h)


def test_inconsistent_equalities_are_infeasible():
    """Inconsistent equalities are infeasible."""
    e = np.array([[1.0, 1.0], [2.0, 2.0]])
    assert solve_qp(np.eye(2), np.zeros(2), e, np.array([1.0, 3.0])).status == ak.INFEASIBLE


def test_infeasible_inequalities():
    """Infeasible inequalities."""
    g = np.array([[1.0, 0.0], [-1.0, 0.0]])
    assert solve_qp(np.eye(2), np.zeros(2), g=g, h=np.array([-1.0, -1.0])).status == ak.INFEASIBLE


def test_unbounded_lp_and_semidefinite_qp():
    """Unbounded lp and semidefinite qp."""
    # min -x0 s.t. x1 <= 1  (x0 free)
    assert solve_qp(np.zeros((2, 2)), np.array([-1.0, 0.0]), g=np.array([[0.0, 1.0]]), h=[1.0]).status == ak.UNBOUNDED
    # min x1^2 - x0, x0 free
    assert solve_qp(np.diag([0.0, 1.0]), np.array([-1.0, 0.0])).status == ak.UNBOUNDED


def test_lp_vertex():
    """Lp vertex."""
    # max x + y s.t. x + 2y <= 4, 3x + y <= 6, x, y >= 0  -> (1.6, 1.2)
    g = np.array([[1.0, 2.0], [3.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    res = solve_qp(np.zeros((2, 2)), np.array([-1.0, -1.0]), g=g, h=np.array([4.0, 6.0, 0.0, 0.0]))
    np.testing.assert_allclose(res.x, [1.6, 1.2])
    np.testing.assert_allclose(res.y_ineq, [0.4, 0.2, 0.0, 0.0], atol=1e-12)


def test_degenerate_vertex():
    """Degenerate vertex."""
    # many constraints active at the optimum (0, 0)
    angles = np.linspace(0.1, np.pi - 0.1, 15)
    g = -np.column_stack([np.cos(angles), np.sin(angles)])
    res = solve_qp(np.eye(2), np.array([0.0, 1.0]), g=g, h=np.zeros(15), x0=np.array([0.0, 5.0]))
    assert res.status == ak.OPTIMAL
    np.testing.assert_allclose(res.x, 0.0, atol=1e-9)


def test_warm_start_reduces_iterations():
    """Warm start reduces iterations."""
    rng = np.random.default_rng(7)
    p, q, e, f, g, h = _random_qp(rng, 30, 1, 120)
    cold = solve_qp(p, q, e, f, g, h)
    q2 = q + 1e-3 * rng.standard_normal(q.size)
    warm = solve_qp(p, q2, e, f, g, h, x0=cold.x, working_set=cold.working_set)
    ref = solve_qp(p, q2, e, f, g, h)
    np.testing.assert_allclose(warm.x, ref.x, atol=1e-7)
    assert warm.iterations < ref.iterations


def test_max_iter():
    """Max iter."""
    rng = np.random.default_rng(0)
    p, q, e, f, g, h = _random_qp(rng, 10, 0, 40)
    assert solve_qp(p, q, e, f, g, h, max_iter=1).status == ak.USER_LIMIT
