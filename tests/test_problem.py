"""End-to-end problems compared with cvxpy, solver dispatch, statuses and duals."""

import cvxpy as cp
import numpy as np
import pytest

import akme as ak


@pytest.fixture
def market():
    """Random covariance matrix and expected returns."""
    rng = np.random.default_rng(42)
    n = 12
    f = rng.standard_normal((n, n))
    return n, f @ f.T / n + 0.01 * np.eye(n), 0.1 * rng.standard_normal(n)


def _both(build_ak, build_cp):
    """Solve the akme and the cvxpy formulation; return both problems."""
    p_ak, p_cp = build_ak(), build_cp()
    p_ak.solve()
    p_cp.solve(solver=cp.CLARABEL)
    return p_ak, p_cp


# --------------------------------------------------------------- polyhedral
def test_portfolio_qp(market):
    """Long-only portfolio QP with box bounds (active-set)."""
    n, s, mu = market
    w, wc = ak.Variable(n), cp.Variable(n)
    p_ak, p_cp = _both(
        lambda: ak.Problem(ak.Minimize(ak.quad_form(w, s) - mu @ w), [ak.sum(w) == 1, w >= 0, w <= 0.25]),
        lambda: cp.Problem(cp.Minimize(cp.quad_form(wc, s) - mu @ wc), [cp.sum(wc) == 1, wc >= 0, wc <= 0.25]),
    )
    assert p_ak.solver_stats.solver_name == ak.ACTIVE_SET
    assert p_ak.status == ak.OPTIMAL
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-6)
    np.testing.assert_allclose(w.value, wc.value, atol=1e-5)
    for c_ak, c_cp in zip(p_ak.constraints, p_cp.constraints, strict=True):
        np.testing.assert_allclose(c_ak.dual_value, c_cp.dual_value, atol=1e-5)


def test_lasso_with_abs_is_polyhedral():
    """Lasso with abs is polyhedral."""
    rng = np.random.default_rng(0)
    a, b = rng.standard_normal((30, 8)), rng.standard_normal(30)
    x, xc = ak.Variable(8), cp.Variable(8)
    p_ak, p_cp = _both(
        lambda: ak.Problem(ak.Minimize(ak.sum_squares(a @ x - b) + 2 * ak.norm1(x))),
        lambda: cp.Problem(cp.Minimize(cp.sum_squares(a @ xc - b) + 2 * cp.norm1(xc))),
    )
    assert p_ak.solver_stats.solver_name == ak.ACTIVE_SET
    assert p_ak.value == pytest.approx(p_cp.value, rel=1e-6)
    np.testing.assert_allclose(x.value, xc.value, atol=1e-5)


def test_piecewise_linear_atoms():
    """Piecewise linear atoms."""
    rng = np.random.default_rng(5)
    a, b = rng.standard_normal((10, 4)), rng.standard_normal(10)
    x, xc = ak.Variable(4), cp.Variable(4)
    p_ak, p_cp = _both(
        lambda: ak.Problem(
            ak.Minimize(ak.norm_inf(a @ x - b) + ak.sum(ak.pos(x - 0.5)) + ak.max(x) - ak.min(x)),
            [ak.sum(ak.maximum(x, -1)) <= 2, ak.sum(ak.minimum(x, 1)) >= -2],
        ),
        lambda: cp.Problem(
            cp.Minimize(cp.norm_inf(a @ xc - b) + cp.sum(cp.pos(xc - 0.5)) + cp.max(xc) - cp.min(xc)),
            [cp.sum(cp.maximum(xc, -1)) <= 2, cp.sum(cp.minimum(xc, 1)) >= -2],
        ),
    )
    assert p_ak.solver_stats.solver_name == ak.ACTIVE_SET
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-6)


def test_lp_maximize_matrix_variable():
    """LP on a matrix variable with axis sums (active-set)."""
    x, xc = ak.Variable((2, 3), nonneg=True), cp.Variable((2, 3), nonneg=True)
    c = np.arange(6.0).reshape(2, 3)
    p_ak, p_cp = _both(
        lambda: ak.Problem(ak.Maximize(ak.sum(ak.multiply(c, x))), [ak.sum(x, axis=0) <= 1, ak.sum(x, axis=1) <= 2]),
        lambda: cp.Problem(cp.Maximize(cp.sum(cp.multiply(c, xc))), [cp.sum(xc, axis=0) <= 1, cp.sum(xc, axis=1) <= 2]),
    )
    assert p_ak.solver_stats.solver_name == ak.ACTIVE_SET
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-6)


# ---------------------------------------------------------- non-polyhedral
def test_soc_risk_limit(market):
    """Return maximisation under an SOC risk limit (Clarabel)."""
    n, s, mu = market
    lchol = np.linalg.cholesky(s)
    w, wc = ak.Variable(n), cp.Variable(n)
    p_ak, p_cp = _both(
        lambda: ak.Problem(ak.Maximize(mu @ w), [ak.sum(w) == 1, w >= 0, ak.norm2(lchol.T @ w) <= 0.3]),
        lambda: cp.Problem(cp.Maximize(mu @ wc), [cp.sum(wc) == 1, wc >= 0, cp.norm2(lchol.T @ wc) <= 0.3]),
    )
    assert p_ak.solver_stats.solver_name == ak.CLARABEL
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-6)
    np.testing.assert_allclose(p_ak.constraints[2].dual_value, p_cp.constraints[2].dual_value, atol=1e-5)


def test_quadratic_constraint(market):
    """Quadratic constraint."""
    n, s, mu = market
    w, wc = ak.Variable(n), cp.Variable(n)
    p_ak, p_cp = _both(
        lambda: ak.Problem(ak.Maximize(mu @ w), [ak.sum(w) == 1, w >= 0, ak.quad_form(w, s) <= 0.09]),
        lambda: cp.Problem(cp.Maximize(mu @ wc), [cp.sum(wc) == 1, wc >= 0, cp.quad_form(wc, s) <= 0.09]),
    )
    assert p_ak.solver_stats.solver_name == ak.CLARABEL
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-6)


def test_nearest_correlation_psd():
    """Nearest correlation psd."""
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, (5, 5))
    a = (a + a.T) / 2
    np.fill_diagonal(a, 1.0)
    x, xc = ak.Variable((5, 5), PSD=True), cp.Variable((5, 5), PSD=True)
    p_ak, p_cp = _both(
        lambda: ak.Problem(ak.Minimize(ak.sum_squares(x - a)), [ak.diag(x) == 1]),
        lambda: cp.Problem(cp.Minimize(cp.sum_squares(xc - a)), [cp.diag(xc) == 1]),
    )
    assert p_ak.solver_stats.solver_name == ak.CLARABEL
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-6)
    np.testing.assert_allclose(x.value, xc.value, atol=1e-5)
    assert np.linalg.eigvalsh(x.value).min() > -1e-7


def test_lambda_max_and_psd_dual():
    """Lambda max and psd dual."""
    rng = np.random.default_rng(2)
    a0, a1 = (lambda m: (m + m.T) / 2)(rng.standard_normal((4, 4))), np.diag([1.0, -1.0, 2.0, 0.5])
    t, tc = ak.Variable(), cp.Variable()
    p_ak, p_cp = _both(
        lambda: ak.Problem(ak.Minimize(ak.lambda_max(a0 + t * a1) - 0.1 * t), [ak.abs(t) <= 3]),
        lambda: cp.Problem(cp.Minimize(cp.lambda_max(a0 + tc * a1) - 0.1 * tc), [cp.abs(tc) <= 3]),
    )
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-6)

    x = ak.Variable((3, 3), symmetric=True)
    prob = ak.Problem(ak.Minimize(ak.trace(x)), [x >> np.diag([1.0, 2.0, 3.0])])
    assert prob.solve() == pytest.approx(6.0, abs=1e-6)
    np.testing.assert_allclose(prob.constraints[0].dual_value, np.eye(3), atol=1e-5)


def test_entropy_maximization():
    """Entropy maximization."""
    rng = np.random.default_rng(3)
    a, b = rng.uniform(0, 1, (3, 6)), rng.uniform(0.3, 0.6, 3)
    x, xc = ak.Variable(6), cp.Variable(6)
    p_ak, p_cp = _both(
        lambda: ak.Problem(ak.Maximize(ak.sum(ak.entr(x))), [ak.sum(x) == 1, a @ x <= b]),
        lambda: cp.Problem(cp.Maximize(cp.sum(cp.entr(xc))), [cp.sum(xc) == 1, a @ xc <= b]),
    )
    assert p_ak.solver_stats.solver_name == ak.CLARABEL
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-6)


def test_exp_log_lse():
    """Exp log log-sum-exp."""
    x, xc = ak.Variable(3), cp.Variable(3)
    c = np.array([1.0, 2.0, 3.0])
    p_ak, p_cp = _both(
        lambda: ak.Problem(
            ak.Minimize(ak.log_sum_exp(x) + ak.sum(ak.exp(-x)) - ak.sum(ak.log(x + 2))), [c @ x >= 1, x <= 2]
        ),
        lambda: cp.Problem(
            cp.Minimize(cp.log_sum_exp(xc) + cp.sum(cp.exp(-xc)) - cp.sum(cp.log(xc + 2))), [c @ xc >= 1, xc <= 2]
        ),
    )
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-6)


def test_power_cones():
    """Power cones."""
    x, xc = ak.Variable(4), cp.Variable(4)
    b = np.array([1.0, -2.0, 0.5, 3.0])
    p_ak, p_cp = _both(
        lambda: ak.Problem(
            ak.Minimize(ak.pnorm(x - b, 3) + ak.sum(ak.inv_pos(x + 3)) - ak.sum(ak.sqrt(x + 3))),
            [ak.sum(ak.power(x, 1.5)) <= 10, x >= 0],
        ),
        lambda: cp.Problem(
            cp.Minimize(cp.pnorm(xc - b, 3) + cp.sum(cp.inv_pos(xc + 3)) - cp.sum(cp.sqrt(xc + 3))),
            [cp.sum(cp.power(xc, 1.5)) <= 10, xc >= 0],
        ),
    )
    assert p_ak.solver_stats.solver_name == ak.CLARABEL
    assert p_ak.value == pytest.approx(p_cp.value, abs=1e-5)


def test_explicit_cone_constraints():
    """Explicit cone constraints."""
    x = ak.Variable(2)
    t = ak.Variable()
    prob = ak.Problem(ak.Minimize(t), [ak.SOC(t, x - np.array([3.0, 4.0]))])
    assert prob.solve() == pytest.approx(0.0, abs=1e-6)

    y = ak.Variable()
    prob = ak.Problem(ak.Minimize(y), [ak.ExpCone(1.0, 1.0, y)])
    assert prob.solve() == pytest.approx(np.e, abs=1e-6)

    z = ak.Variable()
    prob = ak.Problem(ak.Maximize(z), [ak.PowCone3D(4.0, 1.0, z, 0.5)])
    assert prob.solve() == pytest.approx(2.0, abs=1e-6)


# -------------------------------------------------------------- dispatch etc
def test_solver_override_and_errors():
    """Solver override and errors."""
    x = ak.Variable(3)
    prob = ak.Problem(ak.Minimize(ak.sum_squares(x - 1)), [ak.sum(x) == 1])
    assert prob.is_polyhedral()
    v_as = prob.solve()
    v_cl = prob.solve(solver="clarabel")
    assert prob.solver_stats.solver_name == ak.CLARABEL
    assert v_as == pytest.approx(v_cl, abs=1e-7)

    soc = ak.Problem(ak.Minimize(ak.norm2(x)), [ak.sum(x) == 1])
    assert not soc.is_polyhedral()
    with pytest.raises(ValueError, match="polyhedral"):
        soc.solve(solver="ACTIVE_SET")
    with pytest.raises(ValueError, match="unknown solver"):
        soc.solve(solver="GUROBI")


def test_infeasible_and_unbounded():
    """Infeasible and unbounded."""
    x = ak.Variable(2)
    for solver in (None, ak.CLARABEL):
        prob = ak.Problem(ak.Minimize(ak.sum(x)), [x >= 1, ak.sum(x) <= 1])
        assert prob.solve(solver=solver) == np.inf
        assert prob.status == ak.INFEASIBLE
        assert x.value is None

        prob = ak.Problem(ak.Minimize(ak.sum(x)), [x <= 1])
        assert prob.solve(solver=solver) == -np.inf
        assert prob.status == ak.UNBOUNDED

        prob = ak.Problem(ak.Maximize(ak.sum(x)), [x <= 1])
        assert prob.solve(solver=solver) == pytest.approx(2.0)


def test_warm_start_from_variable_values(market):
    """Warm start from variable values."""
    n, s, mu = market
    w = ak.Variable(n)
    cons = [ak.sum(w) == 1, w >= 0, w <= 0.2, ak.abs(w - 1 / n) <= 0.15]
    cold = ak.Problem(ak.Minimize(ak.quad_form(w, s) - mu @ w), cons)
    cold.solve()
    mu2 = mu + 1e-4
    warm = ak.Problem(ak.Minimize(ak.quad_form(w, s) - mu2 @ w), cons)
    warm.solve(warm_start=True)
    w_warm = w.value
    w.value = None
    ref = ak.Problem(ak.Minimize(ak.quad_form(w, s) - mu2 @ w), cons)
    ref.solve()
    np.testing.assert_allclose(w_warm, w.value, atol=1e-7)
    assert warm.solver_stats.num_iters < ref.solver_stats.num_iters


def test_objective_value_and_constant_objective():
    """Objective value and constant objective."""
    x = ak.Variable(2)
    prob = ak.Problem(ak.Minimize(ak.sum_squares(x) + 3), [x >= 1])
    assert prob.solve() == pytest.approx(5.0)
    assert prob.objective.value == pytest.approx(5.0)
    feas = ak.Problem(ak.Minimize(0), [x >= 1, x <= 2])
    assert feas.solve() == pytest.approx(0.0)
    assert np.all(x.value >= 1 - 1e-9)
