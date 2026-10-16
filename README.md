# acme

cvxpy-style modelling of convex (quadratic) optimisation problems with automatic solver dispatch:

- **Polyhedral feasible set** (affine `==` / `<=` / `>=`, piecewise-linear atoms such as `abs`, `pos`,
  `maximum`, `norm1`, `norm_inf`) with a linear or convex quadratic objective → acme's own
  **primal active-set** QP solver (dense numpy, warm-startable).
- **Anything else** (second-order, PSD, exponential and power cones) → [Clarabel](https://clarabel.org).

## Installation

```bash
uv add acme-convex   # or: pip install acme-convex
```

## Quick start

```pycon
>>> import numpy as np
>>> import acme as ak

>>> n = 10
>>> rng = np.random.default_rng(0)
>>> f = rng.standard_normal((n, n))
>>> sigma, mu = f @ f.T / n, 0.1 * rng.standard_normal(n)

>>> w = ak.Variable(n)
>>> budget = ak.sum(w) == 1

>>> # Polyhedral → active-set
>>> prob = ak.Problem(ak.Minimize(ak.quad_form(w, sigma) - mu @ w), [budget, w >= 0, w <= 0.3])
>>> prob.solve()
-0.00568...
>>> prob.status
'optimal'
>>> prob.solver_stats.solver_name
'ACTIVE_SET'
>>> np.round(w.value, 3) + 0.0  # + 0.0 shows -0.0 as 0.0
array([0.3  , 0.266, 0.151, 0.   , 0.   , 0.   , 0.075, 0.165, 0.   ,
       0.043])
>>> round(float(budget.dual_value), 4)
-0.0508

>>> # Second-order cone risk limit → Clarabel
>>> chol = np.linalg.cholesky(sigma)
>>> prob = ak.Problem(ak.Maximize(mu @ w), [ak.sum(w) == 1, w >= 0, ak.norm2(chol.T @ w) <= 0.2])
>>> prob.solve()
0.04907...
>>> prob.solver_stats.solver_name
'CLARABEL'
```

## Modelling

| Category | Available |
|---|---|
| Variables | `Variable(shape, name, nonneg=, nonpos=, symmetric=, PSD=)` — scalars, vectors, matrices |
| Affine | `+ - * / @`, indexing, `.T`, `sum(x, axis)`, `trace`, `diag`, `reshape`, `vec`, `hstack`, `vstack`, `multiply` |
| Piecewise linear | `abs`, `pos`, `maximum`, `minimum`, `max`, `min`, `norm1`, `norm_inf` |
| Quadratic | `sum_squares`, `square`, `x**2`, `quad_form(x, S)` |
| Second-order cone | `norm2` / `norm(x, 2)`, quadratic atoms used in constraints |
| Exponential cone | `exp`, `log`, `entr`, `log_sum_exp` |
| Power cone | `power(x, p)`, `sqrt`, `inv_pos`, `pnorm(x, p)` |
| Semidefinite | `X >> Y`, `X << Y`, `lambda_max`, `lambda_min` |
| Explicit cones | `SOC(t, x)`, `ExpCone(x, y, z)`, `PowCone3D(x, y, z, alpha)` |

### Rules (light DCP)

- `Minimize` takes a convex objective and `Maximize` a concave one. A constraint `lhs <= rhs` needs
  `lhs - rhs` to be convex, and `==` / `>>` need affine sides.
- Atom arguments have to be **affine**. Nested atoms such as `exp(abs(x))` raise `DCPError`.
- Nonlinear terms can only be combined by sums and by constant linear maps with a single sign
  (for example `2 * sum(exp(x)) - log(y)`).
- `power(x, p)` with `p > 1` means `|x|^p`.

## Solving

```pycon
>>> prob.solve()  # automatic: ACTIVE_SET if polyhedral, else CLARABEL
0.04907...
>>> prob.solve(solver="CLARABEL", tol_gap_rel=1e-10)  # force a backend; extra kwargs are Clarabel settings
0.04907...
>>> prob.solve(warm_start=True)  # active-set starts from the variables' current values
0.04907...
>>> prob.is_polyhedral()  # which backend the automatic choice picks
False
>>> data = prob.get_problem_data()  # conic standard form: min ½x'Px + q'x  s.t. Ax + s = b, s ∈ K
```

After a solve, `prob.status` holds one of the cvxpy statuses (`optimal`, `infeasible`, `unbounded`,
`optimal_inaccurate`, `user_limit`, ...). `prob.value` is the optimal value (`±inf` when the problem
is infeasible or unbounded). Each variable's `.value` and each constraint's `.dual_value` are filled
in, and the duals follow cvxpy's sign conventions.

### Using the active-set solver directly

```pycon
>>> from acme import solve_qp

>>> # min ½x'Px + q'x  s.t.  E x = f,  G x <= h   (P positive semidefinite)
>>> p, q = np.diag([2.0, 4.0]), np.array([-2.0, -4.0])
>>> e, f = np.array([[1.0, 1.0]]), np.array([1.0])
>>> g, h = -np.eye(2), np.zeros(2)
>>> res = solve_qp(p, q, e, f, g, h, x0=None, working_set=None)
>>> res.status, res.working_set
('optimal', [])
>>> res.x
array([0.33333333, 0.66666667])
>>> res.y_eq, res.y_ineq
(array([1.33333333]), array([0., 0.]))
```

The solver uses the primal active-set method (Nocedal & Wright, Alg. 16.3) in the null space of the
working set:

- On a positive definite reduced Hessian it takes a Newton step. Along directions of zero curvature
  it follows a descent ray, which lets it handle LPs and semidefinite `P`. A ray that no constraint
  blocks means the problem is unbounded.
- It finds a feasible starting point with a phase-1 problem `min τ  s.t.  Gx − τ ≤ h`.
- It drops redundant equality rows, and reports inconsistent ones as infeasible.
- It can warm start from a previous point and working set.

It is dense, so it suits small and medium problems.

## Development

```bash
make install   # set up the environment
make test      # pytest + coverage
make fmt       # pre-commit hooks (ruff format + lint, ...)
make all       # every gate, as CI runs it
```
