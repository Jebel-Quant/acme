"""akme -- cvxpy-style modelling of convex (quadratic) problems.

Polyhedral problems are solved with akme's own primal active-set method,
everything else (second-order, PSD, exponential and power cones) with Clarabel.

Example:
    >>> import numpy as np
    >>> import akme as ak
    >>> x = ak.Variable(2)
    >>> prob = ak.Problem(ak.Minimize(ak.sum_squares(x - np.array([1.0, 2.0]))), [x <= 1])
    >>> round(prob.solve(), 6)
    1.0
    >>> prob.solver_stats.solver_name
    'ACTIVE_SET'
"""

from akme.atoms import (
    abs,  # noqa: A004 - cvxpy naming
    diag,
    entr,
    exp,
    hstack,
    inv_pos,
    lambda_max,
    lambda_min,
    log,
    log_sum_exp,
    max,  # noqa: A004 - cvxpy naming
    maximum,
    min,  # noqa: A004 - cvxpy naming
    minimum,
    multiply,
    norm,
    norm1,
    norm2,
    norm_inf,
    pnorm,
    pos,
    power,
    quad_form,
    reshape,
    sqrt,
    square,
    sum,  # noqa: A004 - cvxpy naming
    sum_squares,
    trace,
    vec,
    vstack,
)
from akme.constraints import PSD, SOC, Constraint, Equality, ExpCone, Inequality, PowCone3D
from akme.expressions import Constant, Curvature, DCPError, Expression, Variable
from akme.problem import Maximize, Minimize, Problem, SolverStats
from akme.solvers.active_set import solve_qp
from akme.status import (
    ACTIVE_SET,
    CLARABEL,
    INFEASIBLE,
    INFEASIBLE_INACCURATE,
    OPTIMAL,
    OPTIMAL_INACCURATE,
    SOLVER_ERROR,
    UNBOUNDED,
    UNBOUNDED_INACCURATE,
    USER_LIMIT,
)

__all__ = [
    "ACTIVE_SET",
    "CLARABEL",
    "INFEASIBLE",
    "INFEASIBLE_INACCURATE",
    "OPTIMAL",
    "OPTIMAL_INACCURATE",
    "PSD",
    "SOC",
    "SOLVER_ERROR",
    "UNBOUNDED",
    "UNBOUNDED_INACCURATE",
    "USER_LIMIT",
    "Constant",
    "Constraint",
    "Curvature",
    "DCPError",
    "Equality",
    "ExpCone",
    "Expression",
    "Inequality",
    "Maximize",
    "Minimize",
    "PowCone3D",
    "Problem",
    "SolverStats",
    "Variable",
    "abs",
    "diag",
    "entr",
    "exp",
    "hstack",
    "inv_pos",
    "lambda_max",
    "lambda_min",
    "log",
    "log_sum_exp",
    "max",
    "maximum",
    "min",
    "minimum",
    "multiply",
    "norm",
    "norm1",
    "norm2",
    "norm_inf",
    "pnorm",
    "pos",
    "power",
    "quad_form",
    "reshape",
    "solve_qp",
    "sqrt",
    "square",
    "sum",
    "sum_squares",
    "trace",
    "vec",
    "vstack",
]
