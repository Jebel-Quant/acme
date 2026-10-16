"""Solver status strings (cvxpy compatible) and solver names."""

OPTIMAL = "optimal"
OPTIMAL_INACCURATE = "optimal_inaccurate"
INFEASIBLE = "infeasible"
INFEASIBLE_INACCURATE = "infeasible_inaccurate"
UNBOUNDED = "unbounded"
UNBOUNDED_INACCURATE = "unbounded_inaccurate"
USER_LIMIT = "user_limit"
SOLVER_ERROR = "solver_error"

#: Statuses for which a primal solution is available.
SOLUTION_PRESENT = (OPTIMAL, OPTIMAL_INACCURATE)

ACTIVE_SET = "ACTIVE_SET"
CLARABEL = "CLARABEL"
