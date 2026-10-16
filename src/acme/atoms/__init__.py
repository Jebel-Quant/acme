"""Atoms (nonlinear functions) and affine helper functions.

Every nonlinear atom takes *affine* arguments only. This keeps canonicalization flat:
each atom is replaced by an epigraph (convex) or hypograph (concave) variable plus a few
cone constraints, and the light DCP check only has to track signs of linear maps.

The atoms are grouped by family into submodules; everything public is re-exported here.
"""

from acme.atoms.affine import diag, hstack, reshape, sum, trace, vec, vstack  # noqa: A004
from acme.atoms.exponential import entr, exp, log, log_sum_exp
from acme.atoms.norms import norm, norm1, norm2, norm_inf, pnorm
from acme.atoms.piecewise import abs, max, maximum, min, minimum, pos  # noqa: A004
from acme.atoms.powers import inv_pos, power, sqrt
from acme.atoms.quadratic import quad_form, square, sum_squares
from acme.atoms.spectral import lambda_max, lambda_min
from acme.expressions._ops import ops
from acme.expressions.expression import multiply

__all__ = [
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
    "sqrt",
    "square",
    "sum",
    "sum_squares",
    "trace",
    "vec",
    "vstack",
]

ops.sum = sum
ops.power = power
