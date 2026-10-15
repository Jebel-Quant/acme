"""User-facing validation errors: each raises the documented type with its message."""

import numpy as np
import pytest
import scipy.sparse as sp

import akme as ak
from akme.expressions.expression import LinMapExpr, broadcast_to


def test_expression_errors():
    """Division, products, matmul, linear maps and broadcasting reject invalid operands."""
    x, y = ak.Variable(3), ak.Variable(3)
    with pytest.raises(ak.DCPError, match="can only divide by a constant"):
        x / y
    with pytest.raises(ak.DCPError, match="cannot multiply two non-constant expressions"):
        ak.abs(x) * ak.abs(y)
    with pytest.raises(ValueError, match="matmul does not accept scalars; use \\* instead"):
        x @ 2.0
    with pytest.raises(ValueError, match="linear map of shape"):
        LinMapExpr(ak.abs(x), sp.csr_array(np.ones((2, 2))), (2,))
    with pytest.raises(ValueError, match="cannot broadcast shape"):
        broadcast_to(x, (4,))


def test_atom_errors():
    """Atoms reject parameters and shapes outside their domain."""
    x, m = ak.Variable(3), ak.Variable((2, 3))
    with pytest.raises(ak.DCPError, match="norm requires p >= 1"):
        ak.norm(x, 0.5)
    with pytest.raises(ValueError, match="quad_form: matrix of shape"):
        ak.quad_form(x, np.eye(2))
    with pytest.raises(ValueError, match="axis sums are only supported for matrices"):
        ak.sum(x, axis=0)
    with pytest.raises(ValueError, match="trace requires a square matrix"):
        ak.trace(m)
    with pytest.raises(ValueError, match="diag requires a square matrix or a vector"):
        ak.diag(m)
    with pytest.raises(ValueError, match="lambda_max requires a square matrix"):
        ak.lambda_max(m)


def test_constraint_errors():
    """Cone constraints validate their arguments."""
    x = ak.Variable(3)
    with pytest.raises(ValueError, match="PSD constraints require a square matrix expression"):
        ak.PSD(ak.Variable((2, 3)))
    with pytest.raises(ValueError, match="SOC: t must be a scalar"):
        ak.SOC(x, x)
    with pytest.raises(ValueError, match=r"alpha must lie in \(0, 1\)"):
        ak.PowCone3D(x[0], x[1], x[2], 1.5)


def test_variable_errors():
    """Variables are at most two-dimensional."""
    with pytest.raises(ValueError, match="akme supports at most 2-dimensional variables"):
        ak.Variable((2, 2, 2))
