"""Expression algebra, shapes, values and curvature."""

import numpy as np
import pytest

import akme as ak
from akme.expressions.affine import AffineExpr


@pytest.fixture
def rng():
    """Seeded random generator."""
    return np.random.default_rng(1)


def _set(*pairs):
    for var, value in pairs:
        var.value = value


def test_affine_ops_match_numpy(rng):
    """Affine ops match numpy."""
    x = ak.Variable((3, 2))
    y = ak.Variable(2)
    a = rng.standard_normal((4, 3))
    b = rng.standard_normal(2)
    xv, yv = rng.standard_normal((3, 2)), rng.standard_normal(2)
    _set((x, xv), (y, yv))

    cases = {
        "matmul_left": (a @ x, a @ xv),
        "matmul_right": (x @ b, xv @ b),
        "vec_dot": (b @ y, b @ yv),
        "add_sub": (x[0] + 2 * y - 1, xv[0] + 2 * yv - 1),
        "transpose": (x.T, xv.T),
        "slice": (x[1:, 1], xv[1:, 1]),
        "fancy": (x[[0, 2], :], xv[[0, 2], :]),
        "div": (y / 4, yv / 4),
        "neg": (-x, -xv),
        "multiply": (ak.multiply(b, y), b * yv),
        "sum": (ak.sum(x), xv.sum()),
        "sum0": (ak.sum(x, axis=0), xv.sum(axis=0)),
        "sum1": (x.sum(axis=1), xv.sum(axis=1)),
        "reshape": (ak.reshape(x, (2, 3)), xv.reshape((2, 3), order="F")),
        "vec": (ak.vec(x), xv.flatten(order="F")),
        "hstack": (ak.hstack([y, y]), np.hstack([yv, yv])),
        "vstack": (ak.vstack([x, y.reshape((1, 2))]), np.vstack([xv, yv.reshape(1, 2)])),
        "scalar_broadcast": (x + ak.sum(y), xv + yv.sum()),
        "rsub": (1 - y, 1 - yv),
    }
    for name, (expr, expected) in cases.items():
        assert isinstance(expr, AffineExpr), name
        assert expr.shape == np.shape(expected), name
        np.testing.assert_allclose(expr.value, expected, err_msg=name)


def test_matrix_helpers(rng):
    """Matrix helpers."""
    m = ak.Variable((3, 3))
    mv = rng.standard_normal((3, 3))
    m.value = mv
    np.testing.assert_allclose(ak.trace(m).value, np.trace(mv))
    np.testing.assert_allclose(ak.diag(m).value, np.diag(mv))
    v = ak.Variable(3)
    v.value = [1.0, 2.0, 3.0]
    np.testing.assert_allclose(ak.diag(v).value, np.diag([1.0, 2.0, 3.0]))


def test_atom_values(rng):
    """Atom values."""
    x = ak.Variable(4)
    xv = rng.uniform(0.1, 2.0, 4)
    x.value = xv
    s = np.diag([1.0, 2.0, 3.0, 4.0])
    cases = {
        "abs": (ak.abs(x - 1), np.abs(xv - 1)),
        "pos": (ak.pos(x - 1), np.maximum(xv - 1, 0)),
        "maximum": (ak.maximum(x, 1), np.maximum(xv, 1)),
        "minimum": (ak.minimum(x, 1), np.minimum(xv, 1)),
        "max": (ak.max(x), xv.max()),
        "min": (ak.min(x), xv.min()),
        "norm1": (ak.norm1(x), np.abs(xv).sum()),
        "norm2": (ak.norm2(x), np.linalg.norm(xv)),
        "norm_inf": (ak.norm_inf(x), np.abs(xv).max()),
        "pnorm3": (ak.pnorm(x, 3), np.linalg.norm(xv, 3)),
        "sum_squares": (ak.sum_squares(x), xv @ xv),
        "square": (ak.square(x), xv**2),
        "pow": (x**2, xv**2),
        "quad_form": (ak.quad_form(x, s), xv @ s @ xv),
        "exp": (ak.exp(x), np.exp(xv)),
        "log": (ak.log(x), np.log(xv)),
        "entr": (ak.entr(x), -xv * np.log(xv)),
        "lse": (ak.log_sum_exp(x), np.log(np.exp(xv).sum())),
        "power1.5": (ak.power(x, 1.5), xv**1.5),
        "sqrt": (ak.sqrt(x), np.sqrt(xv)),
        "inv_pos": (ak.inv_pos(x), 1 / xv),
        "lin_combo": (2 * ak.sum(ak.exp(x)) - ak.log(x[0]), 2 * np.exp(xv).sum() - np.log(xv[0])),
    }
    for name, (expr, expected) in cases.items():
        np.testing.assert_allclose(expr.value, expected, err_msg=name)


def test_value_none_without_values():
    """Value none without values."""
    x = ak.Variable(2)
    assert x.value is None
    assert (2 * x).value is None
    assert ak.exp(x).value is None


def test_curvature():
    """Curvature."""
    x = ak.Variable(3)
    assert ak.Constant(1.0).curvature is ak.Curvature.CONSTANT
    assert (2 * x).curvature is ak.Curvature.AFFINE
    assert ak.sum(ak.exp(x)).curvature is ak.Curvature.CONVEX
    assert (-ak.sum(ak.exp(x))).curvature is ak.Curvature.CONCAVE
    assert (ak.sum(ak.log(x)) - ak.sum_squares(x)).curvature is ak.Curvature.CONCAVE
    assert (ak.sum(ak.exp(x)) + ak.sum(ak.log(x))).curvature is ak.Curvature.UNKNOWN
    assert (np.array([1.0, -1.0, 1.0]) @ ak.exp(x)).curvature is ak.Curvature.UNKNOWN
    assert ak.quad_form(x, -np.eye(3)).curvature is ak.Curvature.CONCAVE


def test_dcp_errors():
    """Dcp errors."""
    x = ak.Variable(3)
    with pytest.raises(ak.DCPError):
        ak.exp(ak.abs(x))
    with pytest.raises(ak.DCPError):
        x @ x
    with pytest.raises(ak.DCPError):
        ak.quad_form(x, np.diag([1.0, -1.0, 1.0]))
    with pytest.raises(ak.DCPError):
        ak.Problem(ak.Maximize(ak.sum(ak.exp(x)))).solve()
    with pytest.raises(ak.DCPError):
        ak.Problem(ak.Minimize(0), [ak.norm2(x) >= 1]).solve()
    with pytest.raises(ak.DCPError):
        ak.Problem(ak.Minimize(0), [ak.norm2(x) == 1]).solve()


def test_shape_errors():
    """Shape errors."""
    x = ak.Variable(3)
    with pytest.raises(ValueError, match="incompatible shapes"):
        x + np.ones(2)
    with pytest.raises(ValueError, match="cannot reshape"):
        x.reshape((2, 2))
    with pytest.raises(ValueError, match="scalar"):
        ak.Minimize(x)
    with pytest.raises(ValueError, match="square"):
        ak.Variable((2, 3), PSD=True)


def test_constraint_construction():
    """Constraint construction."""
    x = ak.Variable(2)
    assert isinstance(x == 1, ak.Equality)
    assert isinstance(x <= 1, ak.Inequality)
    assert isinstance(np.ones(2) >= x, ak.Inequality)
    m = ak.Variable((2, 2))
    assert isinstance(m >> 0, ak.PSD)
    assert isinstance(m << np.eye(2), ak.PSD)
