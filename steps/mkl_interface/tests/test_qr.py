import dpctl.tensor as dpt
import numpy as np
import mkl_interface_ext as mi

import pytest

tol_mult = 12


@pytest.fixture(params=[dpt.float32, dpt.float64])
def dt(request):
    return request.param


def skip_unsupported_dt(dt):
    try:
        _ = dpt.empty(tuple(), dtype=dt)
    except ValueError:
        pytest.skip(
            f"Default device does not support dtype={dt}"
        )


def test_square(dt):
    skip_unsupported_dt(dt)

    b, n = 10, 4

    x_np = np.random.randn(b, n, n).astype(dt)
    x = dpt.asarray(x_np, dtype=dt)

    q, r = mi.qr(x)

    assert q.shape == (b, n, n,)
    assert r.shape == x.shape

    res1 = dpt.max(dpt.abs(q.mT @ q - dpt.eye(n, dtype=dt)[dpt.newaxis, ...]))
    res2 = dpt.max(dpt.abs(q @ r - x))

    assert res1 < tol_mult * dpt.finfo(dt).eps
    assert res2 < (tol_mult + dpt.max(dpt.abs(x))) * dpt.finfo(dt).eps


def test_tall(dt):
    skip_unsupported_dt(dt)

    b, n = 10, 4
    m = 2 * n

    x_np = np.random.randn(b, m, n).astype(dt)
    x = dpt.asarray(x_np, dtype=dt)

    q, r = mi.qr(x)

    assert q.shape == (b, m, m,)
    assert r.shape == x.shape

    res1 = dpt.max(dpt.abs(q.mT @ q - dpt.eye(m, dtype=dt)[dpt.newaxis, ...]))
    res2 = dpt.max(dpt.abs(q @ r - x))

    assert res1 < tol_mult * dpt.finfo(dt).eps
    assert res2 < (tol_mult + dpt.max(dpt.abs(x))) * dpt.finfo(dt).eps


def test_short(dt):
    skip_unsupported_dt(dt)

    b, n = 10, 4
    m = n - 1

    x_np = np.random.randn(b, m, n).astype(dt)
    x = dpt.asarray(x_np, dtype=dt)

    q, r = mi.qr(x)

    assert q.shape == (b, m, m,)
    assert r.shape == x.shape

    res1 = dpt.max(dpt.abs(q.mT @ q - dpt.eye(m, dtype=dt)[dpt.newaxis, ...]))
    res2 = dpt.max(dpt.abs(q @ r - x))

    assert res1 < tol_mult * dpt.finfo(dt).eps
    assert res2 < (tol_mult + dpt.max(dpt.abs(x))) * dpt.finfo(dt).eps


def test_single_matrix(dt):
    skip_unsupported_dt(dt)

    n = 4
    x = dpt.eye(n, dtype=dt)

    q, r = mi.qr(x)

    assert q.shape == (n, n,)
    assert r.shape == x.shape

    res1 = dpt.max(dpt.abs(q.mT @ q - dpt.eye(n, dtype=dt)[dpt.newaxis, ...]))
    res2 = dpt.max(dpt.abs(q @ r - x))

    assert res1 < tol_mult * dpt.finfo(dt).eps
    assert res2 < (tol_mult + dpt.max(dpt.abs(x))) * dpt.finfo(dt).eps
