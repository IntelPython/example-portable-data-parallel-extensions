import os

import mkl_interface_ext as mi
import dpctl
import dpctl.tensor as dpt
import numpy as np
import pytest
import timeit

print(f"Using device {dpctl.select_default_device().name}")

# check that extension produces correct results
pytest.main(["--no-header", os.path.abspath(os.path.dirname(__file__)) + "/tests/"])

# now benchmark using a single large matrix
dt = dpt.float32
tol = 12

n = 3000

print(f"QR decomposition for matrix of size = ({n}, {n})")

x_np = np.eye(n, dtype=dt)
x = dpt.asarray(x_np)

t0 = timeit.default_timer()

q, r = mi.qr(x)

t1 = timeit.default_timer()

assert q.shape == (n, n,)
assert r.shape == x.shape

res1 = dpt.max(dpt.abs(q.mT @ q - dpt.eye(n, dtype=dt)[dpt.newaxis, ...]))
res2 = dpt.max(dpt.abs(q @ r - x))

assert res1 < tol * dpt.finfo(dt).eps
assert res2 < (tol + dpt.max(dpt.abs(x))) * dpt.finfo(dt).eps

# benchmark with Numpy

t2 = timeit.default_timer()

q_np, r_np = np.linalg.qr(x_np)

t3 = timeit.default_timer()

assert dpt.allclose(q, dpt.asarray(q_np))
assert dpt.allclose(r, dpt.asarray(r_np))

print("Result agreed.")
print(f"qr took {t1-t0} seconds")
print(f"np.linalg.qr took {t3-t2} seconds")
