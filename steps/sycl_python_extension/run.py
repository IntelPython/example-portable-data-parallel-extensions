import kde_sycl_ext as kse
import dpctl
import dpctl.tensor as dpt
import numpy as np
import timeit

print(f"Using device {dpctl.select_default_device().name}")

dt = dpt.float32
rng = np.random.default_rng()

n_sample = 1_000_000
n_dim = 7
n_est = 17
h = 0.05
print(f"KDE for n_sample = {n_sample}, n_est = {n_est}, n_dim = {n_dim}, h = {h}")

poi_np = rng.uniform(0.1, 0.9, size=(n_est, n_dim)).astype(dt, copy=False)
us_np = rng.uniform(0, 1, size=(n_sample, n_dim)).astype(dt, copy=False)

poi = dpt.asarray(poi_np)
us = dpt.asarray(us_np)

t0 = timeit.default_timer()

f1 = kse.kde_dpctl(poi, us, 0.05)
f1.sycl_queue.wait()

t1 = timeit.default_timer()

# use of atomics and reduction over work-group
f2 = kse.kde_ext(poi, us, h, mode=0)
f2.sycl_queue.wait()

t2 = timeit.default_timer()

# use of atomics
f3 = kse.kde_ext(poi, us, h, mode=1)
f3.sycl_queue.wait()

t3 = timeit.default_timer()

# tree reduction
f4 = kse.kde_ext(poi, us, h, mode=2)
f4.sycl_queue.wait()

t4 = timeit.default_timer()

f5 = kse.kde_numpy(poi_np, us_np, h)

t5 = timeit.default_timer()

assert dpt.allclose(f1, f2)
assert dpt.allclose(f1, f3)
assert dpt.allclose(f1, f4)
assert dpt.allclose(f1, dpt.asarray(f5))

print("Result agreed.")
print(f"kde_dpctl took {t1-t0} seconds")
print(f"kde_ext[mode=0] {t2-t1} seconds")
print(f"kde_ext[mode=1] {t3-t2} seconds")
print(f"kde_ext[mode=2] {t4-t3} seconds")
print(f"kde_numpy {t5-t4} seconds")
