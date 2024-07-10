# Python data-parallel native extension

Here we build Python extension for a QR decomposition using `oneMKL` LAPACK routines.

## Building

Please refer to [building.md](./building.md) for details on how to build the extension.


## Running

Using ``SYCL_CACHE_PERSISTENT=1`` allows DPC++ to save results of compiling intermediate SPIR language for the specific device installed on your
machine and avoid recompiling it from run to run.

```bash
$ SYCL_CACHE_PERSISTENT=1 python
run.py
Using device Intel(R) Graphics [0x9a49]
================================================= test session starts ==================================================
collected 8 items

tests/test_qr.py .s.s.s.s                                                                                        [100%]

============================================= 4 passed, 4 skipped in 1.11s =============================================
QR decomposition for matrix of size = (3000, 3000)
Result agreed.
qr took 0.689391479943879 seconds
np.linalg.qr took 2.144680592115037 seconds
```

The tests can be found in the [/tests/test_qr.py](./tests/test_qr.py) file.

Do note that some tests may be skipped as some devices may not support double-precision (as above).

This sample run was obtained on a laptop with 11th Gen Intel(R) Core(TM) i7-1185G7 CPU @ 3.00GHz, 16 GB of RAM, and the integrated Intel(R) Iris(R) Xe GPU, with stock NumPy 1.26.4, and development build of dpctl 0.18 built with oneAPI DPC++ 2024.2.0.

Numpy benchmark results may also differ in the presence of a LAPACK library. See the [Numpy documentation](https://numpy.org/doc/stable/reference/routines.linalg.html) on the `np.linalg` submodule for more details.
