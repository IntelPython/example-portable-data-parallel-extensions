# Python data-parallel native extension

Here we build Python extension to provide Python API to SYCL implementations of 
Kernel Density Estimation.

## Building

Please refer to [building.md](./building.md) for details on how to build the extension.


## Running

Using ``SYCL_CACHE_PERSISTENT=1`` allows DPC++ to save results of compiling intermediate SPIR language for the specific device installed on your
machine and avoid recompiling it from run to run.

```bash
$ SYCL_CACHE_PERSISTENT=1 python run.py
KDE for n_sample = 1000000, n_est = 17, n_dim = 7, h = 0.05
Result agreed.
kde_dpctl took 0.3404452269896865 seconds
kde_ext[mode=0] 0.02209925901843235 seconds
kde_ext[mode=1] 0.02560457994695753 seconds
kde_ext[mode=2] 0.02815118699800223 seconds
kde_numpy 0.7227164240321144 seconds
```

Mode number maps to implementation as follows:

- Mode 2: ``kernel_density_estimate_temps``, tree reduction with use temporary allocations
- Mode 1: ``kernel_density_estimate_atomic_ref``, use of atomic updates without use of temporaries
- Mode 0: ``kernel_density_estimate_work_group_reduce_and_atomic_ref``, use of atomic updates and combining values held by work-items of the same work-group to reduce contention of atomically updating the same memory address from multiple work-items

This sample run was obtained on a laptop with 11th Gen Intel(R) Core(TM) i7-1185G7 CPU @ 3.00GHz, 32 GB of RAM, and the integrated Intel(R) Iris(R) Xe GPU, with stock NumPy 1.26.4, and development build of dpctl 0.17 built with oneAPI DPC++ 2024.1.0.
