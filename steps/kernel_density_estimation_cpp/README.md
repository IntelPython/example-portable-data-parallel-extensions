# Kernel Density Estimation C++ SYCL application example

## Building

Please refer to [building.md](./building.md) for details on how to build the application using either `CMake` or `Meson`.

## Running

```bash
(dev_dpctl) vm:~/scipy_2024/steps/kernel_density_estimation_cpp/meson_build_dir$ ./kde_app --help
Device: Intel(R) Graphics [0x9a49][1.3.29138]
Usage: kde_app [--help] [--version] [--n_sample VAR] [--dimension VAR] [--points VAR] [--seed VAR] [--smoothing_scale VAR] [--algorithm VAR]

Optional arguments:
  -h, --help         shows help message and exits
  -v, --version      prints version information and exits
  -n, --n_sample     Number of samples from underlying cuboid distribution [nargs=0..1] [default: 1000000]
  -d, --dimension    Dimensionality of samples [nargs=0..1] [default: 4]
  -m, --points       Number of points at which to estimate distribution value [nargs=0..1] [default: 25]
  --seed             Random seed to use for reproducibility [nargs=0..1] [default: 18446744073709551615]
  --smoothing_scale  Kernel density estimation smoothing scale parameter [nargs=0..1] [default: 0.05]
  --algorithm        Kernel implementation to use. Supported choices are [temps, atomic_ref, work_group_reduce_and_atomic_ref] [nargs=0..1] [default: "work_group_reduce_and_atomic_ref"]
```

By default, different set of random inputs are generated. Use `"--seed"` option to compare output of different kernel implementations. For example,

```
(dev_dpctl) vm:~/scipy_2024/steps/kernel_density_estimation_cpp/meson_build_dir$ ./kde_app -m 6 --seed 555 --algorithm temps
Device: Intel(R) Graphics [0x9a49][1.3.29138]
KDE estimation, n_sample: 1000000, dim = 4, n_est = 6
Samples are from 4-dimensional uniform distribution
KDE smoothing parameter: 0.05
Using kernel implementation 'temps'
Estimated density: 0.982785 0.991264 0.977207 0.99606 0.9784 1.01235
```
```
(dev_dpctl) vm:~/scipy_2024/steps/kernel_density_estimation_cpp/meson_build_dir$ ./kde_app -m 6 --seed 555 --algorithm atomic_ref
Device: Intel(R) Graphics [0x9a49][1.3.29138]
KDE estimation, n_sample: 1000000, dim = 4, n_est = 6
Samples are from 4-dimensional uniform distribution
KDE smoothing parameter: 0.05
Using kernel implementation 'atomic_ref'
Estimated density: 0.982785 0.991265 0.977207 0.996059 0.9784 1.01235
```
```
(dev_dpctl) vm:~/scipy_2024/steps/kernel_density_estimation_cpp/meson_build_dir$ ./kde_app -m 6 --seed 555
Device: Intel(R) Graphics [0x9a49][1.3.29138]
KDE estimation, n_sample: 1000000, dim = 4, n_est = 6
Samples are from 4-dimensional uniform distribution
KDE smoothing parameter: 0.05
Using default kernel implementation 'work_group_reduce_and_atomic_ref'
Estimated density: 0.982785 0.991264 0.977206 0.99606 0.9784 1.01235
```