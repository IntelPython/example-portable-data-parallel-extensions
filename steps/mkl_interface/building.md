# oneMKL Interface

[oneMKL Interfaces](https://github.com/oneapi-src/oneMKL) is an open-source implementation of the oneMKL Data Parallel C++ (DPC++) interface according to the [oneMKL specification](https://spec.oneapi.com/versions/latest/elements/oneMKL/source/index.html). It works with multiple devices (backends) using device-specific libraries underneath.

oneMKL is part of the [UXL Foundation](http://www.uxlfoundation.org/).

## Getting the library

The library can be cloned from GitHub:

```bash
git clone https://github.com/oneapi-src/oneMKL.git
```

but it is also included in this repo as a git submodule, in "./oneMKL" subfolder.

## Building 

oneMKL interface library allows for many ways to be built, please see [building oneMKL interface](https://oneapi-src.github.io/oneMKL/building_the_project_with_dpcpp.html) for full details.

Note that building oneMKL interfaces requires cmake, which may also be required to build the extension or dpctl (if targeting NVidia(R)). As such, it may be a good idea to install requirements for the chosen build path ([`scikit-build-core`](#building-c-extension-using-scikit-build-core) or [`meson-python`](#building-c-extension-using-meson-python)) ahead of time.

In order to target Intel(R) GPUs, CPUs, and NVidia(R) GPUs for BLAS, LAPACK, FFT, and RNG domains configure it as follows:

```bash
# activate oneAPI
source /opt/intel/oneapi/setvars.sh
```
```bash
mkdir build
cd build
# configure project
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DMKL_ROOT=${MKLROOT} -DBUILD_FUNCTIONAL_TESTS=OFF              \
    -DENABLE_CUBLAS_BACKEND=True  -DENABLE_CUSOLVER_BACKEND=True -DENABLE_CURAND_BACKEND=True -DENABLE_CUFFT_BACKEND=True
```

When building for ROCm, in addition to specifying `-DENABLE_ROCBLAS_BACKEND=True`, `-DENABLE_ROCRAND_BACKEND=True`, `-DENABLE_ROCFFT_BACKEND=True`,
and `-DENABLE_ROCSOLVER_BACKEND=True` as appropriate, one must also specify the targeted device architecture, e.g., `-DHIP_TARGETS=gfx908`. If only targeting an Intel devices or CPUs, just

```bash
cmake .. -DCMAKE_CXX_COMPILER=icpx -DCMAKE_C_COMPILER=icx -DMKL_ROOT=${MKLROOT} -DBUILD_FUNCTIONAL_TESTS=OFF
```

should work.

```bash
# build
cmake --build .
```

```bash
# install
export MKL_INTERFACE_ROOT=$(pwd)/../install
cmake --install . --prefix=${MKL_INTERFACE_ROOT}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MKL_INTERFACE_ROOT}/lib
cd ..
```

## Building a C++ example that uses the library

A set of examples is included with the library. Let's build an LAPACK example of
solving general triangular linear system. We use `-fsycl-targets`
to specify targets to generate offload sections for, and link to the
library dynamically using `-lonemkl`:

```bash
cd examples/lapack/run_time_dispatching/
icpx -fsycl getrs_usm.cpp                                     \
    -fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown \
    -I${MKL_INTERFACE_ROOT}/include                           \
    -I${MKL_INTERFACE_ROOT}/../examples/include               \
    -L${MKL_INTERFACE_ROOT}/lib -lonemkl                      \
    -Wl,-rpath,${MKL_INTERFACE_ROOT}/lib -o run
```

The ELF executable does contain two offload sections as expected:

```
$ readelf -St run | grep CLANG_OFFLOAD
  [17] __CLANG_OFFLOAD_BUNDLE__sycl-nvptx64
  [19] __CLANG_OFFLOAD_BUNDLE__sycl-spir64
```

## Running on different devices

The default selected device is Intel GPU:

```bash
$ ./run | grep Device
# Device will be selected during runtime.
Device name is: Intel(R) UHD Graphics 770
```

Using ``ONEAPI_DEVICE_SELECTOR`` we can influence the choice of the default-selected device:

```bash
$ ONEAPI_DEVICE_SELECTOR=opencl:gpu ./run | grep Device
# Device will be selected during runtime.
Device name is: Intel(R) UHD Graphics 770
```

```bash
$ ONEAPI_DEVICE_SELECTOR=cuda:gpu ./run | grep Device
# Device will be selected during runtime.
Device name is: NVIDIA GeForce GT 1030
```

```bash
$ ONEAPI_DEVICE_SELECTOR=opencl:cpu ./run | grep Device
# Device will be selected during runtime.
Device name is: 12th Gen Intel(R) Core(TM) i9-12900
```

### Build dpctl for NVidia or AMD

`dpctl` is not currently distributed pre-built with NVidia(R) or AMD support. As such, if a user intends on targeting these devices, we will have to build it from source.

First clone from Github:

```bash
git clone https://github.com/IntelPython/dpctl.git
```

Make sure the necessary requirements are installed for your build path ([`scikit-build-core`](#building-c-extension-using-scikit-build-core) or [`meson-python`](#building-c-extension-using-meson-python) for each build path and installing the requirements). The appropriate requirements file will be named `requirements_build.txt` if building to target non-Intel devices, while the environment file will be `environment_build.yml`.

Now build

```bash
python scripts/build_locally.py --verbose --cmake-opts="-DDPCTL_TARGET_CUDA=ON"
```

See the [`dpctl` documentation](https://intelpython.github.io/dpctl/latest/beginners_guides/installation.html#building-for-custom-sycl-targets) for more information.


### Building C++ extension manually

We chose to demonstrate use of oneMKL interface library from Python by building 
pybind11 extension that computes QR decomposition for a stack of real 
floating-point matrices.

Since oneMKL LAPACK functions [require](https://spec.oneapi.io/versions/latest/elements/oneMKL/source/domains/lapack/lapack.html)
column-major layout and modify input arrays in place, the extension must copy the input stack of matrices accordingly.

If requirements are not installed, install requirements.

For a `conda` / `mamba` env:
```bash
mamba env create -f environment.yml
```

For system Python or a virtual environment:
```bash
pip install -r requirements.txt
```

Now build and install

```bash
icpx -fsycl -fPIC -shared -fno-approx-func -fno-fast-math                 \
    $(python -m pybind11 --includes) -I$MKL_INTERFACE_ROOT/include        \
    $(python -m dpctl --includes) $(python -m dpctl --tensor-includes)    \
    -L$MKL_INTERFACE_ROOT/lib -lonemkl -Wl,-rpath,$MKL_INTERFACE_ROOT/lib \
    src/py.cpp -o mkl_interface_ext/_qr.so
```

Tests of the extension can be executed as follows:

```bash
# work-around a problem with oneMKL library linkage
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${MKL_INTERFACE_ROOT}/lib
PYTHONPATH=$PYTHONPATH:$(pwd)/mkl_interface_ext pytest tests
```

### Building C++ extension using `scikit-build-core`

Make sure to delete the `mkl_interface_ext/_qr.so` from previous step.

If requirements are not installed, install requirements.

For a `conda` / `mamba` env:
```bash
mamba env create -f scikit_build_core_mkl_ext/environment.yml
```

For system Python or a virtual environment:
```bash
pip install -r scikit_build_core_mkl_ext/requirements.txt
```

Now build and install

```bash
# help cmake to find oneMKL library
export ONEMKL_ROOT=$MKL_INTERFACE_ROOT
# help cmake to find dpctl package if built in-place
export Dpctl_ROOT=$(python -m dpctl --cmakedir)
export VERBOSE=1
CXX=icpx pip install -e scikit_build_core_mkl_ext/ --no-deps --no-build-isolation --verbose
```

Now, set `LD_LIBRARY_PATH` as in the previous step, if not done yet, and run

```bash
pytest tests
```

If building for NVidia GPUs, add `-Ccmake.args="-DTARGET_CUDA=ON"`:

```bash
CXX=icpx pip install -e scikit_build_core_mkl_ext/ --no-deps --no-build-isolation --verbose \
    -Ccmake.args="-DTARGET_CUDA=ON"
```

### Building C++ extension using `meson-python`

Make sure to delete the `mkl_interface_ext/_qr.so` if manual building was performed.

If requirements are not installed, install requirements.

For a `conda` / `mamba` env:
```bash
mamba env create -f meson_mkl_ext/environment.yml
```

For system Python or a virtual environment:
```bash
pip install -r meson_mkl_ext/requirements.txt
```

Now build and install

```bash
export VERBOSE=1
CXX=icpx pip install -e meson_mkl_ext/ --no-deps --no-build-isolation --verbose
```

Now, set `LD_LIBRARY_PATH` as in the previous step, if not done yet, and run

```bash
pytest tests
```

If building for NVidia GPUs, add `-Csetup-args="-Dtarget-cuda=true"`:

``bash
CXX=icpx pip install -e meson_mkl_ext/ --no-deps --no-build-isolation --verbose \
    -Csetup-args="-Dtarget-cuda=true"
```
