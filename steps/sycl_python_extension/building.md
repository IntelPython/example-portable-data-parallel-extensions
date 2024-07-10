# Building Python extension

## Build dpctl for NVidia or AMD

`dpctl` is not currently distributed pre-built with NVidia(R) or AMD support. As such, if a user intends on targeting these devices, we will have to build it from source.

First clone from Github:

```bash
git clone https://github.com/IntelPython/dpctl.git
```

Make sure the necessary requirements are installed for your build path ([`scikit-build-core`](#build-and-install-extension-with-scikit-build-core) or [`meson-python`](#build-and-install-extension-with-meson-python) for each build path and installing the requirements). The appropriate requirements file will be named `requirements_build.txt` if building to target non-Intel devices, while the environment file will be `environment_build.yml`.

Now build

```bash
python scripts/build_locally.py --verbose --cmake-opts="-DDPCTL_TARGET_CUDA=ON"
```

See the [`dpctl` documentation](https://intelpython.github.io/dpctl/latest/beginners_guides/installation.html#building-for-custom-sycl-targets) for more information.

## Building extension manually

If requirements are not installed, install requirements. This is made simple with `environment.yml` files for `conda` or `mamba` environments and `requirements.txt` files for system Python or virtual environments.

For a `conda` / `mamba` env:
```bash
mamba env create -f environment.yml
mamba activate kde-ext
```

For system Python or a virtual environment:
```bash
pip install -r requirements.txt
```

Make sure oneAPI is activated and compiler is available:
```bash
$ type -P icpx || source /opt/intel/oneapi/setvars.sh
```

Now build

```bash
icpx -fsycl -fno-fast-math $(python -m pybind11 --includes) $(python -m dpctl --includes --tensor-includes) -I src/ src/py.cpp -fPIC --shared -o kde_sycl_ext/_kde_sycl_ext.so
```

## Build and install extension with scikit-build-core

If requirements are not installed, install requirements.

For a `conda` / `mamba` env:
```bash
mamba env create -f scikit_build_core_sycl_python_extension/environment.yml
mamba activate kde-ext
```

For system Python or a virtual environment:
```bash
pip install -r scikit_build_core_sycl_python_extension/requirements.txt
```

Make sure oneAPI is activated and compiler is available:
```bash
$ type -P icpx || source /opt/intel/oneapi/setvars.sh
```

Now build and install

```bash
VERBOSE=1 CXX=icpx Dpctl_ROOT=$(python -m dpctl --cmakedir) pip install -e \
    scikit_build_core_sycl_python_extension --no-deps --no-build-isolation --verbose
```

To build extension that can offload to NVidia(R) GPUs, we need to set ``TARGET_CUDA`` cmake boolean variable, which can be done 
using `cmake.args` [config-setting variable](https://github.com/scikit-build/scikit-build-core) of `scikit-build-core`:

```bash
VERBOSE=1 CXX=icpx Dpctl_ROOT=$(python -m dpctl --cmakedir) pip install -e \
    scikit_build_core_sycl_python_extension --no-deps --no-build-isolation --verbose \
    -Ccmake.args="-DTARGET_CUDA=ON"
```

## Build and install extension with meson-python

If requirements are not installed, install requirements.

For a `conda` / `mamba` env:
```bash
mamba env create -f meson_sycl_python_extension/environment.yml
mamba activate kde-ext
```

For system Python or a virtual environment:
```bash
pip install -r meson_sycl_python_extension/requirements.txt
```

Make sure oneAPI is activated and compiler is available:
```bash
$ type -P icpx || source /opt/intel/oneapi/setvars.sh
```

Now build and install

```bash
VERBOSE=1 CXX=icpx pip install -e meson_sycl_python_extension --no-deps --no-build-isolation --verbose
```

To build extension that can offload to NVidia(R) GPUs, we need to set ``target-cuda`` meson option, which can be done 
using `setup-args` [setup-args](https://meson-python.readthedocs.io/en/latest/how-to-guides/meson-args.html) of `meson-python`:

```bash
VERBOSE=1 CXX=icpx pip install -e meson_sycl_python_extension --no-deps --no-build-isolation --verbose \
    -Csetup-args="-Dtarget-cuda=true"
```
