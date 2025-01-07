# Building instruction

## Building with CMake

1. Make sure `cmake` of version 3.27 or later is available. The easiest is to install it into conda environment.
2. Make sure oneAPI is activated and compiler is available:
```bash
$ type -P icpx || source /opt/intel/oneapi/setvars.sh
```
3. Configure project
```bash
$ CXX=icpx cmake . -B cmake_build_dir -DCMAKE_INSTALL_PREFIX=cmake_install_dir
```

Use `-DTARGET_CUDA=ON` to build multi-target binary for CUDA.

For HIP, use `-DTARGET_HIP=<ARCH>` where `<ARCH>` is the architecture of the AMD GPU.

To find the architecture, use
```bash
rocminfo | grep 'Name: *gfx.*'
```

which should show something like
```bash
  Name:                    gfx1030
```
where `gfx` followed by four digits the GPU's the architecture.

4. Build and install
The C++ code uses [argparse](https://github.com/p-ranav/argparse) project to support CLI options. 
Make sure to run `git submodule update --init` to retrieve its sources.

```bash
$ cmake --build cmake_build_dir --target install
```
5. Run the project
```bash
$ ./cmake_install_dir/kde_app
```

## Building with Meson

1. Make sure `meson` of version 1.4 or later is available. The easiest is to install it into conda environment.
2. Make sure oneAPI is activated and compiler is available:
```bash
$ type -P icpx || source /opt/intel/oneapi/setvars.sh
```
3. Configure project
```bash
$ CXX=icpx meson setup meson_build_dir
```

Use `-Dtarget-cuda=true` to build multi-target binary for CUDA.

For HIP, use `-DTARGET_HIP=<ARCH>` where `<ARCH>` is the architecture of the AMD GPU.

To find the architecture, use
```bash
rocminfo | grep 'Name: *gfx.*'
```

which should show something like
```bash
  Name:                    gfx1030
```
where `gfx` followed by four digits the GPU's the architecture.

4. Build and install
```bash
$ cd meson_build_dir
$ meson compile
```
5. Run the project
```bash
$ ./kde_app
```
