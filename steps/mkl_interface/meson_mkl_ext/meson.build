project('_qr', 'cpp',
  version : '1.0',
  license: 'Apache 2.0')

fs = import('fs')

sycl_feature_code = '''#include <iostream>
using namespace std;
int main(){
#if defined(SYCL_LANGUAGE_VERSION)
cout << "SYCL_LANGUAGE_VERSION="<<SYCL_LANGUAGE_VERSION<<endl;
#endif
return 0;}
'''

compiler = meson.get_compiler('cpp')
result = compiler.compiles(sycl_feature_code, name : 'test sycl support')

if not result
   error('Compiler does not support SYCL, consider using icpx')
endif

should_target_cuda = get_option('target-cuda')
targeted_hip_arch = get_option('target-hip')
if should_target_cuda
    sycl_targets = 'nvptx64-nvidia-cuda,spir64-unknown-unknown'
if targeted_hip_arch != ''
    if sycl_targets
        sycl_targets = 'amdgcn-amd-amdhsa,' + sycl_targets
    else
        sycl_targets = 'amdgcn-amd-amdhsa,spir64-unknown-unknown'

if sycl_targets != ''
    sycl_compile_opts = ['-fsycl', '-fsycl-targets=' + sycl_targets]
    sycl_link_opts = ['-fsycl', '-fsycl-targets=' + sycl_targets]
    if targeted_hip_arch != ''
        sycl_compile_opts = sycl_compile_opts + ['-Xsycl-target-backend=amdgcn-amd-amdhsa', '--offload-arch=' + targeted_hip_arch]
        sycl_link_opts = sycl_link_opts + ['-Xsycl-target-backend=amdgcn-amd-amdhsa', '--offload-arch=' + targeted_hip_arch]
else
    sycl_compile_opts = '-fsycl'
    sycl_link_opts = '-fsycl'
endif

compile_opts = ['-O3', '-fno-approx-func', '-fno-fast-math'] + sycl_compile_opts

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

pybind11_dep = dependency('pybind11', required: false)
if not pybind11_dep.found()
  pybind11_proj = subproject('pybind11')
  pybind11_dep = pybind11_proj.get_variable('pybind11_dep')
endif

mkl_dir = fs.parent(meson.current_source_dir()) + '/oneMKL'
mkl_lib_dir = mkl_dir + '/install/lib'
mkl_lib = compiler.find_library('onemkl', dirs: mkl_lib_dir)
mkl_incl_dir = mkl_dir + '/install/include'
mkl_incl = include_directories(mkl_incl_dir)

incdir_dpctl_incl = run_command(py,
  ['-m', 'dpctl', '--include-dir'],
  check : true
).stdout().strip()

incdir_dpctl_tensor_incl = fs.parent(incdir_dpctl_incl) + '/tensor/libtensor/include'

incdir_dpctl = include_directories(incdir_dpctl_incl, incdir_dpctl_tensor_incl)

incdir = include_directories('../src')

py.install_sources(
    ['../mkl_interface_ext/__init__.py', '../mkl_interface_ext/_qr_impl.py'],
    subdir: 'mkl_interface_ext'
)

py.extension_module('_qr',
  ['../src/py.cpp'],
  include_directories: [incdir, incdir_dpctl, mkl_incl],
  dependencies : [pybind11_dep, mkl_lib],
  cpp_args : compile_opts,
  link_args: sycl_link_opts,
  install : true,
  install_dir: py.get_install_dir() / 'mkl_interface_ext',
)
