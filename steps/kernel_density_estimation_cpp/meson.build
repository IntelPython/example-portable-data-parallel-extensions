project('kde_app', 'cpp')

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

incdir = include_directories('.')
argparse_incdir = include_directories('./argparse/include')
executable('kde_app', 'app.cpp',
    include_directories: [incdir, argparse_incdir],
    cpp_args : sycl_compile_opts,
    link_args: sycl_link_opts,
    install: true
)

