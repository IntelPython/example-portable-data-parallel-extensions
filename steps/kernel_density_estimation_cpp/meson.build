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
if should_target_cuda
    sycl_compile_opts = ['-fsycl', '-fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown']
    sycl_link_opts = ['-fsycl', '-fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown']
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

