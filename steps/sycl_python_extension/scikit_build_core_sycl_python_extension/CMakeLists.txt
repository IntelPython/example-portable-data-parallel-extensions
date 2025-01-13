cmake_minimum_required(VERSION 3.27...3.29 FATAL_ERROR)

project(${SKBUILD_PROJECT_NAME}
    VERSION ${SKBUILD_PROJECT_VERSION}
    LANGUAGES CXX
    DESCRIPTION "A sample kernel density estimation pybind11 extension for dpctl"
)

option(TARGET_CUDA "Whether to additionally target NVPTX64" OFF)

set(TARGET_HIP
    ""
    CACHE STRING
    "Whether to additionally target a user-specified HIP architecture"
)

find_package(IntelSYCL REQUIRED)
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

include(FetchContent)
FetchContent_Declare(
  pybind11
  URL https://github.com/pybind/pybind11/archive/refs/tags/v2.12.0.tar.gz
  URL_HASH SHA256=bf8f242abd1abcd375d516a7067490fb71abd79519a282d22b6e4d19282185a7
  FIND_PACKAGE_ARGS NAMES pybind11
)
FetchContent_MakeAvailable(pybind11)

find_package(Dpctl CONFIG REQUIRED)

set(py_module_name _kde_sycl_ext)
set(_kde_sycl_ext_sources
    ../src/py.cpp
)

python_add_library(${py_module_name} MODULE ${_kde_sycl_ext_sources} WITH_SOABI)

target_link_libraries(${py_module_name} PRIVATE pybind11::headers)
add_sycl_to_target(TARGET ${py_module_name} SOURCES ${_kde_sycl_ext_sources})

target_compile_options(${py_module_name} PRIVATE -O3 -fno-approx-func -fno-fast-math)
target_include_directories(${py_module_name} PRIVATE ${Dpctl_INCLUDE_DIRS})
target_include_directories(${py_module_name} PRIVATE ${Dpctl_TENSOR_INCLUDE_DIR})

set(_sycl_targets)
set(_hip_targets)
if (${TARGET_CUDA})
    set(_sycl_targets "nvptx64-nvidia-cuda,spir64-unknown-unknown")
endif()
if (NOT "x${TARGET_HIP}" STREQUAL "x")
    set(_hip_targets ${TARGET_HIP})
    if(_sycl_targets)
        set(_sycl_targets "amdgcn-amd-amdhsa,${_dpctl_sycl_targets}")
    else()
        set(_sycl_targets "amdgcn-amd-amdhsa,spir64-unknown-unknown")
    endif()
endif()

set(_sycl_target_compile_options)
set(_sycl_target_link_options)

if (_sycl_targets)
    message(STATUS "Compiling for -fsycl-targets=${_sycl_targets}")
    list(APPEND _sycl_target_compile_options -fsycl-targets=${_sycl_targets})
    list(APPEND _sycl_target_link_options -fsycl-targets=${_sycl_targets})
    if(_hip_targets)
        list(APPEND _sycl_target_compile_options -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${_hip_targets})
        list(APPEND _sycl_target_link_options -Xsycl-target-backend=amdgcn-amd-amdhsa --offload-arch=${_hip_targets})
    endif()
    target_compile_options(${py_module_name} PUBLIC ${_sycl_target_compile_options})
    target_link_options(${py_module_name} PUBLIC ${_sycl_target_link_options})
endif()

install(TARGETS ${py_module_name} DESTINATION ${SKBUILD_PROJECT_NAME})
