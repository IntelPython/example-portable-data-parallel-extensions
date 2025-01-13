cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(kde_app
    VERSION 1
    LANGUAGES CXX
    DESCRIPTION "C++ SYCL sample application to compute Kernel Density Estimation"
)

option(TARGET_CUDA "Whether to additionally target NVPTX64" OFF)

set(TARGET_HIP
    ""
    CACHE STRING
    "Whether to additionally target a user-specified HIP architecture"
)

find_package(IntelSYCL REQUIRED)

add_executable(
    kde_app
    ${CMAKE_SOURCE_DIR}/app.cpp
)
target_include_directories(kde_app PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/argparse/include)
add_sycl_to_target(TARGET kde_app SOURCES ${CMAKE_SOURCE_DIR}/app.cpp)
target_compile_options(kde_app PUBLIC -Wall)

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
    target_compile_options(kde_app PUBLIC ${_sycl_target_compile_options})
    target_link_options(kde_app PUBLIC ${_sycl_target_link_options})
endif()

install(TARGETS kde_app DESTINATION ${CMAKE_INSTALL_PREFIX})
