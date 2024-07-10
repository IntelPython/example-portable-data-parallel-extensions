cmake_minimum_required(VERSION 3.27 FATAL_ERROR)

project(kde_app
    VERSION 1
    LANGUAGES CXX
    DESCRIPTION "C++ SYCL sample application to compute Kernel Density Estimation"
)

option(TARGET_CUDA "Whether to build SYCL target NVPTX64 in addition to default SPIR64 SYCL target" OFF)

find_package(IntelSYCL REQUIRED)

add_executable(
    kde_app
    ${CMAKE_SOURCE_DIR}/app.cpp
)
target_include_directories(kde_app PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_SOURCE_DIR}/argparse/include)
add_sycl_to_target(TARGET kde_app SOURCES ${CMAKE_SOURCE_DIR}/app.cpp)
target_compile_options(kde_app PUBLIC -Wall)

if (${TARGET_CUDA})
    target_compile_options(kde_app PUBLIC -fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown)
    target_link_options(kde_app PUBLIC -fsycl-targets=nvptx64-nvidia-cuda,spir64-unknown-unknown)
endif()

install(TARGETS kde_app DESTINATION ${CMAKE_INSTALL_PREFIX})
