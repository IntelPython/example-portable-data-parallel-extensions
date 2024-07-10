// Copyright 2022-2024 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <iostream>
#include <cassert>

namespace example {

namespace detail {

template <typename T>
T upper_quotient_of(T n, T wg)
{
    return ((n + wg - 1) / wg);
}

template <typename T1, typename T2>
T1 upper_quotient_of(T1 n, T2 wg) {
    return upper_quotient_of(n, static_cast<T1>(wg));
}

template <typename T>
T gaussian_density_scaling_factor(T h, std::int32_t dim)
{
    T two_pi = T(8) * sycl::atan(T(1));
    T gaussian_norm;
    if (dim == 1) {
        // use reciprocal sqrt function for efficiency
        gaussian_norm = sycl::rsqrt(two_pi) / h;
    } else if (dim % 2 == 1) {
        gaussian_norm = (sycl::rsqrt(two_pi) / h) / sycl::pown(two_pi * h * h, dim / 2);
    } else {
        gaussian_norm = T(1) / sycl::pown(two_pi * h * h, dim / 2);
    }

    return gaussian_norm;
}

/*! @brief Evaluate K( dist_sq(y, x)/(h*h) )*/
template <typename T>
T unnormalized_gaussian_density(const T *y, const T*x, T h, std::int32_t dim) {
    T dist_sq(0);
    for(std::int32_t k=0; k < dim; ++k) {
        T diff = y[k] - x[k];
        dist_sq += diff * diff;
    }
    // local_sum += K( (x-x_i)/h ) / (n * h)
    const T dist_sq_half = dist_sq / T(2);
    const T arg = -dist_sq_half / (h*h);
    return sycl::exp(arg);
}

} // namespace detail


template <typename T>
sycl::event
kernel_density_estimate_temps(
    // execution queue
    sycl::queue &exec_q,
    // number of points to evaluate
    size_t m,
    // dimensionality of the data
    std::int32_t dim,
    // points at which KDE is evaluated, content of (m, dims) array
    const T* x_poi,
    // where values of kde(x, h) are written to, content of (m, ) array
    T *f,
    // Number of points in the data-set: sample from an unknown distribution
    size_t n_data,
    // data-set, content of (n_data, dims) array
    const T* data,
    // smoothing parameter
    T h,
    // vector representing execution status of tasks that must be complete
    // before execution of this kernel can begin
    const std::vector<sycl::event> &depends
)
{
    assert(dim > 0);
    constexpr std::uint32_t n_data_per_wi = 256;

    size_t n_blocks = detail::upper_quotient_of(n_data, n_data_per_wi);

    size_t temp_size = m * n_blocks;
    T *temp = sycl::malloc_device<T>(2 * temp_size, exec_q);

    T *partial_sums = temp;
    T *scratch = temp + temp_size;

    sycl::event e_partial_sums =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(depends);

            sycl::range<2> gRange(m, n_blocks);
            cgh.parallel_for(
                gRange,
                [=](sycl::item<2> it) {
                    size_t t = it.get_id(0);
                    size_t i_block = it.get_id(1);

                    const T &gaussian_norm = detail::gaussian_density_scaling_factor(h, dim);
                    T local_sum(0);

                    for(size_t k = 0; k < n_data_per_wi; ++k) {
                        const size_t x_data_id = i_block * n_data_per_wi + k;
                        if (x_data_id < n_data) {

                            const T &term = detail::unnormalized_gaussian_density(
                                x_poi + t * dim,
                                data + x_data_id * dim,
                                h,
                                dim
                            );

                            // local_sum += K( (x-x_i)/h ) / (n * h)
                            local_sum += (gaussian_norm / n_data) * term;
                        }
                    }

                    partial_sums[t * n_blocks + i_block] = local_sum;
                }
            );
        });

    while (n_blocks > n_data_per_wi) {
        size_t local_n_blocks = detail::upper_quotient_of(n_blocks, n_data_per_wi);

        e_partial_sums =
            exec_q.submit([&](sycl::handler &cgh) {
                cgh.depends_on(e_partial_sums);

                sycl::range<2> gRange(m, local_n_blocks);
                cgh.parallel_for(
                    gRange,
                    [=](sycl::item<2> it) {
                        size_t t = it.get_id(0);
                        size_t i_block = it.get_id(1);

                        T local_sum(0);

                        for(size_t k = 0; k < n_data_per_wi; ++k) {
                            const size_t partials_id = i_block * n_data_per_wi + k;
                            if (partials_id < n_blocks) {
                                local_sum += partial_sums[t * n_blocks + partials_id];
                            }
                        }

                        scratch[t * local_n_blocks + i_block] = local_sum;
                    }
                );
            });

        std::swap(partial_sums, scratch);
        n_blocks = local_n_blocks;
    }

    // final reduction from scratch to output array
    e_partial_sums =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e_partial_sums);

            sycl::range<1> gRange(m);
            cgh.parallel_for(
                gRange,
                [=](sycl::item<1> it) {
                    size_t t = it.get_id(0);

                    T local_sum(0);

                    for(size_t k = 0; k < n_blocks; ++k) {
                        local_sum += partial_sums[t * n_blocks + k];
                    }

                    f[t] = local_sum;
                }
            );
        });

    // wait for all kernels to finish execution and
    // free temporary allocation
    e_partial_sums.wait();
    sycl::free(temp, exec_q);

    return e_partial_sums;
}

template <typename T>
sycl::event
kernel_density_estimate_atomic_ref(
    // execution queue
    sycl::queue &exec_q,
    // number of points to evaluate
    size_t m,
    // dimensionality of the data
    std::int32_t dim,
    // points at which KDE is evaluated, content of (m, dims) array
    const T* x_poi,
    // where values of kde(x, h) are written to, content of (m, ) array
    T *f,
    // Number of points in the data-set: sample from an unknown distribution
    size_t n_data,
    // data-set, content of (n_data, dims) array
    const T* data,
    // smoothing parameter
    T h,
    // vector representing execution status of tasks that must be complete
    // before execution of this kernel can begin
    const std::vector<sycl::event> &depends
)
{
    assert(dim > 0);
    constexpr std::uint32_t n_data_per_wi = 256;

    size_t n_blocks = detail::upper_quotient_of(n_data, n_data_per_wi);

    sycl::event e_fill =
        exec_q.fill<T>(f, T(0), m, depends);

    sycl::event e_kde =
        exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e_fill);

            sycl::range<2> gRange(m, n_blocks);
            cgh.parallel_for(
                gRange,
                [=](sycl::item<2> it) {
                    size_t t = it.get_id(0);
                    size_t i_block = it.get_id(1);

                    const T &gaussian_norm = detail::gaussian_density_scaling_factor(h, dim);
                    T local_sum(0);

                    for(size_t k = 0; k < n_data_per_wi; ++k) {
                        const size_t x_data_id = i_block * n_data_per_wi + k;
                        if (x_data_id < n_data) {

                            const T &term = detail::unnormalized_gaussian_density(
                                x_poi + t * dim,
                                data + x_data_id * dim,
                                h,
                                dim
                            );

                            // local_sum += K( (x-x_i)/h ) / (n * h)
                            local_sum += (gaussian_norm / n_data) * term;
                        }
                    }


                    sycl::atomic_ref<T, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space> f_ref(f[t]);
                    f_ref += local_sum;
                }
            );
        });

    return e_kde;
}


/*
    Evaluates

     f(x, h) = sum(
        1/(sqrt(2*pi)*h)**dim * exp( - dist_squared(x, x_data[j])/(2*h*h)),
        0 <= j < n_data)

    writes out f(x, h) for every x.

    Execution target is specified with sycl::queue argument.

    All pointers are expected to be USM pointers bound to the
    sycl::context used to create execution queue.

 */
template <typename T>
sycl::event
kernel_density_estimate_work_group_reduce_and_atomic_ref(
    // execution queue
    sycl::queue &exec_q,
    // number of points to evaluate
    size_t n_evals,
    // dimensionality of the data
    std::int32_t dim,
    // points at which KDE is evaluated, content of (n_evals, dims) array
    const T* x_poi,
    // where values of kde(x, h) are written to, content of (n_evals, ) array
    T *f,
    // Number of points in the data-set: sample from an unknown distribution
    size_t n_data,
    // data-set, content of (n_data, dims) array
    const T* data,
    // smoothing parameter
    T h,
    // vector representing execution status of tasks that must be complete
    // before execution of this kernel can begin
    const std::vector<sycl::event> &depends
)
{
    assert(dim > 0);
    sycl::event e, e_fill;

    // initialize array of function values with zeros
    try {
        e_fill = exec_q.submit(
            [&](sycl::handler &cgh) {
                cgh.depends_on(depends);
                cgh.fill(f, T(0), n_evals);
            }
        );
    } catch (const std::exception &e){
        std::cout << e.what() << std::endl;
        std::rethrow_exception(std::current_exception());
    }

    const std::uint32_t wg = 512;
    constexpr std::uint32_t n_data_per_wi = 128;

    const size_t n_groups = detail::upper_quotient_of<size_t>(n_data, wg * n_data_per_wi);

    sycl::range<2> gRange(n_evals, n_groups * wg);
    sycl::range<2> lRange(1, wg);

    // populate function values
    // perform 2D loop in parallel
    try{
        e =
        exec_q.submit(
            [&](sycl::handler &cgh) {
                cgh.depends_on(e_fill);

                cgh.parallel_for(
                    sycl::nd_range<2>(gRange, lRange),
                    [=](sycl::nd_item<2> it) {
                        auto x_id = it.get_global_id(0);
                        auto x_data_batch_id = it.get_group(1);
                        auto x_data_local_id = it.get_local_id(1);

                        // work-items sums over data-points with indices
                        //   x_data_id = x_data_batch_id * wg * n_data_per_wi + m * wg + x_data_local_id
                        // for 0 <= m < n_wi
                        T local_sum(0);
                        const T &gaussian_norm = detail::gaussian_density_scaling_factor(h, dim);

                        for(size_t m = 0; m < n_data_per_wi; ++m) {
                            size_t x_data_id = x_data_local_id + m * wg + x_data_batch_id * wg * n_data_per_wi;
                            if (x_data_id < n_data) {
                                const T &term = detail::unnormalized_gaussian_density(
                                    x_poi + x_id * dim,
                                    data + x_data_id * dim,
                                    h,
                                    dim
                                );

                                // local_sum += K( (x-x_i)/h ) / (n * h)
                                local_sum += (gaussian_norm / n_data) * term;
                            }
                        }

                        // Combine values held by each work-item of the work-group
                        // in work-item's private variable `local_sum`
                        auto work_group = it.get_group();
                        T sum_over_wg = sycl::reduce_over_group(work_group, local_sum, sycl::plus<T>());

                        // A single representative of the work-group atomically updates function value
                        // stored in the device global memory with
                        if (work_group.leader()) {
                            sycl::atomic_ref<T, sycl::memory_order::relaxed,
                                    sycl::memory_scope::device,
                                    sycl::access::address_space::global_space> f_ref(f[x_id]);
                            f_ref += sum_over_wg;
                        }
                    }
                );
            });
    } catch (const std::exception &e) {
        std::cout << e.what() << std::endl;
        std::rethrow_exception(std::current_exception());
    }

    return e;
}

template <typename T>
sycl::event
kernel_density_estimate(
    // execution queue
    sycl::queue &exec_q,
    // number of points to evaluate
    size_t n,
    // dimensionality of the data
    std::int32_t dim,
    // points at which KDE is evaluated, content of (n, dims) array
    const T* x,
    // where values of kde(x, h) are written to, content of (n, ) array
    T *f,
    // Number of points in the data-set: sample from an unknown distribution
    size_t n_data,
    // data-set, content of (n_data, dims) array
    const T* data,
    // smoothing parameter
    T h,
    // vector representing execution status of tasks that must be complete
    // before execution of this kernel can begin
    const std::vector<sycl::event> &depends
)
{
    /*
       kernel_density_estimate_temps
       kernel_density_estimate_atomic_ref
       kernel_density_estimate_work_group_reduce_and_atomic_ref
    */
    return kernel_density_estimate_work_group_reduce_and_atomic_ref(
        exec_q, n, dim, x, f, n_data, data, h, depends
    );
}


} // namespace example
