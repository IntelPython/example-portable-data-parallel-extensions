#include <sycl/sycl.hpp>
#include <argparse/argparse.hpp>
#include "kde.hpp"

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <random>
#include <memory>
#include <exception>
#include <functional>

std::string get_device_info(const sycl::device &d) {
    std::stringstream ss{};

    ss << "Device: ";
    const auto dev_name = d.get_info<sycl::info::device::name>();
    ss << dev_name;
    const auto driver_ver = d.get_info<sycl::info::device::driver_version>();
    ss << "[" << driver_ver << "]";
    ss << std::endl;

    return ss.str();
}

template <typename T, typename Engine>
std::vector<T> uniform_random_vector(Engine &eng, T a, T b, const size_t n)
{
    std::vector<T> vec{};
    vec.reserve(n);

    std::uniform_real_distribution<T> uniform_dist(a, b);

    for(size_t i = 0; i < n; ++i) {
        const auto &val = uniform_dist(eng);
        vec.emplace_back(val);
    }

    return vec;
}

static const auto &algo_temps = "temps";
static const auto &algo_atomic = "atomic_ref";
static const auto &algo_wgreduce_and_atomic = "work_group_reduce_and_atomic_ref";

static const auto &n_sample_opt = "--n_sample";
static const auto &dimension_opt = "--dimension";
static const auto &points_opt = "--points";
static const auto &seed_opt = "--seed";
static const auto &kde_scale_opt = "--smoothing_scale";
static const auto &algo_opt = "--algorithm";

void parse_args(argparse::ArgumentParser &program, int argc, const char *argv[]) {
    program.add_argument("-n", n_sample_opt)
        .help("Number of samples from underlying cuboid distribution")
        .default_value(size_t(1000000))
        .scan<'d', size_t>();

    program.add_argument("-d", dimension_opt)
        .help("Dimensionality of samples")
        .default_value(size_t(4))
        .scan<'d', size_t>();

    program.add_argument("-m", points_opt)
        .help("Number of points at which to estimate distribution value")
        .default_value(size_t(25))
        .scan<'d', size_t>();

    program.add_argument(seed_opt)
        .help("Random seed to use for reproducibility")
        .default_value(size_t(-1))
        .scan<'d', size_t>();

    program.add_argument(kde_scale_opt)
        .help("Kernel density estimation smoothing scale parameter")
        .default_value(double(1)/double(20))
        .scan<'f', float>();

    program.add_argument(algo_opt)
        .help(std::string("Kernel implementation to use. Supported choices are [") +
            algo_temps + ", " +
            algo_atomic + ", " +
            algo_wgreduce_and_atomic +
        "]")
        .default_value(std::string(algo_wgreduce_and_atomic))
        .choices(algo_temps, algo_atomic, algo_wgreduce_and_atomic);

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        std::exit(1);
    }
}

int main(int argc, const char *argv[]) {
    sycl::queue q{sycl::default_selector_v};

    std::cout << get_device_info(q.get_device());

    argparse::ArgumentParser program("kde_app", "1.0");
    parse_args(program, argc, argv);

    using T = float;

    /* Estimate density from `n_sample` points uniformly sampled
     * from unit `dimension`-dimensional cuboid
     */
    const size_t n_sample = program.get<size_t>(n_sample_opt);
    const size_t n_dims = program.get<size_t>(dimension_opt);

    /* Estimate density at `points` sample points inside the cuboid */
    const size_t n_est = program.get<size_t>(points_opt);;

    std::unique_ptr<std::default_random_engine> rng_uptr;
    if (program.is_used(seed_opt)) {
        rng_uptr = std::make_unique<std::default_random_engine>(program.get<size_t>(seed_opt));
    } else {
        std::random_device dev;
        rng_uptr = std::make_unique<std::default_random_engine>(dev());
    }

    std::default_random_engine rng = *rng_uptr;
    const auto &sample = uniform_random_vector(rng, T(0), T(1), n_sample * n_dims);

    const T &margin = T(1)/T(10);
    const auto &poi = uniform_random_vector(rng, margin, T(1) -  margin, n_est * n_dims);

    // allocated Unified Shared Memory accessible from kernels for random samples
    T *sample_usm = sycl::malloc_device<T>(sample.size(), q);
    // start copying data from host-allocated vector to USM allocation
    // The event represents execution status of this task
    sycl::event sample_copy_ev = q.copy<T>(sample.data(), sample_usm, sample.size());

    // USM allocation for points where PDF value needs to be estimated
    T *poi_usm = sycl::malloc_device<T>(poi.size(), q);
    sycl::event poi_copy_ev = q.copy<T>(poi.data(), poi_usm, poi.size());

    // KDE smoothing parameter
    const T &h = (program.is_used(kde_scale_opt)) ?
        program.get<T>(kde_scale_opt) :
        (margin / 4) * std::sqrt(T(n_dims));

    std::cout << "KDE estimation, n_sample: " << n_sample << ", dim = " << n_dims << ", n_est = " << n_est << std::endl;
    std::cout << "Samples are from " << n_dims << "-dimensional uniform distribution" << std::endl;

    std::cout << "KDE smoothing parameter: " << h << std::endl;

    using impl_fn_t = std::function<sycl::event(sycl::queue &, size_t, size_t, const T*, T*, size_t, T*, T, const std::vector<sycl::event> &)>;
    impl_fn_t impl_fn = example::kernel_density_estimate<T>;

    if (program.is_used(algo_opt)) {
        const auto &algo_name = program.get<std::string>(algo_opt);
        if (algo_name == algo_temps) {
            std::cout << "Using kernel implementation '" << algo_temps << "'" << std::endl;
            impl_fn = example::kernel_density_estimate_temps<T>;
        } else if (algo_name == algo_atomic) {
            std::cout << "Using kernel implementation '" << algo_atomic << "'" << std::endl;
            impl_fn = example::kernel_density_estimate_atomic_ref<T>;
        } else {
            std::cout << "Using kernel implementation '" << algo_wgreduce_and_atomic << "'" << std::endl;
        }
    } else {
        std::cout << "Using default kernel implementation '" << algo_wgreduce_and_atomic << "'" << std::endl;
    }

    // USM for estimated density function values
    T *pdf_usm = sycl::malloc_device<T>(n_est, q);

    // submit tasks for estimation, and obtain execution status
    sycl::event kde_ev =
        impl_fn(
            q,
            n_est,
            n_dims,
            poi_usm,
            pdf_usm,
            n_sample,
            sample_usm,
            h,
            // KDE estimation kernel should begin execution
            // only after tasks of populating USM allocations complete
            {sample_copy_ev, poi_copy_ev}
        );

    // container to copy the density estimateds into
    std::vector<T> f(n_est);

    // copy back and synchronize
    q.copy<T>(pdf_usm, f.data(), n_est, {kde_ev}).wait();

    // Free device allocations
    sycl::free(pdf_usm, q);
    sycl::free(poi_usm, q);
    sycl::free(sample_usm, q);

    // Output estimated values
    std::cout << "Estimated density:";
    for(size_t i=0; i < n_est; ++i) {
        std::cout << " " << f[i];
    }
    std::cout << std::endl;

    return 0;
}