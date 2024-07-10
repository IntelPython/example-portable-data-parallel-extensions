#include <sycl/sycl.hpp>
#include "oneapi/mkl.hpp"

#include <cstdint>
#include <exception>
#include <vector>
#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "dpctl4pybind11.hpp"
#include "utils/type_dispatch.hpp"


namespace py = pybind11;
namespace dpt = dpctl::tensor;

namespace {

template <typename intT>
intT round_up_mult(intT n, intT m) {
    intT q = (n + (m - 1)) / m;
    return q * m;
}

}

/*
    Complete QR decomposition:

    A (m, n, b) ->
        Q (m, m, b) @ R(m, n, b)

    Number of reflectsion max(1, min(m, n)).

    All input arrays have F-contig layout,
    A.strides = [1, m, m * n]
    Q.strides = [1, m, m * n]
    R.strides = [1, m, m * n]
 */
template <typename T>
sycl::event
do_qr(
    sycl::queue &exec_q, 
    std::int64_t m,
    std::int64_t n,
    std::int64_t b,
    T *a,
    T *q,
    T *r,
    const std::vector<sycl::event> &depends)
{
    static_assert(std::is_floating_point_v<T>);

    std::int64_t lda = m;
    std::int64_t mat_size = m * n;
    std::int64_t q_size = m * m;
    std::int64_t tau_size = std::max(std::int64_t(1), std::min(m, n));

    std::int64_t n_linear_streams = (b > 16) ? 4 : ((b > 4 ? 2 : 1));

    std::int64_t scratch_sz_geqrf = 
        oneapi::mkl::lapack::geqrf_scratchpad_size<T>(exec_q, m, n, lda);

    std::int64_t scratch_sz_orgqr = 
        oneapi::mkl::lapack::orgqr_scratchpad_size<T>(exec_q, m, m, tau_size, lda);

    std::int64_t padding = 256 / sizeof(T);
    size_t alloc_tau_sz = round_up_mult(n_linear_streams * tau_size, padding);
    size_t alloc_geqrf_scratch_sz = round_up_mult(n_linear_streams * scratch_sz_geqrf, padding);
    size_t alloc_orgqr_scratch_sz = n_linear_streams * scratch_sz_orgqr;

    size_t alloc_size = 
        alloc_tau_sz + alloc_geqrf_scratch_sz + alloc_orgqr_scratch_sz;

    // allocate memory for temporaries: taus and scratch spaces
    T *blob = sycl::malloc_device<T>(alloc_size, exec_q);

    if (!blob) 
        throw std::runtime_error("Device allocation failed");

    T *taus = blob;
    T *scratch_geqrf = taus + alloc_tau_sz;
    T *scratch_orgqr = scratch_geqrf + alloc_geqrf_scratch_sz;

    // events to manage execution graph, which is `n_linear_stream` of
    // linear graphs which tie up into memory clean-up host task
    std::vector<std::vector<sycl::event>> comp_evs(n_linear_streams, depends);

    std::exception_ptr e_ptr;
    // iterate over batches on host to submit tasks
    for(size_t batch_id = 0; batch_id < b; ++batch_id) {
        std::int64_t stream_id = (batch_id % n_linear_streams);

        T *current_a = a + batch_id * mat_size;
        T *current_q = q + batch_id * q_size;
        T *current_r = r + batch_id * mat_size;

        T *current_tau = taus + stream_id * tau_size;
        T *current_scratch_geqrf = scratch_geqrf + stream_id * scratch_sz_geqrf;
        T *current_scratch_orgqr = scratch_orgqr + stream_id * scratch_sz_orgqr;

        const auto &current_dep = comp_evs[stream_id];  

        // overwrites memory in current_a
        sycl::event e_geqrf;
        try {
            e_geqrf = oneapi::mkl::lapack::geqrf(
                exec_q, m, n, current_a, lda, current_tau, current_scratch_geqrf, scratch_sz_geqrf, current_dep);
        } catch (const oneapi::mkl::lapack::exception &e) {
            std::cerr << "Exception raised by geqrf: " << e.what() << ", info = " << e.info() << std::endl;

            e_ptr = std::current_exception();
            break;
        }

        sycl::event e_copy_r = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e_geqrf);
            sycl::range<2> gRange{
                static_cast<size_t>(n),
                static_cast<size_t>(m)
            };
            cgh.parallel_for(
                gRange,
                [=](sycl::id<2> id) {
                    auto i = id[1];
                    auto j = id[0];
                    auto offset = j * lda + i;
                    current_r[offset] = (i > j) ? T(0) : current_a[offset];
                }
            );
        });

        sycl::event e_copy_q = exec_q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e_geqrf);
            sycl::range<2> gRange{
                static_cast<size_t>(m),
                static_cast<size_t>(m)
            };
            cgh.parallel_for(
                gRange, 
                [=](sycl::id<2> id) {
                    auto i = id[1];
                    auto j = id[0];
                    auto offset = j * lda + i;
                    current_q[offset] = (j < n) ? current_a[offset] : T(0);
                }
            );
        });


        sycl::event e_orgqr; 
        try {
            e_orgqr = oneapi::mkl::lapack::orgqr(
                exec_q, m, m, tau_size, current_q, lda, current_tau, current_scratch_orgqr, scratch_sz_orgqr, {e_copy_q});
        } catch (const oneapi::mkl::lapack::exception &e) {
            std::cerr << "Exception raised by orgqr: " << e.what() << ", info = " << e.info() << std::endl;

            e_ptr = std::current_exception();
            break;
        }


        comp_evs[stream_id] = {e_orgqr, e_copy_r};
    }

    if (e_ptr) {
        sycl::free(blob, exec_q);
        std::rethrow_exception(e_ptr); 
    } 

    sycl::event ht_ev = 
        exec_q.submit([&](sycl::handler &cgh) {
            for(const auto &el : comp_evs) {
                cgh.depends_on(el);
            }
            const auto ctx = exec_q.get_context();

            cgh.host_task([ctx, blob] {
                sycl::free(blob, ctx); 
            });
        });

    return ht_ev;
}

const auto &unexpected_dims0_msg = "Unexpected dimensions of input arrays. All arrays must be 3D, for stack of matrices";
const auto &unexpected_dims1_msg = "Unexpected dimensions of input arrays. All stacks of matrices must have equal number of matrices";
const auto &unexpected_dims2_msg = "Unexpected dimensions of input arrays. All matrices in stacks must have consistent dimensions";
const auto &unexpected_types_msg = "All arrays must have the same data type";
const auto &unexpected_input_layout_msg = "All input arrays must be F-contiguous, indexed by (height_id, width_id, batch_id)";
const auto &incompatible_queues_msg = "All arrays must has the same queue associated with them";
const auto &empty_inputs_msg = "Non-empty input arrays are expected";

std::pair<sycl::event, sycl::event>
py_qr(
    dpt::usm_ndarray &stack_of_mats,
    dpt::usm_ndarray &stack_of_qs,
    dpt::usm_ndarray &stack_of_rs,
    const std::vector<sycl::event> &depends
)
{
    auto mats_ndim = stack_of_mats.get_ndim();
    auto qs_ndim = stack_of_qs.get_ndim();
    auto rs_ndim = stack_of_rs.get_ndim();

    if (rs_ndim != 3 || qs_ndim != 3 || mats_ndim != 3)
        throw py::value_error(unexpected_dims0_msg);

    py::ssize_t s0_mats = stack_of_mats.get_shape(0);
    py::ssize_t s1_mats = stack_of_mats.get_shape(1);
    py::ssize_t b_mats = stack_of_mats.get_shape(2);

    py::ssize_t s0_qs = stack_of_qs.get_shape(0);
    py::ssize_t s1_qs = stack_of_qs.get_shape(1);
    py::ssize_t b_qs = stack_of_qs.get_shape(2);

    py::ssize_t s0_rs = stack_of_rs.get_shape(0);
    py::ssize_t s1_rs = stack_of_rs.get_shape(1);
    py::ssize_t b_rs = stack_of_qs.get_shape(2);

    if (b_mats != b_qs || b_mats != b_rs)
        throw py::value_error(unexpected_dims1_msg);

    if(s0_mats != s0_qs || s1_qs != s0_rs || s1_mats != s1_rs)
        throw py::value_error(unexpected_dims2_msg);

    if (b_mats == 0 || s0_mats == 0 || s1_mats == 0) 
        throw py::value_error(empty_inputs_msg);

    int mats_tnum = stack_of_mats.get_typenum();
    int qs_tnum = stack_of_qs.get_typenum();
    int rs_tnum = stack_of_rs.get_typenum();

    if (mats_tnum != qs_tnum || mats_tnum != rs_tnum)
        throw py::value_error(unexpected_types_msg);

    bool all_f_contig = stack_of_mats.is_f_contiguous();
    all_f_contig = all_f_contig && stack_of_qs.is_f_contiguous();
    all_f_contig = all_f_contig && stack_of_rs.is_f_contiguous();

    if (!all_f_contig)
        throw py::value_error(unexpected_input_layout_msg);

    sycl::queue m_q = stack_of_mats.get_queue();
    const sycl::queue &q_q = stack_of_qs.get_queue();
    const sycl::queue &r_q = stack_of_rs.get_queue();

    if (!dpctl::utils::queues_are_compatible(m_q, {q_q, r_q}))
        throw py::value_error(incompatible_queues_msg);

    sycl::queue &exec_q = m_q;
    py::ssize_t m = s0_mats;
    py::ssize_t n = s1_mats;
    py::ssize_t b = b_mats;

    auto const &array_types = dpt::type_dispatch::usm_ndarray_types();
    const int inp_typeid = array_types.typenum_to_lookup_id(mats_tnum);
    
    sycl::event qr_ev;

    if (inp_typeid == static_cast<int>(dpt::type_dispatch::typenum_t::FLOAT)) {
        using T = float;
        T *a_data = stack_of_mats.get_data<T>();
        T *q_data = stack_of_qs.get_data<T>();
        T *r_data = stack_of_rs.get_data<T>();

        qr_ev = do_qr<T>(
            exec_q, 
            m, n, b,
            a_data, q_data, r_data,  
            depends
        );
    } else if (inp_typeid == static_cast<int>(dpt::type_dispatch::typenum_t::DOUBLE)) {
        using T = double;

        T *a_data = stack_of_mats.get_data<T>();
        T *q_data = stack_of_qs.get_data<T>();
        T *r_data = stack_of_rs.get_data<T>();

        qr_ev = do_qr<T>(
            exec_q, 
            m, n, b,
            a_data, q_data, r_data,  
            depends
        );
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    sycl::event ht_ev = 
        dpctl::utils::keep_args_alive(exec_q, {stack_of_mats, stack_of_qs, stack_of_qs}, {qr_ev});

    return std::make_pair(ht_ev, qr_ev);
}

PYBIND11_MODULE(_qr, m) {
    m.def("_qr", &py_qr, 
        "Compute QR decomposition on stack of real floating-point F-contiguous arrays",
        py::arg("stack_of_as"), 
        py::arg("stack_of_qs"), 
        py::arg("stack_of_rs"), 
        py::arg("depends") = py::list()
    );
}