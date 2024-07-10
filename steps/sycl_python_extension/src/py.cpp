#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "dpctl4pybind11.hpp"

#include "utils/type_dispatch.hpp"
#include "kde.hpp"

#include <vector>
#include <utility>

namespace py = pybind11;
namespace dpt = dpctl::tensor;

typedef std::intptr_t ssize_t;

const auto &unexpected_shape_msg = "Unexpected shapes of array arguments";
const auto &unexpected_types_msg = "Unexpected types of array arguments: expected arrays of the same real floating type";
const auto &unexpected_layout_msg = "All input arrays must be C-contiguous";
const auto &incompatible_queue_msg = "Unable to deduce execution queue, queues associated with input arrays are not the same";
const auto &expected_writable_msg = "Output array must be writable";

template <typename T>
sycl::event 
call_kde(
    sycl::queue &exec_q,
    size_t m,
    size_t dim,
    const T* poi_ptr,
    T *pdf_ptr,
    size_t n,
    const T* sample_ptr,
    T h,
    int mode,
    const std::vector<sycl::event> &depends
)
{
    if (mode == 0) {
        return example::kernel_density_estimate_work_group_reduce_and_atomic_ref<T>(
            exec_q, m, dim, poi_ptr, pdf_ptr, n, sample_ptr, h, depends);
    } else if (mode == 1) {
        return example::kernel_density_estimate_atomic_ref<T>(
            exec_q, m, dim, poi_ptr, pdf_ptr, n, sample_ptr, h, depends);
    } else if (mode == 2) {
        return example::kernel_density_estimate_temps<T>(
            exec_q, m, dim, poi_ptr, pdf_ptr, n, sample_ptr, h, depends);
    } else {
        throw std::runtime_error("Invalid mode parameter");
    }
}

std::pair<sycl::event, sycl::event>
py_kde_ext(
    const dpt::usm_ndarray &poi,
    const dpt::usm_ndarray &sample,
    py::object h,
    const dpt::usm_ndarray &pdf,
    int mode,
    const std::vector<sycl::event> &depends
) {

    if (poi.get_ndim() != 2 || sample.get_ndim() != 2 || pdf.get_ndim() != 1) {
        throw py::value_error(unexpected_shape_msg);
    }

    ssize_t m = poi.get_shape(0);
    ssize_t d1 = poi.get_shape(1);

    ssize_t n = sample.get_shape(0);
    ssize_t d2 = sample.get_shape(1);

    ssize_t pdf_len = pdf.get_shape(0);

    if ((d1 != d2) || (pdf_len != m)) {
        throw py::value_error(unexpected_shape_msg);
    }

    int poi_tn = poi.get_typenum();
    int sample_tn = sample.get_typenum();
    int pdf_tn = pdf.get_typenum();

    if ((poi_tn != sample_tn) || (poi_tn != pdf_tn)) {
        throw py::value_error(unexpected_types_msg);
    }

    if (!poi.is_c_contiguous() || !sample.is_c_contiguous() || !pdf.is_c_contiguous()) {
        throw py::value_error(unexpected_layout_msg);
    }

    if (!pdf.is_writable()) {
        throw py::value_error(expected_writable_msg);
    }

    sycl::queue q_poi = poi.get_queue();
    sycl::queue q_sample = sample.get_queue(); 
    sycl::queue q_pdf = pdf.get_queue();

    if (!dpctl::utils::queues_are_compatible(q_poi, {q_sample, q_pdf})) {
        throw py::value_error(incompatible_queue_msg);
    }

    sycl::queue &exec_q = q_poi;

    if (mode < 0 || mode > 2) {
        throw py::value_error("Supported mode selector values are 0, 1, 2");
    }

    auto const &array_types = dpt::type_dispatch::usm_ndarray_types();
    int inp_typeid = array_types.typenum_to_lookup_id(poi_tn);

    sycl::event e_comp;
    if (inp_typeid == static_cast<int>(dpctl::tensor::type_dispatch::typenum_t::FLOAT)) {
        using T = float;

        T h_sc = py::cast<T>(h);
        e_comp = 
            call_kde<T>(exec_q, m, d1, poi.get_data<T>(), pdf.get_data<T>(), n, sample.get_data<T>(), h_sc, mode, depends);

    } else if (inp_typeid == static_cast<int>(dpctl::tensor::type_dispatch::typenum_t::DOUBLE)) {
        using T = double;

        T h_sc = py::cast<T>(h);
        e_comp = 
            call_kde<T>(exec_q, m, d1, poi.get_data<T>(), pdf.get_data<T>(), n, sample.get_data<T>(), h_sc, mode, depends);

    } else {
        throw py::value_error(unexpected_types_msg);
    }

    sycl::event ht_ev = 
        dpctl::utils::keep_args_alive(exec_q, {poi, sample, pdf}, {e_comp});

    return std::make_pair(ht_ev, e_comp);
}


PYBIND11_MODULE(_kde_sycl_ext, m) {
    m.def(
        "_kde", 
        py_kde_ext, 
        "Kernel density estimation",
        py::arg("poi"),
        py::arg("sample"),
        py::arg("h"),
        py::arg("pdf"),
        py::arg("mode"),
        py::arg("depends")
    );
}
