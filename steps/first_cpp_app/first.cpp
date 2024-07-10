#include <sycl/sycl.hpp>
#include <iostream>
#include <iomanip>

/*! @brief Output array elements */
void output_array(const int *data, size_t n) {
    size_t line_lenths = 20;

    for(size_t line_id = 0; line_id < n; line_id += line_lenths) {
        size_t el_max = std::min(line_id + line_lenths, n);

        for(size_t el_id = line_id; el_id < el_max; ++el_id) {
            std::cout << std::setw(4) << data[el_id];
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void output_device(const sycl::device &d) {
    const auto &name =
      d.get_info<sycl::info::device::name>();
    const auto &dver =
      d.get_info<sycl::info::device::driver_version>();
    std::cout << "Device: " << name << " [" << dver << "]"
              << std::endl;

    return;
}

int main(void)
{

    // queue to enqueue work to default-selected device 
    sycl::queue q{sycl::default_selector_v};

    output_device(q.get_device());

    // allocation device 
    size_t data_size = 256;
    int *data = sycl::malloc_device<int>(data_size, q);

    // submit a task
    sycl::event e_fill = 
        q.fill<int>(data, 42, data_size);

    sycl::event e_comp = 
        q.submit([&](sycl::handler &cgh) {
            cgh.depends_on(e_fill);

            sycl::range<1> global_iter_range{data_size};
            cgh.parallel_for(
                global_iter_range,
                [=](sycl::item<1> it) {
                    int i = it.get_id(0);
                    data[i] += i;
                }
            );
        });

    int *host_data = new int[data_size];
    q.copy<int>(data, host_data, data_size, {e_comp}).wait();

    sycl::free(data, q);

    output_array(host_data, data_size);

    delete[] host_data;

    return 0;
}
