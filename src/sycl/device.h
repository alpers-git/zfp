#include <CL/sycl.hpp>
#ifndef ZFP_SYCL_DEVICE_H
#define ZFP_SYCL_DEVICE_H

#define ZFP_MAGIC 0x7a667000u

namespace zfp {
namespace sycl {
namespace internal {

    // Device init kernel in SYCL
class device_init_kernel;

bool device_init()
{
    // Get a SYCL device queue
    sycl::queue deviceQueue;

    // Allocate device memory
    sycl::buffer<unsigned int, 1> d_word(sycl::range<1>(1));

    // Submit a command group to the device queue
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto d_word_ptr = d_word.get_access<sycl::access::mode::write>(cgh);

        // Launch a dummy kernel
        cgh.parallel_for(sycl::range<1>(1), [=](sycl::item<1> item) {
            *d_word_ptr.get_pointer() = ZFP_MAGIC;
        });
    });

    // Allocate host memory
    unsigned int* h_word = static_cast<unsigned int*>(malloc(sizeof(*h_word)));

    // Copy from device to host
    deviceQueue.submit([&](sycl::handler& cgh) {
        auto d_word_ptr = d_word.get_access<sycl::access::mode::read>(cgh);
        sycl::accessor h_word_accessor(h_word, cgh);

        cgh.copy(d_word_ptr, h_word_ptr);
    });

    // Check the result
    bool success = (*h_word == ZFP_MAGIC);

    // Free host memory
    free(h_word);

    return success;
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
