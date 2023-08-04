#include <CL/sycl.hpp>
#ifndef ZFP_SYCL_DEVICE_H
#define ZFP_SYCL_DEVICE_H

#define ZFP_MAGIC 0x7a667000u

#define SYCL_ERR(function_call, custom_msg, return_value) \
try { \
    function_call; \
} catch (const cl::sycl::exception& e) { \
    std::cerr << "zfp_sycl : " << custom_msg << e.what() << std::endl; \
    return return_value; \
} 

namespace zfp {
namespace sycl {
namespace internal {

using namespace ::sycl;

bool device_init()
{
    bool success = true;
    try{
        // Get a SYCL device queue
        //TODO: how to get the preferred device type?
        queue device_q(cpu_selector_v);
    
        // allocate a buffer to store the magic number on the device
        buffer<unsigned int, 1> d_word_buf(NULL, 1);

        //launch a kernel to initialize the magic number
        device_q.submit([&](handler& cgh) {
            auto d_word = d_word_buf.get_access<access::mode::write>(cgh);
            cgh.single_task<class device_init_kernel>([=]() {
                d_word[0] = ZFP_MAGIC;
            });
        });
        device_q.wait();

        //copy the magic number back to the host
        unsigned int h_word = d_word_buf.get_access<access::mode::read>()[0];
        if (h_word != ZFP_MAGIC) {
            std::cerr<<"zfp_sycl : zfp device init failed"<<std::endl;
            success = false;
        }
    }catch (cl::sycl::exception const &e) {
        std::cerr<<"zfp_sycl : zfp device init "<< e.what() << std::endl;
        success = false;}


    return success;
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
