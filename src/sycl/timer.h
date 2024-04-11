#ifndef ZFP_SYCL_TIMER_CUH
#define ZFP_SYCL_TIMER_CUH

#define DPCT_PROFILING_ENABLED
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>
#include <chrono>

namespace zfp {
namespace sycl {
namespace internal {

// timer for measuring encode/decode throughput
class Timer {
public:
    Timer()
    {}

    // start timer
    void start()
    {
        dpct::device_ext& dev_ct1 = dpct::get_current_device();
        ::sycl::queue q_ct1((::sycl::device)dev_ct1, ::sycl::property::queue::enable_profiling{});
        e_start = q_ct1.ext_oneapi_submit_barrier();//ugh...instead of timing the actual kernel, we time the barrier
    }

    // stop timer
    void stop()
    {
        dpct::device_ext& dev_ct1 = dpct::get_current_device();
        ::sycl::queue q_ct1((::sycl::device)dev_ct1, ::sycl::property::queue::enable_profiling{});
        e_stop = q_ct1.ext_oneapi_submit_barrier();//ugh...instead of timing the actual kernel, we time the barrier
        e_stop.wait_and_throw();
        q_ct1.wait();
    }

    // print throughput in GB/s
    template <typename Scalar, typename RangeType>
    void print_throughput(const char* task, const char* subtask, RangeType dims) const
    {   
        //since we are timing barriers we will get end barrier's starting time and start barrier's ending time...ugh
        float ns = (e_stop.get_profiling_info<::sycl::info::event_profiling::command_start>() -
            e_start.get_profiling_info<::sycl::info::event_profiling::command_end>());
        double seconds = double(ns) / 1e9;
        size_t bytes = dims.size() * sizeof(Scalar);
        double throughput = bytes / seconds;
        throughput /= 1024 * 1024 * 1024;
        std::cerr << task << " elapsed time: " << std::fixed << std::setprecision(6) << seconds << std::endl;
        std::cerr << "# " << subtask << " rate: " << std::fixed << std::setprecision(2) << throughput << " (GB / sec)" << std::endl;
    }

protected:
    ::sycl::event e_start, e_stop;
};

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif