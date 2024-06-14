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
    
namespace Timer {
template <typename Scalar, typename RangeType>
static void print_throughput(::sycl::event queue_event, const char* task, const char* subtask, RangeType dims)
{   
    float ns = (queue_event.get_profiling_info<::sycl::info::event_profiling::command_end>() -
        queue_event.get_profiling_info<::sycl::info::event_profiling::command_start>());
    double seconds = double(ns) / 1e9;
    size_t bytes = dims.size() * sizeof(Scalar);
    double throughput = bytes / seconds;
    throughput /= 1024 * 1024 * 1024;
    std::cerr << task << " elapsed time: " << std::fixed << std::setprecision(6) << seconds << std::endl;
    std::cerr << "# " << subtask << " rate: " << std::fixed << std::setprecision(2) << throughput << " (GB / sec)" << std::endl;
}
} // namespace Timer

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif