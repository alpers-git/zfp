#ifndef SYCLZFP_POINTERS_H
#define SYCLZFP_POINTERS_H

#define DPCT_PROFILING_ENABLED
#define DPCT_COMPAT_RT_VERSION 11060
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "ErrorCheck.h"
#include <iostream>


namespace syclZFP
{
// https://gitlab.kitware.com/third-party/nvpipe/blob/master/encode.c
bool is_gpu_ptr(const void *ptr) try {
  dpct::pointer_attributes atts;
  const dpct::err0 perr = DPCT_CHECK_ERROR(atts.init(ptr));

  // clear last error so other error checking does
  // not pick it up
  /*
  DPCT1010:19: SYCL uses exceptions to report errors and does not use the error
  codes. The call was replaced with 0. You need to rewrite this code.
  */
  dpct::err0 error = 0;
#if DPCT_COMPAT_RT_VERSION >= 10000
  return perr == 0 && (atts.get_memory_type() == sycl::usm::alloc::device ||
                       atts.get_memory_type() == sycl::usm::alloc::shared);
#else
  return perr == cudaSuccess && atts.memoryType == cudaMemoryTypeDevice;
#endif
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

} // namespace syclZFP

#endif
