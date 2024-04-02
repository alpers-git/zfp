#ifndef SYCLZFP_POINTERS_H
#define SYCLZFP_POINTERS_H

#define DPCT_PROFILING_ENABLED
#define DPCT_COMPAT_RT_VERSION 11060
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
// #include "ErrorCheck.h"
#include <iostream>


namespace syclZFP
{
// https://gitlab.kitware.com/third-party/nvpipe/blob/master/encode.c
bool is_gpu_ptr(const void *ptr) try {
  dpct::pointer_attributes atts;
  bool err = false;
  try {
    atts.init(ptr);
  } catch (sycl::exception const &exc) {
    err = true;
  }
  return (!err) && (atts.get_memory_type() == sycl::usm::alloc::device ||
                       atts.get_memory_type() == sycl::usm::alloc::shared);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

} // namespace syclZFP

#endif
