#include <iostream>
#include "interface.h"
#include "shared.h"
//#include "device.h"

#include <CL/sycl.hpp>

zfp_bool
zfp_internal_sycl_init(zfp_exec_params_sycl* params)
{
  // ensure device word size equals CPU word size
  if (sizeof(Word) != sizeof(bitstream_word))
    return false;
  
  // perform expensive query of device properties only once
  static bool initialized = false;
  sycl::device d;
  //support cpu only for now
  try {
    params->device = zfp_sycl_cpu;
    d = sycl::device(sycl::cpu_selector_v);
  }
  catch (cl::sycl::exception const &e) {
    params->device = zfp_sycl_default;
    d = sycl::device(sycl::default_selector_v);
    std::cerr<<"SYCL exception caught: "<<e.what()<< "(using the default device selector)" << std::endl;
  }

  if (!initialized)
    return zfp_false;
  initialized = true;

  params->processors = d.get_info<sycl::info::device::max_compute_units>();
  const sycl::id<3> grid_sizes = d.get_info<sycl::ext::oneapi::experimental::info::device::max_work_groups<3>>();
  params->grid_size[0] = grid_sizes[0];
  params->grid_size[1] = grid_sizes[1];
  params->grid_size[2] = grid_sizes[2];


  //return (zfp_bool)zfp::sycl::internal::device_init();
}

size_t
zfp_internal_sycl_compress(zfp_stream* stream, const zfp_field* field)
{
  return 0;
}

size_t
zfp_internal_sycl_decompress(zfp_stream* stream, zfp_field* field)
{
    return 0;
}
    
