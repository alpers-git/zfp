#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <iostream>
#include "interface.h"
#include "../share/device.h"
#include "shared.h"
#include "error.h"
#include "traits.h"
#include "device.h"
#include "writer.h"
#include "encode.h"
#include "encode1.h"
#include "encode2.h"
#include "encode3.h"
#include "variable.h"
#include "reader.h"
#include "decode.h"
#include "decode1.h"
#include "decode2.h"
#include "decode3.h"

#include <CL/sycl.hpp>
#include <dpct/dpct.hpp>

zfp_bool
zfp_internal_sycl_init(zfp_exec_params_sycl* params) try {

  // ensure GPU word size equals CPU word size
  if (sizeof(Word) != sizeof(bitstream_word))
    return false;

  params->device = zfp_sycl_cpu;

  // perform expensive query of device properties only once
  static bool initialized = false;
  static dpct::device_info prop;
  /*
  DPCT1003:36: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  if (!initialized &&
      (dpct::dev_mgr::instance().get_device(0).get_device_info(prop), 0) != 0)
    return zfp_false;
  initialized = true;

  // cache device properties
  params->processors = prop.get_max_compute_units();
  /*
  DPCT1022:37: There is no exact match between the maxGridSize and the
  max_nd_range size. Verify the correctness of the code.
  */
  params->grid_size[0] = prop.get_max_nd_range_size<int *>()[0];
  /*
  DPCT1022:38: There is no exact match between the maxGridSize and the
  max_nd_range size. Verify the correctness of the code.
  */
  params->grid_size[1] = prop.get_max_nd_range_size<int *>()[1];
  /*
  DPCT1022:39: There is no exact match between the maxGridSize and the
  max_nd_range size. Verify the correctness of the code.
  */
  params->grid_size[2] = prop.get_max_nd_range_size<int *>()[2];

  // launch device warm-up kernel
  return (zfp_bool)zfp::sycl::internal::device_init();
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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
    
