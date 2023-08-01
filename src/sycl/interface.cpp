#include <iostream>
#include "interface.h"
//#include <CL/sycl.hpp>

zfp_bool
zfp_internal_sycl_init(zfp_exec_params_sycl* params)
{
    static bool initialized = false;
    return zfp_false;
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
    