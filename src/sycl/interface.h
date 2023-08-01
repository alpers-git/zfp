#ifndef ZFP_SYCL_INTERFACE_H
#define ZFP_SYCL_INTERFACE_H

#include "zfp.h"

#ifdef __cplusplus
extern "C" {
#endif

// these functions should not be called directly; use zfp_(de)compress()
zfp_bool zfp_internal_sycl_init(zfp_exec_params_sycl* params);
size_t zfp_internal_hip_compress(zfp_stream* stream, const zfp_field* field);
size_t zfp_internal_hip_decompress(zfp_stream* stream, zfp_field* field);

#ifdef __cplusplus
}
#endif

#endif

