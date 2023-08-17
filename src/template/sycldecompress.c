#ifdef ZFP_WITH_SYCL

#include "../sycl/interface.h"

/* decompress 1d contiguous array */
static void
_t2(decompress_sycl, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  zfp_internal_sycl_decompress(stream, field);
}

/* compress 1d strided array */
static void
_t2(decompress_strided_sycl, Scalar, 1)(zfp_stream* stream, zfp_field* field)
{
  zfp_internal_sycl_decompress(stream, field);
}

/* compress 2d strided array */
static void
_t2(decompress_strided_sycl, Scalar, 2)(zfp_stream* stream, zfp_field* field)
{
  zfp_internal_sycl_decompress(stream, field);
}

/* compress 3d strided array */
static void
_t2(decompress_strided_sycl, Scalar, 3)(zfp_stream* stream, zfp_field* field)
{
  zfp_internal_sycl_decompress(stream, field);
}

#endif
