#include <iostream>
#include "interface.h"
#include <CL/sycl.hpp>

zfp_bool
zfp_internal_sycl_init(zfp_exec_params_sycl* params)
{
    static bool initialized = false;
    
    return zfp_false;
}

size_t
zfp_internal_sycl_compress(zfp_stream* stream, const zfp_field* field)
{
    // determine compression mode and ensure it is supported
  bool variable_rate = false;
//   switch (zfp_stream_compression_mode(stream)) {
//     case zfp_mode_fixed_rate:
//       break;
//     case zfp_mode_fixed_precision:
//     case zfp_mode_fixed_accuracy:
//     case zfp_mode_expert:
//       variable_rate = true;
//       break;
//     default:
//       // unsupported compression mode
//       return 0;
//   }

  // determine field dimensions
  size_t size[3];
  size[0] = field->nx;
  size[1] = field->ny;
  size[2] = field->nz;

  // determine field strides
  ptrdiff_t stride[3];
  stride[0] = field->sx ? field->sx : 1;
  stride[1] = field->sy ? field->sy : (ptrdiff_t)field->nx;
  stride[2] = field->sz ? field->sz : (ptrdiff_t)field->nx * (ptrdiff_t)field->ny;

  // copy field to device if not already there
  void* d_begin = NULL;
  void* d_data = zfp::sycl::internal::setup_device_field_compress(field, d_begin);

  // null means the array is non-contiguous host memory, which is not supported
  if (!d_data)
    return 0;

  // allocate compressed buffer
  Word* d_stream = zfp::sycl::internal::setup_device_stream_compress(stream);
  // TODO: populate stream->index even in fixed-rate mode if non-null
  ushort* d_index = variable_rate ? zfp::sycl::internal::setup_device_index_compress(stream, field) : NULL;

  // determine minimal slot needed to hold a compressed block
  uint maxbits = (uint)zfp_maximum_block_size_bits(stream, field);

  // encode data
  const bitstream_offset pos = stream_wtell(stream->stream);
  const zfp_exec_params_sycl* params = static_cast<zfp_exec_params_sycl*>(stream->exec.params);
  unsigned long long bits_written = 0;
  switch (field->type) {
    case zfp_type_int32:
      bits_written = zfp::sycl::encode((int*)d_data, size, stride, params, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    case zfp_type_int64:
      bits_written = zfp::sycl::encode((long long int*)d_data, size, stride, params, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    case zfp_type_float:
      bits_written = zfp::sycl::encode((float*)d_data, size, stride, params, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    case zfp_type_double:
      bits_written = zfp::sycl::encode((double*)d_data, size, stride, params, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    default:
      break;
  }

  // compact stream of variable-length blocks stored in fixed-length slots
//   if (variable_rate) {
//     const size_t blocks = zfp_field_blocks(field);
//     bits_written = zfp::sycl::internal::compact_stream(d_stream, maxbits, d_index, blocks, params->processors);
//   }

  const size_t stream_bytes = zfp::sycl::internal::round_up((bits_written + CHAR_BIT - 1) / CHAR_BIT, sizeof(Word));

  if (d_index) {
    const size_t size = zfp_field_blocks(field) * sizeof(ushort);
    // TODO: assumes index stores block sizes
    zfp::sycl::internal::cleanup_device(stream->index ? stream->index->data : NULL, d_index, size);
  }

  // copy stream from device to host if needed and free temporary buffers
  zfp::sycl::internal::cleanup_device(stream->stream->begin, d_stream, stream_bytes);
  zfp::sycl::internal::cleanup_device(zfp_field_begin(field), d_begin);

  // update bit stream to point just past produced data
  if (bits_written)
    stream_wseek(stream->stream, pos + bits_written);

  return bits_written;
}

size_t
zfp_internal_sycl_decompress(zfp_stream* stream, zfp_field* field)
{
    return 0;
}
    