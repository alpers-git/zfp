#include <iostream>
#include "interface.h"
#include "../share/device.h"
#include "shared.cuh"
#include "error.cuh"
#include "traits.cuh"
#include "device.cuh"
#include "writer.cuh"
#include "encode.cuh"
#include "encode1.cuh"
#include "encode2.cuh"
#include "encode3.cuh"
#include "variable.cuh"
#include "reader.cuh"
#include "decode.cuh"
#include "decode1.cuh"
#include "decode2.cuh"
#include "decode3.cuh"

zfp_bool
zfp_internal_cuda_init(zfp_exec_params_cuda* params)
{
  // ensure GPU word size equals CPU word size
  if (sizeof(Word) != sizeof(bitstream_word))
    return false;

  // perform expensive query of device properties only once
  static bool initialized = false;
  static cudaDeviceProp prop;
  if (!initialized && cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
    return zfp_false;
  initialized = true;

  // cache device properties
  params->processors = prop.multiProcessorCount;
  params->grid_size[0] = prop.maxGridSize[0];
  params->grid_size[1] = prop.maxGridSize[1];
  params->grid_size[2] = prop.maxGridSize[2];

  // launch device warm-up kernel
  return (zfp_bool)zfp::cuda::internal::device_init();
}

size_t
zfp_internal_cuda_compress(zfp_stream* stream, const zfp_field* field)
{
  // determine compression mode and ensure it is supported
  bool variable_rate = false;
  switch (zfp_stream_compression_mode(stream)) {
    case zfp_mode_fixed_rate:
      break;
    case zfp_mode_fixed_precision:
    case zfp_mode_fixed_accuracy:
    case zfp_mode_expert:
      variable_rate = true;
      break;
    default:
      // unsupported compression mode
      return 0;
  }

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
  void* d_data = zfp::cuda::internal::setup_device_field_compress(field, d_begin);

  // null means the array is non-contiguous host memory, which is not supported
  if (!d_data)
    return 0;

  // allocate compressed buffer
  Word* d_stream = zfp::cuda::internal::setup_device_stream_compress(stream);
  // TODO: populate stream->index even in fixed-rate mode if non-null
  ushort* d_index = variable_rate ? zfp::cuda::internal::setup_device_index_compress(stream, field) : NULL;

  // determine minimal slot needed to hold a compressed block
  uint maxbits = (uint)zfp_maximum_block_size_bits(stream, field);

  // encode data
  const bitstream_offset pos = stream_wtell(stream->stream);
  const zfp_exec_params_cuda* params = static_cast<zfp_exec_params_cuda*>(stream->exec.params);
  unsigned long long bits_written = 0;
  switch (field->type) {
    case zfp_type_int32:
      bits_written = zfp::cuda::encode((int*)d_data, size, stride, params, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    case zfp_type_int64:
      bits_written = zfp::cuda::encode((long long int*)d_data, size, stride, params, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    case zfp_type_float:
      bits_written = zfp::cuda::encode((float*)d_data, size, stride, params, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    case zfp_type_double:
      bits_written = zfp::cuda::encode((double*)d_data, size, stride, params, d_stream, d_index, stream->minbits, maxbits, stream->maxprec, stream->minexp);
      break;
    default:
      break;
  }

  // compact stream of variable-length blocks stored in fixed-length slots
  if (variable_rate) {
    const size_t blocks = zfp_field_blocks(field);
    bits_written = zfp::cuda::internal::compact_stream(d_stream, maxbits, d_index, blocks, params->processors);
  }

  const size_t stream_bytes = zfp::cuda::internal::round_up((bits_written + CHAR_BIT - 1) / CHAR_BIT, sizeof(Word));

  if (d_index) {
    const size_t size = zfp_field_blocks(field) * sizeof(ushort);
    // TODO: assumes index stores block sizes
    zfp::cuda::internal::cleanup_device(stream->index ? stream->index->data : NULL, d_index, size);
  }

  // copy stream from device to host if needed and free temporary buffers
  zfp::cuda::internal::cleanup_device(stream->stream->begin, d_stream, stream_bytes);
  zfp::cuda::internal::cleanup_device(zfp_field_begin(field), d_begin);

  // update bit stream to point just past produced data
  if (bits_written)
    stream_wseek(stream->stream, pos + bits_written);

  return bits_written;
}

size_t
zfp_internal_cuda_decompress(zfp_stream* stream, zfp_field* field)
{
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

  void* d_begin;
  void* d_data = zfp::cuda::internal::setup_device_field_decompress(field, d_begin);

  // null means the array is non-contiguous host memory, which is not supported
  if (!d_data)
    return 0;

  Word* d_stream = zfp::cuda::internal::setup_device_stream_decompress(stream);
  Word* d_index = NULL;

  // decode_parameter differs per execution policy
  zfp_mode mode = zfp_stream_compression_mode(stream);
  zfp_index_type index_type = zfp_index_none;
  uint granularity;

  switch (mode) {
    case zfp_mode_fixed_rate:
      granularity = 1;
      break;
    case zfp_mode_fixed_precision:
    case zfp_mode_fixed_accuracy:
      if (!stream->index) {
#ifdef ZFP_DEBUG
        std::cerr << "zfp_cuda : variable-rate decompression requires block index" << std::endl;
#endif
        return 0;
      }
      index_type = stream->index->type;
      if (index_type != zfp_index_offset && index_type != zfp_index_hybrid) {
#ifdef ZFP_DEBUG
        std::cerr << "zfp_cuda : index type not supported on GPU" << std::endl;
#endif
        return 0;
      }
      granularity = stream->index->granularity;
      d_index = zfp::cuda::internal::setup_device_index_decompress(stream);
      break;
    default:
      // TODO: clean up device to avoid memory leak
#ifdef ZFP_DEBUG
      std::cerr << "zfp_cuda : compression mode not supported on GPU" << std::endl;
#endif
      return 0;
  }

  // decode compressed data
  const bitstream_offset pos = stream_rtell(stream->stream);
  const zfp_exec_params_cuda* params = static_cast<zfp_exec_params_cuda*>(stream->exec.params);
  const uint minbits = stream->minbits;
  const uint maxbits = stream->maxbits;
  const uint maxprec = stream->maxprec;
  const int minexp = stream->minexp;
  unsigned long long bits_read = 0;
  switch (field->type) {
    case zfp_type_int32:
      bits_read = zfp::cuda::decode((int*)d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    case zfp_type_int64:
      bits_read = zfp::cuda::decode((long long int*)d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    case zfp_type_float:
      bits_read = zfp::cuda::decode((float*)d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    case zfp_type_double:
      bits_read = zfp::cuda::decode((double*)d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    default:
      break;
  }

  // copy field from device to host if needed and free temporary buffers
  size_t field_bytes = zfp_field_size_bytes(field);
  zfp::cuda::internal::cleanup_device(zfp_field_begin(field), d_begin, field_bytes);
  zfp::cuda::internal::cleanup_device(stream->stream->begin, d_stream);
  if (d_index)
    zfp::cuda::internal::cleanup_device(stream->index->data, d_index);

  // update bit stream to point just past consumed data
  if (bits_read)
    stream_rseek(stream->stream, pos + bits_read);

  return bits_read;
}