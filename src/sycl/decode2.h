#include <sycl/sycl.hpp>
#ifndef ZFP_SYCL_DECODE2_H
#define ZFP_SYCL_DECODE2_H

namespace zfp {
namespace sycl {
namespace internal {
template <typename Scalar>
inline 
void scatter2(const Scalar* q, Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
{
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx)
      *p = *q++;
}

template <typename Scalar>
inline 
void scatter_partial2(const Scalar* q, Scalar* p, uint nx, uint ny, ptrdiff_t sx, ptrdiff_t sy)
{
  for (uint y = 0; y < 4; y++)
    if (y < ny) {
      for (uint x = 0; x < 4; x++)
        if (x < nx) {
          *p = q[x + 4 * y];
          p += sx;
        }
      p += sy - nx * sx;
    }
}

// decode kernel
template <typename Scalar>

void
decode2_kernel(
  Scalar* d_data,
  size2 size,
  ptrdiff2 stride,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
  unsigned long long int* max_offset,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
,
  const ::sycl::nd_item<1> &item_ct1)
{
  const size_t chunk_idx = item_ct1.get_global_linear_id();

  // number of zfp blocks
  const size_t bx = (size.x() + 3) / 4;
  const size_t by = (size.y() + 3) / 4;
  const size_t blocks = bx * by;

  // first and last zfp block assigned to thread
  size_t block_idx = chunk_idx * granularity;
  const size_t block_end =
      ::sycl::min((size_t)(block_idx + granularity), blocks);

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // compute bit offset to compressed block
  unsigned long long int bit_offset;
  if (minbits == maxbits)
    bit_offset = chunk_idx * maxbits;
  else
    bit_offset = block_offset(d_index, index_type, chunk_idx, item_ct1);
  BlockReader reader(d_stream, bit_offset);

  // decode blocks assigned to this thread 
  //TODO:FIX ME
  for (; block_idx < block_end; block_idx++) {
    /*Inplace<*/Scalar/*>*/ fblock[ZFP_2D_BLOCK_SIZE] = { 0 };
    decode_block</*Inplace<*/Scalar/*>*/, ZFP_2D_BLOCK_SIZE>()(fblock, reader, minbits, maxbits,
                                              maxprec, minexp);

    // logical position in 2d array
    size_t pos = block_idx;
    const ptrdiff_t x = (pos % bx) * 4; pos /= bx;
    const ptrdiff_t y = (pos % by) * 4; pos /= by;

    // offset into field
    const ptrdiff_t data_offset = x * stride.x() + y * stride.y();

    // scatter data from contiguous block
    const uint nx = (uint)::sycl::min(size_t(size.x() - x), size_t(4));
    const uint ny = (uint)::sycl::min(size_t(size.y() - y), size_t(4));
    if (nx * ny < ZFP_2D_BLOCK_SIZE)
      scatter_partial2((Scalar*)fblock, d_data + data_offset, nx, ny, stride.x(), stride.y());
    else
      scatter2((Scalar*)fblock, d_data + data_offset, stride.x(), stride.y());
  }

  if(block_idx == blocks)
    *max_offset = reader.rtell();
}

// launch decode kernel
template <typename Scalar, int SgSize>
unsigned long long
decode2(Scalar *d_data, const size_t size[], const ptrdiff_t stride[],
        const zfp_exec_params_sycl *params, const Word *d_stream, uint minbits,
        uint maxbits, uint maxprec, int minexp, const Word *d_index,
        zfp_index_type index_type, uint granularity)
{
  ::sycl::queue q(zfp::sycl::internal::zfp_dev_selector
#ifdef ZFP_WITH_SYCL_PROFILE
  , ::sycl::property_list{::sycl::property::queue::enable_profiling()}
#endif
  );
  // block size is fixed to 32 in this version for hybrid index
  const int sycl_block_size = SgSize;

  // number of zfp blocks to decode
  const size_t blocks = ((size[0] + 3) / 4) *
                        ((size[1] + 3) / 4);

  // number of chunks of blocks
  const size_t chunks = (blocks + granularity - 1) / granularity;

  // determine execution range for sycl kernel
  auto kernel_range = calculate_kernel_size(params, chunks, sycl_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* offset;
  offset = (unsigned long long int*)::sycl::malloc_shared(
      sizeof(*offset), q);
  offset[0] = 0;

  // launch GPU kernel
  /*DPCT1049:17: Resolved*/
  auto kernel = q.submit([&](::sycl::handler& cgh) {

    auto data_size = 
      make_size2(size[0], size[1]);
    auto data_stride = 
      make_ptrdiff2(stride[0], stride[1]);

    cgh.parallel_for(kernel_range,
      [=](::sycl::nd_item<1> item_ct1)
      [[intel::reqd_sub_group_size(SgSize)]] {
        decode2_kernel<Scalar>(
          d_data, data_size, data_stride, 
          d_stream, minbits, maxbits, maxprec, 
          minexp, offset, d_index, index_type,
          granularity, item_ct1);
      });
    });
  kernel.wait();
  #ifdef ZFP_WITH_SYCL_PROFILE
    Timer::print_throughput<Scalar>(kernel, "Decode", "decode2",
                                  ::sycl::range<2>(size[0], size[1]));
  #endif
  
  return *offset;
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
