#include <CL/sycl.hpp>
#ifndef ZFP_SYCL_ENCODE1_H
#define ZFP_SYCL_ENCODE1_H

namespace zfp {
namespace sycl {
namespace internal {
    
template <typename Scalar>
inline 
void gather1(Scalar* q, const Scalar* p, ptrdiff_t sx)
{
  for (uint x = 0; x < 4; x++, p += sx)
    *q++ = *p;
}

template <typename Scalar>
inline 
void gather_partial1(Scalar* q, const Scalar* p, uint nx, ptrdiff_t sx)
{
  for (uint x = 0; x < 4; x++)
    if (x < nx)
      q[x] = p[x * sx];
  pad_block(q, nx, 1);
}

// encode kernel
template <typename Scalar>

void encode1_kernel(
  const Scalar* d_data, // field data device pointer
  size_t size,          // field dimensions
  ptrdiff_t stride,     // field stride
  Word* d_stream,       // compressed bit stream device pointer
  ushort* d_index,      // block index
  uint minbits,         // min compressed #bits/block
  uint maxbits,         // max compressed #bits/block
  uint maxprec,         // max uncompressed #bits/value
  int minexp            ,
  const ::sycl::nd_item<3> &item_ct1,
  unsigned char *perm_1,
  unsigned char *perm_2,
  unsigned char *perm_3// min bit plane index
)
{
  const size_t blockId =
      item_ct1.get_group(2) +
      (size_t)item_ct1.get_group_range(2) *
          (item_ct1.get_group(1) +
           (size_t)item_ct1.get_group_range(1) * item_ct1.get_group(0));

  // each thread gets a block; block index = global thread index
  const size_t block_idx =
      blockId * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

  // number of zfp blocks
  const size_t blocks = (size + 3) / 4;

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // logical position in 1d array
  const size_t pos = block_idx;
  const ptrdiff_t x = pos * 4;

  // offset into field
  const ptrdiff_t offset = x * stride;

  // initialize block writer
  BlockWriter::Offset bit_offset = block_idx * maxbits;
  BlockWriter writer(d_stream, bit_offset);

  // gather data into a contiguous block
  Scalar fblock[ZFP_1D_BLOCK_SIZE];
  const uint nx = (uint)::sycl::min(size_t(size - x), size_t(4));
  if (nx < ZFP_1D_BLOCK_SIZE)
    gather_partial1(fblock, d_data + offset, nx, stride);
  else
    gather1(fblock, d_data + offset, stride);

  uint bits = encode_block<Scalar, ZFP_1D_BLOCK_SIZE>()(
      fblock, writer, minbits, maxbits, maxprec, minexp, perm_1, perm_2,
      perm_3);

  if (d_index)
    d_index[block_idx] = (ushort)bits;
}

// launch encode kernel
template <typename Scalar>
unsigned long long
encode1(
  const Scalar* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_sycl* params,
  Word* d_stream,
  ushort* d_index,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  ::sycl::queue &q_ct1 = dev_ct1.default_queue();
  const int sycl_block_size = 128;
  const ::sycl::range<3> block_size = ::sycl::range<3>(1, 1, sycl_block_size);

  // number of zfp blocks to encode
  const size_t blocks = (size[0] + 3) / 4;

  // determine grid of thread blocks
  const ::sycl::range<3> grid_size =
      calculate_grid_size(params, blocks, sycl_block_size);

  // zero-initialize bit stream (for atomics)
  const size_t stream_bytes = calculate_device_memory(blocks, maxbits);
  q_ct1.memset(d_stream, 0, stream_bytes).wait();

#ifdef ZFP_WITH_SYCL_PROFILE
  Timer timer;
  timer.start();
#endif

  // launch GPU kernel
  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  q_ct1.submit([&](::sycl::handler &cgh) {
    extern dpct::global_memory<const unsigned char, 1> perm_1;
    extern dpct::global_memory<const unsigned char, 1> perm_2;
    extern dpct::global_memory<const unsigned char, 1> perm_3;

    perm_1.init();
    perm_2.init();
    perm_3.init();

    auto perm_1_ptr_ct1 = perm_1.get_ptr();
    auto perm_2_ptr_ct1 = perm_2.get_ptr();
    auto perm_3_ptr_ct1 = perm_3.get_ptr();

    auto size_ct1 = size[0];
    auto stride_ct2 = stride[0];

    cgh.parallel_for(::sycl::nd_range<3>(grid_size * block_size, block_size),
                     [=](::sycl::nd_item<3> item_ct1) {
                       encode1_kernel<Scalar>(
                           d_data, size_ct1, stride_ct2, d_stream, d_index,
                           minbits, maxbits, maxprec, minexp, item_ct1,
                           perm_1_ptr_ct1, perm_2_ptr_ct1, perm_3_ptr_ct1);
                     });
  });

#ifdef ZFP_WITH_SYCL_PROFILE
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode1", dim3(size[0]));
#endif

  return (unsigned long long)stream_bytes * CHAR_BIT;
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif