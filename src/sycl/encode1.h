#include <sycl/sycl.hpp>
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
  int minexp,           // min bit plane index
  const ::sycl::nd_item<1> &item_ct1
  //::sycl::stream os
)
{
  const size_t blockId = item_ct1.get_group(0);

  // each thread gets a block; block index = global thread index
  const size_t block_idx = blockId * item_ct1.get_local_range(0) + item_ct1.get_local_id(0);

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
      fblock, writer, minbits, maxbits, maxprec, minexp);
  //* fblock checked: no problem?
  //* x and offset checked: no problem
  //* perm checked: no problem
  //* bits checked: no problem
  //* writer checked: no problem?
  //! d_stream checked: incorrect values
  // os << "<"<< blockId << "," << block_idx << ">:" << d_stream[0] << " "<<
  //  d_stream[1] << " " << d_stream[2] << " " << d_stream[3] << ::sycl::endl;
  // static const CONSTANT char FMT[] = "<%d,%d>:%llu %llu %llu %llu\n";
  // ::sycl::ext::oneapi::experimental::printf(FMT, blockId, block_idx,
  //   d_stream[0], d_stream[1], d_stream[2], d_stream[3]);

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
  ::sycl::queue q(zfp::sycl::internal::zfp_dev_selector
#ifdef ZFP_WITH_SYCL_PROFILE
  , ::sycl::property_list{::sycl::property::queue::enable_profiling()}
#endif
  );
  const int sycl_block_size = 128;

  // number of zfp blocks to encode
  const size_t blocks = (size[0] + 3) / 4;

  // determine execution range for sycl kernel
  auto kernel_range = calculate_kernel_size(params, blocks, sycl_block_size);

  // zero-initialize bit stream (for atomics)
  const size_t stream_bytes = calculate_device_memory(blocks, maxbits);
  //std::memset(d_stream, 0, stream_bytes);
  auto e1 = q.memset(d_stream, 0, stream_bytes);

  // launch GPU kernel
  /*
  DPCT1049:1: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  
  //* size, stride, minbits, maxbits, maxprec, minexp, stream_bytes, blocks checked: no problem
  //* d_data and d_stream checked: no problem
  auto kernel = q.submit([&](::sycl::handler &cgh) {
    cgh.depends_on({e1});
    auto size_ct1 = size[0];
    auto stride_ct2 = stride[0];

    cgh.parallel_for(kernel_range,
                     [=](::sycl::nd_item<1> item_ct1) {
                       encode1_kernel<Scalar>(
                           d_data, size_ct1, stride_ct2, d_stream, d_index,
                           minbits, maxbits, maxprec, minexp,
                           item_ct1);
                     });
  });
  kernel.wait();
#ifdef ZFP_WITH_SYCL_PROFILE
  Timer::print_throughput<Scalar>(kernel, "Encode", "encode1",
                                 ::sycl::range<1>(size[0]));
#endif

  //* d_data checked: no problem
  //! d_stream checked: incorrect values for CPU

  return (unsigned long long)stream_bytes * CHAR_BIT;
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif