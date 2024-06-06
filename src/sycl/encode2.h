#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef ZFP_SYCL_ENCODE2_H
#define ZFP_SYCL_ENCODE2_H

namespace zfp {
namespace sycl {
namespace internal {

template <typename Scalar>
inline 
void gather2(Scalar* q, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy)
{
  for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
    for (uint x = 0; x < 4; x++, p += sx)
      *q++ = *p;
}

template <typename Scalar>
inline 
void gather_partial2(Scalar* q, const Scalar* p, uint nx, uint ny, ptrdiff_t sx, ptrdiff_t sy)
{
  for (uint y = 0; y < 4; y++)
    if (y < ny) {
      for (uint x = 0; x < 4; x++)
        if (x < nx) {
          q[4 * y + x] = *p;
          p += sx;
        }
      pad_block(q + 4 * y, nx, 1);
      p += sy - (ptrdiff_t)nx * sx;
    }
  for (uint x = 0; x < 4; x++)
    pad_block(q + x, ny, 4);
}

// encode kernel
template <typename Scalar>

void
encode2_kernel(
  const Scalar* d_data, // field data device pointer
  size2 size,           // field dimensions
  ptrdiff2 stride,      // field strides
  Word* d_stream,       // compressed bit stream device pointer
  ushort* d_index,      // block index
  uint minbits,         // min compressed #bits/block
  uint maxbits,         // max compressed #bits/block
  uint maxprec,         // max uncompressed #bits/value
  int minexp,           // min bit plane index

  const ::sycl::nd_item<1> &item_ct1,
  ::sycl::local_accessor<Scalar, 1> fblock)
{
  // each thread gets a block; block index = global thread index
  const size_t block_idx = item_ct1.get_global_linear_id();

  // number of zfp blocks
  const size_t bx = (size.x() + 3) / 4;
  const size_t by = (size.y() + 3) / 4;
  const size_t blocks = bx * by;

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // logical position in 2d array
  size_t pos = block_idx;
  const ptrdiff_t x = (pos % bx) * 4; pos /= bx;
  const ptrdiff_t y = (pos % by) * 4; pos /= by;

  // offset into field
  const ptrdiff_t offset = x * stride.x() + y * stride.y();

  // initialize block writer
  BlockWriter::Offset bit_offset = block_idx * maxbits;
  BlockWriter writer(d_stream, bit_offset);

  // gather data into a contiguous block
  const size_t fblock_offset = item_ct1.get_local_linear_id() * ZFP_2D_BLOCK_SIZE; //to use SLM
  Scalar* fblock_ptr = fblock.get_pointer() + fblock_offset;
  const uint nx = (uint)::sycl::min(size_t(size.x() - x), size_t(4));
  const uint ny = (uint)::sycl::min(size_t(size.y() - y), size_t(4));
  if (nx * ny < ZFP_2D_BLOCK_SIZE)
    gather_partial2(fblock_ptr, d_data + offset, nx, ny, stride.x(), stride.y());
  else
    gather2(fblock_ptr, d_data + offset, stride.x(), stride.y());

  //set cache for block
  fblock_ptr = fblock.get_pointer() + fblock_offset;

  uint bits = encode_block<Scalar, ZFP_2D_BLOCK_SIZE>()(
      fblock_ptr, writer, minbits, maxbits, maxprec, minexp);

  if (d_index)
    d_index[block_idx] = (ushort)bits;
}

// launch encode kernel
template <typename Scalar>
unsigned long long
encode2(
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
  const size_t blocks = ((size[0] + 3) / 4) *
                        ((size[1] + 3) / 4);

  // determine execution range for sycl kernel
  auto kernel_range = calculate_kernel_size(params, blocks, sycl_block_size);

  // zero-initialize bit stream (for atomics)
  const size_t stream_bytes = calculate_device_memory(blocks, maxbits);
  auto e1 = q.memset(d_stream, 0, stream_bytes);

  // launch GPU kernel
  /*
  DPCT1049:2: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */

  auto kernel = q.submit([&](::sycl::handler &cgh) {
    cgh.depends_on({e1});

    ::sycl::local_accessor<Scalar, 1> fblock_slm(::sycl::range<1>(sycl_block_size * ZFP_2D_BLOCK_SIZE), cgh);

    auto make_size2_size_size_ct1 = make_size2(size[0], size[1]);
    auto make_ptrdiff2_stride_stride_ct2 = make_ptrdiff2(stride[0], stride[1]);

    cgh.parallel_for(kernel_range,
                     [=](::sycl::nd_item<1> item_ct1) {
                       encode2_kernel<Scalar>(
                           d_data, make_size2_size_size_ct1,
                           make_ptrdiff2_stride_stride_ct2, d_stream, d_index,
                           minbits, maxbits, maxprec, minexp,
                           item_ct1, fblock_slm);
                     });
  });
  kernel.wait();
#ifdef ZFP_WITH_SYCL_PROFILE
  Timer::print_throughput<Scalar>(kernel, "Encode", "encode2", range<2>(size[0], size[1]));
#endif

  return (unsigned long long)stream_bytes * CHAR_BIT;
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
