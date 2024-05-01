#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef ZFP_SYCL_ENCODE3_H
#define ZFP_SYCL_ENCODE3_H

namespace zfp {
namespace sycl {
namespace internal {

template <typename Scalar>
inline 
void gather3(Scalar* q, const Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
{
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx)
        *q++ = *p;
}

template <typename Scalar>
inline 
void gather_partial3(Scalar* q, const Scalar* p, uint nx, uint ny, uint nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
{
  for (uint z = 0; z < 4; z++)
    if (z < nz) {
      for (uint y = 0; y < 4; y++)
        if (y < ny) {
          for (uint x = 0; x < 4; x++)
            if (x < nx) {
              q[16 * z + 4 * y + x] = *p;
              p += sx;
            }
          p += sy - (ptrdiff_t)nx * sx;
          pad_block(q + 16 * z + 4 * y, nx, 1);
        }
      for (uint x = 0; x < 4; x++)
        pad_block(q + 16 * z + x, ny, 4);
      p += sz - (ptrdiff_t)ny * sy;
    }
  for (uint y = 0; y < 4; y++)
    for (uint x = 0; x < 4; x++)
      pad_block(q + 4 * y + x, nz, 16);
}

// encode kernel
template <typename Scalar>

// avoid register spillage
void
encode3_kernel(
  const Scalar* d_data, // field data device pointer
  size3 size,           // field dimensions
  ptrdiff3 stride,      // field strides
  Word* d_stream,       // compressed bit stream device pointer
  ushort* d_index,      // block index
  uint minbits,         // min compressed #bits/block
  uint maxbits,         // max compressed #bits/block
  uint maxprec,         // max uncompressed #bits/value
  int minexp,           // min bit plane index
  const ::sycl::nd_item<3> &item_ct1,
  unsigned char *perm)
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
  const size_t bx = (size.x() + 3) / 4;
  const size_t by = (size.y() + 3) / 4;
  const size_t bz = (size.z() + 3) / 4;
  const size_t blocks = bx * by * bz;

  // return if thread has no blocks assigned
  if (block_idx >= blocks)
    return;

  // logical position in 2d array
  size_t pos = block_idx;
  const ptrdiff_t x = (pos % bx) * 4; pos /= bx;
  const ptrdiff_t y = (pos % by) * 4; pos /= by;
  const ptrdiff_t z = (pos % bz) * 4; pos /= bz;

  // offset into field
  const ptrdiff_t offset = x * stride.x() + y * stride.y() + z * stride.z();

  // initialize block writer
  BlockWriter::Offset bit_offset = block_idx * maxbits;
  BlockWriter writer(d_stream, bit_offset);

  // gather data into a contiguous block
  Scalar fblock[ZFP_3D_BLOCK_SIZE];
  const uint nx = (uint)::sycl::min(size_t(size.x() - x), size_t(4));
  const uint ny = (uint)::sycl::min(size_t(size.y() - y), size_t(4));
  const uint nz = (uint)::sycl::min(size_t(size.z() - z), size_t(4));
  if (nx * ny * nz < ZFP_3D_BLOCK_SIZE)
    gather_partial3(fblock, d_data + offset, nx, ny, nz, stride.x(), stride.y(),
                    stride.z());
  else
    gather3(fblock, d_data + offset, stride.x(), stride.y(), stride.z());

  uint bits = encode_block<Scalar, ZFP_3D_BLOCK_SIZE>()(
      fblock, writer, minbits, maxbits, maxprec, minexp, perm);

  if (d_index)
    d_index[block_idx] = (ushort)bits;
}

// launch encode kernel
template <typename Scalar>
unsigned long long
encode3(
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
  ::sycl::queue q(zfp::sycl::internal::zfp_dev_selector);
  const int sycl_block_size = 128;
  const ::sycl::range<3> block_size = ::sycl::range<3>(1, 1, sycl_block_size);

  // number of zfp blocks to encode
  const size_t blocks = ((size[0] + 3) / 4) *
                        ((size[1] + 3) / 4) *
                        ((size[2] + 3) / 4);

  // determine grid of thread blocks
  const ::sycl::range<3> grid_size =
      calculate_grid_size(params, blocks, sycl_block_size);

  // zero-initialize bit stream (for atomics)
  const size_t stream_bytes = calculate_device_memory(blocks, maxbits);
  auto e1 = q.memset(d_stream, 0, stream_bytes);

#ifdef ZFP_WITH_SYCL_PROFILE
  Timer timer;
  timer.start();
  ::sycl::event e = 
#endif

  // launch GPU kernel
  /*
  DPCT1049:3: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  unsigned char* perm_3_data = ::sycl::malloc_shared<unsigned char>(64, q);

  // Initialize perm_3 data
  //memcpy(perm_3_data, perm_3, 64 * sizeof(unsigned char));
  auto e2 = q.memcpy(perm_3_data, perm_3, 64 * sizeof(unsigned char));

  q.submit([&](::sycl::handler &cgh) {
    cgh.depends_on({e1,e2});

    auto make_size3_size_size_size_ct1 = make_size3(size[0], size[1], size[2]);
    auto make_ptrdiff3_stride_stride_stride_ct2 =
        make_ptrdiff3(stride[0], stride[1], stride[2]);

    cgh.parallel_for(::sycl::nd_range<3>(grid_size * block_size, block_size),
                     [=](::sycl::nd_item<3> item_ct1) {
                       encode3_kernel<Scalar>(
                           d_data, make_size3_size_size_size_ct1,
                           make_ptrdiff3_stride_stride_stride_ct2, d_stream,
                           d_index, minbits, maxbits, maxprec, minexp,
                          item_ct1, perm_3_data);
                     });
  }).wait();

  ::sycl::free(perm_3_data, q);

#ifdef ZFP_WITH_SYCL_PROFILE
  e.wait();
  timer.stop();
  timer.print_throughput<Scalar>("Encode", "encode3", range<3>(size[0], size[1], size[2]));
#endif

  return (unsigned long long)stream_bytes * CHAR_BIT;
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
