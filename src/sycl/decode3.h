#include <sycl/sycl.hpp>
#ifndef ZFP_SYCL_DECODE3_H
#define ZFP_SYCL_DECODE3_H

namespace zfp {
namespace sycl {
namespace internal {

template <typename Scalar>
inline 
void scatter3(const Scalar* q, Scalar* p, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
{
  for (uint z = 0; z < 4; z++, p += sz - 4 * sy)
    for (uint y = 0; y < 4; y++, p += sy - 4 * sx)
      for (uint x = 0; x < 4; x++, p += sx)
        *p = *q++;
}

template <typename Scalar>
inline 
void scatter_partial3(const Scalar* q, Scalar* p, uint nx, uint ny, uint nz, ptrdiff_t sx, ptrdiff_t sy, ptrdiff_t sz)
{
  for (uint z = 0; z < 4; z++)
    if (z < nz) {
      for (uint y = 0; y < 4; y++)
        if (y < ny) {
          for (uint x = 0; x < 4; x++)
            if (x < nx) {
              *p = q[x + 4 * y + 16 * z];
              p += sx;
            }
          p += sy - nx * sx;
        }
      p += sz - ny * sy;
    }
}

// decode kernel
template <typename Scalar>

// avoid register spillage
void
decode3_kernel(
  Scalar* d_data,
  size3 size,
  ptrdiff3 stride,
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
  const ::sycl::nd_item<1> &item_ct1,
  ::sycl::local_accessor<uint64, 1> offset)
{
  const size_t chunk_idx = item_ct1.get_global_linear_id();

  // number of zfp blocks
  const size_t bx = (size.x() + 3) / 4;
  const size_t by = (size.y() + 3) / 4;
  const size_t bz = (size.z() + 3) / 4;
  const size_t blocks = bx * by * bz;

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
    bit_offset = block_offset(d_index, index_type, chunk_idx, item_ct1, offset);
  BlockReader reader(d_stream, bit_offset);

  // decode blocks assigned to this thread
  for (; block_idx < block_end; block_idx++) {
    Scalar fblock[ZFP_3D_BLOCK_SIZE] = { 0 };
    decode_block<Scalar, ZFP_3D_BLOCK_SIZE>()(fblock, reader, minbits, maxbits,
                                              maxprec, minexp);

    // logical position in 3d array
    size_t pos = block_idx;
    const ptrdiff_t x = (pos % bx) * 4; pos /= bx;
    const ptrdiff_t y = (pos % by) * 4; pos /= by;
    const ptrdiff_t z = (pos % bz) * 4; pos /= bz;

    // offset into field
    const ptrdiff_t data_offset = x * stride.x() + y * stride.y() + z * stride.z();

    // scatter data from contiguous block
    const uint nx = (uint)::sycl::min(size_t(size.x() - x), size_t(4));
    const uint ny = (uint)::sycl::min(size_t(size.y() - y), size_t(4));
    const uint nz = (uint)::sycl::min(size_t(size.z() - z), size_t(4));
    if (nx * ny * nz < ZFP_3D_BLOCK_SIZE)
      scatter_partial3(fblock, d_data + data_offset, nx, ny, nz, stride.x(),
                       stride.y(), stride.z());
    else
      scatter3(fblock, d_data + data_offset, stride.x(), stride.y(), stride.z());
  }

  // record maximum bit offset reached by any thread
  if(block_idx == blocks)
    *max_offset = reader.rtell();
}

// launch decode kernel
template <typename Scalar>
unsigned long long
decode3(Scalar *d_data, const size_t size[], const ptrdiff_t stride[],
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
  const int sycl_block_size = 32;

  // number of zfp blocks to decode
  const size_t blocks = ((size[0] + 3) / 4) *
                        ((size[1] + 3) / 4) *
                        ((size[2] + 3) / 4);

  // number of chunks of blocks
  const size_t chunks = (blocks + granularity - 1) / granularity;

  // determine execution range for sycl kernel
  auto kernel_range = calculate_kernel_size(params, chunks, sycl_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* offset;
  offset = (unsigned long long int*)::sycl::malloc_shared(
      sizeof(*offset), q);


  // launch GPU kernel
  /*DPCT1049:17: Resolved*/
  auto kernel = q.submit([&](::sycl::handler& cgh) {
    ::sycl::local_accessor<uint64, 1> offset_acc_ct1(::sycl::range<1>(32), cgh);

    auto data_dims = 
      make_size3(size[0], size[1], size[2]);
    auto data_stride =
      make_ptrdiff3(stride[0], stride[1], stride[2]);

    cgh.parallel_for(kernel_range,
      [=](::sycl::nd_item<1> item_ct1) {
        decode3_kernel<Scalar>(
          d_data, data_dims, data_stride, 
          d_stream, minbits, maxbits, maxprec, 
          minexp, offset, d_index, index_type,
          granularity, item_ct1, offset_acc_ct1);
      });
    });
  kernel.wait();
#ifdef ZFP_WITH_SYCL_PROFILE
  Timer::print_throughput<Scalar>(kernel, "Decode", "decode3",
    ::sycl::range<3>(size[0], size[1], size[2]));
#endif

  return *offset;
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
