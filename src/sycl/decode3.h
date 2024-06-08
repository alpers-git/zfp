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
  ::sycl::local_accessor<uint64, 1> offset,
  ::sycl::local_accessor<Scalar, 1> fblock)
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
  unsigned long long bit_offset;
  if (minbits == maxbits)
    bit_offset = chunk_idx * maxbits;
  else
    bit_offset = block_offset(d_index, index_type, chunk_idx, item_ct1, offset);
  BlockReader reader(d_stream, bit_offset);

  // decode blocks assigned to this thread
  const size_t fblock_offset = item_ct1.get_local_linear_id() * ZFP_3D_BLOCK_SIZE;
  Scalar* fblock_ptr = fblock.get_pointer() + fblock_offset;
  for (; block_idx < block_end; block_idx++) {
    // Scalar fblock[ZFP_3D_BLOCK_SIZE] = { 0 };
    decode_block<Scalar, ZFP_3D_BLOCK_SIZE>()(fblock_ptr, reader, minbits, maxbits,
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
      scatter_partial3(fblock_ptr, d_data + data_offset, nx, ny, nz, stride.x(),
                       stride.y(), stride.z());
    else
      scatter3(fblock_ptr, d_data + data_offset, stride.x(), stride.y(), stride.z());
  }

  // record maximum bit offset reached by any thread
  bit_offset = reader.rtell();
  dpct::atomic_fetch_max<::sycl::access::address_space::generic_space>(
      max_offset, bit_offset);
}

// launch decode kernel
template <typename Scalar>
unsigned long long
decode3(Scalar *d_data, const size_t size[], const ptrdiff_t stride[],
        const zfp_exec_params_sycl *params, const Word *d_stream, uint minbits,
        uint maxbits, uint maxprec, int minexp, const Word *d_index,
        zfp_index_type index_type, uint granularity) //try 
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
  auto kernel_range = calculate_kernel_size(params, blocks, sycl_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* d_offset;
  d_offset = (unsigned long long int*)::sycl::malloc_device(
      sizeof(*d_offset), q);
  auto e1 = q.memset(d_offset, 0, sizeof(*d_offset));


  // launch GPU kernel
  /*
  DPCT1049:17: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  auto kernel = q.submit([&](::sycl::handler &cgh) {

    ::sycl::local_accessor<uint64, 1> offset_acc_ct1(::sycl::range<1>(32), cgh);
    ::sycl::local_accessor<Scalar, 1> fblock_slm(::sycl::range<1>(sycl_block_size * ZFP_3D_BLOCK_SIZE), cgh);

    auto make_size3_size_size_size_ct1 = make_size3(size[0], size[1], size[2]);
    auto make_ptrdiff3_stride_stride_stride_ct2 =
        make_ptrdiff3(stride[0], stride[1], stride[2]);

    cgh.depends_on({e1});
    cgh.parallel_for(kernel_range,
                     [=](::sycl::nd_item<1> item_ct1) {
                        decode3_kernel<Scalar>(
                           d_data, make_size3_size_size_size_ct1,
                           make_ptrdiff3_stride_stride_stride_ct2, d_stream,
                           minbits, maxbits, maxprec, minexp, d_offset, d_index,
                           index_type, granularity, item_ct1,
                           offset_acc_ct1, fblock_slm);
                     });
  });
  kernel.wait();
#ifdef ZFP_WITH_SYCL_PROFILE
  Timer::print_throughput<Scalar>(kernel, "Decode", "decode3", range<3>(size[0], size[1], size[2]));
#endif

  // copy bit offset
  unsigned long long int offset;
  q.memcpy(&offset, d_offset, sizeof(offset)).wait();
  ::sycl::free(d_offset, q);

  return offset;
}
// catch (::sycl::exception const &exc) {
//   std::cerr << exc.what() << "Exception caught at file:" << __FILE__
//             << ", line:" << __LINE__ << std::endl;
//   std::exit(1);
// }

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
