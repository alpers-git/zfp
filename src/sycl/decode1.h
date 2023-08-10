#include <CL/sycl.hpp>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef ZFP_SYCL_DECODE1_H
#define ZFP_SYCL_DECODE1_H

namespace zfp {
namespace sycl {
namespace internal {

template <typename Scalar>
inline 
void scatter1(const Scalar* q, Scalar* p, ptrdiff_t sx)
{
  for (uint x = 0; x < 4; x++, p += sx)
    *p = *q++;
}

template <typename Scalar>
inline 
void scatter_partial1(const Scalar* q, Scalar* p, uint nx, ptrdiff_t sx)
{
  for (uint x = 0; x < 4; x++)
    if (x < nx)
      p[x * sx] = q[x];
}

// decode kernel
template <typename Scalar>

void
decode1_kernel(
  Scalar* d_data,
  size_t size,
  ptrdiff_t stride,
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
  const ::sycl::nd_item<3> &item_ct1,
  unsigned char *perm_1,
  unsigned char *perm_2,
  unsigned char *perm_3,
  uint64 *offset)
{
  const size_t blockId =
      item_ct1.get_group(2) +
      (size_t)item_ct1.get_group_range(2) *
          (item_ct1.get_group(1) +
           (size_t)item_ct1.get_group_range(1) * item_ct1.get_group(0));
  const size_t chunk_idx =
      item_ct1.get_local_id(2) + item_ct1.get_local_range(2) * blockId;

  // number of zfp blocks
  const size_t blocks = (size + 3) / 4;

  // first and last zfp block assigned to thread
  size_t block_idx = chunk_idx * granularity;
  const size_t block_end =
      ::sycl::min((const size_t)(block_idx + granularity), blocks);

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
  for (; block_idx < block_end; block_idx++) {
    Scalar fblock[ZFP_1D_BLOCK_SIZE] = { 0 };
    decode_block<Scalar, ZFP_1D_BLOCK_SIZE>()(fblock, reader, minbits, maxbits,
                                              maxprec, minexp, perm_1, perm_2,
                                              perm_3);

    // logical position in 1d array
    size_t pos = block_idx;
    const ptrdiff_t x = pos * 4;

    // offset into field
    const ptrdiff_t offset = x * stride;

    // scatter data from contiguous block
    const uint nx = (uint)::sycl::min(size_t(size - x), size_t(4));
    if (nx < ZFP_1D_BLOCK_SIZE)
      scatter_partial1(fblock, d_data + offset, nx, stride);
    else
      scatter1(fblock, d_data + offset, stride);
  }

  // record maximum bit offset reached by any thread
  bit_offset = reader.rtell();
  dpct::atomic_fetch_max<::sycl::access::address_space::generic_space>(
      max_offset, bit_offset);
}

// launch decode kernel
template <typename Scalar>
unsigned long long
decode1(Scalar *d_data, const size_t size[], const ptrdiff_t stride[],
        const zfp_exec_params_sycl *params, const Word *d_stream, uint minbits,
        uint maxbits, uint maxprec, int minexp, const Word *d_index,
        zfp_index_type index_type, uint granularity) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  ::sycl::queue &q_ct1 = dev_ct1.default_queue();
  // block size is fixed to 32 in this version for hybrid index
  const int SYCL_block_size = 32;
  const ::sycl::range<3> block_size = ::sycl::range<3>(1, 1, SYCL_block_size);

  // number of zfp blocks to decode
  const size_t blocks = (size[0] + 3) / 4;

  // number of chunks of blocks
  const size_t chunks = (blocks + granularity - 1) / granularity;

  // determine grid of thread blocks
  const ::sycl::range<3> grid_size =
      calculate_grid_size(params, chunks, SYCL_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* d_offset;
  /*
  DPCT1003:33: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  if ((d_offset =
           (unsigned long long *)::sycl::malloc_device(sizeof(*d_offset), q_ct1),
       0) != 0)
    return 0;
  q_ct1.memset(d_offset, 0, sizeof(*d_offset)).wait();

#ifdef ZFP_WITH_SYCL_PROFILE
  Timer timer;
  timer.start();
#endif

  // launch GPU kernel
  /*
  DPCT1049:15: The work-group size passed to the SYCL kernel may exceed the
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

    ::sycl::local_accessor<uint64, 1> offset_acc_ct1(::sycl::range<1>(32), cgh);

    auto size_ct1 = size[0];
    auto stride_ct2 = stride[0];

    cgh.parallel_for(::sycl::nd_range<3>(grid_size * block_size, block_size),
                     [=](::sycl::nd_item<3> item_ct1) {
                       decode1_kernel<Scalar>(
                           d_data, size_ct1, stride_ct2, d_stream, minbits,
                           maxbits, maxprec, minexp, d_offset, d_index,
                           index_type, granularity, item_ct1, perm_1_ptr_ct1,
                           perm_2_ptr_ct1, perm_3_ptr_ct1,
                           offset_acc_ct1.get_pointer());
                     });
  });

#ifdef ZFP_WITH_SYCL_PROFILE
  timer.stop();
  timer.print_throughput<Scalar>("Decode", "decode1", dim3(size[0]));
#endif

  // copy bit offset
  unsigned long long int offset;
  q_ct1.memcpy(&offset, d_offset, sizeof(offset)).wait();
  ::sycl::free(d_offset, q_ct1);

  return offset;
}
catch (::sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

} // namespace internal
} // namespace hip
} // namespace zfp

#endif