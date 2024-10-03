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

template <typename Scalar, int BlockSize>
inline 
void scatter3(const SplitMem<Inplace<Scalar>, BlockSize> q, Scalar* const p, const uint nx, 
    const uint ny, const uint nz, 
    const ptrdiff_t sx, const ptrdiff_t sy, const ptrdiff_t sz)
{
    for (uint z = 0; z < nz; z++) {
        for (uint y = 0; y < ny; y++) {
            for (uint x = 0; x < nx; x++) {
                p[x*sx + y*sy + z*sz] = q[x + 4 * y + 16 * z].scalar;
            }
        }
    }
}

// decode kernel
template <typename Scalar>
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
  const ::sycl::nd_item<1> &item_ct1)
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
    bit_offset = block_offset(d_index, index_type, chunk_idx, item_ct1);
  BlockReader reader(d_stream, bit_offset);

  // decode blocks assigned to this thread
  for (; block_idx < block_end; block_idx++) {
    Inplace<Scalar> fblock[ZFP_3D_BLOCK_SIZE] = { 0 };
    decode_block<Inplace<Scalar>, ZFP_3D_BLOCK_SIZE>()(fblock, reader, minbits, maxbits,
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
      scatter_partial3((Scalar*)fblock, d_data + data_offset, nx, ny, nz, stride.x(),
                       stride.y(), stride.z());
    else
      scatter3((Scalar*)fblock, d_data + data_offset, stride.x(), stride.y(), stride.z());
  }

  // record maximum bit offset reached by any thread
  if(block_idx == blocks)
    *max_offset = reader.rtell();
}

// decode kernel register spill optimized
template <typename Scalar>
void
decode3_kernel(
  Scalar* d_data,
  const size3 size,
  const ptrdiff3 stride,
  int3 b,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
  unsigned long long int* max_offset,
  const Word* d_index,
  zfp_index_type index_type
  //uint granularity
,
  const ::sycl::nd_item<1> &item_ct1)
{
  #ifdef __SYCL_DEVICE_ONLY__
    const int block_idx = item_ct1.get_global_linear_id();

    // number of zfp blocks. Since each block holds 64 values, 
    // having more than 2147483647 blocks (max int value) does not make sense.
    // since it would be ~64*2*10^9*sizeof(type) bytes (even for 1 byte types that is 128GB)
    // We therefore stick to int for block_idx and blocks
    const int blocks = b.x() * b.y() * b.z();

    // return if thread has no blocks assigned
    if (block_idx >= blocks)
        return;

    // compute bit offset to compressed block
    unsigned long long int bit_offset;
    if (minbits == maxbits)
        bit_offset = block_idx * maxbits;
    else if (index_type == zfp_index_offset)
        bit_offset = d_index[block_idx];
    else 
        bit_offset = block_offset(d_index, index_type, block_idx, item_ct1);
        
    // logical position in 3d array
    const int x = (block_idx % b.x()) * 4; 
    const int y = ((block_idx/b.x()) % b.y()) * 4; 
    const int z = (block_idx/(b.x()*b.y()) % b.z()) * 4;

    // offset into field
    const ptrdiff_t data_offset = x * stride.x() + y * stride.y() + z * stride.z();
    auto d_data_offset = d_data + data_offset;

    // scatter data from contiguous block
    const uint nx = std::min<uint>(size.x() - x, 4);
    const uint ny = std::min<uint>(size.y() - y, 4);
    const uint nz = std::min<uint>(size.z() - z, 4);

    BlockReader reader(d_stream, bit_offset);
    //std::array<ScalarUnion<Scalar>, ZFP_3D_BLOCK_SIZE> fblock = {(Scalar)0};
    
    //offset slm for each thread, +1 to avoid bank conflicts
    //ScalarUnion<Scalar>* const fblock = &fblock_slm[item_ct1.get_sub_group().get_local_linear_id() * ZFP_3D_BLOCK_SIZE];
    SplitMem<Inplace<Scalar>, ZFP_3D_BLOCK_SIZE> fblock;
    for (int iter = 0; iter < ZFP_3D_BLOCK_SIZE; iter++) {
        fblock[iter].scalar = (Scalar)0;
    }

    decode_block<SplitMem<Inplace<Scalar>, ZFP_3D_BLOCK_SIZE>, ZFP_3D_BLOCK_SIZE>()(fblock, reader, minbits, maxbits,
                                              maxprec, minexp);

    
    scatter3(fblock, d_data_offset, nx, ny, nz, stride.x(), stride.y(), stride.z());

    // record maximum bit offset reached by any thread
    if(block_idx == blocks-1) max_offset[0] = reader.rtell();
#endif
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
  const int sycl_block_size = 512;

  const int3 b = make_int3((size[0] + 3) / 4, 
                            (size[1] + 3) / 4, 
                            (size[2] + 3) / 4);
  // number of zfp blocks to decode
  const size_t blocks = (size_t) b.x() * b.y() * b.z();

  // number of chunks of blocks
  const size_t chunks = (blocks + granularity - 1) / granularity;

  // determine execution range for sycl kernel
  auto kernel_range = calculate_kernel_size(params, chunks, sycl_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* offset;
  offset = (unsigned long long int*)::sycl::malloc_shared(
      sizeof(*offset), q);


  // launch GPU kernel
  auto kernel = q.submit([&](::sycl::handler& cgh) {

    auto data_size = 
      make_size3(size[0], size[1], size[2]);
    auto data_stride =
      make_ptrdiff3(stride[0], stride[1], stride[2]);

    cgh.parallel_for(kernel_range,
      [=](::sycl::nd_item<1> item_ct1)
      [[intel::reqd_sub_group_size(SG_SIZE)]] {
        decode3_kernel<Scalar>(
          d_data, data_size, data_stride, b,
          d_stream, minbits, maxbits, maxprec, 
          minexp, offset, d_index, index_type,
          /*granularity,*/ item_ct1);
      });
    });
  kernel.wait();
#ifdef ZFP_WITH_SYCL_PROFILE
  Timer::print_throughput<Scalar>(kernel, "Decode", "decode3",
    ::sycl::range<3>(size[0], size[1], size[2]));
#endif

  return *offset;
}


// launch a different kernel for 65-bit types to reduce reg-spill
template <>
unsigned long long
decode3(double *d_data, const size_t size[], const ptrdiff_t stride[],
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
  const int sycl_block_size = 512;

  const int3 b = make_int3((size[0] + 3) / 4, 
                            (size[1] + 3) / 4, 
                            (size[2] + 3) / 4);
  // number of zfp blocks to decode
  const size_t blocks = (size_t) b.x() * b.y() * b.z();

  // number of chunks of blocks
  const size_t chunks = (blocks + granularity - 1) / granularity;

  // determine execution range for sycl kernel
  auto kernel_range = calculate_kernel_size(params, chunks, sycl_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* offset;
  offset = (unsigned long long int*)::sycl::malloc_shared(
      sizeof(*offset), q);


  // launch GPU kernel
  auto kernel = q.submit([&](::sycl::handler& cgh) {

    auto data_size = 
      make_size3(size[0], size[1], size[2]);
    auto data_stride =
      make_ptrdiff3(stride[0], stride[1], stride[2]);

    cgh.parallel_for(kernel_range,
      [=](::sycl::nd_item<1> item_ct1)
      [[intel::reqd_sub_group_size(SG_SIZE)]] {
        decode3_kernel<double>(
          d_data, data_size, data_stride,
          d_stream, minbits, maxbits, maxprec, 
          minexp, offset, d_index, index_type,
          granularity, item_ct1);
      });
    });
  kernel.wait();
#ifdef ZFP_WITH_SYCL_PROFILE
  Timer::print_throughput<double>(kernel, "Decode", "decode3",
    ::sycl::range<3>(size[0], size[1], size[2]));
#endif

  return *offset;
}

template <>
unsigned long long
decode3(long long int *d_data, const size_t size[], const ptrdiff_t stride[],
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
  const int sycl_block_size = 512;

  const int3 b = make_int3((size[0] + 3) / 4, 
                            (size[1] + 3) / 4, 
                            (size[2] + 3) / 4);
  // number of zfp blocks to decode
  const size_t blocks = (size_t) b.x() * b.y() * b.z();

  // number of chunks of blocks
  const size_t chunks = (blocks + granularity - 1) / granularity;

  // determine execution range for sycl kernel
  auto kernel_range = calculate_kernel_size(params, chunks, sycl_block_size);

  // storage for maximum bit offset; needed to position stream
  unsigned long long int* offset;
  offset = (unsigned long long int*)::sycl::malloc_shared(
      sizeof(*offset), q);


  // launch GPU kernel
  auto kernel = q.submit([&](::sycl::handler& cgh) {

    auto data_size = 
      make_size3(size[0], size[1], size[2]);
    auto data_stride =
      make_ptrdiff3(stride[0], stride[1], stride[2]);

    cgh.parallel_for(kernel_range,
      [=](::sycl::nd_item<1> item_ct1)
      [[intel::reqd_sub_group_size(SG_SIZE)]] {
        decode3_kernel<long long int>(
          d_data, data_size, data_stride,
          d_stream, minbits, maxbits, maxprec, 
          minexp, offset, d_index, index_type,
          granularity, item_ct1);
      });
    });
  kernel.wait();
#ifdef ZFP_WITH_SYCL_PROFILE
  Timer::print_throughput<long long int>(kernel, "Decode", "decode3",
    ::sycl::range<3>(size[0], size[1], size[2]));
#endif

  return *offset;
}


} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
