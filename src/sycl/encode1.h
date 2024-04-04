#ifndef CUZFP_ENCODE1_CUH
#define CUZFP_ENCODE1_CUH

#define DPCT_PROFILING_ENABLED
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "syclZFP.h"
#include "shared.h"
#include "encode.h"
#include "type_info.h"

#include <iostream>
#define ZFP_1D_BLOCK_SIZE 4 

namespace syclZFP
{

template<typename Scalar> 
inline 
void gather_partial1(Scalar* q, const Scalar* p, int nx, int sx)
{
  uint x;
  for (x = 0; x < 4; x++)
    if (x < nx) q[x] = p[x * sx];
  pad_block(q, nx, 1);
}

template<typename Scalar> 
inline 
void gather1(Scalar* q, const Scalar* p, int sx)
{
  uint x;
  for (x = 0; x < 4; x++, p += sx)
    *q++ = *p;
}

template<class Scalar>

void 
syclEncode1(const uint maxbits,
           const Scalar* scalars,
           Word *stream,
           const uint dim,
           const int sx,
           const uint padded_dim,
           const uint tot_blocks,
           const sycl::nd_item<3> &item_ct1,
           unsigned char *perm_3d,
           unsigned char *perm_1,
           unsigned char *perm_2)
{

  typedef unsigned long long int ull;
  typedef long long int ll;
  const ull blockId = item_ct1.get_group(2) +
                      item_ct1.get_group(1) * item_ct1.get_group_range(2) +
                      item_ct1.get_group_range(2) *
                          item_ct1.get_group_range(1) * item_ct1.get_group(0);

  // each thread gets a block so the block index is 
  // the global thread index
  const uint block_idx =
      blockId * item_ct1.get_local_range(2) + item_ct1.get_local_id(2);

  if(block_idx >= tot_blocks)
  {
    // we can't launch the exact number of blocks
    // so just exit if this isn't real
    return;
  }

  uint block_dim;
  block_dim = padded_dim >> 2; 

  // logical pos in 3d array
  uint block;
  block = (block_idx % block_dim) * 4; 

  const ll offset = (ll)block * sx; 

  Scalar fblock[ZFP_1D_BLOCK_SIZE]; 

  bool partial = false;
  if(block + 4 > dim) partial = true;
 
  if(partial) 
  {
    uint nx = 4 - (padded_dim - dim);
    gather_partial1(fblock, scalars + offset, nx, sx);
  }
  else
  {
    gather1(fblock, scalars + offset, sx);
  }

  zfp_encode_block<Scalar, ZFP_1D_BLOCK_SIZE>(fblock, maxbits, block_idx,
                                              stream, perm_3d, perm_1, perm_2);
}
//
// Launch the encode kernel
//
template<class Scalar>
size_t encode1launch(uint dim, 
                     int sx,
                     const Scalar *d_data,
                     Word *stream,
                     const int maxbits)
{
  sycl::queue q_ct1{syclZFP::internal_device_selector{}};
  const int sycl_block_size = 128;
  sycl::range<3> block_size = sycl::range<3>(1, 1, sycl_block_size);

  uint zfp_pad(dim); 
  if(zfp_pad % 4 != 0) zfp_pad += 4 - dim % 4;
  //TODO: THIS BLOCK SIZE CALCULATION MAY NOT MATCH THE SYCL IMPLEMENTATION GO OVER
  const uint zfp_blocks = (zfp_pad) / 4; 
  //
  // we need to ensure that we launch a multiple of the 
  // sycl block size
  //
  int block_pad = 0; 
  if(zfp_blocks % sycl_block_size != 0)
  {
    block_pad = sycl_block_size - zfp_blocks % sycl_block_size; 
  }

  size_t total_blocks = block_pad + zfp_blocks;

  sycl::range<3> grid_size = calculate_grid_size(total_blocks, sycl_block_size);

  //
  size_t stream_bytes = calc_device_mem1d(zfp_pad, maxbits);
  // ensure we have zeros
  q_ct1.memset(stream, 0, stream_bytes).wait();

//TODO: make alterations for SYCL
#ifdef SYCL_ZFP_RATE_PRINT
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
#endif

  {
    //dpct::has_capability_or_fail(q_ct1.get_device(), {sycl::aspect::fp64});//! LOOKS PROBLEMATIC 
    q_ct1.submit([&](sycl::handler &cgh) {//! Crashes here on opencl:gpu
      perm_3d.init();
      perm_1.init();
      perm_2.init();

      auto perm_3d_ptr_ct1 = perm_3d.get_ptr();
      auto perm_1_ptr_ct1 = perm_1.get_ptr();
      auto perm_2_ptr_ct1 = perm_2.get_ptr();

      cgh.parallel_for(sycl::nd_range<3>(grid_size * block_size, block_size),
                       [=](sycl::nd_item<3> item_ct1) {
                         syclEncode1<Scalar>(maxbits, d_data, stream, dim, sx,
                                             zfp_pad, zfp_blocks, item_ct1,
                                             perm_3d_ptr_ct1, perm_1_ptr_ct1,
                                             perm_2_ptr_ct1);
                       });
    });
  }

//TODO: make alterations for SYCL
#ifdef CUDA_ZFP_RATE_PRINT
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaStreamSynchronize(0);

  float milliseconds = 0.f;
  cudaEventElapsedTime(&milliseconds, start, stop);
  float seconds = milliseconds / 1000.f;
  float gb = (float(dim) * float(sizeof(Scalar))) / (1024.f * 1024.f * 1024.f);
  float rate = gb / seconds;
  printf("Encode elapsed time: %.5f (s)\n", seconds);
  printf("# encode1 rate: %.2f (GB / sec) %d\n", rate, maxbits);
#endif
  return stream_bytes;
}

//
// Encode a host vector and output a encoded device vector
//
template<class Scalar>
size_t encode1(int dim,
               int sx,
               Scalar *d_data,
               Word *stream,
               const int maxbits)
{
  return encode1launch<Scalar>(dim, sx, d_data, stream, maxbits);
}

}

#endif
