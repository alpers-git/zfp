#ifndef ZFP_SYCL_SHARED_H
#define ZFP_SYCL_SHARED_H


#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <cstdio>
#include "zfp.h"
#include "traits.h"
#include "constants.h"
#ifdef ZFP_WITH_SYCL_PROFILE
  #include "timer.h"
#endif

// we need to know about bitstream, but we don't want duplicate symbols
#ifndef inline_
  #define inline_ inline
#endif

#include "zfp/bitstream.inl"

// bit stream word/buffer type; granularity of stream I/O operations
typedef unsigned long long Word;

#define ZFP_1D_BLOCK_SIZE 4
#define ZFP_2D_BLOCK_SIZE 16
#define ZFP_3D_BLOCK_SIZE 64
#define ZFP_4D_BLOCK_SIZE 256 // not yet supported

namespace zfp {
namespace sycl {
namespace internal {

typedef ::sycl::ulonglong2 size2;
typedef ::sycl::ulonglong3 size3;
typedef ::sycl::longlong2 ptrdiff2;
typedef ::sycl::longlong3 ptrdiff3;

#define make_size2(x, y) ::sycl::ulonglong2(x, y)
#define make_ptrdiff2(x, y) ::sycl::longlong2(x, y)
#define make_size3(x, y, z) ::sycl::ulonglong3(x, y, z)
#define make_ptrdiff3(x, y, z) ::sycl::longlong3(x, y, z)

// round size up to the next multiple of unit
inline 
size_t round_up(size_t size, size_t unit)
{
  size += unit - 1;
  size -= size % unit;
  return size;
}

// size / unit rounded up to the next integer
inline 
size_t count_up(size_t size, size_t unit)
{
  return (size + unit - 1) / unit;
}

// true if max compressed size exceeds maxbits
template <int BlockSize>
inline 
bool with_maxbits(uint maxbits, uint maxprec)
{
  return (maxprec + 1) * BlockSize - 1 > maxbits;
}

size_t calculate_device_memory(size_t blocks, size_t bits_per_block)
{
  const size_t bits_per_word = sizeof(Word) * CHAR_BIT;
  const size_t bits = blocks * bits_per_block;
  const size_t words = count_up(bits, bits_per_word);
  return words * sizeof(Word);
}

::sycl::range<3> calculate_grid_size(const zfp_exec_params_sycl *params,
                                   size_t threads, size_t sycl_block_size)
{
  // compute minimum number of thread blocks needed
  const size_t blocks = count_up(threads, sycl_block_size);
  const ::sycl::range<3> max_grid_dims(params->grid_size[2], params->grid_size[1],
                                     params->grid_size[0]);

  // compute grid dimensions
  if (blocks <= (size_t)max_grid_dims[2]) {
    // 1D grid
    return ::sycl::range<3>(1, 1, blocks);
  } else if (blocks <= (size_t)max_grid_dims[2] * max_grid_dims[1]) {
    // 2D grid
    const size_t base = (size_t)std::sqrt((double)blocks);
    return ::sycl::range<3>(1, round_up(blocks, base), base);
  } else if (blocks <=
             (size_t)max_grid_dims[2] * max_grid_dims[1] * max_grid_dims[0]) {
    // 3D grid
    const size_t base = (size_t)std::cbrt((double)blocks);
    return ::sycl::range<3>(round_up(blocks, base * base), base, base);
  }
  else {
    // too many thread blocks
    return ::sycl::range<3>(0, 0, 0);
  }
}

// coefficient permutations
template <int BlockSize>
inline 
const unsigned char* get_perm(const unsigned char *perm_1,
                              const unsigned char *perm_2,
                              const unsigned char *perm_3);

template <>
inline 
const unsigned char* get_perm<4>(const unsigned char *perm_1,
                                 const unsigned char *perm_2,
                                 const unsigned char *perm_3)
{
  return perm_1;
}

template <>
inline 
const unsigned char* get_perm<16>(const unsigned char *perm_1,
                                  const unsigned char *perm_2,
                                  const unsigned char *perm_3)
{
  return perm_2;
}

template <>
inline 
const unsigned char* get_perm<64>(const unsigned char *perm_1,
                                  const unsigned char *perm_2,
                                  const unsigned char *perm_3)
{
  return perm_3;
}

// maximum number of bit planes to encode/decode
inline 
uint precision(int maxexp, uint maxprec, int minexp, int dims)
{
#if (ZFP_ROUNDING_MODE != ZFP_ROUND_NEVER) && defined(ZFP_WITH_TIGHT_ERROR)
  return min(maxprec, max(0, maxexp - minexp + 2 * dims + 1));
#else
  return ::sycl::min(maxprec, (unsigned int)(::sycl::max(
                                0, (int)(maxexp - minexp + 2 * dims + 2))));
#endif
}

template <int BlockSize>
inline 
uint precision(int maxexp, uint maxprec, int minexp);

template <>
inline 
uint precision<4>(int maxexp, uint maxprec, int minexp)
{
  return precision(maxexp, maxprec, minexp, 1);
}

template <>
inline 
uint precision<16>(int maxexp, uint maxprec, int minexp)
{
  return precision(maxexp, maxprec, minexp, 2);
}

template <>
inline 
uint precision<64>(int maxexp, uint maxprec, int minexp)
{
  return precision(maxexp, maxprec, minexp, 3);
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
