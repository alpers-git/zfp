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

typedef ::sycl::ulong2 size2;
typedef ::sycl::ulong3 size3;
typedef ::sycl::long2 ptrdiff2;
typedef ::sycl::long3 ptrdiff3;
typedef ::sycl::int2 int2;
typedef ::sycl::int3 int3;

#define make_size2(x, y) ::sycl::ulong2(x, y)
#define make_ptrdiff2(x, y) ::sycl::long2(x, y)
#define make_size3(x, y, z) ::sycl::ulong3(x, y, z)
#define make_ptrdiff3(x, y, z) ::sycl::long3(x, y, z)
#define make_int2(x, y) ::sycl::int2(x, y)
#define make_int3(x, y, z) ::sycl::int3(x, y, z)

//create a union of types Scalar, Int, and UInt
template <typename Scalar>
union Inplace{
  typedef Scalar ScalarType;
  Scalar scalar;
  traits<Scalar>::Int intVal;
  traits<Scalar>::UInt uintVal;
  Inplace(Scalar s) : scalar(s) {}
  Inplace() {}
};

//Split the bigger size arrays into smaller arrays
//to avoid register spillage
template <typename T, int BlockSize>
class SplitMem {
public:
    T& operator[](const int i) { return reg[i]; }
    const T& operator[](const int i) const { return reg[i]; }
private:
    T reg[BlockSize];
};

template<>
class SplitMem<float, 64> {
public:
    float& operator[](const int i) { 
        if (i < 16) return reg[i];
        else if (i < 32) return reg2[i - 16];
        else if (i < 48) return reg3[i - 32];
        else return reg4[i - 48];
    }
    const float& operator[](const int i) const { 
        if (i < 16) return reg[i];
        else if (i < 32) return reg2[i - 16];
        else if (i < 48) return reg3[i - 32];
        else return reg4[i - 48];
    }
private:
    float reg[16];
    float reg2[16];
    float reg3[16];
    float reg4[16];
};

template<>
class SplitMem<Inplace<float>, 64> {
public:
    Inplace<float>& operator[](const int i) { 
        if (i < 16) return reg[i];
        else if (i < 32) return reg2[i - 16];
        else if (i < 48) return reg3[i - 32];
        else return reg4[i - 48];
    }
    const Inplace<float>& operator[](const int i) const { 
        if (i < 16) return reg[i];
        else if (i < 32) return reg2[i - 16];
        else if (i < 48) return reg3[i - 32];
        else return reg4[i - 48];
    }
private:
    Inplace<float> reg[16];
    Inplace<float> reg2[16];
    Inplace<float> reg3[16];
    Inplace<float> reg4[16];
};

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
bool with_maxbits(const uint maxbits, const uint maxprec)
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

::sycl::nd_range<1> calculate_kernel_size(const zfp_exec_params_sycl *params,
                                   size_t work_items, size_t sycl_block_size)
{
  // compute minimum number of work-groups needed
  size_t work_group_num = count_up(work_items, sycl_block_size);
  const size_t total_num_work_items = work_group_num * sycl_block_size;

  const size_t work_group_size = ::sycl::min(sycl_block_size, (size_t)params->max_work_group_size);

  work_group_num = std::ceil((float)total_num_work_items / (float)work_group_size);

  return ::sycl::nd_range<1>(work_group_num * work_group_size, work_group_size);
}

// coefficient permutations
template <int BlockSize>
inline 
const unsigned char* get_perm();

template <>
inline 
const unsigned char* get_perm<4>()
{
  return perm_1.get();
}

template <>
inline 
const unsigned char* get_perm<16>()
{
  return perm_2.get();
}

template <>
inline 
const unsigned char* get_perm<64>()
{
  return perm_3.get();
}

// maximum number of bit planes to encode/decode
inline 
uint precision(const int maxexp, const uint maxprec, const int minexp, const int dims)
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
uint precision(const int maxexp, const uint maxprec, const int minexp);

template <>
inline 
uint precision<4>(const int maxexp, const uint maxprec, const int minexp)
{
  return precision(maxexp, maxprec, minexp, 1);
}

template <>
inline 
uint precision<16>(const int maxexp, const uint maxprec, const int minexp)
{
  return precision(maxexp, maxprec, minexp, 2);
}

template <>
inline 
uint precision<64>(const int maxexp, const uint maxprec, const int minexp)
{
  return precision(maxexp, maxprec, minexp, 3);
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
