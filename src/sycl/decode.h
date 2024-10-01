#include <sycl/sycl.hpp>
#ifndef ZFP_SYCL_DECODE_H
#define ZFP_SYCL_DECODE_H

#include "shared.h"

namespace zfp {
namespace sycl {
namespace internal {
// map exponent e to dequantization scale factor
template <typename Scalar>
inline 
Scalar dequantize_factor(int e);

template <>
inline 
double dequantize_factor<double>(int e)
{
  return ::sycl::ldexp(1.0, e - (int)(traits<double>::precision - 2));
}

template <>
inline 
float dequantize_factor<float>(int e)
{
  return ::sycl::ldexp(1.0f, e - (int)(traits<float>::precision - 2));
}

// inverse block-floating-point transform from signed integers
template <typename Scalar, typename Int, int BlockSize>
inline 
void inv_cast(const Int *iblock, Scalar *fblock, int emax)
{
  const Scalar scale = dequantize_factor<Scalar>(emax);

#pragma unroll BlockSize
  for (int i = 0; i < BlockSize; i++)
    fblock[i] = scale * (Scalar)iblock[i];
}

// inverse lifting transform of 4-vector
template <class Int, uint s>
inline 
void inv_lift(Int* p)
{
  Int x, y, z, w;
  x = *p; p += s;
  y = *p; p += s;
  z = *p; p += s;
  w = *p; p += s;

  // non-orthogonal transform
  //       ( 4  6 -4 -1) (x)
  // 1/4 * ( 4  2  4  5) (y)
  //       ( 4 -2  4 -5) (z)
  //       ( 4 -6 -4  1) (w)
  y += w >> 1; w -= y >> 1;
  y += w; w <<= 1; w -= y;
  z += x; x <<= 1; x -= z;
  y += z; z <<= 1; z -= y;
  w += x; x <<= 1; x -= w;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

// inverse decorrelating transform (partial specialization via functor)
template <typename Int, int BlockSize>
struct inv_xform;

template <typename Int>
struct inv_xform<Int, 4> {
  inline 
  void operator()(Int* p) const
  {
    inv_lift<Int, 1>(p);
  }
};

template <typename Int>
struct inv_xform<Int, 16> {
  inline 
  void operator()(Int* p) const
  {
    // transform along y
    for (uint x = 0; x < 4; x++)
      inv_lift<Int, 4>(p + 1 * x);
    // transform along x
    for (uint y = 0; y < 4; y++)
      inv_lift<Int, 1>(p + 4 * y);
  }
};

template <typename Int>
struct inv_xform<Int, 64> {
  inline 
  void operator()(Int* p) const
  {
    // transform along z
    for (uint y = 0; y < 4; y++)
      for (uint x = 0; x < 4; x++)
        inv_lift<Int, 16>(p + 1 * x + 4 * y);
    // transform along y
    for (uint x = 0; x < 4; x++)
      for (uint z = 0; z < 4; z++)
        inv_lift<Int, 4>(p + 16 * z + 1 * x);
    // transform along x
    for (uint z = 0; z < 4; z++)
      for (uint y = 0; y < 4; y++)
        inv_lift<Int, 1>(p + 4 * y + 16 * z);
  }
};

#if ZFP_ROUNDING_MODE == ZFP_ROUND_LAST
// bias values such that truncation is equivalent to round to nearest
template <typename UInt, uint BlockSize>
inline __device__
void inv_round(UInt* ublock, uint m, uint prec)
{
  // add 1/6 ulp to unbias errors
  if (prec < (uint)(traits<UInt>::precision - 1)) {
    // the first m values (0 <= m <= n) have one more bit of precision
    uint n = BlockSize - m;
    while (m--) *ublock++ += ((traits<UInt>::nbmask >> 2) >> prec);
    while (n--) *ublock++ += ((traits<UInt>::nbmask >> 1) >> prec);
  }
}
#endif

// map negabinary unsigned integer to two's complement signed integer
template <typename Int, typename UInt>
inline 
Int uint2int(UInt x)
{
  return (Int)((x ^ traits<UInt>::nbmask) - traits<UInt>::nbmask);
}

template <typename Int, typename UInt, int BlockSize>
inline 
void inv_order(InplaceScalar<Int>* block)
{
  #define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))
  block[index(0, 0, 0)].intVal = uint2int<Int, UInt>(block[0].uintVal); // 0<-0
  block[index(1, 0, 0)].intVal = uint2int<Int, UInt>(block[1].uintVal); // 1<-1

  UInt temp = block[index(0, 1, 0)].uintVal; // hold 4s value
  block[index(0, 1, 0)].intVal = uint2int<Int, UInt>(block[2].uintVal); // 4<-2
  block[index(2, 0, 0)].intVal = uint2int<Int, UInt>(block[index(3, 1, 0)].uintVal); // 2<-7
  block[index(3, 1, 0)].intVal = uint2int<Int, UInt>(block[index(2, 2, 1)].uintVal); // 7<-26
  block[index(2, 2, 1)].intVal = uint2int<Int, UInt>(block[index(2, 0, 2)].uintVal); // 26<-34
  block[index(2, 0, 2)].intVal = uint2int<Int, UInt>(block[index(0, 2, 1)].uintVal); // 34<-24
  block[index(0, 2, 1)].intVal = uint2int<Int, UInt>(block[index(1, 3, 0)].uintVal); // 24<-13
  block[index(1, 3, 0)].intVal = uint2int<Int, UInt>(block[index(1, 3, 1)].uintVal); // 13<-29
  block[index(1, 3, 1)].intVal = uint2int<Int, UInt>(block[index(0, 1, 2)].uintVal); // 29<-36
  block[index(0, 1, 2)].intVal = uint2int<Int, UInt>(block[index(0, 0, 1)].uintVal); // 36<-16
  block[index(0, 0, 1)].intVal = uint2int<Int, UInt>(block[index(3, 0, 0)].uintVal); // 16<-3 
  block[index(3, 0, 0)].intVal = uint2int<Int, UInt>(block[index(1, 0, 1)].uintVal); // 3<-17
  block[index(1, 0, 1)].intVal = uint2int<Int, UInt>(block[index(1, 1, 0)].uintVal); // 17<-5 
  block[index(1, 1, 0)].intVal = uint2int<Int, UInt>(block[index(2, 1, 0)].uintVal); // 5<-6 
  block[index(2, 1, 0)].intVal = uint2int<Int, UInt>(block[index(3, 2, 0)].uintVal); // 6<-11
  block[index(3, 2, 0)].intVal = uint2int<Int, UInt>(block[index(2, 1, 2)].uintVal); // 11<-38
  block[index(2, 1, 2)].intVal = uint2int<Int, UInt>(block[index(1, 0, 2)].uintVal); // 38<-33
  block[index(1, 0, 2)].intVal = uint2int<Int, UInt>(block[index(3, 3, 0)].uintVal); // 33<-15
  block[index(3, 3, 0)].intVal = uint2int<Int, UInt>(block[index(1, 1, 3)].uintVal); // 15<-53
  block[index(1, 1, 3)].intVal = uint2int<Int, UInt>(block[index(1, 1, 2)].uintVal); // 53<-37
  block[index(1, 1, 2)].intVal = uint2int<Int, UInt>(block[index(2, 1, 1)].uintVal); // 37<-22
  block[index(2, 1, 1)].intVal = uint2int<Int, UInt>(block[index(0, 1, 1)].uintVal); // 22<-20
  block[index(0, 1, 1)].intVal = uint2int<Int, UInt>(temp); // 20<-4   

  block[index(0, 2, 0)].intVal = uint2int<Int, UInt>(block[8].intVal); // 8<-8

  
  temp = block[index(0, 0, 2)].uintVal; // hold 32s value
  block[index(0, 0, 2)].intVal = uint2int<Int, UInt>(block[9].uintVal); // 32<-9
  block[index(1, 2, 0)].intVal = uint2int<Int, UInt>(block[index(2, 3, 0)].uintVal);// 9<-14
  block[index(2, 3, 0)].intVal = uint2int<Int, UInt>(block[index(1, 2, 2)].uintVal); // 14<-41
  block[index(1, 2, 2)].intVal = uint2int<Int, UInt>(temp); // 41<-32

  temp = block[index(1, 1, 1)].uintVal; // hold 21s value 
  block[index(1, 1, 1)].intVal = uint2int<Int, UInt>(block[10].uintVal); // 21<-10
  block[index(2, 2, 0)].intVal = uint2int<Int, UInt>(block[index(1, 2, 1)].uintVal); // 10<-25
  block[index(1, 2, 1)].intVal = uint2int<Int, UInt>(temp); // 25<-21

  temp = block[index(2, 0, 1)].uintVal; // hold 18s value
  block[index(2, 0, 1)].intVal = uint2int<Int, UInt>(block[12].uintVal); // 18<-12
  block[index(0, 3, 0)].intVal = uint2int<Int, UInt>(temp); // 12<-18

  temp = block[index(0, 0, 3)].uintVal; // hold 48s value
  block[index(0, 0, 3)].intVal = uint2int<Int, UInt>(block[19].uintVal); // 48<-19
  block[index(3, 0, 1)].intVal = uint2int<Int, UInt>(block[index(3, 2, 1)].uintVal); // 19<-27
  block[index(3, 2, 1)].intVal = uint2int<Int, UInt>(block[index(1, 3, 2)].uintVal); // 27<-45
  block[index(1, 3, 2)].intVal = uint2int<Int, UInt>(block[index(3, 3, 2)].uintVal); // 45<-47
  block[index(3, 3, 2)].intVal = uint2int<Int, UInt>(block[index(2, 3, 3)].uintVal); // 47<-62
  block[index(2, 3, 3)].intVal = uint2int<Int, UInt>(block[index(0, 3, 3)].uintVal); // 62<-60
  block[index(0, 3, 3)].intVal = uint2int<Int, UInt>(block[index(3, 0, 3)].uintVal); // 60<-51
  block[index(3, 0, 3)].intVal = uint2int<Int, UInt>(block[index(0, 1, 3)].uintVal); // 51<-52
  block[index(0, 1, 3)].intVal = uint2int<Int, UInt>(block[index(3, 3, 1)].uintVal); // 52<-31
  block[index(3, 3, 1)].intVal = uint2int<Int, UInt>(block[index(3, 2, 3)].uintVal); // 31<-59
  block[index(3, 2, 3)].intVal = uint2int<Int, UInt>(block[index(1, 3, 3)].uintVal); // 59<-61
  block[index(1, 3, 3)].intVal = uint2int<Int, UInt>(block[index(1, 2, 3)].uintVal); // 61<-57
  block[index(1, 2, 3)].intVal = uint2int<Int, UInt>(block[index(2, 0, 3)].uintVal); // 57<-50
  block[index(2, 0, 3)].intVal = uint2int<Int, UInt>(block[index(2, 2, 2)].uintVal); // 50<-42
  block[index(2, 2, 2)].intVal = uint2int<Int, UInt>(block[index(0, 3, 2)].uintVal); // 42<-44
  block[index(0, 3, 2)].intVal = uint2int<Int, UInt>(block[index(0, 2, 2)].uintVal); // 44<-40
  block[index(0, 2, 2)].intVal = uint2int<Int, UInt>(block[index(3, 1, 1)].uintVal); // 40<-23
  block[index(3, 1, 1)].intVal = uint2int<Int, UInt>(block[index(3, 0, 2)].uintVal); // 23<-35
  block[index(3, 0, 2)].intVal = uint2int<Int, UInt>(block[index(3, 1, 2)].uintVal); // 35<-39
  block[index(3, 1, 2)].intVal = uint2int<Int, UInt>(block[index(2, 3, 2)].uintVal); // 39<-46
  block[index(2, 3, 2)].intVal = uint2int<Int, UInt>(block[index(3, 1, 3)].uintVal); // 46<-55
  block[index(3, 1, 3)].intVal = uint2int<Int, UInt>(block[index(2, 2, 3)].uintVal); // 55<-58
  block[index(2, 2, 3)].intVal = uint2int<Int, UInt>(block[index(0, 2, 3)].uintVal); // 58<-56
  block[index(0, 2, 3)].intVal = uint2int<Int, UInt>(block[index(3, 2, 2)].uintVal); // 56<-43
  block[index(3, 2, 2)].intVal = uint2int<Int, UInt>(block[index(2, 1, 3)].uintVal); // 43<-54
  block[index(2, 1, 3)].intVal = uint2int<Int, UInt>(block[index(1, 0, 3)].uintVal); // 54<-49
  block[index(1, 0, 3)].intVal = uint2int<Int, UInt>(block[index(2, 3, 1)].uintVal); // 49<-30
  block[index(2, 3, 1)].intVal = uint2int<Int, UInt>(temp); // 30<-48

  block[index(0, 3, 1)].intVal = uint2int<Int, UInt>(block[28].uintVal); // 28<-28

  block[index(3, 3, 3)].intVal = uint2int<Int, UInt>(block[63].uintVal); // 63<-63
  #undef index
}

template <typename Int, typename UInt, int BlockSize>
inline 
void inv_order(const UInt* ublock, Int* iblock)
{
//   const auto perm = get_perm<BlockSize>();


// #pragma unroll BlockSize
//   for (int i = 0; i < BlockSize; i++)
//     iblock[perm[i]] = uint2int<Int, UInt>(ublock[i]);
  if constexpr(BlockSize == 4)
  {
    iblock[0] = uint2int<Int, UInt>(ublock[0]);
    iblock[1] = uint2int<Int, UInt>(ublock[1]);
    iblock[2] = uint2int<Int, UInt>(ublock[2]);
    iblock[3] = uint2int<Int, UInt>(ublock[3]);
  }
  else if (BlockSize == 16)
  {
#define index(i, j) ((i) + 4 * (j))
    iblock[index(0, 0)] = uint2int<Int, UInt>(ublock[0]);
    iblock[index(1, 0)] = uint2int<Int, UInt>(ublock[1]);
    iblock[index(0, 1)] = uint2int<Int, UInt>(ublock[2]);
    iblock[index(1, 1)] = uint2int<Int, UInt>(ublock[3]);
    iblock[index(2, 0)] = uint2int<Int, UInt>(ublock[4]);
    iblock[index(0, 2)] = uint2int<Int, UInt>(ublock[5]);
    iblock[index(2, 1)] = uint2int<Int, UInt>(ublock[6]);
    iblock[index(1, 2)] = uint2int<Int, UInt>(ublock[7]);
    iblock[index(3, 0)] = uint2int<Int, UInt>(ublock[8]);
    iblock[index(0, 3)] = uint2int<Int, UInt>(ublock[9]);
    iblock[index(2, 2)] = uint2int<Int, UInt>(ublock[10]);
    iblock[index(3, 1)] = uint2int<Int, UInt>(ublock[11]);
    iblock[index(1, 3)] = uint2int<Int, UInt>(ublock[12]);
    iblock[index(3, 2)] = uint2int<Int, UInt>(ublock[13]);
    iblock[index(2, 3)] = uint2int<Int, UInt>(ublock[14]);
    iblock[index(3, 3)] = uint2int<Int, UInt>(ublock[15]);
#undef index
  }
  else
   {
    const auto perm = get_perm<BlockSize>();
    for (int i = 0; i < BlockSize; i++)
      iblock[perm[i]] = uint2int<Int, UInt>(ublock[i]);
   }
}

template <typename UInt, int BlockSize>
inline 
uint decode_ints(UInt* ublock, BlockReader& reader, uint maxbits, uint maxprec)
{
  const uint intprec = traits<UInt>::precision;
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;
  uint k, m, n;

  for (k = intprec, m = n = 0; bits && k-- > kmin;) {
    // decode bit plane
    m = ::sycl::min(n, bits);
    bits -= m;
    ::uint64 x = reader.read_bits(m);
    for (; n < BlockSize && bits && (bits--, reader.read_bit()); x += (::uint64)1 << n++)
      for (; n < BlockSize - 1 && bits && (bits--, !reader.read_bit()); n++)
        ;

    // deposit bit plane (use fixed bound to prevent warp divergence)
    
#pragma unroll BlockSize
    for (int i = 0; i < BlockSize; i++, x >>= 1)
      ublock[i] += (UInt)(x & 1u) << k;
  }

#if ZFP_ROUNDING_MODE == ZFP_ROUND_LAST
  // bias values to achieve proper rounding
  inv_round<UInt, BlockSize>(ublock, m, intprec - k);
#endif

  return maxbits - bits;
}

template <typename UInt, int BlockSize>
inline 
uint decode_ints_prec(UInt* ublock, BlockReader& reader, const uint maxprec)
{
  const BlockReader::Offset offset = reader.rtell();
  const uint intprec = traits<UInt>::precision;
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint k, n;

  for (k = intprec, n = 0; k-- > kmin;) {
    // decode bit plane
    ::uint64 x = reader.read_bits(n);
    for (; n < BlockSize && reader.read_bit(); x += (::uint64)1 << n, n++)
      for (; n < BlockSize - 1 && !reader.read_bit(); n++)
        ;

    // deposit bit plane (use fixed bound to prevent warp divergence)
    
#pragma unroll BlockSize
    for (int i = 0; i < BlockSize; i++, x >>= 1)
      ublock[i] += (UInt)(x & 1u) << k;
  }

#if ZFP_ROUNDING_MODE == ZFP_ROUND_LAST
  // bias values to achieve proper rounding
  inv_round<UInt, BlockSize>(ublock, 0, intprec - k);
#endif

  return (uint)(reader.rtell() - offset);
}

// common integer and floating-point decoder
template <typename Int, int BlockSize>
inline 
uint decode_int_block(
  Int* iblock,
  BlockReader& reader,
  uint minbits,
  uint maxbits,
  uint maxprec)
{
  // decode integer coefficients
  typedef typename traits<Int>::UInt UInt;
  UInt ublock[BlockSize] = { 0 };
  uint bits = with_maxbits<BlockSize>(maxbits, maxprec)
                ? decode_ints<UInt, BlockSize>(ublock, reader, maxbits, maxprec)
                : decode_ints_prec<UInt, BlockSize>(ublock, reader, maxprec);

  // read at least minbits bits
  if (minbits > bits) {
    reader.skip(minbits - bits);
    bits = minbits;
  }

  // reorder unsigned coefficients and convert to signed integer
  inv_order<Int, UInt, BlockSize>(ublock, iblock);

  // perform decorrelating transform
  inv_xform<Int, BlockSize>()(iblock);

  return bits;
}

// decoder specialization for floats and doubles
template <typename Scalar, int BlockSize>
inline 
uint decode_float_block(
  Scalar* fblock,
  BlockReader& reader,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp)
{
  uint bits = 1;
  if (reader.read_bit()) {
    // decode block exponent
    bits += traits<Scalar>::ebits;
    int emax = (int)reader.read_bits(bits - 1) - traits<Scalar>::ebias;
    maxprec = precision<BlockSize>(emax, maxprec, minexp);
    // decode integer block
    typedef typename traits<Scalar>::Int Int;
    Int* iblock = (Int*)fblock;
    bits += decode_int_block<Int, BlockSize>(
        iblock, reader, ::sycl::max(minbits, bits) - bits,
        ::sycl::max(maxbits, bits) - bits, maxprec);
    // perform inverse block-floating-point transform
    inv_cast<Scalar, Int, BlockSize>(iblock, fblock, emax);
  }
  else {
    // read at least minbits bits
    if (minbits > bits) {
      reader.skip(minbits - bits);
      bits = minbits;
    }
  }

  return bits;
}

// inplace integer and floating-point decoder
template <typename Int, int BlockSize>
inline 
uint decode_int_block(
  InplaceScalar<Int>* iblock,
  BlockReader& reader,
  uint minbits,
  uint maxbits,
  uint maxprec)
{

  // decode integer coefficients
  typedef typename traits<Int>::UInt UInt;
  for(size_t i =0; i < BlockSize; i++)
      iblock[i] = (UInt)0;
  uint bits = with_maxbits<BlockSize>(maxbits, maxprec)
                ? decode_ints<UInt, BlockSize>((UInt*)iblock, reader, maxbits, maxprec)
                : decode_ints_prec<UInt, BlockSize>((UInt*)iblock, reader, maxprec);

  // read at least minbits bits
  if (minbits > bits) {
    reader.skip(minbits - bits);
    bits = minbits;
  }

  // reorder unsigned coefficients and convert to signed integer
  inv_order<Int, UInt, BlockSize>((InplaceScalar<Int>*)iblock);

  // perform decorrelating transform
  inv_xform<Int, BlockSize>()((Int*)iblock);

  return bits;
}

// inplace decoder specialization for floats and doubles
template <typename Scalar, int BlockSize>
inline 
uint decode_float_block(
  InplaceScalar<Scalar>* fblock,
  BlockReader& reader,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp)
{
  uint bits = 1;
  if (reader.read_bit()) {
    // decode block exponent
    bits += traits<Scalar>::ebits;
    int emax = (int)reader.read_bits(bits - 1) - traits<Scalar>::ebias;
    maxprec = precision<BlockSize>(emax, maxprec, minexp);
    // decode integer block
    typedef typename traits<Scalar>::Int Int;
    bits += decode_int_block<Int, BlockSize>(
        (InplaceScalar<Int>*)fblock, reader, ::sycl::max(minbits, bits) - bits,
        ::sycl::max(maxbits, bits) - bits, maxprec);
    // perform inverse block-floating-point transform
    inv_cast<Scalar, Int, BlockSize>((Int*)fblock, (Scalar*)fblock, emax);
  }
  else {
    // read at least minbits bits
    if (minbits > bits) {
      reader.skip(minbits - bits);
      bits = minbits;
    }
  }

  return bits;
}

// generic decoder
template <typename Scalar, int BlockSize>
struct decode_block;

// decoder specialization for ints
template <int BlockSize>
struct decode_block<int, BlockSize> {
  inline 
  uint operator()(int* iblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return decode_int_block<int, BlockSize>(iblock, reader, minbits, maxbits, maxprec);
  }
};

// decoder specialization for long longs
template <int BlockSize>
struct decode_block<long long, BlockSize> {
  inline 
  uint operator()(long long* iblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return decode_int_block<long long, BlockSize>(
        iblock, reader, minbits, maxbits, maxprec);
  }
};

// decoder specialization for floats
template <int BlockSize>
struct decode_block<float, BlockSize> {
  inline 
  uint operator()(float* fblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int minexp) const
  {
    return decode_float_block<float, BlockSize>(fblock, reader, minbits,
                                                maxbits, maxprec, minexp);
  }
};

// decoder specialization for doubles
template <int BlockSize>
struct decode_block<double, BlockSize> {
  inline 
  uint operator()(double *fblock, BlockReader &reader,
                                       uint minbits, uint maxbits, uint maxprec,
                                       int minexp) const
  {
    return decode_float_block<double, BlockSize>(fblock, reader, minbits,
                                                 maxbits, maxprec, minexp);
  }
};

// inplace decoder specialization for ints
template <int BlockSize>
struct decode_block<InplaceScalar<int>, BlockSize> {
  inline 
  uint operator()(InplaceScalar<int>* iblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return decode_int_block<int, BlockSize>(iblock, reader, minbits, maxbits, maxprec);
  }
};

// inplace decoder specialization for long longs
template <int BlockSize>
struct decode_block<InplaceScalar<long long>, BlockSize> {
  inline 
  uint operator()(InplaceScalar<long long>* iblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return decode_int_block<long long, BlockSize>(
        iblock, reader, minbits, maxbits, maxprec);
  }
};

// inplace decoder specialization for floats
template <int BlockSize>
struct decode_block<InplaceScalar<float>, BlockSize> {
  inline 
  uint operator()(InplaceScalar<float>* fblock, BlockReader& reader, uint minbits, uint maxbits, uint maxprec, int minexp) const
  {
    return decode_float_block<float, BlockSize>(fblock, reader, minbits,
                                                maxbits, maxprec, minexp);
  }
};

// inplace decoder specialization for doubles
template <int BlockSize>
struct decode_block<InplaceScalar<double>, BlockSize> {
  inline 
  uint operator()(InplaceScalar<double> *fblock, BlockReader &reader,
                                       uint minbits, uint maxbits, uint maxprec,
                                       int minexp) const
  {
    return decode_float_block<double, BlockSize>(fblock, reader, minbits,
                                                 maxbits, maxprec, minexp);
  }
};

// forward declarations
template <typename T>
unsigned long long
decode1(
  T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_sycl* params,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
);

template <typename T>
unsigned long long
decode2(
  T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_sycl* params,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
);

template <typename T>
unsigned long long
decode3(
  T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_sycl* params,
  const Word* d_stream,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp,
  const Word* d_index,
  zfp_index_type index_type,
  uint granularity
);

// compute bit offset to compressed block
//TODO: this function seems very optimized for CUDA consider refactoring
SYCL_EXTERNAL inline unsigned long long
block_offset(const Word *d_index, zfp_index_type index_type, size_t chunk_idx,
             const ::sycl::nd_item<1> &item_ct1, ::sycl::local_accessor<uint64, 1> offset)
{
  if (index_type == zfp_index_offset)
    return d_index[chunk_idx];

  if (index_type == zfp_index_hybrid) {
    const size_t thread_idx = item_ct1.get_local_id(0);
    // TODO: Why subtract thread_idx? And should granularity not matter?
    const size_t warp_idx = (chunk_idx - thread_idx) / 32;
    // warp operates on 32 blocks indexed by one 64-bit offset, 32 16-bit sizes
    const ::uint64* data64 = (const ::uint64*)d_index + warp_idx * 9;
    const ::uint16* data16 = (const ::uint16*)data64 + 3;

    offset[thread_idx] = thread_idx ? data16[thread_idx] : *data64;
    // compute prefix sum in parallel
    for (uint i = 1u; i < 32u; i <<= 1) {
      if (thread_idx + i < 32u)
        offset[thread_idx + i] += offset[thread_idx];
      item_ct1.barrier(::sycl::access::fence_space::local_space);
    }
    return offset[thread_idx];
  }

  return 0;
}

} // namespace internal

// decode field from d_stream to d_data
template <typename T>
unsigned long long
decode(
  T* d_data,                          // field data device pointer
  const size_t size[],                // field dimensions
  const ptrdiff_t stride[],           // field strides
  const zfp_exec_params_sycl* params, // execution parameters
  const Word* d_stream,               // compressed bit stream device pointer
  uint minbits,                       // minimum compressed #bits/block
  uint maxbits,                       // maximum compressed #bits/block
  uint maxprec,                       // maximum uncompressed #bits/value
  int minexp,                         // minimum bit plane index
  const Word* d_index,                // block index device pointer
  zfp_index_type index_type,          // block index type
  uint granularity                    // block index granularity in blocks/entry
)
{
  unsigned long long bits_read = 0;

  //internal::ErrorCheck error;
  const uint dims = size[0] ? size[1] ? size[2] ? 3 : 2 : 1 : 0;
  switch (dims) {
    case 1:
      bits_read = internal::decode1<T>(d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    case 2:
      bits_read = internal::decode2<T>(d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    case 3:
      bits_read = internal::decode3<T>(d_data, size, stride, params, d_stream, minbits, maxbits, maxprec, minexp, d_index, index_type, granularity);
      break;
    default:
      break;
  }
  // if (!error.check("decode"))
  //   bits_read = 0;

  return bits_read;
}

} // namespace sycl
} // namespace zfp

#endif
