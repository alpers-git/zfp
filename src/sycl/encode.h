#ifndef ZFP_SYCL_ENCODE_H
#define ZFP_SYCL_ENCODE_H

#include "shared.h"

namespace zfp {
namespace sycl {
namespace internal {

// pad partial block of width n <= 4 and stride s
template <typename Scalar>
SYCL_EXTERNAL inline void pad_block(Scalar *p, uint n, ptrdiff_t s)
{
  switch (n) {
    case 0:
      p[0 * s] = 0;
      /* FALLTHROUGH */
    case 1:
      p[1 * s] = p[0 * s];
      /* FALLTHROUGH */
    case 2:
      p[2 * s] = p[1 * s];
      /* FALLTHROUGH */
    case 3:
      p[3 * s] = p[0 * s];
      /* FALLTHROUGH */
    default:
      break;
  }
}

template <typename Scalar>
inline 
int get_exponent(Scalar x);

template <>
inline 
int get_exponent(float x)
{
  int e;
  //DPCT1017:24:Resolved. Removed weird address operations
  ::sycl::frexp(x, &e);
  return e;
}

template <>
inline 
int get_exponent(double x)
{
  int e;
  //DPCT1017:25:Resolved. Removed weird address operations
  ::sycl::frexp(x, &e);
  return e;
}

template <typename Scalar>
inline 
int exponent(Scalar x)
{
  int e = -traits<Scalar>::ebias;
#ifdef ZFP_WITH_DAZ
  // treat subnormals as zero; resolves issue #119 by avoiding overflow
  if (x >= get_scalar_min<Scalar>())
    e = get_exponent(x);
#else
  if (x > 0) {
    int e = get_exponent(x);
    // clamp exponent in case x is subnormal
    return ::sycl::max(e, (int)(1 - traits<Scalar>::ebias));
  }
#endif
  return e;
}

template <typename Scalar, int BlockSize>
inline 
int max_exponent(const Scalar* p)
{
  Scalar max_val = 0;
  for (int i = 0; i < BlockSize; i++) {
    Scalar f = ::sycl::fabs((p[i]));
    max_val = ::sycl::max(max_val, f);
  }
  return exponent<Scalar>(max_val);
}

template <typename Scalar, int BlockSize>
inline 
int max_exponent(const Inplace<Scalar>* p)
{
  Scalar max_val = 0;
  for (int i = 0; i < BlockSize; i++) {
    Scalar f = ::sycl::fabs((p[i].scalar));
    max_val = ::sycl::max(max_val, f);
  }
  return exponent<Scalar>(max_val);
}

// map exponent to power-of-two quantization factor
template <typename Scalar>
inline 
Scalar quantize_factor(int exponent);

template <>
inline 
float quantize_factor<float>(int exponent)
{
  return ::sycl::ldexp(1.0f, traits<float>::precision - 2 - exponent);
}

template <>
inline 
double quantize_factor<double>(int exponent)
{
  return ::sycl::ldexp(1.0, traits<double>::precision - 2 - exponent);
}

// forward block-floating-point transform to signed integers
template <typename Scalar, typename Int, int BlockSize>
inline 
void fwd_cast(Int *iblock, const Scalar *fblock, int emax)
{
  const Scalar scale = quantize_factor<Scalar>(emax);

#pragma unroll BlockSize
  for (int i = 0; i < BlockSize; i++)
    iblock[i] = (Int)(scale * fblock[i]);
}

// forward lifting transform of 4-vector
template <class Int, uint s>
inline 
void fwd_lift(Int* p)
{
  Int x = *p; p += s;
  Int y = *p; p += s;
  Int z = *p; p += s;
  Int w = *p; p += s;

  // non-orthogonal transform
  //        ( 4  4  4  4) (x)
  // 1/16 * ( 5  1 -1 -5) (y)
  //        (-4  4  4 -4) (z)
  //        (-2  6 -6  2) (w)
  x += w; x >>= 1; w -= x;
  z += y; z >>= 1; y -= z;
  x += z; x >>= 1; z -= x;
  w += y; w >>= 1; y -= w;
  w += y >> 1; y -= w >> 1;

  p -= s; *p = w;
  p -= s; *p = z;
  p -= s; *p = y;
  p -= s; *p = x;
}

// forward decorrelating transform (partial specialization via functor)
template <typename Int, int BlockSize>
struct fwd_xform;

template <typename Int>
struct fwd_xform<Int, 4> {
  inline 
  void operator()(Int* p) const
  {
    fwd_lift<Int, 1>(p);
  }
};

template <typename Int>
struct fwd_xform<Int, 16> {
  inline 
  void operator()(Int* p) const
  {
    // transform along x
    for (uint y = 0; y < 4; y++)
     fwd_lift<Int, 1>(p + 4 * y);
    // transform along y
    for (uint x = 0; x < 4; x++)
      fwd_lift<Int, 4>(p + 1 * x);
  }
};

template <typename Int>
struct fwd_xform<Int, 64> {
  inline 
  void operator()(Int* p) const
  {
    // transform along x
    for (uint z = 0; z < 4; z++)
      for (uint y = 0; y < 4; y++)
        fwd_lift<Int, 1>(p + 4 * y + 16 * z);
    // transform along y
    for (uint x = 0; x < 4; x++)
      for (uint z = 0; z < 4; z++)
        fwd_lift<Int, 4>(p + 16 * z + 1 * x);
    // transform along z
    for (uint y = 0; y < 4; y++)
      for (uint x = 0; x < 4; x++)
        fwd_lift<Int, 16>(p + 1 * x + 4 * y);
   }
};

#if ZFP_ROUNDING_MODE == ZFP_ROUND_FIRST
// bias values such that truncation is equivalent to round to nearest
template <typename Int, uint BlockSize>
inline __device__
void fwd_round(Int* iblock, uint maxprec)
{
  // add or subtract 1/6 ulp to unbias errors
  if (maxprec < (uint)traits<Int>::precision) {
    Int bias = (traits<Int>::nbmask >> 2) >> maxprec;
    uint n = BlockSize;
    if (maxprec & 1u)
      do *iblock++ += bias; while (--n);
    else
      do *iblock++ -= bias; while (--n);
  }
}
#endif

// map two's complement signed integer to negabinary unsigned integer
template <typename Int, typename UInt>
inline 
UInt int2uint(const Int x)
{
  return ((UInt)x + traits<UInt>::nbmask) ^ traits<UInt>::nbmask;
}

template <typename Int, typename UInt, int BlockSize>
inline
void fwd_order(Inplace<Int>* block) //Cursed, but less reg. pressure
{
  #define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))
  block[0].uintVal = int2uint<Int, UInt>(block[index(0, 0, 0)].intVal); // 0<-0
  block[1].uintVal = int2uint<Int, UInt>(block[index(1, 0, 0)].intVal); // 1<-1

  Int temp = block[2].intVal; // hold 2s value
  block[2].uintVal = int2uint<Int, UInt>(block[index(0, 1, 0)].intVal); // 2<-4
  block[index(0, 1, 0)].uintVal = int2uint<Int, UInt>(block[index(0, 1, 1)].intVal); // 4<-20
  block[index(0, 1, 1)].uintVal = int2uint<Int, UInt>(block[index(2, 1, 1)].intVal); // 20<-22
  block[index(2, 1, 1)].uintVal = int2uint<Int, UInt>(block[index(1, 1, 2)].intVal); // 22<-37
  block[index(1, 1, 2)].uintVal = int2uint<Int, UInt>(block[index(1, 1, 3)].intVal); // 37<-53
  block[index(1, 1, 3)].uintVal = int2uint<Int, UInt>(block[index(3, 3, 0)].intVal); // 53<-15
  block[index(3, 3, 0)].uintVal = int2uint<Int, UInt>(block[index(1, 0, 2)].intVal); // 15<-33
  block[index(1, 0, 2)].uintVal = int2uint<Int, UInt>(block[index(2, 1, 2)].intVal); // 33<-38
  block[index(2, 1, 2)].uintVal = int2uint<Int, UInt>(block[index(3, 2, 0)].intVal); // 38<-11
  block[index(3, 2, 0)].uintVal = int2uint<Int, UInt>(block[index(2, 1, 0)].intVal); // 11<-6
  block[index(2, 1, 0)].uintVal = int2uint<Int, UInt>(block[index(1, 1, 0)].intVal); // 6<-5
  block[index(1, 1, 0)].uintVal = int2uint<Int, UInt>(block[index(1, 0, 1)].intVal); // 5<-17
  block[index(1, 0, 1)].uintVal = int2uint<Int, UInt>(block[index(3, 0, 0)].intVal); // 17<-3
  block[index(3, 0, 0)].uintVal = int2uint<Int, UInt>(block[index(0, 0, 1)].intVal); // 3<-16
  block[index(0, 0, 1)].uintVal = int2uint<Int, UInt>(block[index(0, 1, 2)].intVal); // 16<-36
  block[index(0, 1, 2)].uintVal = int2uint<Int, UInt>(block[index(1, 3, 1)].intVal); // 36<-29
  block[index(1, 3, 1)].uintVal = int2uint<Int, UInt>(block[index(1, 3, 0)].intVal); // 29<-13
  block[index(1, 3, 0)].uintVal = int2uint<Int, UInt>(block[index(0, 2, 1)].intVal); // 13<-24
  block[index(0, 2, 1)].uintVal = int2uint<Int, UInt>(block[index(2, 0, 2)].intVal); // 24<-34
  block[index(2, 0, 2)].uintVal = int2uint<Int, UInt>(block[index(2, 2, 1)].intVal); // 34<-26
  block[index(2, 2, 1)].uintVal = int2uint<Int, UInt>(block[index(3, 1, 0)].intVal); // 26<-7
  block[index(3, 1, 0)].uintVal = int2uint<Int, UInt>(temp);                         // 7<-2

  block[8].uintVal = int2uint<Int, UInt>(block[index(0, 2, 0)].intVal); // 8<-8

  temp = block[9].intVal; // hold 9s value
  block[9].uintVal = int2uint<Int, UInt>(block[index(0, 0, 2)].intVal); // 9<-32
  block[index(0, 0, 2)].uintVal = int2uint<Int, UInt>(block[index(1, 2, 2)].intVal); // 32<-41
  block[index(1, 2, 2)].uintVal = int2uint<Int, UInt>(block[index(2, 3, 0)].intVal); // 41<-14
  block[index(2, 3, 0)].uintVal = int2uint<Int, UInt>(temp);                         // 14<-9
  
  temp = block[10].intVal; // hold 10s value 
  block[10].uintVal = int2uint<Int, UInt>(block[index(1, 1, 1)].intVal); // 10<-21
  block[index(1, 1, 1)].uintVal = int2uint<Int, UInt>(block[index(1, 2, 1)].intVal); // 21<-25
  block[index(1, 2, 1)].uintVal = int2uint<Int, UInt>(temp); // 25<-10

  temp = block[12].intVal; // hold 12s value
  block[12].uintVal = int2uint<Int, UInt>(block[index(2, 0, 1)].intVal); // 12<-18
  block[index(2, 0, 1)].uintVal = int2uint<Int, UInt>(temp); // 18<-12

  temp = block[19].intVal; // hold 19s value
  block[19].uintVal = int2uint<Int, UInt>(block[index(0, 0, 3)].intVal); // 19<-48
  block[index(0, 0, 3)].uintVal = int2uint<Int, UInt>(block[index(2, 3, 1)].intVal); // 48<-30
  block[index(2, 3, 1)].uintVal = int2uint<Int, UInt>(block[index(1, 0, 3)].intVal); // 30<-49
  block[index(1, 0, 3)].uintVal = int2uint<Int, UInt>(block[index(2, 1, 3)].intVal); // 49<-54
  block[index(2, 1, 3)].uintVal = int2uint<Int, UInt>(block[index(3, 2, 2)].intVal); // 54<-43
  block[index(3, 2, 2)].uintVal = int2uint<Int, UInt>(block[index(0, 2, 3)].intVal); // 43<-56
  block[index(0, 2, 3)].uintVal = int2uint<Int, UInt>(block[index(2, 2, 3)].intVal); // 56<-58
  block[index(2, 2, 3)].uintVal = int2uint<Int, UInt>(block[index(3, 1, 3)].intVal); // 58<-55
  block[index(3, 1, 3)].uintVal = int2uint<Int, UInt>(block[index(2, 3, 2)].intVal); // 55<-46
  block[index(2, 3, 2)].uintVal = int2uint<Int, UInt>(block[index(3, 1, 2)].intVal); // 46<-39
  block[index(3, 1, 2)].uintVal = int2uint<Int, UInt>(block[index(3, 0, 2)].intVal); // 39<-35
  block[index(3, 0, 2)].uintVal = int2uint<Int, UInt>(block[index(3, 1, 1)].intVal); // 35<-23
  block[index(3, 1, 1)].uintVal = int2uint<Int, UInt>(block[index(0, 2, 2)].intVal); // 23<-40
  block[index(0, 2, 2)].uintVal = int2uint<Int, UInt>(block[index(0, 3, 2)].intVal); // 40<-44
  block[index(0, 3, 2)].uintVal = int2uint<Int, UInt>(block[index(2, 2, 2)].intVal); // 44<-42
  block[index(2, 2, 2)].uintVal = int2uint<Int, UInt>(block[index(2, 0, 3)].intVal); // 42<-50
  block[index(2, 0, 3)].uintVal = int2uint<Int, UInt>(block[index(1, 2, 3)].intVal); // 50<-57
  block[index(1, 2, 3)].uintVal = int2uint<Int, UInt>(block[index(1, 3, 3)].intVal); // 57<-61
  block[index(1, 3, 3)].uintVal = int2uint<Int, UInt>(block[index(3, 2, 3)].intVal); // 61<-59
  block[index(3, 2, 3)].uintVal = int2uint<Int, UInt>(block[index(3, 3, 1)].intVal); // 59<-31
  block[index(3, 3, 1)].uintVal = int2uint<Int, UInt>(block[index(0, 1, 3)].intVal); // 31<-52
  block[index(0, 1, 3)].uintVal = int2uint<Int, UInt>(block[index(3, 0, 3)].intVal); // 52<-51
  block[index(3, 0, 3)].uintVal = int2uint<Int, UInt>(block[index(0, 3, 3)].intVal); // 51<-60
  block[index(0, 3, 3)].uintVal = int2uint<Int, UInt>(block[index(2, 3, 3)].intVal); // 60<-62
  block[index(2, 3, 3)].uintVal = int2uint<Int, UInt>(block[index(3, 3, 2)].intVal); // 62<-47
  block[index(3, 3, 2)].uintVal = int2uint<Int, UInt>(block[index(1, 3, 2)].intVal); // 47<-45
  block[index(1, 3, 2)].uintVal = int2uint<Int, UInt>(block[index(3, 2, 1)].intVal); // 45<-27
  block[index(3, 2, 1)].uintVal = int2uint<Int, UInt>(temp); // 27<-19

  block[28].uintVal = int2uint<Int, UInt>(block[index(0, 3, 1)].intVal); // 28<-28

  block[63].uintVal = int2uint<Int, UInt>(block[index(3, 3, 3)].intVal); // 63<-63
  #undef index
}

template <typename Int, typename UInt, int BlockSize>
inline
void fwd_order(UInt* ublock, const Int* iblock)
{
//   const auto perm = get_perm<BlockSize>();

// #pragma unroll BlockSize
//   for (int i = 0; i < BlockSize; i++)
//     ublock[i] = int2uint<Int, UInt>(iblock[perm[i]]);
  if constexpr(BlockSize == 4)
  {
    ublock[0] = int2uint<Int, UInt>(iblock[0]);
    ublock[1] = int2uint<Int, UInt>(iblock[1]);
    ublock[2] = int2uint<Int, UInt>(iblock[2]);
    ublock[3] = int2uint<Int, UInt>(iblock[3]);
  }
  else if (BlockSize == 16)
  {
#define index(i, j) ((i) + 4 * (j))
    ublock[0] = int2uint<Int, UInt>(iblock[index(0, 0)]);
    ublock[1] = int2uint<Int, UInt>(iblock[index(1, 0)]);
    ublock[2] = int2uint<Int, UInt>(iblock[index(0, 1)]);
    ublock[3] = int2uint<Int, UInt>(iblock[index(1, 1)]);
    ublock[4] = int2uint<Int, UInt>(iblock[index(2, 0)]);
    ublock[5] = int2uint<Int, UInt>(iblock[index(0, 2)]);
    ublock[6] = int2uint<Int, UInt>(iblock[index(2, 1)]);
    ublock[7] = int2uint<Int, UInt>(iblock[index(1, 2)]);
    ublock[8] = int2uint<Int, UInt>(iblock[index(3, 0)]);
    ublock[9] = int2uint<Int, UInt>(iblock[index(0, 3)]);
    ublock[10] = int2uint<Int, UInt>(iblock[index(2, 2)]);
    ublock[11] = int2uint<Int, UInt>(iblock[index(3, 1)]);
    ublock[12] = int2uint<Int, UInt>(iblock[index(1, 3)]);
    ublock[13] = int2uint<Int, UInt>(iblock[index(3, 2)]);
    ublock[14] = int2uint<Int, UInt>(iblock[index(2, 3)]);
    ublock[15] = int2uint<Int, UInt>(iblock[index(3, 3)]);
#undef index
  }
  else
  {
    const auto perm = get_perm<BlockSize>();

  #pragma unroll BlockSize
    for (int i = 0; i < BlockSize; i++)
      ublock[i] = int2uint<Int, UInt>(iblock[perm[i]]);
  }
}

template <typename UInt, int BlockSize>
inline 
uint encode_ints(UInt const * const ublock, BlockWriter& writer, const uint maxbits, const uint maxprec)
{
  constexpr int intprec = traits<UInt>::precision;
  const int kmin = ::sycl::max<int>(intprec - (int)maxprec, 0);//intprec > maxprec ? intprec - maxprec : 0;
  int bits = maxbits;

  int n = 0;
  UInt mask = (UInt)1 << (intprec-1);
  for (int k = intprec-1; bits && k >= kmin; k--, mask >>= 1) {
    // step 1: extract bit plane #k to x
    uint64 x = 0;

    // #pragma unroll BlockSize
    for (int i = 0; i < BlockSize; i++) {
      if (ublock[i] & mask) x |= (uint64)1 << i;
      //x += (uint64)((ublock[i] >> k) & 1u) << i;
    }
    // step 2: encode first n bits of bit plane
    const int m = ::sycl::min(n, bits);
    bits -= m;
    x = writer.write_bits(x, m);
    // step 3: unary run-length encode remainder of bit plane
    for (; n < BlockSize && bits && (bits--, writer.write_bit(!!x)); x >>= 1, n++) {
      for (; n < BlockSize - 1 && bits && (bits--, !writer.write_bit(x & 1u)); x >>= 1, n++) {
      }
    }
  }

  // output any buffered bits
  writer.flush();

  return maxbits - bits;
}

template <typename UInt, int BlockSize>
inline 
uint encode_ints_prec(UInt const * const ublock, BlockWriter& writer, uint maxprec)
{
  const BlockWriter::Offset offset = writer.wtell();
  constexpr int intprec = traits<UInt>::precision;
  const int kmin = ::sycl::max<int>(intprec - (int)maxprec, 0);//intprec > maxprec ? intprec - maxprec : 0;

  int n = 0;
  UInt mask = (UInt)1 << (intprec-1);
  for (int k = intprec-1; k >= kmin; k--, mask >>= 1) {
    // step 1: extract bit plane #k to x
    uint64 x = 0;
    
    
    //#pragma unroll BlockSize
    for (int i = 0; i < BlockSize; i++) {
        if (ublock[i] & mask) x |= (uint64)1 << i;
        
        //x += (uint64)((ublock[i] >> k) & 1u) << i;
    }
       
    // step 2: encode first n bits of bit plane
    x = writer.write_bits(x, n);

    // step 3: unary run-length encode remainder of bit plane
    for (; n < BlockSize && writer.write_bit(!!x); x >>= 1, n++)
      for (; n < BlockSize - 1 && !writer.write_bit(x & 1u); x >>= 1, n++)
        ;
  }

  // compute number of bits written
  const uint bits = (uint)(writer.wtell() - offset);

  // output any buffered bits
  writer.flush();

  return bits;
}

// common integer and block-floating-point encoder
template <typename Int, int BlockSize>
inline 
uint encode_int_block(
  Int* iblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec)
{
  // perform decorrelating transform
  fwd_xform<Int, BlockSize>()(iblock);

#if ZFP_ROUNDING_MODE == ZFP_ROUND_FIRST
  // bias values to achieve proper rounding
  fwd_round<Int, BlockSize>(iblock, maxprec);
#endif

  // reorder signed coefficients and convert to unsigned integer
  typedef typename traits<Int>::UInt UInt;
  UInt ublock[BlockSize];
  fwd_order<Int, UInt, BlockSize>(ublock, iblock);

  // encode integer coefficients
  uint bits = with_maxbits<BlockSize>(maxbits, maxprec)
                ? encode_ints<UInt, BlockSize>(ublock, writer, maxbits, maxprec)
                : encode_ints_prec<UInt, BlockSize>(ublock, writer, maxprec);

  return ::sycl::max(minbits, bits);
}

// generic encoder for floating point
template <typename Scalar, int BlockSize>
inline 
uint encode_float_block(
  Scalar* fblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp)
{
  uint bits = 1;
  // compute maximum exponent
  const int emax = max_exponent<Scalar, BlockSize>(fblock);
  maxprec = precision<BlockSize>(emax, maxprec, minexp);
  uint e = maxprec ? emax + traits<Scalar>::ebias : 0;
  // encode block only if biased exponent is nonzero
  if (e) {
    // encode common exponent
    bits += traits<Scalar>::ebits;
    writer.write_bits(2 * e + 1, bits);
    // perform forward block-floating-point transform
    typedef typename traits<Scalar>::Int Int;
    Int iblock[BlockSize];
    fwd_cast<Scalar, Int, BlockSize>(iblock, fblock, emax);
    // encode integer block
    bits += encode_int_block<Int, BlockSize>(
        iblock, writer, ::sycl::max(minbits, bits) - bits,
        ::sycl::max(maxbits, bits) - bits, maxprec);
  }

  return ::sycl::max(minbits, bits);
}

// common integer and block-floating-point encoder
template <typename Int, int BlockSize>
inline 
uint encode_int_block(
  Inplace<Int>* iblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec)
{
  // perform decorrelating transform
  fwd_xform<Int, BlockSize>()((Int*)iblock);

#if ZFP_ROUNDING_MODE == ZFP_ROUND_FIRST
  // bias values to achieve proper rounding
  fwd_round<Int, BlockSize>(iblock, maxprec);
#endif

  // reorder signed coefficients and convert to unsigned integer
  typedef typename traits<Int>::UInt UInt;
  fwd_order<Int, UInt, BlockSize>(iblock);

  // encode integer coefficients
  uint bits = with_maxbits<BlockSize>(maxbits, maxprec)
                ? encode_ints<UInt, BlockSize>((UInt*)iblock, writer, maxbits, maxprec)
                : encode_ints_prec<UInt, BlockSize>((UInt*)iblock, writer, maxprec);

  return ::sycl::max(minbits, bits);
}

// generic encoder for floating point
template <typename Scalar, int BlockSize>
inline 
uint encode_float_block(
  Inplace<Scalar>* fblock,
  BlockWriter& writer,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp)
{
  uint bits = 1;
  // compute maximum exponent
  const int emax = max_exponent<Scalar, BlockSize>(fblock);
  maxprec = precision<BlockSize>(emax, maxprec, minexp);
  uint e = maxprec ? emax + traits<Scalar>::ebias : 0;
  // encode block only if biased exponent is nonzero
  if (e) {
    // encode common exponent
    bits += traits<Scalar>::ebits;
    writer.write_bits(2 * e + 1, bits);
    // perform forward block-floating-point transform
    typedef typename traits<Scalar>::Int Int;
    fwd_cast<Scalar, Int, BlockSize>((Int*)fblock, (Scalar*)fblock, emax);
    
    // encode integer block
    bits += encode_int_block<Int, BlockSize>(
        (Inplace<Int>*)fblock, writer, ::sycl::max(minbits, bits) - bits,
        ::sycl::max(maxbits, bits) - bits, maxprec);
  }

  return ::sycl::max(minbits, bits);
}


// generic encoder
template <typename Scalar, int BlockSize>
struct encode_block;

// encoder specialization for ints
template <int BlockSize>
struct encode_block<int, BlockSize> {
  inline 
  uint operator()(int* iblock, BlockWriter& writer, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return encode_int_block<int, BlockSize>(iblock, writer, minbits, maxbits,
                                            maxprec);
  }
};

// encoder specialization for long longs
template <int BlockSize>
struct encode_block<long long, BlockSize> {
  inline 
  uint operator()(long long* iblock, BlockWriter& writer, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return encode_int_block<long long, BlockSize>(
        iblock, writer, minbits, maxbits, maxprec);
  }
};

// encoder specialization for floats
template <int BlockSize>
struct encode_block<float, BlockSize> {
  inline 
  uint operator()(float* fblock, BlockWriter& writer, uint minbits, uint maxbits, uint maxprec, int minexp) const
  {
    return encode_float_block<float, BlockSize>(fblock, writer, minbits,
                                                maxbits, maxprec, minexp);
  }
};

// encoder specialization for doubles
template <int BlockSize>
struct encode_block<double, BlockSize> {
  inline uint operator()(double *fblock, BlockWriter &writer,
                                       uint minbits, uint maxbits, uint maxprec,
                                       int minexp) const
  {
    return encode_float_block<double, BlockSize>(fblock, writer, minbits,
                                                 maxbits, maxprec, minexp);
  }
};

// inplace encoder specialization for ints
template <int BlockSize>
struct encode_block<Inplace<int>, BlockSize> {
  inline 
  uint operator()(Inplace<int>* iblock, BlockWriter& writer, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return encode_int_block<int, BlockSize>(iblock, writer, minbits, maxbits,
                                            maxprec);
  }
};

// inplace encoder specialization for long longs
template <int BlockSize>
struct encode_block<Inplace<long long>, BlockSize> {
  inline 
  uint operator()(Inplace<long long>* iblock, BlockWriter& writer, uint minbits, uint maxbits, uint maxprec, int) const
  {
    return encode_int_block<long long, BlockSize>(
        iblock, writer, minbits, maxbits, maxprec);
  }
};

// inplace encoder specialization for floats
template <int BlockSize>
struct encode_block<Inplace<float>, BlockSize> {
  inline 
  uint operator()(Inplace<float>* fblock, BlockWriter& writer, uint minbits, uint maxbits, uint maxprec, int minexp) const
  {
    return encode_float_block<float, BlockSize>(fblock, writer, minbits,
                                                maxbits, maxprec, minexp);
  }
};

// encoder specialization for doubles
template <int BlockSize>
struct encode_block<Inplace<double>, BlockSize> {
  inline 
  uint operator()(Inplace<double> *fblock, BlockWriter &writer,
                                       uint minbits, uint maxbits, uint maxprec,
                                       int minexp) const
  {
    return encode_float_block<double, BlockSize>(fblock, writer, minbits,
                                                 maxbits, maxprec, minexp);
  }
};

// forward declarations
template <typename T>
unsigned long long
encode1(
  const T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_sycl* params,
  Word* d_stream,
  ushort* d_index,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
);

template <typename T>
unsigned long long
encode2(
  const T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_sycl* params,
  Word* d_stream,
  ushort* d_index,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
);

template <typename T, int SgSize>
unsigned long long
encode3(
  const T* d_data,
  const size_t size[],
  const ptrdiff_t stride[],
  const zfp_exec_params_sycl* params,
  Word* d_stream,
  ushort* d_index,
  uint minbits,
  uint maxbits,
  uint maxprec,
  int minexp
);

} // namespace internal

// encode field from d_data to d_stream
template <typename T>
unsigned long long
encode(
  const T* d_data,                    // field data device pointer
  const size_t size[],                // field dimensions
  const ptrdiff_t stride[],           // field strides
  const zfp_exec_params_sycl* params, // execution parameters
  Word* d_stream,                     // compressed bit stream device pointer
  ushort* d_index,                    // block index device pointer
  uint minbits,                       // minimum compressed #bits/block
  uint maxbits,                       // maximum compressed #bits/block
  uint maxprec,                       // maximum uncompressed #bits/value
  int minexp                          // minimum bit plane index
)
{
  unsigned long long bits_written = 0;

  //internal::ErrorCheck error;

  uint dims = size[0] ? size[1] ? size[2] ? 3 : 2 : 1 : 0;
  switch (dims) {
    case 1:
      bits_written = internal::encode1<T>(d_data, size, stride, params, d_stream, d_index, minbits, maxbits, maxprec, minexp);
      break;
    case 2:
      bits_written = internal::encode2<T>(d_data, size, stride, params, d_stream, d_index, minbits, maxbits, maxprec, minexp);
      break;
    case 3:
      switch (params->min_sub_group_size)
      {
      case 8:
        bits_written = internal::encode3<T, 8>(d_data, size, stride, params, d_stream, d_index, minbits, maxbits, maxprec, minexp);
        break;
      case 16:
        bits_written = internal::encode3<T, 16>(d_data, size, stride, params, d_stream, d_index, minbits, maxbits, maxprec, minexp);
        break;
      default:
        throw std::runtime_error("Unsupported sub-group size");
        break;
      }
      break;
    default:
      break;
  }

  return bits_written;
}

} // namespace sycl
} // namespace zfp

#endif
