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
  /*
  DPCT1017:24: The ::sycl::frexp call is used instead of the frexpf call. These
  two calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  ::sycl::frexp(
      x, ::sycl::address_space_cast<::sycl::access::address_space::private_space,
                                  ::sycl::access::decorated::yes, int>(&e));
  return e;
}

template <>
inline 
int get_exponent(double x)
{
  int e;
  /*
  DPCT1017:25: The ::sycl::frexp call is used instead of the frexp call. These two
  calls do not provide exactly the same functionality. Check the potential
  precision and/or performance issues for the generated code.
  */
  ::sycl::frexp(
      x, ::sycl::address_space_cast<::sycl::access::address_space::private_space,
                                  ::sycl::access::decorated::yes, int>(&e));
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
    /*
    DPCT1064:26: Migrated fabs call is used in a macro definition and is not
    valid for all macro uses. Adjust the code.
    */
    Scalar f = ::sycl::fabs((double)(p[i]));
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
#if SYCL_LANGUAGE_VERSION < 8000
#pragma unroll
#else
  #pragma unroll BlockSize
#endif
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
void fwd_order(UInt* ublock, const Int* iblock, unsigned char *perm_1,
               unsigned char *perm_2, unsigned char *perm_3)
{
  const unsigned char *perm = get_perm<BlockSize>(perm_1, perm_2, perm_3);

#if SYCL_LANGUAGE_VERSION < 8000
#pragma unroll
#else
  #pragma unroll BlockSize
#endif
  for (int i = 0; i < BlockSize; i++)
    ublock[i] = int2uint<Int, UInt>(iblock[perm[i]]);
}

template <typename UInt, int BlockSize>
inline 
uint encode_ints(UInt* ublock, BlockWriter& writer, uint maxbits, uint maxprec)
{
  const uint intprec = traits<UInt>::precision;
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;
  uint bits = maxbits;

  for (uint k = intprec, n = 0; bits && k-- > kmin;) {
    // step 1: extract bit plane #k to x
    uint64 x = 0;
#if SYCL_LANGUAGE_VERSION < 8000
#pragma unroll
#else
    #pragma unroll BlockSize
#endif
    for (int i = 0; i < BlockSize; i++)
      x += (uint64)((ublock[i] >> k) & 1u) << i;
    // step 2: encode first n bits of bit plane
    uint m = ::sycl::min(n, bits);
    bits -= m;
    x = writer.write_bits(x, m);
    // step 3: unary run-length encode remainder of bit plane
    for (; n < BlockSize && bits && (bits--, writer.write_bit(!!x)); x >>= 1, n++)
      for (; n < BlockSize - 1 && bits && (bits--, !writer.write_bit(x & 1u)); x >>= 1, n++)
        ;
  }

  // output any buffered bits
  writer.flush();

  return maxbits - bits;
}

template <typename UInt, int BlockSize>
inline 
uint encode_ints_prec(UInt* ublock, BlockWriter& writer, uint maxprec)
{
  const BlockWriter::Offset offset = writer.wtell();
  const uint intprec = traits<UInt>::precision;
  const uint kmin = intprec > maxprec ? intprec - maxprec : 0;

  for (uint k = intprec, n = 0; k-- > kmin;) {
    // step 1: extract bit plane #k to x
    uint64 x = 0;
#if SYCL_LANGUAGE_VERSION < 8000
#pragma unroll
#else
    #pragma unroll BlockSize
#endif
    for (int i = 0; i < BlockSize; i++)
      x += (uint64)((ublock[i] >> k) & 1u) << i;
    // step 2: encode first n bits of bit plane
    x = writer.write_bits(x, n);
    // step 3: unary run-length encode remainder of bit plane
    for (; n < BlockSize && writer.write_bit(!!x); x >>= 1, n++)
      for (; n < BlockSize - 1 && !writer.write_bit(x & 1u); x >>= 1, n++)
        ;
  }

  // compute number of bits written
  uint bits = (uint)(writer.wtell() - offset);

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
  uint maxprec
,
  unsigned char *perm_1,
  unsigned char *perm_2,
  unsigned char *perm_3)
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
  fwd_order<Int, UInt, BlockSize>(ublock, iblock, perm_1, perm_2, perm_3);

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
  int minexp
,
  unsigned char *perm_1,
  unsigned char *perm_2,
  unsigned char *perm_3)
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
        ::sycl::max(maxbits, bits) - bits, maxprec, perm_1, perm_2, perm_3);
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
  uint operator()(int* iblock, BlockWriter& writer, uint minbits, uint maxbits, uint maxprec, int,
                  unsigned char *perm_1, unsigned char *perm_2,
                  unsigned char *perm_3) const
  {
    return encode_int_block<int, BlockSize>(iblock, writer, minbits, maxbits,
                                            maxprec, perm_1, perm_2, perm_3);
  }
};

// encoder specialization for long longs
template <int BlockSize>
struct encode_block<long long, BlockSize> {
  inline 
  uint operator()(long long* iblock, BlockWriter& writer, uint minbits, uint maxbits, uint maxprec, int,
                  unsigned char *perm_1, unsigned char *perm_2,
                  unsigned char *perm_3) const
  {
    return encode_int_block<long long, BlockSize>(
        iblock, writer, minbits, maxbits, maxprec, perm_1, perm_2, perm_3);
  }
};

// encoder specialization for floats
template <int BlockSize>
struct encode_block<float, BlockSize> {
  inline 
  uint operator()(float* fblock, BlockWriter& writer, uint minbits, uint maxbits, uint maxprec, int minexp,
                  unsigned char *perm_1, unsigned char *perm_2,
                  unsigned char *perm_3) const
  {
    return encode_float_block<float, BlockSize>(fblock, writer, minbits,
                                                maxbits, maxprec, minexp,
                                                perm_1, perm_2, perm_3);
  }
};

// encoder specialization for doubles
template <int BlockSize>
struct encode_block<double, BlockSize> {
  SYCL_EXTERNAL inline uint operator()(double *fblock, BlockWriter &writer,
                                       uint minbits, uint maxbits, uint maxprec,
                                       int minexp, unsigned char *perm_1,
                                       unsigned char *perm_2,
                                       unsigned char *perm_3) const
  {
    return encode_float_block<double, BlockSize>(fblock, writer, minbits,
                                                 maxbits, maxprec, minexp,
                                                 perm_1, perm_2, perm_3);
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

template <typename T>
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
      bits_written = internal::encode3<T>(d_data, size, stride, params, d_stream, d_index, minbits, maxbits, maxprec, minexp);
      break;
    default:
      break;
  }

//   if (!error.check("encode"))
//     bits_written = 0;

  return bits_written;
}

} // namespace sycl
} // namespace zfp

#endif