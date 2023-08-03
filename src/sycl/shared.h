#ifndef ZFP_SYCL_SHARED_H
#define ZFP_SYCL_SHARED_H

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

  // typedef ulonglong2 size2;
  // typedef ulonglong3 size3;
  // typedef longlong2 ptrdiff2;
  // typedef longlong3 ptrdiff3;

} // namespace internal
} // namespace sycl
} // namespace zfp
#endif