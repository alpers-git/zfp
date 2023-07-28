#ifndef ZFP_SYCL_SHARED_H
#define ZFP_SYCL_SHARED_H

// report throughput; set via CMake
// #define ZFP_WITH_HIP_PROFILE 1

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


} // namespace internal
} // namespace sycl
} // namespace zfp
#endif