#ifndef ZFP_SYCL_TRAITS_H
#define ZFP_SYCL_TRAITS_H

#include <cfloat>

namespace zfp {
namespace sycl {
namespace internal {

template <typename T>
struct traits;

template <>
struct traits<int> {
  typedef int Int;
  typedef unsigned int UInt;
  static constexpr bool is_int = true;
  static constexpr int ebits = 0;
  static constexpr int ebias = 0;
  static constexpr int precision = 32;
  static constexpr UInt nbmask = 0xaaaaaaaau;
};

template <>
struct traits<long long int> {
  typedef long long int Int;
  typedef unsigned long long int UInt;
  static constexpr bool is_int = true;
  static constexpr int ebits = 0;
  static constexpr int ebias = 0;
  static constexpr int precision = 64;
  static constexpr UInt nbmask = 0xaaaaaaaaaaaaaaaaull;
};

template <>
struct traits<float> {
  typedef int Int;
  typedef unsigned int UInt;
  static constexpr bool is_int = false;
  static constexpr int ebits = 8;
  static constexpr int ebias = 127;
  static constexpr int precision = 32;
  static constexpr UInt nbmask = 0xaaaaaaaau;
};

template <>
struct traits<double> {
  typedef long long int Int;
  typedef unsigned long long int UInt;
  static constexpr bool is_int = false;
  static constexpr int ebits = 11;
  static constexpr int ebias = 1023;
  static constexpr int precision = 64;
  static constexpr UInt nbmask = 0xaaaaaaaaaaaaaaaaull;
};

template <>
struct traits<unsigned int> {
  typedef int Int;
  typedef unsigned int UInt;
  static constexpr bool is_int = true;
  static constexpr int ebits = 0;
  static constexpr int ebias = 0;
  static constexpr int precision = 32;
  static constexpr UInt nbmask = 0xaaaaaaaau;
};

template <>
struct traits<unsigned long long int> {
  typedef long long int Int;
  typedef unsigned long long int UInt;
  static constexpr bool is_int = true;
  static constexpr int ebits = 0;
  static constexpr int ebias = 0;
  static constexpr int precision = 64;
  static constexpr UInt nbmask = 0xaaaaaaaaaaaaaaaaull;
};

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
