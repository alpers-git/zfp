#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef ZFP_SYCL_CONSTANTS_H
#define ZFP_SYCL_CONSTANTS_H

namespace zfp {
namespace sycl {
namespace internal {


static dpct::global_memory<const unsigned char, 1> perm_1(::sycl::range<1>(4),
                                                          {0, 1, 2, 3});

#define index(i, j) ((i) + 4 * (j))

// order coefficients (i, j) by i + j, then i^2 + j^2
static dpct::global_memory<const unsigned char, 1>
    perm_2(::sycl::range<1>(16), {
                                   index(0, 0), //  0 : 0

                                   index(1, 0), //  1 : 1
                                   index(0, 1), //  2 : 1

                                   index(1, 1), //  3 : 2

                                   index(2, 0), //  4 : 2
                                   index(0, 2), //  5 : 2

                                   index(2, 1), //  6 : 3
                                   index(1, 2), //  7 : 3

                                   index(3, 0), //  8 : 3
                                   index(0, 3), //  9 : 3

                                   index(2, 2), // 10 : 4

                                   index(3, 1), // 11 : 4
                                   index(1, 3), // 12 : 4

                                   index(3, 2), // 13 : 5
                                   index(2, 3), // 14 : 5

                                   index(3, 3), // 15 : 6
                               });

#undef index
#define index(x, y, z) ((x) + 4 * ((y) + 4 * (z)))

// order coefficients (i, j, k) by i + j + k, then i^2 + j^2 + k^2
static dpct::global_memory<const unsigned char, 1>
    perm_3(::sycl::range<1>(64), {
                                   index(0, 0, 0), //  0 : 0

                                   index(1, 0, 0), //  1 : 1
                                   index(0, 1, 0), //  2 : 1
                                   index(0, 0, 1), //  3 : 1

                                   index(0, 1, 1), //  4 : 2
                                   index(1, 0, 1), //  5 : 2
                                   index(1, 1, 0), //  6 : 2

                                   index(2, 0, 0), //  7 : 2
                                   index(0, 2, 0), //  8 : 2
                                   index(0, 0, 2), //  9 : 2

                                   index(1, 1, 1), // 10 : 3

                                   index(2, 1, 0), // 11 : 3
                                   index(2, 0, 1), // 12 : 3
                                   index(0, 2, 1), // 13 : 3
                                   index(1, 2, 0), // 14 : 3
                                   index(1, 0, 2), // 15 : 3
                                   index(0, 1, 2), // 16 : 3

                                   index(3, 0, 0), // 17 : 3
                                   index(0, 3, 0), // 18 : 3
                                   index(0, 0, 3), // 19 : 3

                                   index(2, 1, 1), // 20 : 4
                                   index(1, 2, 1), // 21 : 4
                                   index(1, 1, 2), // 22 : 4

                                   index(0, 2, 2), // 23 : 4
                                   index(2, 0, 2), // 24 : 4
                                   index(2, 2, 0), // 25 : 4

                                   index(3, 1, 0), // 26 : 4
                                   index(3, 0, 1), // 27 : 4
                                   index(0, 3, 1), // 28 : 4
                                   index(1, 3, 0), // 29 : 4
                                   index(1, 0, 3), // 30 : 4
                                   index(0, 1, 3), // 31 : 4

                                   index(1, 2, 2), // 32 : 5
                                   index(2, 1, 2), // 33 : 5
                                   index(2, 2, 1), // 34 : 5

                                   index(3, 1, 1), // 35 : 5
                                   index(1, 3, 1), // 36 : 5
                                   index(1, 1, 3), // 37 : 5

                                   index(3, 2, 0), // 38 : 5
                                   index(3, 0, 2), // 39 : 5
                                   index(0, 3, 2), // 40 : 5
                                   index(2, 3, 0), // 41 : 5
                                   index(2, 0, 3), // 42 : 5
                                   index(0, 2, 3), // 43 : 5

                                   index(2, 2, 2), // 44 : 6

                                   index(3, 2, 1), // 45 : 6
                                   index(3, 1, 2), // 46 : 6
                                   index(1, 3, 2), // 47 : 6
                                   index(2, 3, 1), // 48 : 6
                                   index(2, 1, 3), // 49 : 6
                                   index(1, 2, 3), // 50 : 6

                                   index(0, 3, 3), // 51 : 6
                                   index(3, 0, 3), // 52 : 6
                                   index(3, 3, 0), // 53 : 6

                                   index(3, 2, 2), // 54 : 7
                                   index(2, 3, 2), // 55 : 7
                                   index(2, 2, 3), // 56 : 7

                                   index(1, 3, 3), // 57 : 7
                                   index(3, 1, 3), // 58 : 7
                                   index(3, 3, 1), // 59 : 7

                                   index(2, 3, 3), // 60 : 8
                                   index(3, 2, 3), // 61 : 8
                                   index(3, 3, 2), // 62 : 8

                                   index(3, 3, 3), // 63 : 9
                               });

#undef index

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif