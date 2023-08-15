#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>
#include <numeric>

#include "zfp/array2.hpp"
#include "zfp.h"


#include <CL/sycl.hpp>

int main(int argc, char const *argv[])
{
    printf("zfp sycl test\n");

    const uint ny = 128;
    const uint nx = 128;
    const uint rate = 16;
    //create a c++ vector of linearly increasing values over nx*ny
    std::vector<float> v(nx*ny);
    std::iota(v.begin(), v.end(), 0);

    zfp_field* field = zfp_field_2d(v.data(), zfp_type_float, nx, ny);
    size_t blocks = zfp_field_blocks(field);
    size_t scalars = blocks << (2 * 2);
    size_t bufsize = 2 * scalars * sizeof(uint64);
    void* buffer;
    bitstream* stream;
    zfp_stream* zfp;
    buffer = new uint64[2 * scalars];
    stream = stream_open(buffer, bufsize);
    zfp = zfp_stream_open(stream);
    if (!zfp_stream_set_execution(zfp, zfp_exec_sycl)) {
        fprintf(stderr, "execution not available\n");
        return EXIT_FAILURE;
    }


    return 0;
}
