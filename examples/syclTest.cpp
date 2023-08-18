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

    const uint ny = 10;
    const uint nx = 10;
    uint rate = 16;
    const zfp_type type = zfp_type_float;

    //create a c++ vector of linearly increasing values over nx*ny
    std::vector<float> v(nx*ny);
    std::iota(v.begin(), v.end(), 0);

    //create a zfp field and set its type, pointer, and size for 2D
    zfp_field* field = zfp_field_alloc();
    zfp_field_set_type(field, type);
    zfp_field_set_pointer(field, v.data());
    zfp_field_set_size_2d(field, nx, ny);

    //open a stream and allocate memory for compressed data
    zfp_stream* zfp = zfp_stream_open(0);
    rate = zfp_stream_set_rate(zfp, rate, type, 2, zfp_false);
    size_t bufsize = zfp_stream_maximum_size(zfp, field);
    uchar* buffer = new uchar[bufsize];
    bitstream* s = stream_open(buffer, bufsize);
    zfp_stream_set_bit_stream(zfp, s);

    stream_rewind(s); //enter to readmode

    printf("=====Serial execution=====\n");

    printf("----bitream before compression----\n");
    //print the bitstream before compression
    for (size_t i = 0; i < bufsize; i++)
        printf("%02x", buffer[i]);
    
    zfp_stream_rewind(zfp);
    size_t outsize = zfp_compress(zfp, field);
    printf("\n----bitream after compression----\n");
    //print the bitstream after compression
    for (size_t i = 0; i < bufsize; i++)
        printf("%02x", buffer[i]);

    //reset and free the bitstream and zfp stream
    zfp_stream_rewind(zfp);
    stream_close(s);
    zfp_stream_close(zfp);
    //delete[] buffer;

    //create a new bitstream and zfp stream
    uchar* buffer_sycl = new uchar[bufsize];
    s = stream_open(buffer_sycl, bufsize);
    zfp = zfp_stream_open(s);
    zfp_stream_set_rate(zfp, rate, type, 2, zfp_false);
    
    //set execution mode to sycl
    if (!zfp_stream_set_execution(zfp, zfp_exec_sycl)) {
        fprintf(stderr, "execution not available\n");
        return EXIT_FAILURE;
    }

    printf("\n=====SYCL execution=====\n");

    printf("----bitream before compression----\n");
    //print the bitstream before compression
    for (size_t i = 0; i < bufsize; i++)
        printf("%02x", buffer_sycl[i]);

    //compress the data
    zfp_stream_rewind(zfp);
    outsize = zfp_compress(zfp, field);
    printf("\n----bitream after compression----\n");
    //print the bitstream after compression
    for (size_t i = 0; i < bufsize; i++)
        printf("%02x", buffer_sycl[i]);

    return 0;
}
