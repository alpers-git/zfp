#ifndef ZFP_SYCL_VARIABLE_H
#define ZFP_SYCL_VARIABLE_H

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include "shared.h"
#include <cmath>

#include <algorithm>

namespace zfp {
namespace sycl {
namespace internal {

    // *******************************************************************************

    // Copy a chunk of 16-bit stream lengths into the 64-bit offsets array
    // to compute prefix sums. The first value in offsets is the "base" of the prefix sum
    void copy_length(ushort *length,
                                unsigned long long *offsets,
                                unsigned long long first_stream,
                                int nstreams_chunk,
                                const ::sycl::nd_item<3> &item_ct1)
    {
        int index = item_ct1.get_group(2) * item_ct1.get_local_range(2) +
                    item_ct1.get_local_id(2);
        if (index >= nstreams_chunk)
            return;
        offsets[index + 1] = length[first_stream + index];
    }

    void copy_length_launch(ushort *bitlengths,
                            unsigned long long *chunk_offsets,
                            unsigned long long first,
                            int nstreams_chunk)
    {
        ::sycl::range<3> blocks(1, 1, (nstreams_chunk - 1) / 1024 + 1);
        dpct::get_default_queue().parallel_for(
        ::sycl::nd_range<3>(blocks * ::sycl::range<3>(1, 1, 1024),
                        ::sycl::range<3>(1, 1, 1024)),
        [=](::sycl::nd_item<3> item_ct1) {
            copy_length(bitlengths, chunk_offsets, first, nstreams_chunk, item_ct1);
        });
    }

    // *******************************************************************************

    // Each tile loads the compressed but uncompacted data to shared memory.
    // Input alignment can be anything (1-bit) as maxbits is not always a multiple of 8,
    // so the data is aligned on the fly (first bit of the bitstream on bit 0 in shared memory)
    template <uint tile_size>
    inline void load_to_shared(const uint *streams,                   // Input data
                                          uint *sm,                              // Shared memory
                                          const unsigned long long &offset_bits, // Offset in bits for the stream
                                          const uint &length_bits,               // Length in bits for this stream
                                          const int &maxpad32,
                                          const ::sycl::nd_item<3> &item_ct1)                   // Next multiple of 32 of maxbits
    {
        uint misaligned = offset_bits & 31;
        unsigned long long offset_32 = offset_bits / 32;
        for (int i = item_ct1.get_local_id(2); i * 32 < length_bits; i += tile_size)
        {
            // Align even if already aligned
            uint low = streams[offset_32 + i];
            uint high = 0;
            if ((i + 1) * 32 < misaligned + length_bits)
                high = streams[offset_32 + i + 1];
            /*
            DPCT1098:33: The ((upsample(hi, lo) >> (shift & 31)) & 0xFFFFFFFF)
            expression is used instead of the __funnelshift_r call. These two
            expressions do not provide the exact same functionality. Check the
            generated code for potential precision and/or performance issues.
            */
            sm[item_ct1.get_local_id(1) * maxpad32 + i] =
                ((::sycl::upsample<unsigned>(high, low) >> (misaligned & 31)) &
                 0xFFFFFFFF);
        }
    }

    // Read the input bitstreams from shared memory, align them relative to the
    // final output alignment, compact all the aligned bitstreams in sm_out,
    // then write all the data (coalesced) to global memory, using atomics only
    // for the first and last elements
    template <int tile_size, int num_tiles>
    inline void process(bool valid_stream,
                                   unsigned long long &offset0,     // Offset in bits of the first bitstream of the block
                                   const unsigned long long offset, // Offset in bits for this stream
                                   const int &length_bits,          // length of this stream
                                   const int &add_padding,          // padding at the end of the block, in bits
                                   const int &tid,                  // global thread index inside the thread block
                                   uint *sm_in,                     // shared memory containing the compressed input data
                                   uint *sm_out,                    // shared memory to stage the compacted compressed data
                                   uint maxpad32,                   // Leading dimension of the shared memory (padded maxbits)
                                   uint *sm_length,                 // shared memory to compute a prefix-sum inside the block
                                   uint *output,
                                   const ::sycl::nd_item<3> &item_ct1)                    // output pointer
    {
        // All streams in the block will align themselves on the first stream of the block
        int misaligned0 = offset0 & 31;
        int misaligned = offset & 31;
        int off_smin = item_ct1.get_local_id(1) * maxpad32;
        int off_smout = ((int)(offset - offset0) + misaligned0) / 32;
        offset0 /= 32;

        if (valid_stream)
        {
            // Loop on the whole bitstream (including misalignment), 32 bits per thread
            for (int i = item_ct1.get_local_id(2);
                 i * 32 < misaligned + length_bits; i += tile_size)
            {
                // Merge 2 values to create an aligned value
                uint v0 = i > 0 ? sm_in[off_smin + i - 1] : 0;
                uint v1 = sm_in[off_smin + i];
                /*
                DPCT1098:34: The ((upsample(hi, lo) << (shift & 31)) >> 32)
                expression is used instead of the __funnelshift_l call. These
                two expressions do not provide the exact same functionality.
                Check the generated code for potential precision and/or
                performance issues.
                */
                v1 = ((::sycl::upsample<unsigned>(v1, v0) << (misaligned & 31)) >> 32);

                // Mask out neighbor bitstreams
                uint mask = 0xffffffff;
                if (i == 0)
                    mask &= 0xffffffff << misaligned;
                if ((i + 1) * 32 > misaligned + length_bits)
                    mask &= ~(0xffffffff << ((misaligned + length_bits) & 31));

                dpct::atomic_fetch_add<
                    ::sycl::access::address_space::generic_space>(
                    sm_out + off_smout + i, v1 & mask);
            }
        }

        // First thread working on each bistream writes the length in shared memory
        // Add zero-padding bits if needed (last bitstream of last chunk)
        // The extra bits in shared mempory are already zeroed.
        if (item_ct1.get_local_id(2) == 0)
            sm_length[item_ct1.get_local_id(1)] = length_bits + add_padding;

        // This synchthreads protects sm_out and sm_length.
        /*
        DPCT1065:9: Consider replacing ::sycl::nd_item::barrier() with
        ::sycl::nd_item::barrier(::sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();

        // Compute total length for the threadblock
        uint total_length = 0;
        for (int i = tid & 31; i < num_tiles; i += 32)
            total_length += sm_length[i];
        for (int i = 1; i < 32; i *= 2)
            /*
            DPCT1096:42: The right-most dimension of the work-group used in the
            SYCL kernel that calls this function may be less than "32". The
            function "dpct::permute_sub_group_by_xor" may return an unexpected
            result on the CPU device. Modify the size of the work-group to
            ensure that the value of the right-most dimension is a multiple of
            "32".
            */
            total_length += SHFL_XOR(total_length, i);

        // Write the shared memory output data to global memory, using all the threads
        for (int i = tid; i * 32 < misaligned0 + total_length; i += tile_size * num_tiles)
        {
            // Mask out the beginning and end of the block if unaligned
            uint mask = 0xffffffff;
            if (i == 0)
                mask &= 0xffffffff << misaligned0;
            if ((i + 1) * 32 > misaligned0 + total_length)
                mask &= ~(0xffffffff << ((misaligned0 + total_length) & 31));
            // Reset the shared memory to zero for the next iteration.
            uint value = sm_out[i];
            sm_out[i] = 0;
            // Write to global memory. Use atomicCAS for partially masked values
            // Working in-place, the output buffer has not been memset to zero
            if (mask == 0xffffffff)
                output[offset0 + i] = value;
            else
            {
                uint assumed, old = output[offset0 + i];
                do
                {
                    assumed = old;
                    old = dpct::atomic_compare_exchange_strong<
                        ::sycl::access::address_space::generic_space>(
                        output + offset0 + i, assumed,
                        (assumed & ~mask) + (value & mask));
                } while (assumed != old);
            }
        }
    }

    // In-place bitstream concatenation: compacting blocks containing different number
    // of bits, with the input blocks stored in bins of the same size
    // Using a 2D tile of threads,
    // threadIdx.y = Index of the stream
    // threadIdx.x = Threads working on the same stream
    // Must launch dim3(tile_size, num_tiles, 1) threads per block.
    // Offsets has a length of (nstreams_chunk + 1), offsets[0] is the offset in bits
    // where stream 0 starts, it must be memset to zero before launching the very first chunk,
    // and is updated at the end of this kernel.
    template <int tile_size, int num_tiles>
    
        void concat_bitstreams_chunk(uint *__restrict__ streams,
                                                unsigned long long *__restrict__ offsets,
                                                unsigned long long first_stream_chunk,
                                                int nstreams_chunk,
                                                bool last_chunk,
                                                int maxbits,
                                                int maxpad32,
                                                const ::sycl::nd_item<3> &item_ct1,
                                                ::sycl::atomic_ref<unsigned int, ::sycl::memory_order::seq_cst, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space> &sync_ct1,
                                                uint8_t *dpct_local,
                                                uint *sm_length)
    {

        auto sm_in = (uint *)dpct_local; // sm_in[num_tiles * maxpad32]
        uint *sm_out = sm_in + num_tiles * maxpad32; // sm_out[num_tiles * maxpad32 + 2]
        int tid = item_ct1.get_local_id(1) * tile_size + item_ct1.get_local_id(2);
        int grid_stride = item_ct1.get_group_range(2) * num_tiles;
        int first_bitstream_block = item_ct1.get_group(2) * num_tiles;
        int my_stream = first_bitstream_block + item_ct1.get_local_id(1);

        // Zero the output shared memory. Will be reset again inside process().
        for (int i = tid; i < num_tiles * maxpad32 + 2; i += tile_size * num_tiles)
            sm_out[i] = 0;

        // Loop on all the bitstreams of the current chunk, using the whole resident grid.
        // All threads must enter this loop, as they have to synchronize inside.
        for (int i = 0; i < nstreams_chunk; i += grid_stride)
        {
            bool valid_stream = my_stream + i < nstreams_chunk;
            bool active_thread_block = first_bitstream_block + i < nstreams_chunk;
            unsigned long long offset0 = 0;
            unsigned long long offset = 0;
            uint length_bits = 0;
            uint add_padding = 0;
            if (active_thread_block)
                offset0 = offsets[first_bitstream_block + i];

            if (valid_stream)
            {
                offset = offsets[my_stream + i];
                unsigned long long offset_bits = (first_stream_chunk + my_stream + i) * maxbits;
                unsigned long long next_offset_bits = offsets[my_stream + i + 1];
                length_bits = (uint)(next_offset_bits - offset);
                load_to_shared<tile_size>(streams, sm_in, offset_bits,
                                          length_bits, maxpad32, item_ct1);
                if (last_chunk && (my_stream + i == nstreams_chunk - 1))
                {
                    uint partial = next_offset_bits & 63;
                    add_padding = (64 - partial) & 63;
                }
            }

            // Check if there is overlap between input and output at the grid level.
            // Grid sync if needed, otherwise just syncthreads to protect the shared memory.
            // All the threads launched must participate in a grid::sync
            int last_stream = ::sycl::min(nstreams_chunk, i + grid_stride);
            unsigned long long writing_to = (offsets[last_stream] + 31) / 32;
            unsigned long long reading_from = (first_stream_chunk + i) * maxbits;
            if (writing_to >= reading_from)
                dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);
            else
                /*
                DPCT1065:11: Consider replacing ::sycl::nd_item::barrier() with
                ::sycl::nd_item::barrier(::sycl::access::fence_space::local_space)
                for better performance if there is no access to global memory.
                */
                item_ct1.barrier();

            // Compact the shared memory data of the whole thread block and write it to global memory
            if (active_thread_block)
                process<tile_size, num_tiles>(
                    valid_stream, offset0, offset, length_bits, add_padding,
                    tid, sm_in, sm_out, maxpad32, sm_length, streams, item_ct1);
        }

        // Reset the base of the offsets array, for the next chunk's prefix sum
        if (item_ct1.get_group(2) == 0 && tid == 0)
            offsets[0] = offsets[nstreams_chunk];
    }

    void chunk_process_launch(uint *streams,
                              unsigned long long *chunk_offsets,
                              unsigned long long first,
                              int nstream_chunk,
                              bool last_chunk,
                              int nbitsmax,
                              int num_sm)
    {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  ::sycl::queue &q_ct1 = dev_ct1.default_queue();
        int maxpad32 = (nbitsmax + 31) / 32;
        void *kernelArgs[] = {(void *)&streams,
                              (void *)&chunk_offsets,
                              (void *)&first,
                              (void *)&nstream_chunk,
                              (void *)&last_chunk,
                              (void *)&nbitsmax,
                              (void *)&maxpad32};
        // Increase the number of threads per ZFP block ("tile") as nbitsmax increases
        // Compromise between coalescing, inactive threads and shared memory size <= 48KB
        // Total shared memory used = (2 * num_tiles * maxpad + 2) x 32-bit dynamic shared memory
        // and num_tiles x 32-bit static shared memory.
        // The extra 2 elements of dynamic shared memory are needed to handle unaligned output data
        // and potential zero-padding to the next multiple of 64 bits.
        // Block sizes set so that the shared memory stays < 48KB.
        int max_blocks = 0;
        if (nbitsmax <= 352)
        {
            constexpr int tile_size = 1;
            constexpr int num_tiles = 512;
            size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
            // NOTE: Adding num_tiles * sizeof(uint) to account for locally allocated shared memory in in kernel "concat_bitstreams_chunk".
            // See https://oneapi-src.github.io/SYCLomatic/dev_guide/diagnostic_ref/dpct1111.html
            dpct::experimental::calculate_max_active_wg_per_xecore(
                &max_blocks, tile_size * num_tiles,
                shmem + num_tiles * sizeof(uint));
            max_blocks *= num_sm;
            max_blocks = std::min(nstream_chunk, max_blocks);
            ::sycl::range<3> threads(1, num_tiles, tile_size);
            /*
            DPCT1049:12: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
    {
      dpct::global_memory<unsigned int, 0> d_sync_ct1(0);
      unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
      dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();
      q_ct1
          .submit([&](::sycl::handler &cgh) {
            ::sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                ::sycl::range<1>(shmem), cgh);
            ::sycl::local_accessor<uint, 1> sm_length_acc_ct1(
                ::sycl::range<1>(num_tiles), cgh);

            auto streams_ct0 = *(uint *__restrict *)kernelArgs[0];
            auto offsets_ct1 = *(unsigned long long *__restrict *)kernelArgs[1];
            auto first_stream_chunk_ct2 = *(unsigned long long *)kernelArgs[2];
            auto nstreams_chunk_ct3 = *(int *)kernelArgs[3];
            auto last_chunk_ct4 = *(bool *)kernelArgs[4];
            auto maxbits_ct5 = *(int *)kernelArgs[5];
            auto maxpad32_ct6 = *(int *)kernelArgs[6];

            cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(1, 1, max_blocks) * threads,
                                  threads),
                [=](::sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                      auto atm_sync_ct1 = ::sycl::atomic_ref<
                          unsigned int, ::sycl::memory_order::seq_cst,
                          ::sycl::memory_scope::device,
                          ::sycl::access::address_space::global_space>(
                          sync_ct1[0]);
                      concat_bitstreams_chunk<tile_size, num_tiles>(
                          streams_ct0, offsets_ct1, first_stream_chunk_ct2,
                          nstreams_chunk_ct3, last_chunk_ct4, maxbits_ct5,
                          maxpad32_ct6, item_ct1, atm_sync_ct1,
                          dpct_local_acc_ct1.get_pointer(),
                          sm_length_acc_ct1.get_pointer());
                    });
          })
          .wait();
    }
        }
        else if (nbitsmax <= 1504)
        {
            constexpr int tile_size = 4;
            constexpr int num_tiles = 128;
            size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
            // NOTE: Adding num_tiles * sizeof(uint) to account for locally allocated shared memory in in kernel "concat_bitstreams_chunk".
            // See https://oneapi-src.github.io/SYCLomatic/dev_guide/diagnostic_ref/dpct1111.html
            dpct::experimental::calculate_max_active_wg_per_xecore(
                &max_blocks, tile_size * num_tiles,
                shmem + num_tiles * sizeof(uint));
            max_blocks *= num_sm;
            max_blocks = std::min(nstream_chunk, max_blocks);
            ::sycl::range<3> threads(1, num_tiles, tile_size);
            /*
            DPCT1049:13: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
    {
      dpct::global_memory<unsigned int, 0> d_sync_ct1(0);
      unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
      dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();
      q_ct1
          .submit([&](::sycl::handler &cgh) {
            ::sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                ::sycl::range<1>(shmem), cgh);
            ::sycl::local_accessor<uint, 1> sm_length_acc_ct1(
                ::sycl::range<1>(num_tiles), cgh);

            auto streams_ct0 = *(uint *__restrict *)kernelArgs[0];
            auto offsets_ct1 = *(unsigned long long *__restrict *)kernelArgs[1];
            auto first_stream_chunk_ct2 = *(unsigned long long *)kernelArgs[2];
            auto nstreams_chunk_ct3 = *(int *)kernelArgs[3];
            auto last_chunk_ct4 = *(bool *)kernelArgs[4];
            auto maxbits_ct5 = *(int *)kernelArgs[5];
            auto maxpad32_ct6 = *(int *)kernelArgs[6];

            cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(1, 1, max_blocks) * threads,
                                  threads),
                [=](::sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                      auto atm_sync_ct1 = ::sycl::atomic_ref<
                          unsigned int, ::sycl::memory_order::seq_cst,
                          ::sycl::memory_scope::device,
                          ::sycl::access::address_space::global_space>(
                          sync_ct1[0]);
                      concat_bitstreams_chunk<tile_size, num_tiles>(
                          streams_ct0, offsets_ct1, first_stream_chunk_ct2,
                          nstreams_chunk_ct3, last_chunk_ct4, maxbits_ct5,
                          maxpad32_ct6, item_ct1, atm_sync_ct1,
                          dpct_local_acc_ct1.get_pointer(),
                          sm_length_acc_ct1.get_pointer());
                    });
          })
          .wait();
    }
        }
        else if (nbitsmax <= 6112)
        {
            constexpr int tile_size = 16;
            constexpr int num_tiles = 32;
            size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
            // NOTE: Adding num_tiles * sizeof(uint) to account for locally allocated shared memory in in kernel "concat_bitstreams_chunk".
            // See https://oneapi-src.github.io/SYCLomatic/dev_guide/diagnostic_ref/dpct1111.html
            dpct::experimental::calculate_max_active_wg_per_xecore(
                &max_blocks, tile_size * num_tiles,
                shmem + num_tiles * sizeof(uint));
            max_blocks = std::min(nstream_chunk, max_blocks);
            ::sycl::range<3> threads(1, num_tiles, tile_size);
            /*
            DPCT1049:14: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
    {
      dpct::global_memory<unsigned int, 0> d_sync_ct1(0);
      unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
      dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();
      q_ct1
          .submit([&](::sycl::handler &cgh) {
            ::sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                ::sycl::range<1>(shmem), cgh);
            ::sycl::local_accessor<uint, 1> sm_length_acc_ct1(
                ::sycl::range<1>(num_tiles), cgh);

            auto streams_ct0 = *(uint *__restrict *)kernelArgs[0];
            auto offsets_ct1 = *(unsigned long long *__restrict *)kernelArgs[1];
            auto first_stream_chunk_ct2 = *(unsigned long long *)kernelArgs[2];
            auto nstreams_chunk_ct3 = *(int *)kernelArgs[3];
            auto last_chunk_ct4 = *(bool *)kernelArgs[4];
            auto maxbits_ct5 = *(int *)kernelArgs[5];
            auto maxpad32_ct6 = *(int *)kernelArgs[6];

            cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(1, 1, max_blocks) * threads,
                                  threads),
                [=](::sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                      auto atm_sync_ct1 = ::sycl::atomic_ref<
                          unsigned int, ::sycl::memory_order::seq_cst,
                          ::sycl::memory_scope::device,
                          ::sycl::access::address_space::global_space>(
                          sync_ct1[0]);
                      concat_bitstreams_chunk<tile_size, num_tiles>(
                          streams_ct0, offsets_ct1, first_stream_chunk_ct2,
                          nstreams_chunk_ct3, last_chunk_ct4, maxbits_ct5,
                          maxpad32_ct6, item_ct1, atm_sync_ct1,
                          dpct_local_acc_ct1.get_pointer(),
                          sm_length_acc_ct1.get_pointer());
                    });
          })
          .wait();
    }
        }
        else // Up to 24512 bits, so works even for largest 4D.
        {
            constexpr int tile_size = 64;
            constexpr int num_tiles = 8;
            size_t shmem = (2 * num_tiles * maxpad32 + 2) * sizeof(uint);
            // NOTE: Adding num_tiles * sizeof(uint) to account for locally allocated shared memory in in kernel "concat_bitstreams_chunk".
            // See https://oneapi-src.github.io/SYCLomatic/dev_guide/diagnostic_ref/dpct1111.html
            dpct::experimental::calculate_max_active_wg_per_xecore(
                &max_blocks, tile_size * num_tiles,
                shmem + num_tiles * sizeof(uint));
            max_blocks *= num_sm;
            max_blocks = std::min(nstream_chunk, max_blocks);
            ::sycl::range<3> threads(1, num_tiles, tile_size);
            /*
            DPCT1049:15: The work-group size passed to the SYCL kernel may
            exceed the limit. To get the device limit, query
            info::device::max_work_group_size. Adjust the work-group size if
            needed.
            */
    {
      dpct::global_memory<unsigned int, 0> d_sync_ct1(0);
      unsigned *sync_ct1 = d_sync_ct1.get_ptr(dpct::get_default_queue());
      dpct::get_default_queue().memset(sync_ct1, 0, sizeof(int)).wait();
      q_ct1
          .submit([&](::sycl::handler &cgh) {
            ::sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                ::sycl::range<1>(shmem), cgh);
            ::sycl::local_accessor<uint, 1> sm_length_acc_ct1(
                ::sycl::range<1>(num_tiles), cgh);

            auto streams_ct0 = *(uint *__restrict *)kernelArgs[0];
            auto offsets_ct1 = *(unsigned long long *__restrict *)kernelArgs[1];
            auto first_stream_chunk_ct2 = *(unsigned long long *)kernelArgs[2];
            auto nstreams_chunk_ct3 = *(int *)kernelArgs[3];
            auto last_chunk_ct4 = *(bool *)kernelArgs[4];
            auto maxbits_ct5 = *(int *)kernelArgs[5];
            auto maxpad32_ct6 = *(int *)kernelArgs[6];

            cgh.parallel_for(
                ::sycl::nd_range<3>(::sycl::range<3>(1, 1, max_blocks) * threads,
                                  threads),
                [=](::sycl::nd_item<3> item_ct1)
                    [[intel::reqd_sub_group_size(32)]] {
                      auto atm_sync_ct1 = ::sycl::atomic_ref<
                          unsigned int, ::sycl::memory_order::seq_cst,
                          ::sycl::memory_scope::device,
                          ::sycl::access::address_space::global_space>(
                          sync_ct1[0]);
                      concat_bitstreams_chunk<tile_size, num_tiles>(
                          streams_ct0, offsets_ct1, first_stream_chunk_ct2,
                          nstreams_chunk_ct3, last_chunk_ct4, maxbits_ct5,
                          maxpad32_ct6, item_ct1, atm_sync_ct1,
                          dpct_local_acc_ct1.get_pointer(),
                          sm_length_acc_ct1.get_pointer());
                    });
          })
          .wait();
    }
        }
    }

    // *******************************************************************************

    unsigned long long
    compact_stream(
      Word* d_stream,
      uint maxbits,
      ushort* d_index,
      size_t blocks,
      size_t processors
    )
    {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  ::sycl::queue &q_ct1 = dev_ct1.default_queue();
      unsigned long long bits_written = 0;
      unsigned long long *d_offsets;
      size_t chunk_size;
      size_t lcubtemp;
      void *d_cubtemp;

      if (internal::setup_device_chunking(&chunk_size, &d_offsets, &lcubtemp, &d_cubtemp, processors)) {
        // in-place compact variable-length blocks stored as fixed-length records
        for (size_t i = 0; i < blocks; i += chunk_size) {
          int cur_blocks = chunk_size;
          bool last_chunk = false;
          if (i + chunk_size > blocks) {
            cur_blocks = (int)(blocks - i);
            last_chunk = true;
          }
          // copy the 16-bit lengths in the offset array
          copy_length_launch(d_index, d_offsets, i, cur_blocks);

          // prefix sum to turn length into offsets
          oneapi::dpl::inclusive_scan(
              oneapi::dpl::execution::device_policy(q_ct1), d_offsets,
              d_offsets + cur_blocks + 1, d_offsets);

          // compact the stream array in-place
          chunk_process_launch((uint*)d_stream, d_offsets, i, cur_blocks, last_chunk, maxbits, processors);
        }
        // update compressed size and pad to whole words
        q_ct1.memcpy(&bits_written, d_offsets, sizeof(unsigned long long)).wait();
        bits_written = round_up(bits_written, sizeof(Word) * CHAR_BIT);

        // free temporary buffers
        internal::cleanup_device(NULL, d_offsets);
        internal::cleanup_device(NULL, d_cubtemp);
      }

      return bits_written;
    }

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
