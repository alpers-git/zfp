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
// kernel for initializing prefix sum over zfp block lengths

void
copy_length_kernel(
  unsigned long long* d_offset, // block offsets; first is base of prefix sum
  const ushort* d_length,       // block lengths in bits
  uint blocks_per_chunk         ,
  const ::sycl::nd_item<3> &item_ct1// number of blocks in chunk to process
)
{
  const uint block = item_ct1.get_local_id(2) +
                     item_ct1.get_group(2) * item_ct1.get_local_range(2);
  if (block < blocks_per_chunk)
    d_offset[block + 1] = d_length[block];
}

// initialize prefix sum by copying a chunk of 16-bit lengths to 64-bit offsets
void
copy_length_launch(
  unsigned long long* d_offset, // block offsets; first is base of prefix sum
  const ushort* d_length,       // block lengths in bits
  uint blocks_per_chunk         // number of blocks in chunk to process
)
{
  const dpct::dim3 blocks((int)count_up(blocks_per_chunk, 1024), 1, 1);
  /*
  DPCT1049:5: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  dpct::get_in_order_queue().parallel_for(
      ::sycl::nd_range<3>(blocks * ::sycl::range<3>(1, 1, 1024),
                        ::sycl::range<3>(1, 1, 1024)),
      [=](::sycl::nd_item<3> item_ct1) {
        copy_length_kernel(d_offset, d_length, blocks_per_chunk, item_ct1);
      });
}

// load a single unaligned block to a 32-bit aligned slot in shared memory
template <int tile_size>

inline void
load_block(
  uint32* sm_stream,         // shared-memory buffer of 32-bit aligned slots
  uint words_per_slot,       // slot size in number of 32-bit words
  const uint32* d_stream,    // beginning of uncompacted input stream
  unsigned long long offset, // block offset in bits
  uint length                ,
  const ::sycl::nd_item<3> &item_ct1// block length in bits
)
{
  const uint begin = (uint)offset & 31u; // block start within 32-bit word
  const uint end = begin + length;       // block end relative bit offset

  // advance stream to beginning 32-bit word of block
  d_stream += offset / 32;

  // advance shared-memory pointer to where block is to be stored
  sm_stream += item_ct1.get_local_id(1) * words_per_slot;

  // copy compressed data for one block one 32-bit word at a time
  for (uint i = item_ct1.get_local_id(2); i * 32 < length; i += tile_size) {
    // fetch two consecutive words and funnel shift them to one output word
    uint32 lo = d_stream[i];
    uint32 hi = 0;
    if ((i + 1) * 32 < end)
      hi = d_stream[i + 1];
    /*
    DPCT1017:33: The dpct::funnelshift_r call is used instead of the
    __funnelshift_r call. These two calls do not provide exactly the same
    functionality. Check the potential precision and/or performance issues for
    the generated code.
    */
    sm_stream[i] = dpct::funnelshift_r(lo, hi, begin);
  }
}

// copy a single block from its 32-bit aligned slot to its compacted location
template <int tile_size>

inline void
copy_block(
  uint32* sm_out,                 // shared-memory pointer to compacted chunk
  unsigned long long base_offset, // global offset to first block in subchunk
  unsigned long long offset,      // global offset to this block in bits
  uint length,                    // block length in bits
  const uint32* sm_in,            // shared-memory pointer to uncompacted data
  uint words_per_slot             ,
  const ::sycl::nd_item<3> &item_ct1// slot size in number of 32-bit words
)
{
  const uint begin = (uint)offset & 31u; // block start within 32-bit word
  const uint end = begin + length;       // block end relative bit offset

  // advance shared-memory pointer to block source data
  sm_in += item_ct1.get_local_id(1) * words_per_slot;

  // advance pointer to block destination data
  sm_out += offset / 32 - base_offset / 32;

  for (uint i = item_ct1.get_local_id(2); i * 32 < end; i += tile_size) {
    // fetch two consecutive words and funnel shift them to one output word
    uint32 lo = i > 0 ? sm_in[i - 1] : 0;
    uint32 hi = sm_in[i];
    /*
    DPCT1017:34: The dpct::funnelshift_l call is used instead of the
    __funnelshift_l call. These two calls do not provide exactly the same
    functionality. Check the potential precision and/or performance issues for
    the generated code.
    */
    uint32 word = dpct::funnelshift_l(lo, hi, begin);

    // mask out bits from next block
    if ((i + 1) * 32 > end)
      word &= ~(0xffffffffu << (end & 31u));

    // store (partial) word in a thread-safe manner
    dpct::atomic_fetch_add<::sycl::access::address_space::generic_space>(
        sm_out + i, word);
  }
}

// copy one subchunk of num_tiles blocks from shared to global memory
template <int tile_size, int num_tiles>

inline void
store_subchunk(
  uint32* d_stream,          // output pointer
  unsigned long long offset, // bit offset to first block in subchunk
  uint length,               // subchunk length in bits
  uint32* sm_src,            // compacted compressed data in shared memory
  uint tid                   // global thread index inside the thread block
)
{
  // Copy compacted subchunk from shared memory to its final location in global
  // memory using coalesced writes.  Use atomic only for the first and last
  // word of the subchunk.

  const uint begin = offset & 31u; // block start within 32-bit word
  const uint end = begin + length; // block end relative bit offset

  // advance output pointer to first word of subchunk
  d_stream += offset / 32;

  // use all threads to copy compacted subchunk to global memory
  for (uint i = tid; i * 32 < end; i += num_tiles * tile_size) {
    // fetch word and zero out for next subchunk
    uint32 word = sm_src[i];
    sm_src[i] = 0;

    // mask out the beginning and end of subchunk if unaligned
    uint32 mask = 0xffffffffu;
    if (i == 0)
      mask &= 0xffffffffu << begin;
    if ((i + 1) * 32 > end)
      mask &= ~(0xffffffffu << (end & 31u));

    // write masked bits of word to global memory; for partial-word
    // write, use XOR identities x ^ (x ^ y) = y (when mask is on) and
    // x ^ 0 = x (when mask is off) to select bits from x and y
    if (~mask)
      dpct::atomic_fetch_xor<::sycl::access::address_space::generic_space>(
          &d_stream[i], (d_stream[i] ^ word) & mask);
    else
      d_stream[i] = word;
  }
}

// cooperative kernel for stream compaction of one chunk of blocks
template <int tile_size, int num_tiles>


void
compact_stream_kernel(
  uint32* __restrict__ d_stream,             // compressed bit stream
  unsigned long long* __restrict__ d_offset, // destination bit offsets
  size_t first_block,    // global index of first block in chunk
  uint blocks_per_chunk, // number of blocks per chunk
  uint bits_per_slot,    // number of bits per fixed-size slot holding a block
  uint words_per_slot    ,
  const ::sycl::nd_item<3> &item_ct1,
  ::sycl::atomic_ref<unsigned int, ::sycl::memory_order::seq_cst, ::sycl::memory_scope::device, ::sycl::access::address_space::global_space> &sync_ct1,
  uint8_t *dpct_local// number of 32-bit words per slot
)
{
  // In-place stream compaction of variable-length blocks initially stored in
  // d_stream as fixed-length slots of size bits_per_slot.  Compaction is done
  // in parallel in chunks of blocks_per_chunk blocks.  Each chunk is broken
  // into subchunks of num_tiles (one of {8, 32, 128, 512}) blocks each.
  // Compaction first loads a subchunk of blocks to 32-bit aligned slots in
  // shared memory, sm_in, then compacts that subchunk by concatenating the
  // blocks to sm_out, and then copies the compacted data back to global
  // memory in d_stream.  This repeats for all subchunks of the chunk.  The
  // destination offset of each block in the chunk has already been computed
  // as a prefix sum over block lengths and stored in d_offset, which holds
  // blocks_per_chunk + 1 bit offsets.  At the end of the kernel, d_offset[0]
  // is set to point to the beginning of the next chunk.
  //
  // This parallel compaction is executed by num_tiles * tile_size = 512
  // threads, with one such thread block processing one subchunk at a time.
  // Thread indices are:
  //
  //   threadIdx.x = thread among tile_size threads working on the same block
  //   threadIdx.y = block index within subchunk
  //
  // The caller must launch dim3(tile_size, num_tiles, 1) threads per thread
  // block.  The caller also allocates shared memory for sm_in and sm_out.

  auto sm_in = (uint32 *)dpct_local;
  // sm_out[num_tiles * words_per_slot + 2]
  uint32* sm_out = sm_in + num_tiles * words_per_slot;
  // thread within thread block
  const uint tid =
      item_ct1.get_local_id(2) + item_ct1.get_local_id(1) * tile_size;
  // number of blocks per group
  const uint blocks_per_group = item_ct1.get_group_range(2) * num_tiles;
  // first block in this subchunk
  const uint first_subchunk_block = item_ct1.get_group(2) * num_tiles;

  // zero-initialize compacted buffer (also done in store_subchunk())
  for (uint i = tid; i < num_tiles * words_per_slot + 2; i += num_tiles * tile_size)
    sm_out[i] = 0;

  // compact chunk one group at a time
  for (uint i = 0; i < blocks_per_chunk; i += blocks_per_group) {
    // first block in this subchunk
    const uint base_block = first_subchunk_block + i;
    // block assigned to this thread
    const uint block = base_block + item_ct1.get_local_id(1);
    // is this thread block assigned any compressed blocks?
    const bool active_thread_block = (base_block < blocks_per_chunk);
    // is this thread assigned to valid block?
    const bool valid_block = (block < blocks_per_chunk);
    // destination offset to beginning of subchunk in compacted stream
    const unsigned long long base_offset = active_thread_block ? d_offset[base_block] : 0;
    // destination offset within compacted stream
    const unsigned long long offset_out = d_offset[block];
    // bit length of this block
    const uint length = (uint)(d_offset[block + 1] - offset_out);

    if (valid_block) {
      // source offset within uncompacted stream
      const unsigned long long offset_in = (first_block + block) * bits_per_slot;
      // buffer block in fixed-size slot in shared memory
      load_block<tile_size>(sm_in, words_per_slot, d_stream, offset_in, length,
                            item_ct1);
    }

    // synchronize to ensure entire subchunk is loaded
    /*
    DPCT1118:6: SYCL group functions and algorithms must be encountered in
    converged control flow. You may need to adjust the code.
    */
    /*
    DPCT1065:35: Consider replacing ::sycl::nd_item::barrier() with
    ::sycl::nd_item::barrier(::sycl::access::fence_space::local_space) for better
    performance if there is no access to global memory.
    */
    item_ct1.barrier();

    if (valid_block) {
      // compact subchunk by copying block to target location in shared memory
      copy_block<tile_size>(sm_out, base_offset, offset_out, length, sm_in,
                            words_per_slot, item_ct1);
    }

    // synchronize across group if there is overlap between input and output
    const uint last_block = ::sycl::min(i + blocks_per_group, blocks_per_chunk);
    const size_t output_end = d_offset[last_block] / 32;
    const size_t input_begin = (first_block + i) * bits_per_slot / 32;
    if (output_end < input_begin)
      /*
      DPCT1118:7: SYCL group functions and algorithms must be encountered in
      converged control flow. You may need to adjust the code.
      */
      /*
      DPCT1065:36: Consider replacing ::sycl::nd_item::barrier() with
      ::sycl::nd_item::barrier(::sycl::access::fence_space::local_space) for better
      performance if there is no access to global memory.
      */
      item_ct1.barrier();
    else
      dpct::experimental::nd_range_barrier(item_ct1, sync_ct1);

    // copy compacted subchunk from shared memory to global memory
    if (active_thread_block) {
      const unsigned long long last_offset =
          d_offset[::sycl::min(base_block + num_tiles, blocks_per_chunk)];
      const uint subchunk_length = (uint)(last_offset - base_offset);
      // store compacted subchunk to global memory
      store_subchunk<tile_size, num_tiles>(d_stream, base_offset, subchunk_length, sm_out, tid);
    }
  }

  // update the base of the offset array for the next chunk's prefix sum
  if (item_ct1.get_group(2) == 0 && tid == 0)
    d_offset[0] = d_offset[blocks_per_chunk];
}

// launch stream compaction kernel for one chunk
template <int tile_size, int num_tiles>
bool compact_stream_launch(
    uint32 *d_stream,             // compressed bit stream
    unsigned long long *d_offset, // global bit offsets to blocks in chunk
    size_t first_block,           // index of first block in chunk
    uint blocks_per_chunk,        // number of blocks per chunk
    uint bits_per_slot,           // fixed-size slot size in bits
    uint processors               // number of device multiprocessors
    ) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  ::sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  // Assign number of threads ("tile_size") per zfp block in proportion to
  // bits_per_slot.  Compromise between coalescing, keeping threads active,
  // and limiting shared memory usage.  The total dynamic shared memory used
  // equals (2 * num_tiles * words_per_slot + 2) 32-bit words.  The extra
  // two words of shared memory are needed to handle output data that is not
  // aligned on 32-bit words.  The number of zfp blocks per thread block
  // ("num_tiles") is set to ensure that shared memory is at most 48 KB.

  const uint words_per_slot = count_up(bits_per_slot, 32);
  const size_t shmem = (2 * num_tiles * words_per_slot + 2) * sizeof(uint32);

  // compute number of blocks to process concurrently
  int thread_blocks = 0;
  /*
  DPCT1111:37: Please verify the input arguments of
  "dpct::experimental::calculate_max_active_wg_per_xecore" base on the target
  function "compact_stream_kernel<tile_size, num_tiles>".
  */
 int dpct_placeholder = 0; // TODO: remove this when kernel is fixed
  dpct::experimental::calculate_max_active_wg_per_xecore(
      &thread_blocks, tile_size * num_tiles,
      shmem + dpct_placeholder /* total share local memory size */);
  thread_blocks *= processors;
  thread_blocks =
      std::min(thread_blocks, (int)count_up(blocks_per_chunk, num_tiles));

  void* kernel_args[] = {
    (void *)&d_stream,
    (void *)&d_offset,
    (void *)&first_block,
    (void *)&blocks_per_chunk,
    (void *)&bits_per_slot,
    (void *)&words_per_slot
  };

  /*
  DPCT1049:8: The work-group size passed to the SYCL kernel may exceed the
  limit. To get the device limit, query info::device::max_work_group_size.
  Adjust the work-group size if needed.
  */
  /*
  DPCT1123:9: The kernel function pointer cannot be used in the device code. You
  need to call the kernel function with the correct argument(s) directly.
  According to the kernel function definition, adjusting the dimension of the
  ::sycl::nd_item may also be required.
  */
//   return [&]() {
//     q_ct1.parallel_for(
//         ::sycl::nd_range<3>(::sycl::range<3>(1, 1, thread_blocks) *
//                               ::sycl::range<3>(1, num_tiles, tile_size),
//                           ::sycl::range<3>(1, num_tiles, tile_size)),
//         [=](::sycl::nd_item<3> item_ct1) {
//           ((void *)compact_stream_kernel<tile_size, num_tiles>)();
//         });
//     return 0;
//   }() == 0;
//TODO: FIX HERE
}
catch (::sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

// compact a single chunk of blocks
bool
compact_stream_chunk(
  uint32* d_stream,             // compressed bit stream
  unsigned long long* d_offset, // global bit offsets to blocks in chunk
  size_t first_block,           // index of first block in chunk
  uint blocks_per_chunk,        // number of blocks per chunk
  uint bits_per_slot,           // fixed-size slot size in bits
  uint processors               // number of device multiprocessors
)
{
  const uint bytes_per_slot = count_up(bits_per_slot, 32) * sizeof(uint32);
  const size_t shared_memory = 48 * 1024 - 2 * sizeof(uint32);

  // choose number of tiles such that shared memory usage is at most 48 KB
  if (512 * 2 * bytes_per_slot <= shared_memory) // bits_per_slot <= 352
    return compact_stream_launch< 1, 512>(d_stream, d_offset, first_block, blocks_per_chunk, bits_per_slot, processors);
  if (128 * 2 * bytes_per_slot <= shared_memory) // bits_per_slot <= 1504
    return compact_stream_launch< 4, 128>(d_stream, d_offset, first_block, blocks_per_chunk, bits_per_slot, processors);
  if ( 32 * 2 * bytes_per_slot <= shared_memory) // bits_per_slot <= 6112
    return compact_stream_launch<16,  32>(d_stream, d_offset, first_block, blocks_per_chunk, bits_per_slot, processors);
  if (  8 * 2 * bytes_per_slot <= shared_memory) // bits_per_slot <= 24544
    return compact_stream_launch<64,   8>(d_stream, d_offset, first_block, blocks_per_chunk, bits_per_slot, processors);

  // zfp blocks are at most ZFP_MAX_BITS = 16658 bits < 2084 bytes;
  // should never arrive here

  return false;
}

// zero-pad stream to align it on a whole word

void
align_stream_kernel(
  Word* d_stream,              // compacted, compressed stream
  unsigned long long* d_offset // offset to end of stream
)
{
  const size_t alignment = sizeof(Word) * CHAR_BIT;
  const unsigned long long offset = *d_offset;
  const uint shift = (uint)(offset % alignment);

  if (shift) {
    // mask out any nonzero bits at the end and advance offset
    d_stream[offset / alignment] &= ~(~Word(0) << shift);
    *d_offset = round_up(offset, alignment);
  }
}

// compact in place variable-length blocks stored in fixed-length slots
unsigned long long
compact_stream(
  Word* d_stream,         // pointer to compressed blocks
  uint bits_per_slot,     // fixed size of slots holding variable-length blocks
  const ushort* d_length, // lengths of zfp blocks in bits
  size_t blocks,          // number of zfp blocks
  size_t processors       // number of device multiprocessors
)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  ::sycl::queue &q_ct1 = dev_ct1.in_order_queue();
  bool success = true;
  unsigned long long* d_offset;
  void* d_cubtmp;
  size_t blocks_per_chunk;
  size_t cubtmp_size;

  if (!setup_device_compact(&blocks_per_chunk, &d_offset, processors))
    return 0;

  // perform compaction one chunk of blocks at a time
  for (size_t block = 0; block < blocks && success; block += blocks_per_chunk) {
    // determine chunk size
    size_t chunk_size = std::min(blocks_per_chunk, blocks - block);

    // initialize block offsets to block lengths
    copy_length_launch(d_offset, d_length + block, chunk_size);

    // compute prefix sum to turn block lengths into offsets
    oneapi::dpl::inclusive_scan(oneapi::dpl::execution::device_policy(q_ct1),
                                d_offset, d_offset + chunk_size + 1, d_offset);

    // compact the stream in place
    success = compact_stream_chunk((uint32*)d_stream, d_offset, block, chunk_size, bits_per_slot, processors);
  }

  // update compressed size and pad to whole words
  unsigned long long bits_written = 0;
  if (success) {
    q_ct1.parallel_for(
        ::sycl::nd_range<3>(::sycl::range<3>(1, 1, 1), ::sycl::range<3>(1, 1, 1)),
        [=](::sycl::nd_item<3> item_ct1) {
          align_stream_kernel(d_stream, d_offset);
        });
    q_ct1.memcpy(&bits_written, d_offset, sizeof(bits_written)).wait();
  }

  // free temporary buffers
  cleanup_device(d_offset);
  cleanup_device(d_cubtmp);

  return bits_written;
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
