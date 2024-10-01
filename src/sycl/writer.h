#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#ifndef ZFP_SYCL_WRITER_H
#define ZFP_SYCL_WRITER_H

namespace zfp {
namespace sycl {
namespace internal {

class BlockWriter {
private:
  // number of bits in a buffered word
  static constexpr size_t word_size = sizeof(Word) * CHAR_BIT;

  uint bits;         // number of buffered bits (0 <= bits < word_size)
  Word buffer;       // buffer for incoming bits (buffer < 2^bits)
  Word* ptr;         // pointer to next word to be read
  Word* const begin; // beginning of stream

  // use atomic write to avoid write race conditions
  inline void write_word(Word w) {
    dpct::atomic_fetch_add<::sycl::access::address_space::generic_space>(ptr++,
                                                                       w);
    // *ptr++ = w;//! TO AVOID ATOMICS WILL NOT WORK ON VARIABLE-RATE
  }

public:
  typedef unsigned long long int Offset;

    BlockWriter(Word *data, Offset offset = 0) :
    begin(data)
  {
    wseek(offset);
  }

  // return bit offset to next bit to be written
  inline Offset wtell() const {
    return word_size * (Offset)(ptr - begin) + bits;
  }

  // position stream for writing at given bit offset
  inline 
  void wseek(Offset offset)
  {
    uint n = (uint)(offset % word_size);
    ptr = begin + (size_t)(offset / word_size);
    if (n) {
      buffer = *ptr & (((Word)1 << n) - 1);
      bits = n;
    }
    else {
      buffer = 0;
      bits = 0;
    }
  }

  // write single bit (must be 0 or 1)
  inline uint write_bit(uint bit)
  {
    buffer += (Word)bit << bits;
    if (++bits == word_size) {
      write_word(buffer);
      buffer = 0;
      bits = 0;
    }
    return bit;
  }

  // write 0 <= n <= 64 low bits of value and return remaining bits
  inline uint64 write_bits(uint64 value, uint n)
  {
    // append bit string to buffer
    buffer += (Word)(value << bits);
    bits += n;
    // is buffer full?
    if (bits >= word_size) {
      // 1 <= n <= 64; decrement n to ensure valid right shifts below
      value >>= 1;
      n--;
      // assert: 0 <= n < 64; word_size <= bits <= word_size + n
      do {
        // output word_size bits while buffer is full
        bits -= word_size;
        // assert: 0 <= bits <= n
        write_word(buffer);
        // assert: 0 <= n - bits < 64
        buffer = (Word)(value >> (n - bits));
      } while (sizeof(buffer) < sizeof(value) && bits >= word_size);
    }
    // assert: 0 <= bits < word_size
    buffer &= ((Word)1 << bits) - 1;
    // assert: 0 <= n < 64
    return value >> n;
  }

  // append n zero-bits to stream (n >= 0)
  inline 
  void pad(size_t n)
  {
    Offset count = bits;
    for (count += n; count >= word_size; count -= word_size) {
      write_word(buffer);
      buffer = 0;
    }
    bits = (uint)count;
  }

  // write any remaining buffered bits and align stream on next word boundary
  inline uint flush()
  {
    uint count = (word_size - bits) % word_size;
    if (count)
      pad(count);
    return count;
  }
}; // BlockWriter

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
