#include <sycl/sycl.hpp>
#include <oneapi/tbb.h>
#include <oneapi/dpl/experimental/kernel_templates>

namespace kt = oneapi::dpl::experimental::kt;
#ifndef ZFP_SYCL_DEVICE_H
#define ZFP_SYCL_DEVICE_H

#define ZFP_DEBUG

#define ZFP_MAGIC 0x7a667000u

#define SYCL_ERR(function_call, custom_msg, return_value) \
try { \
    function_call; \
} catch (const cl::exception& e) { \
    std::cerr << "zfp::sycl : " << custom_msg << e.what() << std::endl; \
    return return_value; \
} 

namespace zfp {
namespace sycl {
namespace internal {

using namespace ::sycl;

bool device_init()
{
  bool success = true;
  queue q(zfp_dev_selector);
  unsigned int* d_word;
  try {
    // Get a SYCL device queue
    // allocate a buffer to store the magic number on the device
    d_word = malloc_device<unsigned int>(1, q);
   }  
  catch (exception const& e) {
      std::cerr << "zfp::sycl : zfp device init - sycl::malloc_device : " << e.what() << std::endl;
      success = false;
  }

  try {
    //launch a kernel to initialize the magic number
    q.submit([&](handler& cgh) {
      cgh.single_task<class device_init_kernel>([=]() {
        d_word[0] = ZFP_MAGIC;
        });
    });
    q.wait();
  }  
  catch (exception const& e) {
      std::cerr << "zfp::sycl : zfp device init - kernel: " << e.what() << std::endl;
      success = false;
  }

  unsigned int h_word;
  try {
    // copy the magic number back to the host
    q.memcpy(&h_word, d_word, sizeof(unsigned int)).wait();

  }  
  catch (exception const& e) {
      std::cerr << "zfp::sycl : zfp device init - memcpy : " << e.what() << std::endl;
      success = false;
  }

  if (h_word != ZFP_MAGIC) {
    std::cerr << "zfp::sycl : zfp device init - memcpy : ZFP_MAGIC mismatch" << std::endl;
    success = false;
  }

  try {
    // free the device buffer
    free(d_word, q);
  }
  catch (exception const& e) {
      std::cerr << "zfp::sycl : zfp device init - sycl::free : " << e.what() << std::endl;
      success = false;
  }

  return success;
}

// advance pointer from d_begin to address difference between h_ptr and h_begin
template <typename T>
void* device_pointer(void* d_begin, void* h_begin, void* h_ptr)
{
  return (void*)((T*)d_begin + ((T*)h_ptr - (T*)h_begin));
}

void* device_pointer(void* d_begin, void* h_begin, void* h_ptr, zfp_type type)
{
  switch (type) {
  case zfp_type_int32:  return device_pointer<int>(d_begin, h_begin, h_ptr);
  case zfp_type_int64:  return device_pointer<long long int>(d_begin, h_begin, h_ptr);
  case zfp_type_float:  return device_pointer<float>(d_begin, h_begin, h_ptr);
  case zfp_type_double: return device_pointer<double>(d_begin, h_begin, h_ptr);
  default:              return NULL;
  }
}

// clear device memory
template <typename T>
void device_clear(T* d_pointer, size_t size)
{
  queue(zfp_dev_selector).memset(d_pointer, 0, size).wait();
}

// allocate device memory
template <typename T>
bool device_malloc(T** d_pointer, size_t size, const char* what = 0)
{
  bool success = malloc_async(d_pointer, size);

#ifdef ZFP_DEBUG
  if (!success) {
    std::cerr << "zfp::sycl : failed to allocate device memory";
    if (what)
      std::cerr << " for " << what;
    std::cerr << std::endl;
  }
#endif

  return success;
}

// allocate and zero-initialize device memory
template <typename T>
bool device_calloc(T** d_pointer, size_t size, const char* what = 0)
{
  bool success = device_malloc(d_pointer, size, what);
  if (success)
    device_clear(*d_pointer, size);

  return success;
}

// free device memory
template <typename T>
void device_free(T* d_pointer)
{
  free_async(d_pointer);
}


// allocate device memory and copy from host
template <typename T>
bool device_copy_from_host(T** d_pointer, size_t size, void* h_pointer,
  const char* what = 0) try {
  if (!device_malloc(d_pointer, size, what))
    return false;
  queue(zfp_dev_selector).memcpy(*d_pointer, h_pointer, size).wait();
  return true;
}
catch (exception const &exc) {
  #ifdef ZFP_DEBUG
  std::cerr << "zfp::sycl : failed to copy "
            << (what ? what : "data") << ". "<< exc.what() 
            << " exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  #endif
  queue q(zfp_dev_selector);
  free(*d_pointer, q);
  *d_pointer = NULL;
  return false;
}

// copy to host and then deallocate device memory
template <typename T>
bool device_move_to_host(T **d_pointer, size_t size, void *h_pointer,
                         const char *what = 0) try {
  queue q(zfp_dev_selector);
  q.memcpy(h_pointer, *d_pointer, size).wait();
  
  free(*d_pointer, q);
  *d_pointer = NULL;

  return true;
}
catch (exception const &exc) {
#ifdef ZFP_DEBUG
    std::cerr << "zfp::sycl : failed to copy " 
              << (what ? what : "data") << ". "<< exc.what() 
            << " exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
#endif
  return false;
}

Word* setup_device_stream_compress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!is_device_ptr(d_stream)) {
    // allocate device memory for compressed data
    size_t size = stream_capacity(stream->stream);
    device_malloc(&d_stream, size, "stream");
  }

  return d_stream;
}

Word* setup_device_stream_decompress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!is_device_ptr(d_stream)) {
    // copy compressed data to device memory
    size_t size = stream_capacity(stream->stream);
    device_copy_from_host(&d_stream, size, stream->stream->begin, "stream");
  }

  return d_stream;
}

ushort* setup_device_index_compress(zfp_stream* stream, const zfp_field* field)
{
  ushort* d_index = stream->index ? (ushort*)stream->index->data : NULL;
  if (!is_device_ptr(d_index)) {
    // allocate device memory for block index
    size_t size = zfp_field_blocks(field) * sizeof(ushort);
    device_malloc(&d_index, size, "index");
  }

  return d_index;
}

Word* setup_device_index_decompress(zfp_stream* stream)
{
  Word* d_index = (Word*)stream->index->data;
  if (!is_device_ptr(d_index)) {
    // copy index to device memory
    size_t size = stream->index->size;
    device_copy_from_host(&d_index, size, stream->index->data, "index");
  }

  return d_index;
}

// bool setup_device_chunking(size_t* chunk_size, unsigned long long** d_offsets, uint processors)
// {
//   queue q(zfp_dev_selector);
//   // Assuming 1 thread = 1 ZFP block,
//   // launching 1024 threads per SM should give a decent occupancy
//   *chunk_size = processors * 1024;
//   size_t size = (*chunk_size + 1) * sizeof(unsigned long long);
//   if (!device_malloc(d_offsets, size, "offsets"))
//     return false;
//   q.memset(*d_offsets, 0, size).wait(); // ensure offsets are zeroed

//   return true;
// }

bool setup_device_compact(size_t *chunk_size, unsigned long long **d_offset,
                          uint processors) try {
  // use 1K threads per SM for high occupancy (assumes one thread per zfp block)
  const size_t threads_per_sm = 1024;
  *chunk_size = processors * threads_per_sm;

  // allocate and zero-initialize offsets
  const size_t size = (*chunk_size + 1) * sizeof(unsigned long long);
  if (!device_calloc(d_offset, size, "offsets"))
    return false;

  // prefixsum offsets to compact stream
  // auto e = kt::gpu::inclusive_scan(q, *d_offset, *d_offset + *chunk_size + 1,
  //                               *d_offset, std::plus<std::uint32_t>{},
  //                               kt::kernel_param<256, 8>{}); //TODO VERIFY THIS

  return true;
}
catch (exception const &exc) {
  device_free(d_offset);
  *d_offset = NULL;
  return false;
}

void* setup_device_field_compress(const zfp_field* field, void*& d_begin)
{
  void* d_data = field->data;
  if (is_device_ptr(d_data)) {
    // field already resides on device
    d_begin = zfp_field_begin(field);
    return d_data;
  }
  else {
    // GPU implementation currently requires contiguous field
    if (zfp_field_is_contiguous(field)) {
      // copy field from host to device
      size_t size = zfp_field_size_bytes(field);
      void* h_begin = zfp_field_begin(field);
      if (!device_copy_from_host(&d_begin, size, h_begin, "field"))
        return NULL;
      // in case of negative strides, advance device pointer into buffer
      return device_pointer(d_begin, h_begin, d_data, field->type);
    }
    else
      return NULL;
  }
}

void* setup_device_field_decompress(const zfp_field* field, void*& d_begin)
{
  void* d_data = field->data;
  if (is_device_ptr(d_data)) {
    // field has already been allocated on device
    d_begin = zfp_field_begin(field);
    return d_data;
  }
  else {
    // GPU implementation currently requires contiguous field
    if (zfp_field_is_contiguous(field)) {
      // allocate device memory for decompressed field
      size_t size = zfp_field_size_bytes(field);
      if (!device_malloc(&d_begin, size, "field"))
        return NULL;
      void* h_begin = zfp_field_begin(field);
      // in case of negative strides, advance device pointer into buffer
      return device_pointer(d_begin, h_begin, d_data, field->type);
    }
    else
      return NULL;
  }
}

// copy from device to host (if needed) and deallocate device memory
void cleanup_device(void* d_begin, void* begin = 0, size_t bytes = 0)
{
  if (d_begin != begin) {
    // copy data from device to host and free device memory
    if (begin && bytes)
      queue(zfp_dev_selector).memcpy(begin, d_begin, bytes).wait(); //CPU works to an extent with this??
    device_free(d_begin);
  }
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif
