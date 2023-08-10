#include <CL/sycl.hpp>
#ifndef ZFP_SYCL_DEVICE_H
#define ZFP_SYCL_DEVICE_H

#define ZFP_MAGIC 0x7a667000u

#define SYCL_ERR(function_call, custom_msg, return_value) \
try { \
    function_call; \
} catch (const cl::exception& e) { \
    std::cerr << "zfp_sycl : " << custom_msg << e.what() << std::endl; \
    return return_value; \
} 

namespace zfp {
namespace sycl {
namespace internal {

using namespace ::sycl;

bool device_init()
{
    bool success = true;
    try{
        // Get a SYCL device queue
        //TODO: how to get the preferred device type?
        queue device_q(cpu_selector_v);
    
        // allocate a buffer to store the magic number on the device
        buffer<unsigned int, 1> d_word_buf(NULL, 1);

        //launch a kernel to initialize the magic number
        device_q.submit([&](handler& cgh) {
            auto d_word = d_word_buf.get_access<access::mode::write>(cgh);
            cgh.single_task<class device_init_kernel>([=]() {
                d_word[0] = ZFP_MAGIC;
            });
        });
        device_q.wait();

        //copy the magic number back to the host
        unsigned int h_word = d_word_buf.get_access<access::mode::read>()[0];
        if (h_word != ZFP_MAGIC) {
            std::cerr<<"zfp_sycl : zfp device init failed"<<std::endl;
            success = false;
        }
    }catch (exception const &e) {
        std::cerr<<"zfp_sycl : zfp device init "<< e.what() << std::endl;
        success = false;}


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

// allocate device memory
template <typename T>
bool device_malloc(T** d_pointer, size_t size, const char* what = 0)
{
  bool success = malloc_async(d_pointer, size);

#ifdef ZFP_DEBUG
  if (!success) {
    std::cerr << "zfp_cuda : failed to allocate device memory";
    if (what)
      std::cerr << " for " << what;
    std::cerr << std::endl;
  }
#endif

  return success;
}

// allocate device memory and copy from host
template <typename T>
bool device_copy_from_host(T **d_pointer, size_t size, void *h_pointer,
                           const char *what = 0) try {
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  queue &q_ct1 = dev_ct1.default_queue();
  if (!device_malloc(d_pointer, size, what))
    return false;
  /*
  DPCT1003:23: Migrated API does not return error code. (*, 0) is inserted. You
  may need to rewrite this code.
  */
  if ((q_ct1.memcpy(*d_pointer, h_pointer, size).wait(), 0) != 0) {
#ifdef ZFP_DEBUG
    std::cerr << "zfp_cuda : failed to copy " << (what ? what : "data") << " from host to device" << std::endl;
#endif
    free(*d_pointer, q_ct1);
    *d_pointer = NULL;
    return false;
  }
  return true;
}
catch (exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

Word* setup_device_stream_compress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!is_gpu_ptr(d_stream)) {
    // allocate device memory for compressed data
    size_t size = stream_capacity(stream->stream);
    device_malloc(&d_stream, size, "stream");
  }

  return d_stream;
}

Word* setup_device_stream_decompress(zfp_stream* stream)
{
  Word* d_stream = (Word*)stream->stream->begin;
  if (!is_gpu_ptr(d_stream)) {
    // copy compressed data to device memory
    size_t size = stream_capacity(stream->stream);
    device_copy_from_host(&d_stream, size, stream->stream->begin, "stream");
  }

  return d_stream;
}

ushort* setup_device_index_compress(zfp_stream *stream, const zfp_field *field)
{
  ushort* d_index = stream->index ? (ushort*)stream->index->data : NULL;
  if (!is_gpu_ptr(d_index)) {
    // allocate device memory for block index
    size_t size = zfp_field_blocks(field) * sizeof(ushort);
    device_malloc(&d_index, size, "index");
  }

  return d_index;
}

Word* setup_device_index_decompress(zfp_stream* stream)
{
  Word* d_index = (Word*)stream->index->data;
  if (!is_gpu_ptr(d_index)) {
    // copy index to device memory
    size_t size = stream->index->size;
    device_copy_from_host(&d_index, size, stream->index->data, "index");
  }

  return d_index;
}

bool setup_device_chunking(size_t* chunk_size, unsigned long long** d_offsets, size_t* lcubtemp, void** d_cubtemp, uint processors)
{
  dpct::device_ext &dev_ct1 = dpct::get_current_device();
  queue &q_ct1 = dev_ct1.default_queue();
  // TODO : Error handling for CUDA malloc and CUB?
  // Assuming 1 thread = 1 ZFP block,
  // launching 1024 threads per SM should give a decent occupancy
  *chunk_size = processors * 1024;
  size_t size = (*chunk_size + 1) * sizeof(unsigned long long);
  if (!device_malloc(d_offsets, size, "offsets"))
    return false;
  q_ct1.memset(*d_offsets, 0, size).wait(); // ensure offsets are zeroed

  // Using CUB for the prefix sum. CUB needs a bit of temp memory too
  size_t tempsize;
  /*
  DPCT1026:0: The call to cub::DeviceScan::InclusiveSum was removed because this
  call is redundant in SYCL.
  */
  *lcubtemp = tempsize;
  if (!device_malloc(d_cubtemp, tempsize, "offsets")) {
    free(*d_offsets, q_ct1);
    *d_offsets = NULL;
    return false;
  }

  return true;
}

void* setup_device_field_compress(const zfp_field* field, void*& d_begin)
{
  void* d_data = field->data;
  if (is_gpu_ptr(d_data)) {
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
  if (is_gpu_ptr(d_data)) {
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
// TODO: d_begin should be first argument, with begin = NULL as default
void cleanup_device(void* begin, void* d_begin, size_t bytes = 0)
{
  if (d_begin != begin) {
    // copy data from device to host and free device memory
    if (begin && bytes)
      dpct::get_default_queue().memcpy(begin, d_begin, bytes).wait();
    free_async(d_begin);
  }
}

} // namespace internal
} // namespace sycl
} // namespace zfp

#endif