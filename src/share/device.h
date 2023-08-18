#ifndef DEVICE_H
#define DEVICE_H

// device-specific specializations

#if defined(__CUDACC__)
  // CUDA specializations
  #include <cub/cub.cuh>
  #if CUDART_VERSION >= 9000
    #include <cooperative_groups.h>
  #else
    #error "zfp variable-rate compression requires CUDA 9.0 or later"
  #endif

  // __shfl_xor() is deprecated since CUDA 9.0
  #define SHFL_XOR(var, lane_mask) __shfl_xor_sync(0xffffffffu, var, lane_mask)

  namespace zfp {
  namespace cuda {
  namespace internal {

  // determine whether ptr points to device memory
  inline bool is_gpu_ptr(const void* ptr)
  {
    bool status = false;
    cudaPointerAttributes atts;
    if (cudaPointerGetAttributes(&atts, ptr) == cudaSuccess)
      switch (atts.type) {
        case cudaMemoryTypeDevice:
#if CUDART_VERSION >= 10000
        case cudaMemoryTypeManaged:
#endif
          status = true;
          break;
      }
    // clear last error so other error checking does not pick it up
    (void)cudaGetLastError();
    return status;
  }

  // asynchronous memory allocation (when supported)
  template <typename T>
  inline bool malloc_async(T** d_pointer, size_t size)
  {
#if CUDART_VERSION >= 11020
    return cudaMallocAsync(d_pointer, size, 0) == cudaSuccess;
#else
    return cudaMalloc(d_pointer, size) == cudaSuccess;
#endif
  }

  // asynchronous memory deallocation (when supported)
  inline void free_async(void* d_pointer)
  {
#if CUDART_VERSION >= 11020
    cudaFreeAsync(d_pointer, 0);
#else
    cudaFree(d_pointer);
#endif
  }

  } // namespace internal
  } // namespace cuda
  } // namespace zfp
#elif defined(__HIPCC__)
  // HIP specializations
  #include <hipcub/hipcub.hpp>
  #include <hip/hip_cooperative_groups.h>

  // warp shuffle
  #define SHFL_XOR(var, lane_mask) __shfl_xor(var, lane_mask)

  namespace zfp {
  namespace hip {
  namespace internal {

  // determine whether ptr points to device memory
  inline bool is_gpu_ptr(const void* ptr)
  {
    bool status = false;
    hipPointerAttribute_t atts;
    if (hipPointerGetAttributes(&atts, ptr) == hipSuccess)
      status = (atts.memoryType == hipMemoryTypeDevice);
    // clear last error so other error checking does not pick it up
    (void)hipGetLastError();
    return status;
  }

  // memory allocation
  template <typename T>
  inline bool malloc_async(T** d_pointer, size_t size)
  {
    return hipMalloc(d_pointer, size) == hipSuccess;
  }

  // memory deallocation
  inline void free_async(void* d_pointer)
  {
    hipFree(d_pointer);
  }

  } // namespace internal
  } // namespace hip
  } // namespace zfp
#elif(SYCL_LANGUAGE_VERSION)
  // SYCL specializations
#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>

 /*DPCT1023:10: The SYCL sub-group does not support mask options for
dpct::permute_sub_group_by_xor. You can specify
"--use-experimental-features=masked-sub-group-operation" to use the experimental
helper function to migrate __shfl_xor_sync.
*/
#define SHFL_XOR(var, lane_mask)                                               \
  dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), var, lane_mask)

  namespace zfp {
  namespace sycl {
  namespace internal {

  // determine whether ptr points to device memory
  inline bool is_gpu_ptr(const void *ptr){
    dpct::pointer_attributes atts;
    try {
      atts.init(ptr);
      switch (atts.get_memory_type()) {
        case ::sycl::usm::alloc::device:
        /* FALL THROUGH */
        case ::sycl::usm::alloc::shared:
          return true;
        default:
          return false;
      }
    } catch (::sycl::exception const &exc) {
      return false;
    }
    return false;
  }
  // asynchronous memory allocation (when supported)
  template <typename T> inline bool malloc_async(T **d_pointer, size_t size) try {
    return ((*d_pointer = (T *)::sycl::malloc_device(
                                size, dpct::get_default_queue())),0) == 0;//TODO
  }
  catch (::sycl::exception const &exc) {
    std::cerr << exc.what() << "Exception caught at file:" << __FILE__
              << ", line:" << __LINE__ << std::endl;
    std::exit(1);
  }

  // asynchronous memory deallocation (when supported)
  inline void free_async(void* d_pointer)
  {
    ::sycl::free(d_pointer, dpct::get_default_queue());
  }

  } // namespace internal
  } // namespace sycl
  } // namespace zfp
#else
  #error "unknown GPU back-end"
#endif

#endif
