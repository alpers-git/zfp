#ifndef ZFP_ARRAY3_HPP
#define ZFP_ARRAY3_HPP

#include <cstddef>
#include <cstring>
#include <iterator>
#include "zfp/array.hpp"
#include "zfp/constarray3.hpp"
#include "zfp/index.hpp"
#include "zfp/codec/zfpcodec.hpp"
#include "zfp/internal/array/cache3.hpp"
#include "zfp/internal/array/handle3.hpp"
#include "zfp/internal/array/iterator3.hpp"
#include "zfp/internal/array/pointer3.hpp"
#include "zfp/internal/array/reference3.hpp"
#include "zfp/internal/array/store3.hpp"
#include "zfp/internal/array/view3.hpp"

namespace zfp {

// compressed 3D array of scalars
template <
  typename Scalar,
  class Codec = zfp::codec::zfp3<Scalar>,
  class Index = zfp::index::implicit
>
class array3 : public array {
public:
  // types utilized by nested classes
  typedef array3 container_type;
  typedef Scalar value_type;
  typedef Codec codec_type;
  typedef Index index_type;
  typedef zfp::internal::BlockStore3<value_type, codec_type, index_type> store_type;
  typedef zfp::internal::BlockCache3<value_type, store_type> cache_type;
  typedef typename Codec::header header;

  // accessor classes
  typedef zfp::internal::dim3::const_reference<array3> const_reference;
  typedef zfp::internal::dim3::const_pointer<array3> const_pointer;
  typedef zfp::internal::dim3::const_iterator<array3> const_iterator;
  typedef zfp::internal::dim3::const_view<array3> const_view;
  typedef zfp::internal::dim3::private_const_view<array3> private_const_view;
  typedef zfp::internal::dim3::reference<array3> reference;
  typedef zfp::internal::dim3::pointer<array3> pointer;
  typedef zfp::internal::dim3::iterator<array3> iterator;
  typedef zfp::internal::dim3::view<array3> view;
  typedef zfp::internal::dim3::flat_view<array3> flat_view;
  typedef zfp::internal::dim3::nested_view1<array3> nested_view1;
  typedef zfp::internal::dim3::nested_view2<array3> nested_view2;
  typedef zfp::internal::dim3::nested_view2<array3> nested_view3;
  typedef zfp::internal::dim3::nested_view3<array3> nested_view;
  typedef zfp::internal::dim3::private_view<array3> private_view;

  // default constructor
  array3() :
    array(3, Codec::type),
    cache(store)
  {}

  // constructor of nx * ny * nz array using rate bits per value, at least
  // cache_size bytes of cache, and optionally initialized from flat array p
  array3(size_t nx, size_t ny, size_t nz, double rate, const value_type* p = 0, size_t cache_size = 0) :
    array(3, Codec::type),
    store(nx, ny, nz, zfp_config_rate(rate, true)),
    cache(store, cache_size)
  {
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    if (p)
      set(p);
  }

  // constructor, from previously-serialized compressed array
  array3(const zfp::array::header& header, const void* buffer = 0, size_t buffer_size_bytes = 0) :
    array(3, Codec::type, header),
    store(header.size_x(), header.size_y(), header.size_z(), zfp_config_rate(header.rate(), true)),
    cache(store)
  {
    if (buffer) {
      if (buffer_size_bytes && buffer_size_bytes < store.compressed_size())
        throw zfp::exception("buffer size is smaller than required");
      std::memcpy(store.compressed_data(), buffer, store.compressed_size());
    }
  }

  // copy constructor--performs a deep copy
  array3(const array3& a) :
    array(),
    cache(store)
  {
    deep_copy(a);
  }

  // construction from view--perform deep copy of (sub)array
  template <class View>
  array3(const View& v) :
    array(3, Codec::type),
    store(v.size_x(), v.size_y(), v.size_z(), zfp_config_rate(v.rate(), true)),
    cache(store)
  {
    this->nx = v.size_x();
    this->ny = v.size_y();
    this->nz = v.size_z();
    // initialize array in its preferred order
    for (iterator it = begin(); it != end(); ++it)
      *it = v(it.i(), it.j(), it.k());
  }

  // virtual destructor
  virtual ~array3() {}

  // Gets a functor and calls it on each element of this and 'a' and puts the result in 'res'
  template<class F>
  inline array3& gen_binary_operator(const F& f, array3& res, const array3& a)
  {
    // Check if this and a have the same dimensions
    if (nx != a.nx || ny != a.ny || nz != a.nz)
      throw zfp::exception("dimension mismatch between array3s");

    // Get the dimensions of the blocks in the array
    const size_t bx = store.block_size_x();
    const size_t by = store.block_size_y();
    const size_t bz = store.block_size_z();

    value_type block_a[4 * 4 * 4] = {};
    value_type block_this[4 * 4 * 4] = {};
    value_type block_res[4 * 4 * 4] = {};
    // Iterate over each block
    for (size_t block_index = 0; block_index < bx * by * bz; block_index++)
    {
      // Get the current block from this array
      cache.get_block(block_index, block_this, 1, 4, 16);

      // Get the corresponding block from the array 'a'
      a.cache.get_block(block_index, block_a, 1, 4, 16);

      // Apply operator to the corresponding elements of the blocks
      for (size_t i = 0; i < 4 * 4 * 4; i++)
        block_res[i] = f(block_this[i], block_a[i]);

      // Store the updated block back in this array
      res.cache.put_block(block_index, block_res, 1, 4, 16);
    }

    return res;
  }

   // Gets a functor and calls it on each element of this and a scalar and puts the result in 'res'
  template<class F>
  inline array3& gen_binary_operator(const F& f, array3& res, const Scalar& a)
  {
    // Get the dimensions of the blocks in the array
    const size_t bx = store.block_size_x();
    const size_t by = store.block_size_y();
    const size_t bz = store.block_size_z();

    value_type block_this[4 * 4 * 4] = {};
    value_type block_res[4 * 4 * 4] = {};
    // Iterate over each block
    for (size_t block_index = 0; block_index < bx * by * bz; block_index++)
    {
      // Get the current block from this array
      cache.get_block(block_index, block_this, 1, 4, 16);

      // Apply operator to the corresponding elements of the blocks
      for (size_t i = 0; i < 4 * 4 * 4; i++)
        block_res[i] = f(block_this[i], a);

      // Store the updated block back in this array
      res.cache.put_block(block_index, block_res, 1, 4, 16);
    }

    return res;
  }

  // Gets a functor and calls it on each element of this and puts the result in 'res'
  template<class F>
  inline array3& gen_unary_operator(const F& f, array3& res) const
  {
    // Get the dimensions of the blocks in the array
    const size_t bx = store.block_size_x();
    const size_t by = store.block_size_y();
    const size_t bz = store.block_size_z();

    value_type block_this[4 * 4 * 4] = {};
    value_type block_res[4 * 4 * 4] = {};
    // Iterate over each block
    for (size_t block_index = 0; block_index < bx * by * bz; block_index++)
    {
      // Get the current block from this array
      cache.get_block(block_index, block_this, 1, 4, 16);

      // Apply operator to the corresponding elements of the blocks
      for (size_t i = 0; i < 4 * 4 * 4; i++)
        block_res[i] = f(block_this[i]);
      
      // Store the updated block back in this array
      res.cache.put_block(block_index, block_res, 1, 4, 16);
    }

    return res;
  }

  // assignment operator--performs a deep copy
  array3& operator=(const array3& a)
  {
    if (this != &a)
      deep_copy(a);
    return *this;
  }

  // addition operator--adds another array of identical dimensions
  template <typename other_type>
  array3 operator+(const other_type& a) const
  {
    // check this and a have the same dimensions
    if (nx != a.size_x() || ny != a.size_y() || nz != a.size_z())
      throw zfp::exception("dimension mismatch while adding array3s");
    
    //allocate an array named sum in the same dimensions as this
    array3 sum(nx, ny, nz, rate(),0, cache.size());
    sum = *this;
    sum += a;

    return sum;
  }

  // addition assignment operator--adds another array of identical dimensions
  array3& operator+=(const array3& a)
  {
    return gen_binary_operator(functor::plus<value_type>(), *this, a);
  }

  // addition assignment operator--adds another array of identical dimensions
  array3& operator+=(const const_array3<Scalar>& a)
  {
    //check this and a have the same dimensions
    if (nx != a.size_x() || ny != a.size_y() || nz != a.size_z())
      throw zfp::exception("dimension mismatch while adding array3s");
    //add the values of this and a store the result in this
    for (size_t k = 0; k < nz; k++)
      for (size_t j = 0; j < ny; j++)
        for (size_t i = 0; i < nx; i++)
          (*this)(i, j, k) += a(i, j, k);

    return *this;
  }

  // addition assignment operator--adds a scalar to every value in the array
  array3& operator+=(const Scalar& val)
  {
    return gen_binary_operator(functor::plus<value_type>(), *this, val);
  }

  // scaling operator--scales the elements of this by a constant factor
  array3& operator*=(const Scalar& val)
  {
    return gen_binary_operator(functor::multiplies<value_type>(), *this, val);
  }

  // unary negation operator--returns a deep copy with the sign of each element negated
  array3 operator-() const
  {
    array3 res(nx, ny, nz, rate(), 0, cache.size());
    return gen_unary_operator(functor::negate<value_type>(), res);
  }

  // total number of elements in array
  size_t size() const { return nx * ny * nz; }

  // array dimensions
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  size_t size_z() const { return nz; }

  // resize the array (all previously stored data will be lost)
  void resize(size_t nx, size_t ny, size_t nz, bool clear = true)
  {
    cache.clear();
    this->nx = nx;
    this->ny = ny;
    this->nz = nz;
    store.resize(nx, ny, nz, clear);
  }

  // rate in bits per value
  double rate() const { return store.rate(); }

  // set rate in bits per value
  double set_rate(double rate)
  {
    cache.clear();
    return store.set_rate(rate, true);
  }

  // byte size of array data structure components indicated by mask
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    size += store.size_bytes(mask);
    size += cache.size_bytes(mask);
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    return size;
  }

  // number of bytes of compressed data
  size_t compressed_size() const { return store.compressed_size(); }

  // pointer to compressed data for read or write access
  void* compressed_data() const
  {
    cache.flush();
    return store.compressed_data();
  }

  // cache size in number of bytes
  size_t cache_size() const { return cache.size(); }

  // set minimum cache size in bytes (array dimensions must be known)
  void set_cache_size(size_t bytes)
  {
    cache.flush();
    cache.resize(bytes);
  }

  // empty cache without compressing modified cached blocks
  void clear_cache() const { cache.clear(); }

  // flush cache by compressing all modified cached blocks
  void flush_cache() const { cache.flush(); }

  // decompress array and store at p
  void get(value_type* p) const
  {
    const size_t bx = store.block_size_x();
    const size_t by = store.block_size_y();
    const size_t bz = store.block_size_z();
    const ptrdiff_t sx = 1;
    const ptrdiff_t sy = static_cast<ptrdiff_t>(nx);
    const ptrdiff_t sz = static_cast<ptrdiff_t>(nx * ny);
    size_t block_index = 0;
    for (size_t k = 0; k < bz; k++, p += 4 * sy * ptrdiff_t(ny - by))
      for (size_t j = 0; j < by; j++, p += 4 * sx * ptrdiff_t(nx - bx))
        for (size_t i = 0; i < bx; i++, p += 4)
          cache.get_block(block_index++, p, sx, sy, sz);
  }

  // initialize array by copying and compressing data stored at p
  void set(const value_type* p)
  {
    const size_t bx = store.block_size_x();
    const size_t by = store.block_size_y();
    const size_t bz = store.block_size_z();
    size_t block_index = 0;
    if (p) {
      // compress data stored at p
      const ptrdiff_t sx = 1;
      const ptrdiff_t sy = static_cast<ptrdiff_t>(nx);
      const ptrdiff_t sz = static_cast<ptrdiff_t>(nx * ny);
      for (size_t k = 0; k < bz; k++, p += 4 * sy * ptrdiff_t(ny - by))
        for (size_t j = 0; j < by; j++, p += 4 * sx * ptrdiff_t(nx - bx))
          for (size_t i = 0; i < bx; i++, p += 4)
            cache.put_block(block_index++, p, sx, sy, sz);
    }
    else {
      // zero-initialize array
      const value_type block[4 * 4 * 4] = {};
      while (block_index < bx * by * bz)
        cache.put_block(block_index++, block, 1, 4, 16);
    }
  }

  // (i, j, k) accessors
  const_reference operator()(size_t i, size_t j, size_t k) const { return const_reference(const_cast<container_type*>(this), i, j, k); }
  reference operator()(size_t i, size_t j, size_t k) { return reference(this, i, j, k); }

  // flat index accessors
  const_reference operator[](size_t index) const
  {
    size_t i, j, k;
    ijk(i, j, k, index);
    return const_reference(const_cast<container_type*>(this), i, j, k);
  }
  reference operator[](size_t index)
  {
    size_t i, j, k;
    ijk(i, j, k, index);
    return reference(this, i, j, k);
  }

  // random access iterators
  const_iterator cbegin() const { return const_iterator(this, 0, 0, 0); }
  const_iterator cend() const { return const_iterator(this, 0, 0, nz); }
  const_iterator begin() const { return cbegin(); }
  const_iterator end() const { return cend(); }
  iterator begin() { return iterator(this, 0, 0, 0); }
  iterator end() { return iterator(this, 0, 0, nz); }

protected:
  friend class zfp::internal::dim3::const_handle<array3>;
  friend class zfp::internal::dim3::const_reference<array3>;
  friend class zfp::internal::dim3::const_pointer<array3>;
  friend class zfp::internal::dim3::const_iterator<array3>;
  friend class zfp::internal::dim3::const_view<array3>;
  friend class zfp::internal::dim3::private_const_view<array3>;
  friend class zfp::internal::dim3::reference<array3>;
  friend class zfp::internal::dim3::pointer<array3>;
  friend class zfp::internal::dim3::iterator<array3>;
  friend class zfp::internal::dim3::view<array3>;
  friend class zfp::internal::dim3::flat_view<array3>;
  friend class zfp::internal::dim3::nested_view1<array3>;
  friend class zfp::internal::dim3::nested_view2<array3>;
  friend class zfp::internal::dim3::nested_view3<array3>;
  friend class zfp::internal::dim3::private_view<array3>;

  // perform a deep copy
  void deep_copy(const array3& a)
  {
    // copy base class members
    array::deep_copy(a);
    // copy persistent storage
    store.deep_copy(a.store);
    // copy cached data
    cache.deep_copy(a.cache);
  }

  // global index bounds
  size_t min_x() const { return 0; }
  size_t max_x() const { return nx; }
  size_t min_y() const { return 0; }
  size_t max_y() const { return ny; }
  size_t min_z() const { return 0; }
  size_t max_z() const { return nz; }

  // inspector
  value_type get(size_t i, size_t j, size_t k) const { return cache.get(i, j, k); }

  // mutators (called from proxy reference)
  void set(size_t i, size_t j, size_t k, value_type val) { cache.set(i, j, k, val); }
  void add(size_t i, size_t j, size_t k, value_type val) { cache.ref(i, j, k) += val; }
  void sub(size_t i, size_t j, size_t k, value_type val) { cache.ref(i, j, k) -= val; }
  void mul(size_t i, size_t j, size_t k, value_type val) { cache.ref(i, j, k) *= val; }
  void div(size_t i, size_t j, size_t k, value_type val) { cache.ref(i, j, k) /= val; }

  // convert flat index to (i, j, k)
  void ijk(size_t& i, size_t& j, size_t& k, size_t index) const
  {
    i = index % nx; index /= nx;
    j = index % ny; index /= ny;
    k = index;
  }

  store_type store; // persistent storage of compressed blocks
  cache_type cache; // cache of decompressed blocks
};

typedef array3<float> array3f;
typedef array3<double> array3d;

}

#endif
