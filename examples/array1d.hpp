#ifndef ARRAY1D_HPP
#define ARRAY1D_HPP

#include <climits>
#include <vector>

typedef unsigned int uint;

// uncompressed 1D double-precision array (for comparison)
namespace raw {
class array1d {
public:
  // constructors
  array1d() : nx(0) {}
  array1d(size_t nx, double = 0.0, const double* = 0, size_t = 0) : nx(nx), data(nx, 0.0) {}

  // array size
  size_t size() const { return data.size(); }
  size_t size_x() const { return nx; }
  void resize(size_t nx) { this->nx = nx; data.resize(nx, 0.0); }

  // rate in bits/value
  double rate() const { return CHAR_BIT * sizeof(double); }

  // cache size in bytes
  size_t cache_size() const { return 0; }

  // byte size of data structures
  size_t size_bytes(uint mask = ZFP_DATA_ALL) const
  {
    size_t size = 0;
    if (mask & ZFP_DATA_META)
      size += sizeof(*this);
    if (mask & ZFP_DATA_PAYLOAD)
      size += data.size() * sizeof(double);
    return size;
  }

  // addition operator--adds one array to another array of identical dimensions
  array1d operator+(const array1d& a) const
  {
    array1d sum(nx);
    for (size_t i = 0; i < nx; i++)
      sum(i) = (*this)(i) + a(i);
    return sum;
  }

  array1d& operator+=(const array1d &a)
  {
    // 1d loop over array
    for (size_t i = 0; i < nx; i++)
      (*this)(i) += a(i);
    return *this;
  }

  // accessors
  double& operator()(size_t x) { return data[x]; }
  const double& operator()(size_t x) const { return data[x]; }
  double& operator[](size_t index) { return data[index]; }
  const double& operator[](size_t index) const { return data[index]; }

  // minimal-functionality forward iterator
  class iterator {
  public:
    double& operator*() const { return array->operator[](index); }
    iterator& operator++() { index++; return *this; }
    iterator operator++(int) { iterator p = *this; index++; return p; }
    bool operator==(const iterator& it) const { return array == it.array && index == it.index; }
    bool operator!=(const iterator& it) const { return !operator==(it); }
  protected:
    friend class array1d;
    iterator(array1d* array, size_t index) : array(array), index(index) {}
    array1d* array;
    size_t index;
  };

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, nx); }

protected:
  size_t nx;
  std::vector<double> data;
};
}

#endif
