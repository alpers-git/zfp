#ifndef ARRAY3D_HPP
#define ARRAY3D_HPP

#include <climits>
#include <vector>

typedef unsigned int uint;

// uncompressed 3D double-precision array (for comparison)
namespace raw {
class array3d {
public:
  // constructors
  array3d() : nx(0), ny(0), nz(0) {}
  array3d(size_t nx, size_t ny, size_t nz, 
    double = 0.0, const double* = 0, size_t = 0) 
    : nx(nx), ny(ny), nz(nz), data(nx * ny * nz, 0.0) {}

  // array size
  size_t size() const { return data.size(); }
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  size_t size_z() const { return nz; }
  void resize(size_t nx, size_t ny, size_t nz)
  {
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      data.resize(nx * ny * nz, 0.0);
  }

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
  array3d operator+(const array3d &a) const
  {
    array3d sum(nx, ny, nz);
    for (size_t k = 0; k < nz; k++)
      for (size_t j = 0; j < ny; j++)
        for (size_t i = 0; i < nx; i++)
          sum(i, j, k) = (*this)(i, j, k) + a(i, j, k);
    return sum;
  }

  array3d &operator+=(const array3d &a)
  {
    // 3d loop over array
    for (size_t k = 0; k < nz; k++)
      for (size_t j = 0; j < ny; j++)
        for (size_t i = 0; i < nx; i++)
          (*this)(i, j, k) += a(i, j, k);
    return *this;
  }

  // accessors
  double& operator()(size_t x, size_t y, size_t z) { return data[x + nx * y + nx * ny * z]; }
  const double& operator()(size_t x, size_t y, size_t z) const { return data[x + nx * y + nx * ny * z]; }
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
    size_t i() const { return index % array->nx; }
    size_t j() const { return (index / array->nx) % array->ny; }
    size_t k() const { return index / (array->nx * array->ny); }
  protected:
    friend class array3d;
    iterator(array3d* array, size_t index) : array(array), index(index) {}
    array3d* array;
    size_t index;
  };

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, nx * ny * nz); }

protected:
  size_t nx, ny, nz;
  std::vector<double> data;
};
}

#endif
