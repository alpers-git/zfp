#ifndef ARRAY4D_HPP
#define ARRAY4D_HPP

#include <climits>
#include <vector>

typedef unsigned int uint;

// uncompressed 4D double-precision array (for comparison)
namespace raw {
class array4d {
public:
  // constructors
  array4d() : nx(0), ny(0), nz(0), nw(0) {}
  array4d(size_t nx, size_t ny, size_t nz, size_t nw,
    double = 0.0, const double* = 0, size_t = 0) 
    : nx(nx), ny(ny), nz(nz), nw(nw), data(nx * ny * nz * nw, 0.0) {}

  // array size
  size_t size() const { return data.size(); }
  size_t size_x() const { return nx; }
  size_t size_y() const { return ny; }
  size_t size_z() const { return nz; }
  size_t size_w() const { return nw; }
  void resize(size_t nx, size_t ny, size_t nz, size_t nw)
  {
      this->nx = nx;
      this->ny = ny;
      this->nz = nz;
      this->nw = nw;
      data.resize(nx * ny * nz * nw, 0.0);
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
  array4d operator+(const array4d &a) const
  {
    array4d sum(nx, ny, nz, nw);
    for (size_t l = 0; l < nw; l++)
      for (size_t k = 0; k < nz; k++)
        for (size_t j = 0; j < ny; j++)
          for (size_t i = 0; i < nx; i++)
            sum(i, j, k, l) = (*this)(i, j, k, l) + a(i, j, k, l);
    return sum;
  }

  array4d &operator+=(const array4d &a)
  {
    // 4d loop over array
    for (size_t l = 0; l < nw; l++)
      for (size_t k = 0; k < nz; k++)
        for (size_t j = 0; j < ny; j++)
          for (size_t i = 0; i < nx; i++)
            (*this)(i, j, k, l) += a(i, j, k, l);
    return *this;
  }

  // accessors
  double& operator()(size_t x, size_t y, size_t z, size_t w) { return data[x + nx * y + nx * ny * z + nx * ny * nz * w]; }
  const double& operator()(size_t x, size_t y, size_t z, size_t w) const { return data[x + nx * y + nx * ny * z + nx * ny * nz * w]; }
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
    size_t k() const { return (index / (array->nx * array->ny)) % array->nz; }
    size_t l() const { return index / (array->nx * array->ny * array->nz); }
  protected:
    friend class array4d;
    iterator(array4d* array, size_t index) : array(array), index(index) {}
    array4d* array;
    size_t index;
  };

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, nx * ny * nz * nw); }

protected:
  size_t nx, ny, nz, nw;
  std::vector<double> data;
};
}

#endif
