#pragma once

#include "common.cuh"

namespace rmagine
{

  /**
   * @brief Vector3 type
   *
   */
  template <size_t N, typename T>
  struct VectorN
  {
    T data[N];

    JLIO_INLINE_FUNCTION
    VectorN<N, T> add(const VectorN<N, T> &b) const
    {
      VectorN<N, T> res;
      res.setZeros();
#pragma unroll
      for (size_t i = 0; i < N; ++i)
      {
        res[i] += data[i] + b[i];
      }

      return res;
    }

    JLIO_INLINE_FUNCTION
    void addInplace(const VectorN<N, T> &b)
    {
#pragma unroll
      for (size_t i = 0; i < N; ++i)
      {
        data[i] += b[i];
      }
    }

    JLIO_INLINE_FUNCTION
    void addInplace(volatile VectorN<N, T> &b) volatile
    {
#pragma unroll
      for (size_t i = 0; i < N; ++i)
      {
        data[i] += b[i];
      }
    }

    JLIO_INLINE_FUNCTION
    void setZeros()
    {
      memset(data, 0, sizeof(data));
    }

    // OPERATORS
    JLIO_INLINE_FUNCTION
    VectorN<N, T> operator+(const VectorN<N, T> &b) const
    {
      return add(b);
    }

    JLIO_INLINE_FUNCTION
    const T &operator[](size_t i) const
    {
      return data[i];
    }

    JLIO_INLINE_FUNCTION
    volatile T &operator[](size_t i) volatile
    {
      return data[i];
    }

    JLIO_INLINE_FUNCTION
    const T &operator()(size_t i) const
    {
      return data[i];
    }

    JLIO_INLINE_FUNCTION
    T &operator()(size_t i)
    {
      return data[i];
    }

    JLIO_INLINE_FUNCTION
    VectorN<N, T> &operator+=(const VectorN<N, T> &b)
    {
      addInplace(b);
      return *this;
    }

    JLIO_INLINE_FUNCTION
    volatile VectorN<N, T> &operator+=(volatile VectorN<N, T> &b) volatile
    {
      addInplace(b);
      return *this;
    }
  };

} // namespace rmagine