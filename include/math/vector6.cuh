#pragma once

#include "common.cuh"

namespace rmagine
{

  template <typename T>
  struct Vector6
  {
    T data[6];

    Vector6() = default;

    JLIO_INLINE_FUNCTION
    Vector6(T a, T b, T c, T d, T e, T f)
        : data{a, b, c, d, e, f}
    {
    }

    template <class... Args>
    JLIO_INLINE_FUNCTION
    Vector6(const Args &...args)
        : data{args...}
    {
    }

    // FUNCTIONS
    JLIO_INLINE_FUNCTION
    Vector6<T> add(const Vector6<T> &b) const
    {
      return {data[0] + b.data[0], data[1] + b.data[1], data[2] + b.data[2], data[3] + b.data[3], data[4] + b.data[4], data[5] + b.data[5]};
    }

    JLIO_INLINE_FUNCTION
    void addInplace(const Vector6<T> &b)
    {
      data[0] += b.data[0];
      data[1] += b.data[1];
      data[2] += b.data[2];
      data[3] += b.data[3];
      data[4] += b.data[4];
      data[5] += b.data[5];
    }

    JLIO_INLINE_FUNCTION
    void addInplace(volatile Vector6<T> &b) volatile
    {
      data[0] += b.data[0];
      data[1] += b.data[1];
      data[2] += b.data[2];
      data[3] += b.data[3];
      data[4] += b.data[4];
      data[5] += b.data[5];
    }

    JLIO_INLINE_FUNCTION
    static Vector6<T> Zero()
    {
      return Vector6{};
    }

    // OEPRATORS
    JLIO_INLINE_FUNCTION
    const T &operator[](size_t i) const
    {
      return data[i];
    }

    JLIO_INLINE_FUNCTION
    Vector6<T> operator+(const Vector6<T> &b) const
    {
      return add(b);
    }

    JLIO_INLINE_FUNCTION
    Vector6<T> &operator+=(const Vector6<T> &b)
    {
      addInplace(b);
      return *this;
    }

    JLIO_INLINE_FUNCTION
    volatile Vector6<T> &operator+=(volatile Vector6<T> &b) volatile
    {
      addInplace(b);
      return *this;
    }

    JLIO_INLINE_FUNCTION
    T &at(int i)
    {
      return data[i];
    }

    JLIO_INLINE_FUNCTION
    const T &at(int i) const
    {
      return data[i];
    }

    JLIO_INLINE_FUNCTION
    void set(T val)
    {
      data[0] = val;
      data[1] = val;
      data[2] = val;
      data[3] = val;
      data[4] = val;
      data[5] = val;
    }

    JLIO_INLINE_FUNCTION
    void setZeros()
    {
      set(0);
    }

    ~Vector6() = default;
  };

  using Vector6d = Vector6<double>;
  using Vector6f = Vector6<float>;
  using Point6l = Vector6<long>;
  using Point6d = Vector6<double>;

} // end namespace rmagine