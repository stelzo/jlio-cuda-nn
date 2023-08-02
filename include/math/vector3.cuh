#pragma once

#include "common.cuh"

namespace rmagine
{

  /**
   * @brief Vector3 type
   *
   */
  template <typename T>
  struct Vector3
  {
    T x;
    T y;
    T z;

    JLIO_INLINE_FUNCTION
    Vector3()
        : x(0), y(0), z(0)
    {
    }

    JLIO_INLINE_FUNCTION
    Vector3(T x, T y, T z)
        : x(x), y(y), z(z)
    {
    }

    JLIO_INLINE_FUNCTION
    Vector3(const Vector3<T> &other)
    {
      x = other.x;
      y = other.y;
      z = other.z;
    }

    JLIO_INLINE_FUNCTION
    static const Vector3<T> Constant(T val)
    {
      return Vector3<T>(val, val, val);
    }

    // FUNCTIONS
    JLIO_INLINE_FUNCTION
    Vector3<T> add(const Vector3<T> &b) const
    {
      return {x + b.x, y + b.y, z + b.z};
    }

    JLIO_INLINE_FUNCTION
    void addInplace(const Vector3<T> &b)
    {
      x += b.x;
      y += b.y;
      z += b.z;
    }

    JLIO_INLINE_FUNCTION
    void addInplace(volatile Vector3<T> &b) volatile
    {
      x += b.x;
      y += b.y;
      z += b.z;
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> sub(const Vector3<T> &b) const
    {
      return {x - b.x, y - b.y, z - b.z};
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> negation() const
    {
      return {-x, -y, -z};
    }

    template <typename S>
    JLIO_INLINE_FUNCTION
        Vector3<S>
        cast() const
    {
      return {static_cast<S>(x), static_cast<S>(y), static_cast<S>(z)};
    }

    JLIO_INLINE_FUNCTION
    void negate()
    {
      x = -x;
      y = -y;
      z = -z;
    }

    JLIO_INLINE_FUNCTION
    void subInplace(const Vector3<T> &b)
    {
      x -= b.x;
      y -= b.y;
      z -= b.z;
    }

    JLIO_INLINE_FUNCTION
    T dot(const Vector3<T> &b) const
    {
      return x * b.x + y * b.y + z * b.z;
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> cross(const Vector3<T> &b) const
    {
      return {
          y * b.z - z * b.y,
          z * b.x - x * b.z,
          x * b.y - y * b.x};
    }

    JLIO_INLINE_FUNCTION
    T mult(const Vector3<T> &b) const
    {
      return dot(b);
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> mult(const T &s) const
    {
      return {x * s, y * s, z * s};
    }

    JLIO_INLINE_FUNCTION
    void multInplace(const T &s)
    {
      x *= s;
      y *= s;
      z *= s;
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> div(const T &s) const
    {
      return {x / s, y / s, z / s};
    }

    JLIO_INLINE_FUNCTION
    void divInplace(const T &s)
    {
      x /= s;
      y /= s;
      z /= s;
    }

    JLIO_INLINE_FUNCTION
    T l2normSquared() const
    {
      return x * x + y * y + z * z;
    }

    JLIO_INLINE_FUNCTION
    T l2norm() const
    {
      return sqrtf(l2normSquared());
    }

    JLIO_INLINE_FUNCTION
    T sum() const
    {
      return x + y + z;
    }

    JLIO_INLINE_FUNCTION
    T prod() const
    {
      return x * y * z;
    }

    JLIO_INLINE_FUNCTION
    T l1norm() const
    {
      return fabs(x) + fabs(y) + fabs(z);
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> cwiseAbs() const
    {
      return Vector3<T>(abs(x), abs(y), abs(z));
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> normalized() const
    {
      return div(l2norm());
    }

    JLIO_INLINE_FUNCTION
    void normalize()
    {
      divInplace(l2norm());
    }

    JLIO_INLINE_FUNCTION
    void setZeros()
    {
      x = 0.0;
      y = 0.0;
      z = 0.0;
    }

    // OPERATORS
    JLIO_INLINE_FUNCTION
    Vector3<T> operator+(const Vector3<T> &b) const
    {
      return add(b);
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> &operator+=(const Vector3<T> &b)
    {
      addInplace(b);
      return *this;
    }

    JLIO_INLINE_FUNCTION
    volatile Vector3<T> &operator+=(volatile Vector3<T> &b) volatile
    {
      addInplace(b);
      return *this;
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> operator-(const Vector3<T> &b) const
    {
      return sub(b);
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> &operator-=(const Vector3<T> &b)
    {
      subInplace(b);
      return *this;
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> operator-() const
    {
      return negation();
    }

    JLIO_INLINE_FUNCTION
    T operator*(const Vector3<T> &b) const
    {
      return mult(b);
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> operator*(const T &s) const
    {
      return mult(s);
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> &operator*=(const T &s)
    {
      multInplace(s);
      return *this;
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> operator/(const T &s) const
    {
      return div(s);
    }

    JLIO_INLINE_FUNCTION
    Vector3<T> operator/=(const T &s)
    {
      divInplace(s);
      return *this;
    }

    JLIO_INLINE_FUNCTION
    bool operator==(const Vector3<T> &other)
    {
      return other.x == x && other.y == y && other.z == z;
    }
  };

  template <typename T>
  bool operator==(const Vector3<T> one, const Vector3<T> &other)
  {
    return other.x == one.x && other.y == one.y && other.z == one.z;
  }

  template <typename T, typename S>
  Vector3<T> operator*(S scalar, const Vector3<T> one)
  {
    return one * (T)scalar;
  }

  using Vector3f = Vector3<float>;
  using Vector3d = Vector3<double>;
  using Vector3i = Vector3<int>;
  using Pointf = Vector3<float>;
  using Pointi = Vector3<int>;
  using Pointl = Vector3<long>;
  using Pointll = Vector3<long long>;

} // namespace rmagine