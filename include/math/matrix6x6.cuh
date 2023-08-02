#pragma once
#include "common.cuh"
#include "vector3.cuh"

namespace rmagine
{
  /**
   * @brief Matrix6x6<S> class
   *
   * Same order than Eigen::Matrix3f default -> Can be reinterpret-casted or mapped.
   *
   * Storage order ()-operator
   * (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), ...
   *
   * Storage order []-operator
   * [0][0], [0][1], [0][2], [0][3], [0][4], [0][5], [1][0], [1][1], [1][2], ...
   *
   */
  template <typename S>
  struct Matrix6x6
  {
    // DASA
    S data[6][6];

    JLIO_INLINE_FUNCTION
    static Matrix6x6<S> Zero()
    {
      return Matrix6x6{};
    }

    JLIO_INLINE_FUNCTION
    static Matrix6x6<S> Identity()
    {
      Matrix6x6 m = Zero();
      for (int i = 0; i < 6; ++i)
      {
        m.at(i, i) = 1;
      }
    }

    JLIO_INLINE_FUNCTION
    S &at(unsigned int i, unsigned int j)
    {
      return data[j][i];
    }

    JLIO_INLINE_FUNCTION
    volatile S &at(unsigned int i, unsigned int j) volatile
    {
      return data[j][i];
    }

    JLIO_INLINE_FUNCTION
    S at(unsigned int i, unsigned int j) const
    {
      return data[j][i];
    }

    JLIO_INLINE_FUNCTION
    S at(unsigned int i, unsigned int j) volatile const
    {
      return data[j][i];
    }

    JLIO_INLINE_FUNCTION
    void set(S val)
    {
#pragma unroll
      for (int i = 0; i < 6; ++i)
      {
#pragma unroll
        for (int j = 0; j < 6; ++j)
        {
          at(i, j) = val;
        }
      }
    }

    JLIO_INLINE_FUNCTION
    void setZeros();

    JLIO_INLINE_FUNCTION
    Matrix6x6<S> add(const Matrix6x6<S> &M) const;

    JLIO_INLINE_FUNCTION
    void addInplace(const Matrix6x6<S> &M);

    JLIO_INLINE_FUNCTION
    void addInplace(volatile Matrix6x6<S> &M) volatile;

    // JLIO_INLINE_FUNCTION
    // void addInplace(volatile Matrix6x6<S>& M) volatile;

    JLIO_INLINE_FUNCTION
    Matrix6x6<S> div(const S &s) const;

    JLIO_INLINE_FUNCTION
    void divInplace(const S &s);

    JLIO_INLINE_FUNCTION
    S &operator()(unsigned int i, unsigned int j)
    {
      return at(i, j);
    }

    JLIO_INLINE_FUNCTION
    volatile S &operator()(unsigned int i, unsigned int j) volatile
    {
      return at(i, j);
    }

    JLIO_INLINE_FUNCTION
    S operator()(unsigned int i, unsigned int j) const
    {
      return at(i, j);
    }

    JLIO_INLINE_FUNCTION
    S operator()(unsigned int i, unsigned int j) volatile const
    {
      return at(i, j);
    }

    JLIO_INLINE_FUNCTION
    S *operator[](const unsigned int i)
    {
      return data[i];
    }

    JLIO_INLINE_FUNCTION
    const S *operator[](const unsigned int i) const
    {
      return data[i];
    }

    JLIO_INLINE_FUNCTION
    Matrix6x6<S> operator+(const Matrix6x6<S> &M) const;

    JLIO_INLINE_FUNCTION
    Matrix6x6<S> &operator+=(const Matrix6x6<S> &M);

    JLIO_INLINE_FUNCTION
    Matrix6x6<S> operator/(const S &s) const;

    JLIO_INLINE_FUNCTION
    Matrix6x6<S> &operator/=(const S &s);

    JLIO_INLINE_FUNCTION
    volatile Matrix6x6<S> &operator+=(volatile Matrix6x6<S> &M) volatile;
  };

  template <typename S>
  JLIO_INLINE_FUNCTION void Matrix6x6<S>::setZeros()
  {
    set(0);
  }

  template <typename S>
  JLIO_INLINE_FUNCTION
      Matrix6x6<S>
      Matrix6x6<S>::add(const Matrix6x6<S> &M) const
  {
    Matrix6x6<S> ret;
#pragma unroll
    for (int i = 0; i < 6; ++i)
    {
#pragma unroll
      for (int j = 0; j < 6; ++j)
      {
        ret.at(i, j) = at(i, j) + M.at(i, j);
      }
    }

    return ret;
  }

  template <typename S>
  JLIO_INLINE_FUNCTION void Matrix6x6<S>::addInplace(const Matrix6x6<S> &M)
  {
#pragma unroll
    for (int i = 0; i < 6; ++i)
    {
#pragma unroll
      for (int j = 0; j < 6; ++j)
      {
        at(i, j) += M.at(i, j);
      }
    }
  }

  template <typename S>
  JLIO_INLINE_FUNCTION void Matrix6x6<S>::addInplace(volatile Matrix6x6<S> &M) volatile
  {
#pragma unroll
    for (int i = 0; i < 6; ++i)
    {
#pragma unroll
      for (int j = 0; j < 6; ++j)
      {
        at(i, j) += M.at(i, j);
      }
    }
  }

  template <typename S>
  JLIO_INLINE_FUNCTION
      Matrix6x6<S>
      Matrix6x6<S>::div(const S &s) const
  {
    Matrix6x6<S> ret;
#pragma unroll
    for (int i = 0; i < 6; ++i)
    {
#pragma unroll
      for (int j = 0; j < 6; ++j)
      {
        ret(i, j) = at(i, j) / s;
      }
    }
    return ret;
  }

  template <typename S>
  JLIO_INLINE_FUNCTION void Matrix6x6<S>::divInplace(const S &s)
  {
#pragma unroll
    for (int i = 0; i < 6; ++i)
    {
#pragma unroll
      for (int j = 0; j < 6; ++j)
      {
        at(i, j) /= s;
      }
    }
  }

  template <typename S>
  JLIO_INLINE_FUNCTION
      Matrix6x6<S>
      Matrix6x6<S>::operator+(const Matrix6x6<S> &M) const
  {
    return add(M);
  }

  template <typename S>
  JLIO_INLINE_FUNCTION
      Matrix6x6<S> &
      Matrix6x6<S>::operator+=(const Matrix6x6<S> &M)
  {
    addInplace(M);
    return *this;
  }

  template <typename S>
  JLIO_INLINE_FUNCTION volatile Matrix6x6<S> &Matrix6x6<S>::operator+=(volatile Matrix6x6<S> &M) volatile
  {
    addInplace(M);
    return *this;
  }

  template <typename S>
  JLIO_INLINE_FUNCTION
      Matrix6x6<S>
      Matrix6x6<S>::operator/(const S &s) const
  {
    return div(s);
  }

  template <typename S>
  JLIO_INLINE_FUNCTION
      Matrix6x6<S> &
      Matrix6x6<S>::operator/=(const S &s)
  {
    divInplace(s);
    return *this;
  }

  using Matrix6x6i = Matrix6x6<int>;
  using Matrix6x6l = Matrix6x6<long>;
  using Matrix6x6f = Matrix6x6<float>;
  using Matrix6x6d = Matrix6x6<double>;

} // end namespace rmagine