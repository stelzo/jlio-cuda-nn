#pragma once
#include "vector3.cuh"

namespace rmagine
{
    /**
     * @brief Matrix3x3<S> class
     *
     * Same order than Eigen::Matrix3f default -> Can be reinterpret-casted or mapped.
     *
     * Storage order ()-operator
     * (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), ...
     *
     * Storage order []-operator
     * [0][0], [0][1], [0][2], [1][0], [1][1], [1][2], ...
     *
     */
    template <typename S>
    struct Matrix3x3
    {
        // DASA
        S data[3][3];

        // ACCESS
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
        S &operator()(unsigned int i, unsigned int j)
        {
            return data[j][i];
        }

        JLIO_INLINE_FUNCTION
        volatile S &operator()(unsigned int i, unsigned int j) volatile
        {
            return data[j][i];
        }

        JLIO_INLINE_FUNCTION
        S operator()(unsigned int i, unsigned int j) const
        {
            return data[j][i];
        }

        JLIO_INLINE_FUNCTION
        S operator()(unsigned int i, unsigned int j) volatile const
        {
            return data[j][i];
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

        // FUNCSIONS
        JLIO_INLINE_FUNCTION
        void setIdentity()
        {
            at(0, 0) = 1.0f;
            at(0, 1) = 0.0f;
            at(0, 2) = 0.0f;
            at(1, 0) = 0.0f;
            at(1, 1) = 1.0f;
            at(1, 2) = 0.0f;
            at(2, 0) = 0.0f;
            at(2, 1) = 0.0f;
            at(2, 2) = 1.0f;
        }

        JLIO_INLINE_FUNCTION
        void setZeros()
        {
            at(0, 0) = 0.0f;
            at(0, 1) = 0.0f;
            at(0, 2) = 0.0f;
            at(1, 0) = 0.0f;
            at(1, 1) = 0.0f;
            at(1, 2) = 0.0f;
            at(2, 0) = 0.0f;
            at(2, 1) = 0.0f;
            at(2, 2) = 0.0f;
        }

        JLIO_INLINE_FUNCTION
        void setOnes()
        {
            at(0, 0) = 1.0f;
            at(0, 1) = 1.0f;
            at(0, 2) = 1.0f;
            at(1, 0) = 1.0f;
            at(1, 1) = 1.0f;
            at(1, 2) = 1.0f;
            at(2, 0) = 1.0f;
            at(2, 1) = 1.0f;
            at(2, 2) = 1.0f;
        }

        // JLIO_INLINE_FUNCTION
        // void set(const Quaternion& q);

        // JLIO_INLINE_FUNCTION
        // void set(const EulerAngles& e);

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> transpose() const
        {
            Matrix3x3<S> ret;

            ret(0, 0) = at(0, 0);
            ret(0, 1) = at(1, 0);
            ret(0, 2) = at(2, 0);

            ret(1, 0) = at(0, 1);
            ret(1, 1) = at(1, 1);
            ret(1, 2) = at(2, 1);

            ret(2, 0) = at(0, 2);
            ret(2, 1) = at(1, 2);
            ret(2, 2) = at(2, 2);

            return ret;
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> T() const
        {
            return transpose();
        }

        JLIO_INLINE_FUNCTION
        void transposeInplace()
        {
            // use only one S as additional memory
            S swap_mem;
            // can we do this without additional memory?

            swap_mem = at(0, 1);
            at(0, 1) = at(1, 0);
            at(1, 0) = swap_mem;

            swap_mem = at(0, 2);
            at(0, 2) = at(2, 0);
            at(2, 0) = swap_mem;

            swap_mem = at(1, 2);
            at(1, 2) = at(2, 1);
            at(2, 1) = swap_mem;
        }

        JLIO_INLINE_FUNCTION
        S trace() const
        {
            return at(0, 0) + at(1, 1) + at(2, 2);
        }

        JLIO_INLINE_FUNCTION
        S det() const
        {
            return at(0, 0) * (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) -
                   at(0, 1) * (at(1, 0) * at(2, 2) - at(1, 2) * at(2, 0)) +
                   at(0, 2) * (at(1, 0) * at(2, 1) - at(1, 1) * at(2, 0));
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> inv() const
        {
            Matrix3x3<S> ret;

            const S invdet = 1 / det();

            ret(0, 0) = (at(1, 1) * at(2, 2) - at(2, 1) * at(1, 2)) * invdet;
            ret(0, 1) = (at(0, 2) * at(2, 1) - at(0, 1) * at(2, 2)) * invdet;
            ret(0, 2) = (at(0, 1) * at(1, 2) - at(0, 2) * at(1, 1)) * invdet;
            ret(1, 0) = (at(1, 2) * at(2, 0) - at(1, 0) * at(2, 2)) * invdet;
            ret(1, 1) = (at(0, 0) * at(2, 2) - at(0, 2) * at(2, 0)) * invdet;
            ret(1, 2) = (at(1, 0) * at(0, 2) - at(0, 0) * at(1, 2)) * invdet;
            ret(2, 0) = (at(1, 0) * at(2, 1) - at(2, 0) * at(1, 1)) * invdet;
            ret(2, 1) = (at(2, 0) * at(0, 1) - at(0, 0) * at(2, 1)) * invdet;
            ret(2, 2) = (at(0, 0) * at(1, 1) - at(1, 0) * at(0, 1)) * invdet;

            return ret;
        }

        /**
         * @brief Assuming Matrix3x3<S> to be a rotation matrix. then M.inv = M.transpose
         *
         * @return Matrix3x3<S>
         */
        JLIO_INLINE_FUNCTION
        Matrix3x3<S> invRigid() const
        {
            return S();
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> mult(const S &s) const
        {
            Matrix3x3<S> ret;
            ret(0, 0) = at(0, 0) * s;
            ret(0, 1) = at(0, 1) * s;
            ret(0, 2) = at(0, 2) * s;
            ret(1, 0) = at(1, 0) * s;
            ret(1, 1) = at(1, 1) * s;
            ret(1, 2) = at(1, 2) * s;
            ret(2, 0) = at(2, 0) * s;
            ret(2, 1) = at(2, 1) * s;
            ret(2, 2) = at(2, 2) * s;
            return ret;
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> div(const S &s) const
        {
            Matrix3x3<S> ret;
            ret(0, 0) = at(0, 0) / s;
            ret(0, 1) = at(0, 1) / s;
            ret(0, 2) = at(0, 2) / s;
            ret(1, 0) = at(1, 0) / s;
            ret(1, 1) = at(1, 1) / s;
            ret(1, 2) = at(1, 2) / s;
            ret(2, 0) = at(2, 0) / s;
            ret(2, 1) = at(2, 1) / s;
            ret(2, 2) = at(2, 2) / s;
            return ret;
        }

        JLIO_INLINE_FUNCTION
        void divInplace(const S &s)
        {
            at(0, 0) /= s;
            at(0, 1) /= s;
            at(0, 2) /= s;
            at(1, 0) /= s;
            at(1, 1) /= s;
            at(1, 2) /= s;
            at(2, 0) /= s;
            at(2, 1) /= s;
            at(2, 2) /= s;
        }

        JLIO_INLINE_FUNCTION
        void multInplace(const S &s)
        {
            at(0, 0) *= s;
            at(0, 1) *= s;
            at(0, 2) *= s;
            at(1, 0) *= s;
            at(1, 1) *= s;
            at(1, 2) *= s;
            at(2, 0) *= s;
            at(2, 1) *= s;
            at(2, 2) *= s;
        }

        JLIO_INLINE_FUNCTION
        Vector3<S> mult(const Vector3<S> &p) const
        {
            return {
                at(0, 0) * p.x + at(0, 1) * p.y + at(0, 2) * p.z,
                at(1, 0) * p.x + at(1, 1) * p.y + at(1, 2) * p.z,
                at(2, 0) * p.x + at(2, 1) * p.y + at(2, 2) * p.z};
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> mult(const Matrix3x3<S> &M) const
        {
            Matrix3x3<S> res;
            res.setZeros();
#pragma unroll
            for (unsigned int row = 0; row < 3; row++)
            {

#pragma unroll
                for (unsigned int col = 0; col < 3; col++)
                {

#pragma unroll
                    for (unsigned int inner = 0; inner < 3; inner++)
                    {
                        res(row, col) += at(row, inner) * M(inner, col);
                    }
                }
            }
            return res;
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> add(const Matrix3x3<S> &M) const
        {
            Matrix3x3<S> ret;
            ret(0, 0) = at(0, 0) + M(0, 0);
            ret(0, 1) = at(0, 1) + M(0, 1);
            ret(0, 2) = at(0, 2) + M(0, 2);
            ret(1, 0) = at(1, 0) + M(1, 0);
            ret(1, 1) = at(1, 1) + M(1, 1);
            ret(1, 2) = at(1, 2) + M(1, 2);
            ret(2, 0) = at(2, 0) + M(2, 0);
            ret(2, 1) = at(2, 1) + M(2, 1);
            ret(2, 2) = at(2, 2) + M(2, 2);
            return ret;
        }

        JLIO_INLINE_FUNCTION
        void addInplace(const Matrix3x3<S> &M)
        {
            at(0, 0) += M(0, 0);
            at(0, 1) += M(0, 1);
            at(0, 2) += M(0, 2);
            at(1, 0) += M(1, 0);
            at(1, 1) += M(1, 1);
            at(1, 2) += M(1, 2);
            at(2, 0) += M(2, 0);
            at(2, 1) += M(2, 1);
            at(2, 2) += M(2, 2);
        }

        JLIO_INLINE_FUNCTION
        void addInplace(volatile Matrix3x3<S> &M) volatile
        {
            at(0, 0) += M(0, 0);
            at(0, 1) += M(0, 1);
            at(0, 2) += M(0, 2);
            at(1, 0) += M(1, 0);
            at(1, 1) += M(1, 1);
            at(1, 2) += M(1, 2);
            at(2, 0) += M(2, 0);
            at(2, 1) += M(2, 1);
            at(2, 2) += M(2, 2);
        }

        // OPERASORS
        JLIO_INLINE_FUNCTION
        Matrix3x3<S> operator*(const S &s) const
        {
            return mult(s);
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> &operator*=(const S &s)
        {
            multInplace(s);
            return *this;
        }

        JLIO_INLINE_FUNCTION
        Vector3<S> operator*(const Vector3<S> &p) const
        {
            return mult(p);
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> operator*(const Matrix3x3<S> &M) const
        {
            return mult(M);
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> operator/(const S &s) const
        {
            return div(s);
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> &operator/=(const S &s)
        {
            divInplace(s);
            return *this;
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> operator+(const Matrix3x3<S> &M) const
        {
            return add(M);
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> &operator+=(const Matrix3x3<S> &M)
        {
            addInplace(M);
            return *this;
        }

        JLIO_INLINE_FUNCTION
        volatile Matrix3x3<S> &operator+=(volatile Matrix3x3<S> &M) volatile
        {
            addInplace(M);
            return *this;
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3<S> operator~() const
        {
            return inv();
        }

        // JLIO_INLINE_FUNCTION
        // void operator=(const Quaternion& q)
        // {
        //     set(q);
        // }

        // JLIO_INLINE_FUNCTION
        // void operator=(const EulerAngles& e)
        // {
        //     set(e);
        // }
    };

    using Matrix3x3i = Matrix3x3<int>;
    using Matrix3x3f = Matrix3x3<float>;
    using Matrix3x3d = Matrix3x3<double>;

} // end namespace rmagine