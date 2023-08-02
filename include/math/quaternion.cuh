#pragma once

#include "common.cuh"
#include "vectorn.cuh"
#include "vector3.cuh"
#include "matrix3x3.cuh"

namespace rmagine
{
    template <typename T>
    class Quaternion // TODO can currently not be reinterpret casted
    {
    public:
        T w, x, y, z;

        JLIO_INLINE_FUNCTION
        Quaternion() : w(1.0), x(0.0), y(0.0), z(0.0) {}

        JLIO_INLINE_FUNCTION
        Quaternion(T w_, T x_, T y_, T z_) : w(w_), x(x_), y(y_), z(z_) {}

        JLIO_INLINE_FUNCTION
        Quaternion<T> operator*(const Quaternion<T> &other) const
        {
            return Quaternion<T>(
                w * other.y - x * other.z + y * other.w + z * other.x,
                w * other.z + x * other.y - y * other.x + z * other.w,
                w * other.w - x * other.x - y * other.y - z * other.z,
                w * other.x + x * other.w + y * other.z - z * other.y);
        }

        JLIO_INLINE_FUNCTION
        Vector3<T> operator*(const Vector3<T> &v) const
        {
            Vector3<T> qv(x, y, z);
            Vector3<T> t = 2 * qv.cross(v);
            return v + w * t + qv.cross(t);
        }

        JLIO_INLINE_FUNCTION
        Quaternion<T> conjugate() const
        {
            return Quaternion<T>(w, -x, -y, -z);
        }

        JLIO_INLINE_FUNCTION
        double norm() const
        {
            return std::sqrt(w * w + x * x + y * y + z * z);
        }

        JLIO_INLINE_FUNCTION
        Quaternion normalized() const
        {
            double n = 1.0 / this->norm();
            return Quaternion<T>(w * n, x * n, y * n, z * n);
        }

        JLIO_INLINE_FUNCTION
        void normalize()
        {
            double n = 1.0 / this->norm();
            w *= n;
            x *= n;
            y *= n;
            z *= n;
        }

        JLIO_INLINE_FUNCTION
        Matrix3x3d toRotationMatrix() const
        {
            Matrix3x3d rotation_matrix;
            rotation_matrix(0, 0) = w * w + x * x - y * y - z * z;
            rotation_matrix(0, 1) = 2 * x * y - 2 * w * z;
            rotation_matrix(0, 2) = 2 * x * z + 2 * w * y;

            rotation_matrix(1, 0) = 2 * x * y + 2 * w * z;
            rotation_matrix(1, 1) = w * w - x * x + y * y - z * z;
            rotation_matrix(1, 2) = 2 * y * z - 2 * w * x;

            rotation_matrix(2, 0) = 2 * x * z - 2 * w * y;
            rotation_matrix(2, 1) = 2 * y * z + 2 * w * x;
            rotation_matrix(2, 2) = w * w - x * x - y * y + z * z;

            return rotation_matrix;
        }
    };

    using Quaterniond = Quaternion<double>;
    using Quaternionf = Quaternion<float>;

} // namespace rmagine