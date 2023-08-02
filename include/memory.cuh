#pragma once

#include "common.cuh"

namespace jlio
{
    __host__ void malloc(void **ptr, size_t size);

    __host__ void memset(void *ptr, int value, size_t count);

    __host__ void memcpy(void *dst, const void *src, size_t count, int kind);

    __host__ void free(void *ptr);

    __host__ enum cudaMemcpyKind {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4
    };

} // namespace jlio
