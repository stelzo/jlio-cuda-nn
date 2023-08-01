#pragma once

#ifdef USE_CUDA
#include <common.h>
#else
#include <cstring>
#include <cstdlib>
#include <vector>
#endif

#include <iostream>

namespace jlio
{
    void malloc(void **ptr, size_t size);

    void memset(void *ptr, int value, size_t count);

#ifndef USE_CUDA
    std::vector<size_t> indexIota(size_t size);
#endif

    void memcpy(void *dst, const void *src, size_t count, int kind);

    void free(void *ptr);

    enum cudaMemcpyKind
    {
        cudaMemcpyHostToHost = 0,
        cudaMemcpyHostToDevice = 1,
        cudaMemcpyDeviceToHost = 2,
        cudaMemcpyDeviceToDevice = 3,
        cudaMemcpyDefault = 4
    };

} // namespace jlio

