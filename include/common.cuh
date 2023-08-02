#pragma once

#include <cuda_runtime.h>

#define CHECK_LAST_CUDA_ERROR() checkLastCudaError(__FILE__, __LINE__)
#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char *const func, const char *const file, const int line);
void checkLastCudaError(const char *const file, const int line);

#define JLIO_FUNCTION __device__
#define JLIO_KERNEL __global__
#define JLIO_INLINE_FUNCTION __forceinline__ JLIO_FUNCTION
#define JLIO_INLINE_DEVICE_HOST __forceinline__ __host__ __device__
