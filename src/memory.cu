#include <memory.cuh>
#include <point.cuh>

namespace jlio
{
    void malloc(void **ptr, size_t size)
    {
        CHECK_CUDA_ERROR(cudaMallocManaged(ptr, size));
    }

    void memset(void *ptr, int value, size_t count)
    {
        CHECK_CUDA_ERROR(cudaMemset(ptr, value, count));
    }

    void memcpy(void *dst, const void *src, size_t count, int kind)
    {
        ::cudaMemcpyKind _kind;
        switch (kind)
        {
        case 0:
            _kind = ::cudaMemcpyHostToHost;
            break;
        case 1:
            _kind = ::cudaMemcpyHostToDevice;
            break;
        case 2:
            _kind = ::cudaMemcpyDeviceToHost;
            break;
        case 3:
            _kind = ::cudaMemcpyDeviceToDevice;
            break;
        case 4:
            _kind = ::cudaMemcpyDefault;
            break;
        default:
            _kind = ::cudaMemcpyDefault;
        }
        CHECK_CUDA_ERROR(cudaMemcpy(dst, src, count, _kind));
        // dst = src; // TODO only on jetson: no need to copy to and from device when using unified memory; but just assigning the pointer would make free fail
    }

    void free(void *ptr)
    {
        if (ptr == NULL)
        {
            return;
        }

        CHECK_CUDA_ERROR(cudaFree(ptr));
    }

} // namespace jlio
