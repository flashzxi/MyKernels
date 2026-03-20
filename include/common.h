#pragma once

#include <assert.h>
#include <vector>
#include <cstddef>
#include <cstdio>
#include <array>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace my_kernels
{
#define MYKERNEL_HOST_DEVICE __host__ __device__

#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        cudaError_t _err = (err);                                              \
        if (_err != cudaSuccess) {                                             \
            std::fprintf(stderr,                                               \
                        "CUDA error at %s:%d: %s failed: %s\n",                \
                        __FILE__,                                              \
                        __LINE__,                                              \
                        #err,                                                  \
                        cudaGetErrorString(_err));                             \
            exit(-1);                                                          \
        }                                                                      \
    } while (0)

constexpr int MAX_DIMS = 8;
typedef std::array<int, MAX_DIMS> Shape;
typedef std::array<int, MAX_DIMS> Stride;
typedef std::array<int, MAX_DIMS> Coord;

int getMaxThreadsPerBlock();

template <typename... Ts>
MYKERNEL_HOST_DEVICE
Shape make_array(Ts... xs)
{
    static_assert(sizeof...(Ts) <= MAX_DIMS, "supports at most 8 arguments");
    static_assert((std::is_integral_v<Ts> && ...), "arguments must be integral");

    Shape shape{};
    int values[] = { static_cast<int>(xs)... };
    for (int i = 0; i < static_cast<int>(sizeof...(Ts)); ++i) {
        shape[i] = values[i];
    }
    return shape;
}

template <typename... Ts>
MYKERNEL_HOST_DEVICE
Shape make_shape(Ts... xs)
{
    return make_array(xs...);
}

template <typename... Ts>
MYKERNEL_HOST_DEVICE
Stride make_stride(Ts... xs)
{
    return make_array(xs...);
}

template <typename... Ts>
MYKERNEL_HOST_DEVICE
Coord make_coord(Ts... xs)
{
    return make_array(xs...);
}

MYKERNEL_HOST_DEVICE
Shape make_shape(const std::vector<int>& shape_vec);

MYKERNEL_HOST_DEVICE
Stride make_stride(const std::vector<int>& stride_vec);

MYKERNEL_HOST_DEVICE
Stride make_coord(const std::vector<int>& coord_vec);

template <typename T>
MYKERNEL_HOST_DEVICE
T floatConvTo(float fv)
{
    if constexpr (std::is_same_v<T, __half>)
    {
        return __float2half(fv);
    } else if constexpr (std::is_same_v<T, __bf16>)
    {
        return __float2bfloat16(fv);
    } else if constexpr (std::is_same_v<T, float>)
    {
        return fv;
    }
}

template <typename T>
MYKERNEL_HOST_DEVICE
float convToFloat(T fv)
{
    if constexpr (std::is_same_v<T, __half>)
    {
        return __half2float(fv);
    } else if constexpr (std::is_same_v<T, __bf16>)
    {
        return __bfloat162float(fv);
    } else if constexpr (std::is_same_v<T, float>)
    {
        return fv;
    }
}

enum class DataType : int {
    DATATYPE_FLOAT32 = 0,
    DATATYPE_FLOAT16 = 1,
    DATATYPE_BF16    = 2,
};

int dataTypeSize(const DataType& type)
{
    static const int size_in_bytes[] = {
        /* DATATYPE_FLOAT32 */ sizeof(float),
        /* DATATYPE_FLOAT16 */ sizeof(__fp16),
        /* DATATYPE_BF16*/     sizeof(__bf16),
    };
    return size_in_bytes[static_cast<int>(type)];
}

enum class CudaBlockSize : int {
    CUDA_BLOCK_SIZE_256 = 256,
    CUDA_BLOCK_SIZE_512 = 512,
    CUDA_BLOCK_SIZE_1024 = 1024,
    CUDA_BLOCK_SIZE_2048 = 2048,
    CUDA_BLOCK_SIZE_4096 = 4096,
};

enum class DeviceType: int {
    POS_DEVICE,
    POS_HOST,
};

enum class MyKernelStatus : int
{
    MYKERNEL_SUCCESS,
    MYKERNEL_NOT_SUPPORT,
};

struct Layout
{
    Shape shape;
    Stride stride;
    int rank = 0;

    Layout(Shape shape, Stride stride, int rank): shape(shape), stride(stride), rank(rank) {}

    Layout(Shape shape, int rank);

    MYKERNEL_HOST_DEVICE
    [[nodiscard]] size_t locate(const Coord& coord) const;

    // following CuTe’s left-to-right priority
    MYKERNEL_HOST_DEVICE
    [[nodiscard]] size_t convTo1D(const Coord& coord) const;

    MYKERNEL_HOST_DEVICE
    [[nodiscard]] Coord convToCoord(size_t coord1d) const;

    MYKERNEL_HOST_DEVICE
    [[nodiscard]] size_t size() const;

    MYKERNEL_HOST_DEVICE
    [[nodiscard]] size_t data_size() const;

    [[nodiscard]] bool is_contiguous() const;
};

class HostTensor
{
private:
    void* data_host = nullptr;
    void* data_device = nullptr;
    Layout layout;
    DataType dateType;

public:
    HostTensor(void* data, Layout layout, DataType dateType, DeviceType deviceType = DeviceType::POS_HOST);

    cudaError_t copy_to_device();

    cudaError_t copy_to_host();

    cudaError_t free_host();

    cudaError_t free_device();

    cudaError_t allocate_device();

    template <typename T>
    T* get_host_data()
    {
        return reinterpret_cast<T*>(data_host);
    }

    template <typename T>
    T* get_device_data()
    {
        return reinterpret_cast<T*>(data_device);
    }

    ~HostTensor();
};
}
