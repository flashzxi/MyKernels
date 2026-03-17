#pragma once

#include <assert.h>
#include <vector>
#include <cstddef>
#include <cstdio>
#include <array>

namespace my_kernels
{
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
            return _err;                                                       \
        }                                                                      \
    } while (0)

constexpr int MAX_DIMS = 8;
typedef std::array<int, MAX_DIMS> Shape;
typedef std::array<int, MAX_DIMS> Stride;

template <typename... Ts>
Shape make_shape(Ts... xs)
{
    static_assert(sizeof...(Ts) <= MAX_DIMS, "make_shape supports at most 8 arguments");
    static_assert((std::is_integral_v<Ts> && ...), "make_shape arguments must be integral");

    Shape shape{};
    int values[] = { static_cast<int>(xs)... };
    for (int i = 0; i < static_cast<int>(sizeof...(Ts)); ++i) {
        shape[i] = values[i];
    }
    return shape;
}

template <typename... Ts>
Shape make_stride(Ts... xs)
{
    static_assert(sizeof...(Ts) <= MAX_DIMS, "make_stride supports at most 8 arguments");
    static_assert((std::is_integral_v<Ts> && ...), "make_stride arguments must be integral");

    Shape shape{};
    int values[] = { static_cast<int>(xs)... };
    for (int i = 0; i < static_cast<int>(sizeof...(Ts)); ++i) {
        shape[i] = values[i];
    }
    return shape;
}

Shape make_shape(std::vector<int> shape);
Shape make_stride(std::vector<int> stride);

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

struct Layout
{
    Shape shape;
    Stride stride;
    int rank = 0;

    Layout(Shape shape, Stride stride, int rank): shape(shape), stride(stride), rank(rank) {}

    Layout(Shape shape, int rank);

    [[nodiscard]] size_t size() const;

    [[nodiscard]] size_t data_size() const;
};

struct Tensor
{
    void* data_host = nullptr;
    void* data_device = nullptr;
    Layout layout;
    DataType dateType;

public:
    Tensor(void* data, Layout layout, DataType dateType, DeviceType deviceType = DeviceType::POS_HOST);

    cudaError_t copy_to_device();

    cudaError_t copy_to_host();

    cudaError_t free_host();

    cudaError_t free_device();

    cudaError_t allocate_device();
};
}
