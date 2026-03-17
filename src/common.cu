#include "common.h"

namespace my_kernels
{
size_t Layout::size() const
{
    size_t size = 1;
    for (auto s: shape)
    {
        size *= s;
    }
    return size;
}

size_t Layout::data_size() const
{
    int max_stride_idx = -1;
    for (int i = 0; i < rank(); ++i)
    {
        if (stride[i] > stride[max_stride_idx])
        {
            max_stride_idx = i;
        }
    }
    return shape[max_stride_idx] * stride[max_stride_idx];
}

Layout::Layout(Shape shape, int rank): shape(shape), rank(rank)
{
    stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i)
    {
        stride[i] = stride[i + 1] * shape[i];
    }
}

// Tensor implementations
Tensor::Tensor(void* data, Layout layout, DataType dateType, DeviceType deviceType):
    layout(std::move(layout)),
    dateType(dateType)
{
    if (deviceType == DeviceType::POS_HOST)
    {
        data_host = data;
    } else
    {
        data_device = data;
    }
}

cudaError_t Tensor::copy_to_device()
{
    assert(data_host != nullptr);
    CUDA_CHECK(cudaMalloc(&data_device, layout.data_size() * dataTypeSize(dateType)));
    CUDA_CHECK(cudaMemcpy(data_device, data_host, layout.data_size() * dataTypeSize(dateType), cudaMemcpyHostToDevice));
    return cudaSuccess;
}

cudaError_t Tensor::copy_to_host()
{
    assert(data_device != nullptr);
    data_host = new char[layout.data_size() * dataTypeSize(dateType)];
    CUDA_CHECK(cudaMemcpy(data_host, data_device, layout.data_size() * dataTypeSize(dateType), cudaMemcpyDeviceToHost));
    return cudaSuccess;
}

cudaError_t Tensor::free_host()
{
    delete [] (char*) data_host;
    data_host = nullptr;
    return cudaSuccess;
}

cudaError_t Tensor::free_device()
{
    CUDA_CHECK(cudaFree(data_device));
    data_device = nullptr;
    return cudaSuccess;
}

cudaError_t Tensor::allocate_device()
{
    CUDA_CHECK(cudaMalloc(&data_device, layout.data_size() * dataTypeSize(dateType)));
    return cudaSuccess;
}

Shape make_shape(std::vector<int> shape_vec)
{
    assert(shape_vec.size() <= MAX_DIMS, "make_shape supports at most 8 arguments");
    Shape shape{};
    for (int i = 0; i < shape_vec.size(); ++i)
    {
        shape[i] = shape_vec[i];
    }
    return shape;
}
Shape make_stride(std::vector<int> stride_vec)
{
    assert(stride_vec.size() <= MAX_DIMS, "make_stride supports at most 8 arguments");
    Stride stride{};
    for (int i = 0; i < stride_vec.size(); ++i)
    {
        stride[i] = stride_vec[i];
    }
    return stride;
}
}