#include "common.h"

namespace my_kernels
{
MYKERNEL_HOST_DEVICE
size_t Layout::size() const
{
    size_t size = 1;
    for (auto s: shape)
    {
        size *= s;
    }
    return size;
}

MYKERNEL_HOST_DEVICE
size_t Layout::data_size() const
{
    if (rank == 0) return 0;

    int max_stride_idx = 0;
    for (int i = 1; i < rank; ++i)
    {
        if (stride[i] > stride[max_stride_idx])
        {
            max_stride_idx = i;
        }
    }
    return shape[max_stride_idx] * stride[max_stride_idx];
}

MYKERNEL_HOST_DEVICE
size_t tid2idx(size_t tid, int* shape, int* stride, int rank)
{
    size_t offset = 0;
    for (int i = 0; i < rank; ++i)
    {
        offset += stride[i] * (tid % shape[i]);
        tid /= shape[i];
    }
    return offset;
}

bool Layout::is_contiguous() const
{
    return data_size() == size();
}

Layout::Layout(Shape shape, int rank): shape(shape), rank(rank)
{
    stride[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; --i)
    {
        stride[i] = stride[i + 1] * shape[i];
    }
}

MYKERNEL_HOST_DEVICE
size_t Layout::locate(const Coord& coord) const
{
    size_t offset = 0;
    for (int i = 0; i < rank; ++i)
    {
        offset += stride[i] * coord[i];
    }
    return offset;
}

MYKERNEL_HOST_DEVICE
size_t Layout::convTo1D(const Coord& coord) const
{
    size_t cord1d = rank > 0 ? coord[0] : 0;
    for (int i = 1; i < rank; ++i)
    {
        cord1d += shape[i] * coord[i];
    }
    return cord1d;
}

MYKERNEL_HOST_DEVICE
Coord Layout::convToCoord(size_t coord1d) const
{
    Coord coord{};
    for (int i = 0; i < rank; ++i)
    {
        coord[i] = coord1d % shape[i];
        coord1d /= shape[i];
    }
    return coord;
}

// Tensor implementations
HostTensor::HostTensor(void* data, Layout layout, DataType dateType, DeviceType deviceType):
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

cudaError_t HostTensor::copy_to_device()
{
    assert(data_host != nullptr);
    CUDA_CHECK(cudaMalloc(&data_device, layout.data_size() * dataTypeSize(dateType)));
    CUDA_CHECK(cudaMemcpy(data_device, data_host, layout.data_size() * dataTypeSize(dateType), cudaMemcpyHostToDevice));
    return cudaSuccess;
}

cudaError_t HostTensor::copy_to_host()
{
    assert(data_device != nullptr);
    data_host = new char[layout.data_size() * dataTypeSize(dateType)];
    CUDA_CHECK(cudaMemcpy(data_host, data_device, layout.data_size() * dataTypeSize(dateType), cudaMemcpyDeviceToHost));
    return cudaSuccess;
}

cudaError_t HostTensor::free_host()
{
    delete [] (char*) data_host;
    data_host = nullptr;
    return cudaSuccess;
}

cudaError_t HostTensor::free_device()
{
    CUDA_CHECK(cudaFree(data_device));
    data_device = nullptr;
    return cudaSuccess;
}

cudaError_t HostTensor::allocate_device()
{
    CUDA_CHECK(cudaMalloc(&data_device, layout.data_size() * dataTypeSize(dateType)));
    return cudaSuccess;
}

Shape make_array(const std::vector<int>& shape_vec)
{
    assert(shape_vec.size() <= MAX_DIMS, "supports at most 8 arguments");
    Shape shape{};
    for (int i = 0; i < shape_vec.size(); ++i)
    {
        shape[i] = shape_vec[i];
    }
    return shape;
}

Shape make_shape(const std::vector<int>& shape_vec)
{
    return make_array(shape_vec);
}

Stride make_stride(const std::vector<int>& stride_vec)
{
    return make_array(stride_vec);
}

Stride make_coord(const std::vector<int>& coord_vec)
{
    return make_array(coord_vec);
}

int getMaxThreadsPerBlock()
{
    int dev = 0;
    CUDA_CHECK(cudaSetDevice(dev));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    return prop.maxThreadsPerBlock;
}
}