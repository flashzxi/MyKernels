#include "Softmax.h"

#include <bits/valarray_before.h>
#include <cub/block/block_reduce.cuh>

namespace my_kernels
{
// for safe soft_max
struct __align__(8) IntermediateData {
    float SUM;
    float MAX;
};

__device__ __forceinline__
IntermediateData reduce_intermediate(const IntermediateData& a, const IntermediateData& b)
{
    bool a_is_larger = a.MAX > b.MAX;
    IntermediateData greater = a_is_larger ? a : b;
    IntermediateData lesser = a_is_larger ? b : a;
    greater.SUM = lesser.SUM * __expf(lesser.MAX - greater.MAX) + greater.SUM;
    return greater;
}

/**
 * 每个block处理一行
 * @tparam DataType 传入的数据类型
 * @tparam BLOCK_SIZE
 * @param input 输入数据
 * @param output 输出数据
 * @param dim 沿哪个方向做softmax
 * @param layout input的数据布局
 */
template <typename DataType, unsigned int BLOCK_SIZE>
void __global__ blockSoftMax(const DataType* input, DataType* output, int dim, const Layout& layout)
{
    int blockdim = layout.shape[dim];
    int stride = layout.stride[dim];
    int row_idx = blockIdx.x;
    Coord start_coord{};
    for (int i = 0; i < layout.rank; ++i)
    {
        if (i != dim)
        {
            start_coord[i] = row_idx % layout.shape[i];
            row_idx /= layout.shape[i];
        }
    }
    size_t slice_offset = layout.locate(start_coord);
    IntermediateData intermediate_partial{ .SUM = 0.0f, .MAX = -MAXFLOAT };
    for (uint32_t idx = threadIdx.x; idx < blockdim; idx += BLOCK_SIZE)
    {
        IntermediateData tmp{
            .SUM = 1.0f,
            .MAX = static_cast<float>(input[slice_offset + idx * stride])
        };
        intermediate_partial = reduce_intermediate(intermediate_partial, tmp);
    }

    typedef cub::BlockReduce<IntermediateData, BLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage tmp_storage;
    __shared__ IntermediateData intermediate_global;

    IntermediateData intermediate_block = BlockReduce(tmp_storage).Reduce(intermediate_partial, reduce_intermediate);
    if (threadIdx.x == 0)
    {
        intermediate_global = intermediate_block;
    }
    __syncthreads();

    for (uint32_t idx = threadIdx.x; idx < blockdim; idx += BLOCK_SIZE)
    {
        output[slice_offset + idx * stride] = floatConvTo<DataType>(
            __fdividef(__expf(input[slice_offset + idx * stride] - intermediate_global.MAX), intermediate_global.SUM)
        );
    }
}

/**
 * 每个block 处理BLOCK_SIZE_y行，每行BLOCK_SIZE_x个线程处理
 * @tparam DataType
 * @tparam BLOCK_SIZE_x 多少个线程处理一行
 * @tparam BLOCK_SIZE_y 一个block处理多少行
 * @tparam numPerThread 每个线程处理多少个元素（用来预分配寄存器内存）
 * @param input 输入数据
 * @param output 输出数据
 * @param dim 沿哪个方向做softmax
 * @param layout input的数据布局
 */
template <typename DataType, int BLOCK_SIZE_x, int BLOCK_SIZE_y, int numPerThread>
void __global__ warpSoftMax(const DataType* input, DataType* output, int dim, const Layout& layout, int total_rows)
{
    static_assert(BLOCK_SIZE_x > 0 && BLOCK_SIZE_x <= 32 && (BLOCK_SIZE_x & (BLOCK_SIZE_x - 1)) == 0,
              "BLOCK_SIZE_X must be a power of two and between 1 and 32");
    int row_idx = blockIdx.x * BLOCK_SIZE_y + threadIdx.y;
    float threadData[numPerThread];

    int blockdim = layout.shape[dim];
    int stride = layout.stride[dim];
    Coord start_coord{};
    for (int i = 0; i < layout.rank; ++i)
    {
        if (i != dim)
        {
            start_coord[i] = row_idx % layout.shape[i];
            row_idx /= layout.shape[i];
        }
    }
    size_t slice_offset = layout.locate(start_coord);

    float max_local = -MAXFLOAT;
    float sum_local = 0.0f;
    if (row_idx < total_rows)
    {
        for (int idx = 0; threadIdx.x + idx * BLOCK_SIZE_x < blockdim; ++idx)
        {
            threadData[idx] = convToFloat(input[slice_offset + (threadIdx.x + idx * BLOCK_SIZE_x) * stride]);
            max_local = max(max_local, threadData[idx]);
        }

        for (int mask = BLOCK_SIZE_x / 2; mask != 0; mask >>= 1)
        {
            max_local = max(max_local, __shfl_xor_sync(0xffffffff, max_local, mask));
        }

        for (int idx = 0; threadIdx.x + idx * BLOCK_SIZE_x < blockdim; ++idx)
        {
            threadData[idx] = __exp(threadData[idx] - max_local);
            sum_local += threadData[idx];
        }

        for (int mask = BLOCK_SIZE_x / 2; mask != 0; mask >>= 1)
        {
            sum_local = sum_local + __shfl_xor_sync(0xffffffff, sum_local, mask);
        }

        for (int idx = 0; threadIdx.x + idx * BLOCK_SIZE_x < blockdim; ++idx)
        {
            output[slice_offset + (threadIdx.x + idx * BLOCK_SIZE_x) * stride]
                = floatConvTo<DataType>(__fdividef(threadData[idx], sum_local));
        }
    }

}

template<int BLOCK_SIZE>
MyKernelStatus launch_kernel(const void* input,
        void* output,
        Layout layout,
        int dim,
        DataType datatype,
        cudaStream_t stream)
{
    int rank = layout.rank;
    dim = dim < 0 ? dim + rank : dim;
    int dimSize = layout.shape[dim];
    int othersize = layout.size() / dimSize;

    if (datatype == DataType::DATATYPE_FLOAT16)
    {
        if (dimSize > 1024)
        {
            blockSoftMax<__half, BLOCK_SIZE>
                    <<<othersize, BLOCK_SIZE, 0, stream>>>(
                            static_cast<const __half*>(input),
                            static_cast<__half*>(output),
                            dim,
                            layout);
        } else if (dimSize > 31)
        {
            constexpr int BLOCK_SIZE_X = 32;
            constexpr int BLOCK_SIZE_Y = BLOCK_SIZE / BLOCK_SIZE_X;
            constexpr int numPerThread = 32;
            dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridDim((othersize + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
            warpSoftMax<__half, BLOCK_SIZE_X, BLOCK_SIZE_Y, numPerThread>
                    <<<gridDim, blockDim, 0, stream>>>(
                            static_cast<const __half*>(input),
                            static_cast<__half*>(output),
                            dim,
                            layout,
                            othersize);
        } else
        {
            constexpr int BLOCK_SIZE_X = 16;
            constexpr int BLOCK_SIZE_Y = BLOCK_SIZE / BLOCK_SIZE_X;
            constexpr int numPerThread = 2;
            dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridDim((othersize + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
            warpSoftMax<__half, BLOCK_SIZE_X, BLOCK_SIZE_Y, numPerThread>
                    <<<gridDim, blockDim, 0, stream>>>(
                            static_cast<const __half*>(input),
                            static_cast<__half*>(output),
                            dim,
                            layout,
                            othersize);
        }
    } else if (datatype == DataType::DATATYPE_BF16)
    {
        if (dimSize > 1024)
        {
            blockSoftMax<__bf16, BLOCK_SIZE>
                    <<<othersize, BLOCK_SIZE, 0, stream>>>(
                            static_cast<const __bf16*>(input),
                            static_cast<__bf16*>(output),
                            dim,
                            layout);
        } else if (dimSize > 31)
        {
            constexpr int BLOCK_SIZE_X = 32;
            constexpr int BLOCK_SIZE_Y = BLOCK_SIZE / BLOCK_SIZE_X;
            constexpr int numPerThread = 32;
            dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridDim((othersize + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
            warpSoftMax<__bf16, BLOCK_SIZE_X, BLOCK_SIZE_Y, numPerThread>
                    <<<gridDim, blockDim, 0, stream>>>(
                            static_cast<const __bf16*>(input),
                            static_cast<__bf16*>(output),
                            dim,
                            layout,
                            othersize);
        } else
        {
            constexpr int BLOCK_SIZE_X = 16;
            constexpr int BLOCK_SIZE_Y = BLOCK_SIZE / BLOCK_SIZE_X;
            constexpr int numPerThread = 2;
            dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridDim((othersize + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
            warpSoftMax<__bf16, BLOCK_SIZE_X, BLOCK_SIZE_Y, numPerThread>
                    <<<gridDim, blockDim, 0, stream>>>(
                            static_cast<const __bf16*>(input),
                            static_cast<__bf16*>(output),
                            dim,
                            layout,
                            othersize);
        }
    } else if (datatype == DataType::DATATYPE_FLOAT32)
    {
        if (dimSize > 1024)
        {
            blockSoftMax<float, BLOCK_SIZE>
                    <<<othersize, BLOCK_SIZE, 0, stream>>>(
                            static_cast<const float*>(input),
                            static_cast<float*>(output),
                            dim,
                            layout);
        } else if (dimSize > 31)
        {
            constexpr int BLOCK_SIZE_X = 32;
            constexpr int BLOCK_SIZE_Y = BLOCK_SIZE / BLOCK_SIZE_X;
            constexpr int numPerThread = 32;
            dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridDim((othersize + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
            warpSoftMax<float, BLOCK_SIZE_X, BLOCK_SIZE_Y, numPerThread>
                    <<<gridDim, blockDim, 0, stream>>>(
                            static_cast<const float*>(input),
                            static_cast<float*>(output),
                            dim,
                            layout,
                            othersize);
        } else
        {
            constexpr int BLOCK_SIZE_X = 16;
            constexpr int BLOCK_SIZE_Y = BLOCK_SIZE / BLOCK_SIZE_X;
            constexpr int numPerThread = 2;
            dim3 blockDim(BLOCK_SIZE_X, BLOCK_SIZE_Y);
            dim3 gridDim((othersize + BLOCK_SIZE_Y - 1) / BLOCK_SIZE_Y);
            warpSoftMax<float, BLOCK_SIZE_X, BLOCK_SIZE_Y, numPerThread>
                    <<<gridDim, blockDim, 0, stream>>>(
                            static_cast<const float*>(input),
                            static_cast<float*>(output),
                            dim,
                            layout,
                            othersize);
        }
    } else
    {
        return MyKernelStatus::MYKERNEL_NOT_SUPPORT;
    }
}

MyKernelStatus softmax(const void* input, void* output, Layout layout, int dim, DataType datatype, cudaStream_t stream)
{
    int maxThreadsPerBlock = getMaxThreadsPerBlock();

    if (maxThreadsPerBlock == 1024)
    {
        launch_kernel<1024>(input, output, layout, dim, datatype, stream);
    } else if (maxThreadsPerBlock == 512)
    {
        launch_kernel<512>(input, output, layout, dim, datatype, stream);
    } else if (maxThreadsPerBlock == 256)
    {
        launch_kernel<256>(input, output, layout, dim, datatype, stream);
    } else
    {
        launch_kernel<128>(input, output, layout, dim, datatype, stream);
    }
}
}