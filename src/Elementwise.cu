#include "Elementwise.h"
#include <cuda_stdint.h>
#include <cuda/std/__type_traits/rank.h>

namespace my_kernels
{

struct AddOp
{
    constexpr int INPUT_SIZE = 2;

    template <typename T>
    MYKERNEL_HOST_DEVICE
    static T calc(T a, T b)
    {
        return a + b;
    }
};

// 拼接参数到一起
struct ElementwiseInfo
{
private:
    static constexpr int MAX_RANK = sizeof(Shape) / sizeof(int);
    static constexpr int SHAPE_SIZE = sizeof(Shape);

    std::vector<uint8_t> _encoded_params;
    int _input_size;
    int _rank;

public:
    [[nodiscard]] size_t getInfoMemSize() const
    {
        return _encoded_params.size() * sizeof(uint8_t);
    }

    [[nodiscard]] const uint8_t* getInfoStart() const
    {
        return _encoded_params.data();
    }

    [[nodiscard]] int getInputSize() const
    {
        return _input_size;
    }

    [[nodiscard]] int getRank() const
    {
        return _rank;
    }

    [[nodiscard]] size_t getOutputShapeOffset() const
    {
        return 0;
    }

    [[nodiscard]] size_t getOutputStrideOffset() const
    {
        return getOutputShapeOffset() + SHAPE_SIZE;
    }

    [[nodiscard]] size_t getAllInputsShapeOffset() const
    {
        return getOutputStrideOffset() + SHAPE_SIZE;
    }

    [[nodiscard]] size_t getAllInputsStrideOffset() const
    {
        return getAllInputsShapeOffset() + SHAPE_SIZE * _input_size;
    }

    [[nodiscard]] size_t getAllInputsDataOffset() const
    {
        return getAllInputsStrideOffset() + SHAPE_SIZE * _input_size;
    }

    static ElementwiseInfo create(const std::vector<Layout>& input_layouts, const Layout& output_layout, std::vector<void*> input_ptrs)
    {
        ElementwiseInfo info;
        info._rank = -1;
        info._input_size = input_layouts.size();
        info._encoded_params.resize((2 + 2 * info._input_size) * SHAPE_SIZE + info._input_size * sizeof(void*));

        uint8_t* start_ptr = info._encoded_params.data();
        uint8_t* output_shape_ptr = start_ptr;
        uint8_t* output_stride_ptr = output_shape_ptr + SHAPE_SIZE;
        uint8_t* input_shape_ptr = output_stride_ptr + SHAPE_SIZE;
        uint8_t* input_stride_ptr = output_stride_ptr + SHAPE_SIZE * info._input_size;
        uint8_t* input_data_ptr = input_stride_ptr + SHAPE_SIZE * info._input_size;

        memcpy(output_shape_ptr, output_layout.shape.data(), SHAPE_SIZE);
        memcpy(output_stride_ptr + SHAPE_SIZE, output_layout.stride.data(), SHAPE_SIZE);

        for (int i = 0; i < input_layouts.size(); ++i)
        {
            info._rank = max(info._rank, input_layouts[i].rank);
            memcpy(input_shape_ptr + i * SHAPE_SIZE, input_layouts[i].shape.data(), SHAPE_SIZE);
            memcpy(input_stride_ptr + i * SHAPE_SIZE, input_layouts[i].stride.data(), SHAPE_SIZE);
            reinterpret_cast<void**>(input_data_ptr)[i] = input_ptrs[i];
        }
    }
};

template <typename ELEMENT_OP, typename T,typename... Args>
MYKERNEL_HOST_DEVICE
void elementwise_kernel(int input_size, T** inputs, int** input_shape, int** input_stride,
        T* output, int* output_shape, int* output_stride, int total_size, Args ...args)
{
    assert(T::INPUT_SIZE == input_size);
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t step = blockDim.x * gridDim.x;

    for (; tid < total_size; tid += step)
    {
        ELEMENT_OP::calc();
    }
}

template <typename ELEMENT_OP, typename ...Args>
MyKernelStatus elementwiseExecute(
        std::vector<void*> inputs,
        std::vector<Layout> input_layouts,
        void* output,
        Layout output_layout,
        DataType dataType,
        cudaStream_t stream,
        Args... args)
{
    int* input_nums = nullptr;
    int* output_shape = nullptr;
    int* output_stride = nullptr;
    int** input_shapes = nullptr;
    int** input_strides = nullptr;
    int** input_datas = nullptr;

    std::vector<void*> d_input_ptrs;
    for (int i = 0; i < inputs.size(); ++i)
    {
        void* d_input = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_input, input_layouts[i].data_size() * dataTypeSize(dataType), stream));
        CUDA_CHECK(cudaMemcpyAsync(
            d_input, inputs[i], input_layouts[i].data_size() * dataTypeSize(dataType), cudaMemcpyHostToDevice, stream));
        d_input_ptrs.emplace_back(d_input);
    }

    ElementwiseInfo info = ElementwiseInfo::create(input_layouts, output_layout, d_input_ptrs);

    void* d_info_ptr;
    CUDA_CHECK(cudaMallocAsync(&d_info_ptr, info.getInfoMemSize(), stream));
    CUDA_CHECK(cudaMemcpyAsync(
        d_info_ptr, info.getInfoStart(), info.getInputSize(), cudaMemcpyHostToDevice, stream));

    void* d_output_ptr;
    CUDA_CHECK(cudaMallocAsync(&d_output_ptr, output_layout.data_size() * dataTypeSize(dataType), stream));

    output_shape = reinterpret_cast<int*>(d_info_ptr + info.getOutputShapeOffset());
    output_stride = reinterpret_cast<int*>(d_info_ptr + info.getOutputStrideOffset());
    input_shapes = reinterpret_cast<int**>(d_info_ptr + info.getAllInputsShapeOffset());
    input_strides = reinterpret_cast<int**>(d_info_ptr + info.getAllInputsStrideOffset());
    input_datas = reinterpret_cast<int**>(d_info_ptr + info.getAllInputsDataOffset());



    CUDA_CHECK(cudaMemcpyAsync(
        output, d_output_ptr, output_layout.data_size() * dataTypeSize(dataType), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaFreeAsync(d_info_ptr, stream));
    CUDA_CHECK(cudaFreeAsync(d_output_ptr, stream));
    for (auto d_input: d_input_ptrs)
    {
        CUDA_CHECK(cudaFreeAsync(d_input, stream));
    }
    return MyKernelStatus::MYKERNEL_SUCCESS;

}

template MyKernelStatus elementwiseExecute<AddOp>(
        std::vector<void*> inputs,
        std::vector<Layout> input_layouts,
        void* output,
        Layout output_layout,
        DataType dataType,
        cudaStream_t stream);
}
