#include "Elementwise.cuh"

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

namespace my_kernels
{
MyKernelStatus add(void* input1, void* input2, void* output,
        Layout input1_layout, Layout input2_layout, Layout output_layout, DataType dataType, cudaStream_t stream)
{
    if (dataType == DataType::DATATYPE_FLOAT16)
    {
        return elementwiseExecute<512, AddOp, __half>(
            {input1, input2},
            {input1_layout, input2_layout},
            output,
            output_layout,
            dataType,
            stream);
    }
    if (dataType == DataType::DATATYPE_BF16)
    {
        return elementwiseExecute<512, AddOp, __bf16>(
            {input1, input2},
            {input1_layout, input2_layout},
            output,
            output_layout,
            dataType,
            stream);
    }
    if (dataType == DataType::DATATYPE_FLOAT32)
    {
        return elementwiseExecute<256, AddOp, float>(
            {input1, input2},
            {input1_layout, input2_layout},
            output,
            output_layout,
            dataType,
            stream);
    }
    return MyKernelStatus::MYKERNEL_NOT_SUPPORT;
}
}