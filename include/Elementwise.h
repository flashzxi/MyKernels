#pragma once

#include "common.h"

namespace my_kernels
{
MyKernelStatus add(void* input1, void* input2, void* output,
        Layout input1_layout, Layout input2_layout, Layout output_layout, DataType dataType, cudaStream_t stream = nullptr);
}