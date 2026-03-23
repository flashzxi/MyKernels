#pragma once

#include "common.h"

namespace my_kernels
{
MyKernelStatus softmax(const void* input, void* output, Layout layout, int dim, DataType datatype, cudaStream_t stream);
}
