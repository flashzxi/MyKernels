#pragma once

#include "common.h"

namespace my_kernels
{
MyKernelStatus softmax(const void* input, Layout layout, int dim, DataType datatype, HostTensor& output, cudaStream_t stream = nullptr);
}
