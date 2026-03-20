#pragma once

#include "common.h"

namespace my_kernels
{
template <typename ELEMENT_OP>
MyKernelStatus elementwiseExecute(void* input1, void* input2, void* output, Layout layout, DataType dataType);
}