#pragma once

#include "common.h"

namespace my_kernels
{
int softmax(const Tensor& input, int dim, DataType datatype, Tensor& output);
int safe_softmax(const Tensor& input, int dim, DataType datatype, Tensor& output);
}
