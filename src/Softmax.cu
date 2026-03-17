#include "Softmax.h"

namespace my_kernels
{

template<int BLOCK_SIZE>
int launch_kernel(void* data, int blocknum, int rank, )
{

}

int softmax(const Tensor& input, int dim, DataType datatype, Tensor& output)
{
    auto layout = input.layout;
    int rank = layout.rank();
    dim = dim < 0 ? dim + rank : dim;
    int blocksize = input.layout.shape[dim];
    int othersize = layout.size() / blocksize;

    ptrdiff_t stride = layout.stride[dim];
    return 0;
}
}