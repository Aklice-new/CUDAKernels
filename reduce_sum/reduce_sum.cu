#include "ATen/core/TensorBody.h"
#include "c10/util/ArrayRef.h"
#include <cassert>
#include <torch/extension.h>
#include <torch/types.h>

__device__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset >= 1; offset /= 2)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int WARP_SIZE = 32>
__global__ void reduce_sum_kernel1(float* output, const float* input, int N, int C)
{

    extern __shared__ float s_sum[]; // blockDim.x / warp_size
    int row = blockIdx.x;
    int tid = threadIdx.x;

    input += row * C;

    int land_id = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;

    float sum = 0;
    for (int i = tid; i < C; i += blockDim.x)
    {
        sum += input[i];
    }
    sum = warp_reduce_sum(sum);
    if (land_id == 0)
    {
        s_sum[warp_id] = sum;
    }
    __syncthreads();
    // then block level reduce blockDim.x / WARP_SIZE elements if (tid == 0)
    {
        float res = 0;
        for (int i = 0; i < blockDim.x / WARP_SIZE; i++)
        {
            res += s_sum[i];
        }
        output[row] = res;
    }
}

/*

*/
void reduce_sum_forward1(at::Tensor input, at::Tensor& output)
{
    auto shape = input.sizes();
    int N = shape[0];
    int C = shape[1];
    dim3 thread_per_block(256);
    dim3 block_per_grid(N);
    int warp_size = 32;
    int shm_memory_size = (256 / warp_size) * sizeof(float);
    reduce_sum_kernel1<<<block_per_grid, thread_per_block>>>(output.data_ptr<float>(), input.data_ptr<float>(), N, C);
}

/*
blockDim.x == WARP_SIZE
*/
__global__ void reduce_sum_kernel2(float* output, const float* input, int N, int C)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int warp_size = blockDim.x;
    input += row * C;
    int land_id = tid % warp_size;
    float sum = 0;
    for (int i = tid; i < C; i += warp_size)
    {
        sum += input[i];
    }
    sum = warp_reduce_sum(sum);
    if (land_id == 0)
    {
        output[row] = sum;
    }
}

/*

*/
void reduce_sum_forward2(at::Tensor input, at::Tensor& output)
{
    auto shape = input.sizes();
    int N = shape[0];
    int C = shape[1];
    dim3 thread_per_block(32);
    dim3 block_per_grid(N);
    reduce_sum_kernel2<<<block_per_grid, thread_per_block>>>(output.data_ptr<float>(), input.data_ptr<float>(), N, C);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("reduce_sum1", &reduce_sum_forward1, "reduce_sum");
    m.def("reduce_sum2", &reduce_sum_forward2, "reduce_sum");
}
