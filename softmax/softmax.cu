#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros_like.h"
#include "c10/util/ArrayRef.h"
#include <cassert>
#include <cmath>
#include <cstddef>
#include <torch/extension.h>
#include <torch/types.h>

#define WARP_SIZE 32

struct __align__(8) MaxSum
{
    float exp_sum;
    float mx;
};

__device__ MaxSum warp_reduce_max_sum(MaxSum val)
{
#pragma unroll
    for (int offset = WARP_SIZE >> 1; offset >= 1; offset >>= 1)
    {
        MaxSum other;
        other.mx = __shfl_down_sync(0xffffffff, val.mx, offset);
        other.exp_sum = __shfl_down_sync(0xffffffff, val.exp_sum, offset);

        float new_mx = max(other.mx, val.mx);
        float new_exp_sum = other.exp_sum * __expf(other.mx - new_mx) + val.exp_sum * __expf(val.mx - new_mx);
        val.mx = new_mx;
        val.exp_sum = new_exp_sum;
    }
    return val;
}

template <int BLOCK_SIZE = 512>
__device__ MaxSum block_reduce_max_sum(MaxSum val)
{
    static __shared__ MaxSum sram[BLOCK_SIZE / WARP_SIZE];
    int warp_id = threadIdx.x >> 5;
    int land_id = threadIdx.x & 0x1f;
    int tid = threadIdx.x;
    val = warp_reduce_max_sum(val);
    if (land_id == 0)
        sram[warp_id] = val;
    __syncthreads();
    val = tid < (BLOCK_SIZE / WARP_SIZE) ? sram[land_id] : MaxSum{0, -INFINITY};
    val = warp_reduce_max_sum(val);
    return val;
}

__global__ void softmax_kernel(float* output, float* input, int N, int C)
{
    __shared__ MaxSum max_sum;
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    output += bx * C;
    input += bx * C;

    MaxSum now{0, -INFINITY};
    for (int i = tid; i < C; i += blockDim.x)
    {
        float x = input[i];
        float new_mx = max(x, now.mx);
        now.exp_sum = now.exp_sum * __expf(now.mx - new_mx) + __expf(x - new_mx);
        now.mx = new_mx;
    }
    now = block_reduce_max_sum(now);
    if (tid == 0)
    {
        max_sum = now;
    }
    __syncthreads();
    for (int i = tid; i < C; i += blockDim.x)
    {
        output[i] = __expf(input[i] - max_sum.mx) / max_sum.exp_sum;
    }
}

at::Tensor softmax_forward(at::Tensor input)
{
    auto shapes = input.sizes();
    int N = shapes[0];
    int C = shapes[1];

    auto output = torch::zeros_like(input);
    dim3 thread_per_block(512);
    dim3 block_per_grid(N);
    softmax_kernel<<<block_per_grid, thread_per_block>>>(output.data_ptr<float>(), input.data_ptr<float>(), N, C);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("softmax", &softmax_forward, "softmax_forward");
}