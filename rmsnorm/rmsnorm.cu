#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros_like.h"
#include "c10/util/ArrayRef.h"
#include <cassert>
#include <torch/extension.h>
#include <torch/types.h>

__device__ float warp_reduce_sum(float val)
{
    for (int offset = 16; offset >= 1; offset >>= 1)
    {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template <int BLOCK_SIZE = 256, int WARP_SIZE = 32>
__device__ float block_reduce_sum(float sum)
{
    static __shared__ float sram[BLOCK_SIZE / WARP_SIZE];
    int tid = threadIdx.x;
    int land_id = tid & 0x1f;
    int warp_id = tid >> 5;
    sum = warp_reduce_sum(sum);
    if (land_id == 0)
    {
        sram[warp_id] = sum;
    }
    __syncthreads();
    sum = land_id < BLOCK_SIZE / WARP_SIZE ? sram[land_id] : 0.f;
    sum = warp_reduce_sum(sum);
    return sum;
}

template <int BLOCK_SIZE = 256, int WARP_SIZE = 32>
__global__ void rmsnorm_kernel(float* output, const float* input, const float* gamma, int N, int C)
{
    __shared__ float scale;
    // __shared__ float sram[BLOCK_SIZE / WARP_SIZE];
    int tid = threadIdx.x;
    int bx = blockIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 0x1f;
    int row = bx;
    const float epsilon = 1e-5;

    input += row * C;
    output += row * C;

    float sum = 0;
    for (int i = tid; i < C; i += blockDim.x)
    {
        float x = input[i];
        sum += x * x;
    }
    sum = block_reduce_sum(sum);
    if (tid == 0)
    {
        scale = rsqrtf(sum / C + epsilon);
    }
    __syncthreads();
    for (int i = tid; i < C; i += blockDim.x)
    {
        output[i] = input[i] * scale * gamma[i];
    }
}

at::Tensor rmsnorm_forward(at::Tensor input, at::Tensor gamma)
{
    auto input_shape = input.sizes();
    int N = input_shape[0];
    int C = input_shape[1];

    at::Tensor output = torch::zeros_like(input);

    dim3 thread_per_block(256);
    dim3 block_per_grid(N);

    rmsnorm_kernel<<<block_per_grid, thread_per_block>>>(
        output.data_ptr<float>(), input.data_ptr<float>(), gamma.data_ptr<float>(), N, C);
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("rmsnorm", &rmsnorm_forward, "rmsnorm");
}
