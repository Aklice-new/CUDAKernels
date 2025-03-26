#include "ATen/core/TensorBody.h"
#include "ATen/ops/zeros_like.h"
#include "c10/util/ArrayRef.h"
#include <cassert>
#include <torch/extension.h>
#include <torch/types.h>

#define CEIV_DIV(a, b) (((a) + (b) -1) / (b))
#define FLOAT4(val) (reinterpret_cast<float4*>(&(val))[0])

__global__ void vector_add_kernel(float* out, float* a, float* b, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int idx = tid * 4;
    if (idx < N)
    {
        float4 vec_a = FLOAT4(a[idx]);
        float4 vec_b = FLOAT4(b[idx]);
        float4 res;
        res.x = vec_a.x + vec_b.x;
        res.y = vec_a.y + vec_b.y;
        res.z = vec_a.z + vec_b.z;
        res.w = vec_a.w + vec_b.w;
        FLOAT4(out[idx]) = res;
    }
}

at::Tensor vector_add_forward(at::Tensor a, at::Tensor b)
{

    int N = a.numel();
    auto out = torch::zeros_like(a);
    dim3 thread_per_block(512 / 4);
    dim3 block_per_grid(CEIV_DIV(N, 512));
    vector_add_kernel<<<block_per_grid, thread_per_block>>>(
        out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), N);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("vector_add", &vector_add_forward, "vector_add");
}
