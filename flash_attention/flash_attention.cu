#include "ATen/core/TensorBody.h"
#include "c10/core/DeviceType.h"
#include "c10/util/ArrayRef.h"
#include <cassert>
#include <cmath>
#include <torch/extension.h>
#include <torch/types.h>

void cuda_check(cudaError_t error, const char* file, const int line)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%i: %s\n", file, line, cudaGetErrorString(error));
        exit(-1);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))
#define CEIL_DIV(a, b) (((a) + (b - 1)) / (b))

__global__ void flash_attention_kernel(float* O, int N, int d, int Tc, int Tr, int Bc, int Br, float scale,
    const float* Q, const float* K, const float* V, float* l, float* m)
{
    int tid = threadIdx.x;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int head_num = gridDim.y;

    extern __shared__ float sram[];

    int qkvo_offset = bx * head_num * N * d + by * N * d;
    int lm_offset = bx * head_num * N + by * N;

    Q += qkvo_offset;
    K += qkvo_offset;
    V += qkvo_offset;
    O += qkvo_offset;
    l += lm_offset;
    m += lm_offset;

    int Q_tile_size = Br * d;
    int KV_tile_size = Bc * d;

    float* Qi = sram;                      // [Br, d]
    float* Kj = sram + Br * d;             // [Bc, d]
    float* Vj = sram + Br * d + Bc * d;    // [Bc, d]
    float* S = sram + Br * d + 2 * Bc * d; // [Br, Bc]

    // outer loop K_j, V_j from 1 to Tc
    for (int j = 0; j < Tc; j++)
    {
        // load K_j, V_j to sram
        for (int x = 0; x < d; x++)
        {
            Kj[(tid * d) + x] = K[KV_tile_size * j + tid * d + x];
            Vj[(tid * d) + x] = V[KV_tile_size * j + tid * d + x];
        }
        __syncthreads();
        // load Q_i, O_i, l_i, m_i
        for (int i = 0; i < Tr; i++)
        {
            for (int x = 0; x < d; x++)
            {
                Qi[(tid * d) + x] = Q[Q_tile_size * i + tid * d + x];
            }
            // __syncthreads();
            float row_m_pre = m[Br * i + tid];
            float row_l_pre = l[Br * i + tid];
            // compute Q @ K^T

            // 当前线程计算S[tid, ...]这一行中的所有值，并记录下来了这一行的最大值
            float row_m = -INFINITY;
            for (int y = 0; y < Bc; y++)
            {
                float sum = 0;
                for (int x = 0; x < d; x++)
                {
                    sum += Qi[tid * d + x] * Kj[y * d + x]; // 固定Q_i的这一行
                }
                sum *= scale;
                S[tid * Bc + y] = sum;

                if (sum > row_m)
                {
                    row_m = sum;
                }
            }

            // 然后计算这一行的 \sum exp(x_i - max)
            float row_l = 0;
            for (int y = 0; y < Bc; y++)
            {
                S[tid * Bc + y] = __expf(S[tid * Bc + y] - row_m);
                row_l += S[tid * Bc + y];
            }
            // compute new m, l
            float row_m_new = max(row_m_pre, row_m);

            float row_l_new = (__expf(row_m_pre - row_m_new) * row_l_pre) + (__expf(row_m - row_m_new) * row_l);
            // compute S_ij, O j
            for (int x = 0; x < d; x++)
            {
                float sum = 0;
                for (int y = 0; y < Bc; y++)
                {
                    sum += S[tid * Bc + y] * Vj[y * d + tid];
                }
                float old_val = O[Q_tile_size * i + tid * d + x];
                O[Q_tile_size * i + tid * d + x]
                    = (old_val * row_l_pre * __expf(row_m_pre - row_m_new) + sum * __expf(row_m - row_m_new))
                    / row_l_new;
            }
            m[Br * i + tid] = row_m_new;
            l[Br * i + tid] = row_l_new;
        }
        __syncthreads();
    }
}

torch::Tensor flash_attention_forward(at::Tensor Q, at::Tensor K, at::Tensor V)
{
    auto q_shape = Q.sizes();
    auto k_shape = K.sizes();
    auto v_shape = V.sizes();

    // assert(q_shape[2] == k_shape[2]);
    // assert(q_shape[3] == k_shape[3]);
    // assert(q_shape[2] == v_shape[2]);
    // assert(q_shape[3] == v_shape[3]);

    int B = q_shape[0];
    int head_num = q_shape[1];

    int N = q_shape[2];
    int d = q_shape[3];

    int Br = 32;
    int Bc = 32;
    int Tr = CEIL_DIV(N, Br);
    int Tc = CEIL_DIV(N, Bc);
    int shared_memory_size = (Br * d + Bc * d + Bc * d + Br * Bc) * sizeof(float);

    int max_sram_size;
    cudaDeviceGetAttribute(&max_sram_size, cudaDevAttrMaxSharedMemoryPerBlock, 0);
    printf("Max shared memory: %d, requested shared memory: %d \\n", max_sram_size, shared_memory_size);

    auto O = torch::zeros_like(Q);
    auto l = torch::zeros({B, head_num, N});
    auto m = torch::full({B, head_num, N}, -INFINITY);

    torch::Device device(torch::kCUDA);
    l = l.to(device);
    m = m.to(device);

    const float scale = 1.0 / sqrt(d);

    dim3 thread_per_block(Br);
    dim3 block_per_grid(B, head_num);
    flash_attention_kernel<<<block_per_grid, thread_per_block, shared_memory_size>>>(O.data_ptr<float>(), N, d, Tc, Tr,
        Bc, Br, scale, Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(), l.data_ptr<float>(),
        m.data_ptr<float>());
    cudaDeviceSynchronize();

    return O;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("flash_attention", &flash_attention_forward, "reduce_sum");
}
