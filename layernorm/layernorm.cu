#include <cmath>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

#include "common.h"

void layernorm_cpu(
    const floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C)
{
    floatX eps = 1e-5;
    for (int i = 0; i < N; i++)
    {
        const floatX* in_ptr = in + i * C;
        floatX* out_ptr = out + i * C;
        // calculate mean
        floatX sum = 0.f;
        for (int j = 0; j < C; j++)
        {
            sum += in_ptr[j];
        }
        floatX m = sum / C;
        floatX v = 0.f;
        for (int j = 0; j < C; j++)
        {
            floatX diff = in_ptr[j] - m;
            v += diff * diff;
        }
        floatX std = v / C;
        floatX s = 1.f / sqrtf(std + eps);
        for (int j = 0; j < C; j++)
        {
            // normalize
            floatX norm = (in_ptr[j] - m) * s;
            // scale and shift
            out_ptr[j] = norm * weight[j] + bias[j];
        }
        mean[i] = m;
        rstd[i] = s;
    }
}

// layernorm kernel 1: parallel in N, one thread for one row
__global__ void layernorm_kernel1(
    const floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C)
{
    floatX eps = 1e-5;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
    {
        return;
    }
    const floatX* in_ptr = in + idx * C;
    floatX* out_ptr = out + idx * C;
    floatX m = 0.f, s = 0.f;
    // calculate mean
    for (int i = 0; i < C; i++)
    {
        m += in_ptr[i];
    }
    m /= C;
    for (int i = 0; i < C; i++)
    {
        floatX diff = in_ptr[i] - m;
        s += diff * diff;
    }
    s /= C;
    s = 1.f / sqrtf(s + eps);
    for (int i = 0; i < C; i++)
    {
        // normalize
        float norm = (in_ptr[i] - m) * s;
        // scale and shift
        out_ptr[i] = norm * weight[i] + bias[i];
    }
    mean[idx] = m;
    rstd[idx] = s;
}

// layernorm kernel2 : 把layernorm分成三个小的kernel来做, Mean, Rstd, Normlize
// with shm, one block for one row
__global__ void mean_kernel(const floatX* in, floatX* mean, int N, int C)
{
    // input : [N, C]
    extern __shared__ floatX s_data[]; // [0, blockDim)
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    if (tid >= C)
    {
        return;
    }
    const floatX* in_ptr = in + idx * C;
    floatX sum = 0.f;
    // 线程粗化
    for (int i = tid; i < C; i += blockDim.x)
    {
        sum += in_ptr[i];
    }
    s_data[tid] = sum;
    __syncthreads();
    // block-level reduction
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1)
    {
        if (tid < stride)
        {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        mean[idx] = s_data[0] / C;
    }
}
// calculate rstd kernel
// logic is same as mean
__global__ void rstd_kernel(const floatX* in, floatX* mean, floatX* rstd, int N, int C)
{
    extern __shared__ floatX s_data[]; // [0, blockDim.x)
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    const floatX* in_ptr = in + idx * C;
    if (tid > C)
    {
        return;
    }
    floatX diff_sum = 0.f;
    floatX m = mean[idx];
    // 线程粗化
    for (int i = tid; i < C; i += blockDim.x)
    {
        floatX diff = in_ptr[tid] - m;
        diff_sum += diff * diff;
    }
    s_data[tid] = diff_sum;
    __syncthreads();
    // block-level reducion
    for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1)
    {
        if (tid < stride)
        {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        rstd[idx] = 1.f / sqrtf(s_data[0] / C + 1e-5f);
    }
}
// normalize
// one thread for one element
__global__ void norm_kernel(
    const floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int row = idx / C;
    int col = idx % C;
    floatX m = mean[row];
    floatX s = rstd[row];
    floatX xi = in[idx];
    floatX n = s * (xi - m);
    floatX o = n * weight[col] + bias[col];

    out[idx] = o;
}

// kernel3: one warp for one row, implement with cooperative groups
__global__ void layernorm_kernel3(const floatX* __restrict__ in, floatX* __restrict__ out, floatX* __restrict__ mean,
    floatX* __restrict__ rstd, floatX* __restrict__ weight, floatX* __restrict__ bias, int N, int C)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N)
    {
        return;
    }
    const floatX* in_ptr = in + idx * C;
    floatX* out_ptr = out + idx * C;
    // 线程粗化
    floatX sum = 0.f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        sum += in_ptr[i];
    }
    // reduction sum
    sum = cg::reduce(warp, sum, cg::plus<floatX>{});
    floatX m = sum / C;
    floatX diff_sum = 0.f;
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        floatX diff = in_ptr[i] - m;
        diff_sum = diff * diff;
    }
    diff_sum = cg::reduce(warp, diff_sum, cg::plus<floatX>{});
    floatX s = 1.f / sqrtf(diff_sum / C + 1e-5f);
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        // normalize
        floatX norm = (in_ptr[i] - m) * s;
        // scale and shift
        out_ptr[i] = norm * weight[i] + bias[i];
    }
    if (warp.thread_rank() == 0)
    {
        mean[idx] = m;
        rstd[idx] = s;
    }
}
// kernel4 : use var(x) = mean(x**2) - mean(x)**2, and implement without cooperative-groups
// one warp for one row
__global__ void layernorm_kernel4(
    const floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C)
{
    // input : [N, C]
    // warp-level reduction
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    int warpPerBlock = blockDim.x / WARP_SIZE;
    int row = blockIdx.x * warpPerBlock + warp_id;
    if (row >= N)
    {
        return;
    }
    const floatX* in_ptr = in + row * C;
    floatX* out_ptr = out + row * C;

    floatX sum = 0.f;
    floatX sum_2 = 0.f;
    // 线程粗化
    for (int i = lane_id; i < C; i += WARP_SIZE)
    {
        sum += in_ptr[i];
        sum_2 += in_ptr[i] * in_ptr[i];
    }
    // warp-level reduction
    for (int offset = WARP_SIZE / 2; offset >= 1; offset >>= 1)
    {
        __syncwarp();
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
        sum_2 += __shfl_down_sync(0xFFFFFFFF, sum_2, offset);
    }
    floatX m = sum / C;
    floatX m_2 = sum_2 / C;
    floatX s = 1.f / sqrtf(m_2 - m * m + 1e-5f);
    // broadcast to thread in the same warp
    __syncwarp();
    for (int i = lane_id; i < C; i += WARP_SIZE)
    {
        floatX norm = (in_ptr[i] - m) * s;
        out_ptr[i] = norm * weight[i] + bias[i];
    }

    if (lane_id == 0)
    {
        mean[row] = m;
        rstd[row] = s;
    }
}

//_________________________KERNEL LAUNCHER____________________________//

void layernorm1(
    floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C, int block_size)
{
}
void layernorm2(
    floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C, int block_size)
{
}
void layernorm3(
    floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C, int block_size)
{
}
void layernorm4(
    floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C, int block_size)
{
}

//__________________________KERNLE DISPATCHER_________________________//
void layernorm(int kernel_id, floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N,
    int C, int block_size)
{
    switch (kernel_id)
    {
    case 1: layernorm1(in, out, mean, rstd, weight, bias, N, C, block_size); break;
    case 2: layernorm2(in, out, mean, rstd, weight, bias, N, C, block_size); break;
    case 3: layernorm3(in, out, mean, rstd, weight, bias, N, C, block_size); break;
    case 4: layernorm4(in, out, mean, rstd, weight, bias, N, C, block_size); break;
    default: printf("Invalid kernel id: %d\n", kernel_id); break;
    }
}
//__________________________MAIN______________________________________//
