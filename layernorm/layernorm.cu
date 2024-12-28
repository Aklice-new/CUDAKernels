#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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
    // extern __shared__ floatX s_data[]; // [0, blockDim)
    // int idx = blockIdx.x;
    // int tid = threadIdx.x;
    // if (tid >= C)
    // {
    //     return;
    // }
    // const floatX* in_ptr = in + idx * C;
    // floatX sum = 0.f;
    // // 线程粗化
    // for (int i = tid; i < C; i += blockDim.x)
    // {
    //     sum += in_ptr[i];
    // }
    // s_data[tid] = sum;
    // __syncthreads();
    // // block-level reduction
    // for (int stride = blockDim.x / 2; stride >= 1; stride >>= 1)
    // {
    //     if (tid < stride)
    //     {
    //         s_data[tid] += s_data[tid + stride];
    //     }
    //     __syncthreads();
    // }
    // if (tid == 0)
    // {
    //     mean[idx] = s_data[0] / C;
    // }
    extern __shared__ float shared[];
    int idx = blockIdx.x;  // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = in + idx * C;
    int block_size = blockDim.x;
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size)
    {
        sum += x[i];
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0)
    {
        mean[idx] = shared[0] / C;
    }
}
// calculate rstd kernel
// logic is same as mean
__global__ void rstd_kernel(const floatX* in, floatX* mean, floatX* rstd, int N, int C)
{
    extern __shared__ float shared[];
    int idx = blockIdx.x;  // range [0, B*T)
    int tid = threadIdx.x; // range [0, block_size)
    const float* x = in + idx * C;
    int block_size = blockDim.x;
    float m = mean[idx];
    // thread coarsening
    float sum = 0.0f;
    for (int i = tid; i < C; i += block_size)
    {
        float diff = x[i] - m;
        sum += diff * diff;
    }
    shared[tid] = sum;
    __syncthreads();
    // reductions
    for (int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        __syncthreads();
        if (tid < stride)
        {
            shared[tid] += shared[tid + stride];
        }
    }
    // write the final result (at thread 0) to global memory
    if (tid == 0)
    {
        rstd[idx] = 1.0f / sqrtf(shared[0] / C + 1e-5f);
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
    floatX n = (xi - m) * s;
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
        diff_sum += diff * diff;
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
    floatX s = rsqrtf(m_2 - m * m + 1e-5f);
    // broadcast to thread in the same warp
    m = __shfl_sync(0xFFFFFFFF, m, 0);
    s = __shfl_sync(0xFFFFFFFF, s, 0);
    for (int i = lane_id; i < C; i += WARP_SIZE)
    {
        // normalize
        floatX norm = (in_ptr[i] - m) * s;
        // scale and shift
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
    int grid_size = CEIL_DIV(N, block_size);
    layernorm_kernel1<<<grid_size, block_size>>>(in, out, mean, rstd, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}
void layernorm2(
    floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C, int block_size)
{

    // mean
    int grid_size = N;
    size_t shm_size = block_size * sizeof(floatX);
    mean_kernel<<<grid_size, block_size, shm_size>>>(in, mean, N, C);
    cudaCheck(cudaGetLastError());
    // rstd
    rstd_kernel<<<grid_size, block_size, shm_size>>>(in, mean, rstd, N, C);
    cudaCheck(cudaGetLastError());
    // norm
    const int block_size2 = 256;
    grid_size = CEIL_DIV(N * C, block_size2);
    norm_kernel<<<grid_size, block_size2>>>(in, out, mean, rstd, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}
void layernorm3(
    floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C, int block_size)
{
    int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_kernel3<<<grid_size, block_size>>>(in, out, mean, rstd, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}
void layernorm4(
    floatX* in, floatX* out, floatX* mean, floatX* rstd, floatX* weight, floatX* bias, int N, int C, int block_size)
{
    int grid_size = CEIL_DIV(N * 32, block_size);
    layernorm_kernel4<<<grid_size, block_size>>>(in, out, mean, rstd, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
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

int main(int argc, char* argv[])
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    int kernel_id = 1;
    if (argc > 1)
    {
        kernel_id = atoi(argv[1]);
    }
    printf("Using kernel %d\n", kernel_id);

    // create input and output
    // a + b = c
    int N = 8 * 1024;
    int C = 768;
    // host memory
    thrust::host_vector<float> h_in(N * C);
    thrust::host_vector<float> h_out(N * C);
    thrust::host_vector<float> h_mean(N);
    thrust::host_vector<float> h_rstd(N);
    thrust::host_vector<float> h_weight(C);
    thrust::host_vector<float> h_bias(C);

    make_random_float(h_in.data(), N * C);
    make_random_float(h_out.data(), N * C);
    make_zeros_float(h_mean.data(), N);
    make_zeros_float(h_rstd.data(), N);
    make_random_float(h_weight.data(), C);
    make_random_float(h_bias.data(), C);

    // device memory
    thrust::device_vector<floatX> d_in(N * C);
    thrust::device_vector<floatX> d_out(N * C);
    thrust::device_vector<float> d_mean(N);
    thrust::device_vector<float> d_rstd(N);
    thrust::device_vector<float> d_weight(C);
    thrust::device_vector<float> d_bias(C);
    cudaCheck(type_convert_memcpy(d_in.data().get(), h_in.data(), N * C));
    cudaCheck(type_convert_memcpy(d_out.data().get(), h_out.data(), N * C));
    cudaCheck(type_convert_memcpy(d_mean.data().get(), h_mean.data(), N));
    cudaCheck(type_convert_memcpy(d_rstd.data().get(), h_rstd.data(), N));
    cudaCheck(type_convert_memcpy(d_weight.data().get(), h_weight.data(), C));
    cudaCheck(type_convert_memcpy(d_bias.data().get(), h_bias.data(), C));

    // cpu
    layernorm_cpu(h_in.data(), h_out.data(), h_mean.data(), h_rstd.data(), h_weight.data(), h_bias.data(), N, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        layernorm(kernel_id, d_in.data().get(), d_out.data().get(), d_mean.data().get(), d_rstd.data().get(),
            d_weight.data().get(), d_bias.data().get(), N, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_mean.data().get(), h_mean.data(), "mean", N, tol);
        validate_result(d_rstd.data().get(), h_rstd.data(), "rstd", N, tol);
        validate_result(d_out.data().get(), h_out.data(), "out", N * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    //_______________BENCHMARK___________________//
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, layernorm, kernel_id, d_in.data().get(), d_out.data().get(),
            d_mean.data().get(), d_rstd.data().get(), d_weight.data().get(), d_bias.data().get(), N, C, block_size);

        long memory_ops = N * C * 4 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    return 0;
}