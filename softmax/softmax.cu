#include <cuda.h>
#include <cstdio>
#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <float.h>

#include "common.h"

/**
 * @brief softmax function
 * @param in [N * C]
 * @param out [N * C]
 * @param N number of elements
 * @param C number of classes
    out[i] = exp(in[i] - max(in)) / sum(exp(in[j] - max(in)))
*/
void softmax_cpu(float* in, float* out, int N, int C)
{
    for (int i = 0; i < N; i++)
    {
        const float* in_ptr = in + i * C;
        float* out_ptr = out + i * C;
        float max_ = FLT_MIN;
        for (int i = 0; i < C; i++)
        {
            max_ = std::max(max_, in_ptr[i]);
        }
        float sum = 0.f;
        for (int i = 0; i < C; i++)
        {
            out_ptr[i] = exp(in_ptr[i] - max_);
            sum += out_ptr[i];
        }
        float norm = 1.f / sum;
        for (int i = 0; i < C; i++)
        {
            out_ptr[i] *= norm;
        }
    }
}

// kernle1: parallelize over B, T, loops over C
__global__ void softmax_kernel1(floatX* in, floatX* out, int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        floatX* in_ptr = in + idx * C;
        floatX* out_ptr = out + idx * C;
        floatX max_ = -FLT_MAX;
        floatX sum = 0.f;
        for (int i = 0; i < C; i++)
        {
            max_ = fmaxf(max_, in_ptr[i]);
        }
        for (int i = 0; i < C; i++)
        {
            out_ptr[i] = expf(in_ptr[i] - max_);
            sum += out_ptr[i];
        }
        floatX norm = (floatX) 1 / sum;
        for (int i = 0; i < C; i++)
        {
            out_ptr[i] *= norm;
        }
    }
}
// kernle2: parallelize over B, T, C, one block over one row. So two reduction ops, one for max, one for sum
// this two reduction is over block level
__global__ void softmax_kernel2(floatX* in, floatX* out, int N, int C)
{
    extern __shared__ floatX s_data[]; // the size is the same as blockDim.x

    int row = blockIdx.x;  // range: [0, N)
    int tid = threadIdx.x; // range: [0, block_size)
    const floatX* in_ptr = in + row * C;
    floatX* out_ptr = out + row * C;
    floatX max_val = -INFINITY;
    // get max_val in this row, stride is blockDim.x
    for (int i = tid; i < C; i += blockDim.x)
    {
        max_val = fmaxf(max_val, in_ptr[i]);
    }
    s_data[tid] = max_val;
    __syncthreads();
    // reduction in [0, blockDim.x)
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        if (tid < stride)
        {
            s_data[tid] = fmaxf(s_data[tid], s_data[tid + stride]);
        }
        __syncthreads();
    }
    // finish max_val reduction and now it is s_data[0]
    __syncthreads();
    floatX offset = s_data[0];
    floatX sum_val = 0;
    for (int i = tid; i < C; i += blockDim.x)
    {
        out_ptr[i] = expf(in_ptr[i] - offset);
        sum_val += out_ptr[i];
    }
    s_data[tid] = sum_val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2)
    {
        if (tid < stride)
        {
            s_data[tid] += s_data[tid + stride];
        }
        __syncthreads();
    }
    // finish sum_val reduction and now it is s_data[0]
    floatX sum = s_data[0];
    for (int i = tid; i < C; i += blockDim.x)
    {
        out_ptr[i] = (floatX) out_ptr[i] / sum;
    }
}

// kernle3: parallelize over B, T, C, one block over one row.
// this two reduction is base on warp-level and block level
// so its is more efficient than kernel2

__device__ floatX warpReduceMax(floatX value)
{
    for (int offset = WARP_SIZE / 2; offset >= 1; offset /= 2)
    {
        value = fmaxf(value, __shfl_down_sync(0xFFFFFFFF, value, offset));
    }
    return value;
}
__device__ floatX warpReduceSum(floatX value)
{
    for (int offset = WARP_SIZE / 2; offset >= 1; offset /= 2)
    {
        value += __shfl_down_sync(0xFFFFFFFF, value, offset);
    }
    return value;
}
__global__ void softmax_kernel3(floatX* in, floatX* out, int N, int C)
{
    extern __shared__ floatX s_data[]; // [0, blockDim.x / WARP_SIZE]
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int land_id = tid % WARP_SIZE;
    int warpsPerBlock = blockDim.x / WARP_SIZE;

    const floatX* in_ptr = in + idx * C;
    floatX* out_ptr = out + idx * C;
    floatX max_val = -INFINITY;
    for (int i = tid; i < C; i += blockDim.x)
    {
        max_val = fmaxf(max_val, in_ptr[i]);
    }
    // warp reduction
    max_val = warpReduceMax(max_val);
    if (land_id == 0)
    {
        s_data[warp_id] = max_val;
    }
    __syncthreads();
    // // then block reduction
    // for (int stride = warpsPerBlock / 2; stride >= 1; stride /= 2)
    // {
    //     if (warp_id < stride)
    //     {
    //         s_data[warp_id] = fmaxf(s_data[warp_id], s_data[warp_id + stride]);
    //     }
    //     __syncthreads();
    // }
    // __syncthreads();
    //上面这样写是错的，因为这个warpPerBlock可能不是2的幂次方，所以不能用这种方法来reduction
    if (tid == 0)
    {
        floatX max_val = -INFINITY;
        for (int i = 0; i < warpsPerBlock; i++)
        {
            max_val = fmaxf(max_val, s_data[i]);
        }
        s_data[0] = max_val;
    }
    __syncthreads();
    floatX offset = s_data[0];
    floatX sum_val = 0;
    for (int i = tid; i < C; i += blockDim.x)
    {
        out_ptr[i] = expf(in_ptr[i] - offset);
        sum_val += out_ptr[i];
    }
    // warp reduction sum
    sum_val = warpReduceSum(sum_val);
    if (land_id == 0)
    {
        s_data[warp_id] = sum_val;
    }
    __syncthreads();
    // block reduction sum
    // for (int stride = warpsPerBlock / 2; stride >= 1; stride /= 2)
    // {
    //     if (warp_id < stride)
    //     {
    //         s_data[warp_id] += s_data[warp_id + stride];
    //     }
    //     __syncthreads();
    // }
    // 同样这里也不行，因为warpPerBlock可能不是2的幂次方
    if (tid == 0)
    {
        floatX sum_val = 0;
        for (int i = 0; i < warpsPerBlock; i++)
        {
            sum_val += s_data[i];
        }
        s_data[0] = sum_val;
    }
    __syncthreads();
    floatX sum = s_data[0];
    for (int i = tid; i < C; i += blockDim.x)
    {
        out_ptr[i] = (floatX) out_ptr[i] / sum;
    }
}
// kernel4: same as kernel3, but use cooperative-group to implement the warp-level and block-level reduction
__global__ void softmax_kernel4(floatX* in, floatX* out, int N, int C)
{
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<WARP_SIZE> warp = cg::tiled_partition<WARP_SIZE>(block);
    int idx = 0;
}
/**
 * @brief softmax online 版本
 *
 */
void softmax_online_cpu(floatX* in, floatX* out, int N, int C)
{
    for (int i = 0; i < N; i++)
    {
        floatX* in_ptr = in + i * C;
        float* out_ptr = out + i * C;
        floatX max_val = -FLT_MAX;
        floatX sum = 0.f;
        for (int i = 0; i < C; i++)
        {
            floatX cur = in_ptr[i];
            if (cur > max_val)
            {
                floatX max_pre = max_val;
                max_val = cur;
                sum = sum * expf(max_pre - max_val) + expf(cur - max_val);
            }
            else
            {
                sum += expf(cur - max_val);
            }
        }
        floatX norm = 1.f / sum;
        for (int i = 0; i < C; i++)
        {
            out_ptr[i] = expf(in_ptr[i] - max_val) * norm;
        }
    }
}
/**
 * @brief softmax online 版本， parallel on B, T， loop over C
 *
 */
__global__ void softmax_online_kernel1(floatX* in, floatX* out, int N, int C)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        floatX* in_ptr = in + idx * C;
        floatX* out_ptr = out + idx * C;
        floatX max_val = -FLT_MAX;
        floatX sum = 0.f;
        for (int i = 0; i < C; i++)
        {
            floatX cur = in_ptr[i];
            if (cur > max_val)
            {
                floatX max_pre = max_val;
                max_val = cur;
                sum = sum * expf(max_pre - max_val) + expf(cur - max_val);
            }
            else
            {
                sum += expf(cur - max_val);
            }
        }
        floatX norm = 1.f / sum;
        for (int i = 0; i < C; i++)
        {
            out_ptr[i] = expf(in_ptr[i] - max_val) * norm;
        }
    }
}
/**
 * @brief softmax online 版本, parallel on B, T, C, with warp-level and block-level reduction
 *
 */
struct __align__(8) SumMax
{
    floatX sum;
    floatX max_val;
};
__device__ __forceinline__ SumMax reduce_sum_and_max(SumMax a, SumMax b)
{
    bool a_is_max = a.max_val > b.max_val;
    SumMax bigger = a_is_max ? a : b;
    SumMax smaller = a_is_max ? b : a;
    SumMax res;
    res.max_val = bigger.max_val;
    // 这里的这一次合并和之前不一样，就是因为这里最后一项其实是1，现在要把bigger这边的都加上
    res.sum = smaller.sum * expf(smaller.max_val - bigger.max_val) + bigger.sum;
    return res;
}
// 这里有所不同的是，每一行由一个warp来处理，一个warp的线程数是32
// 所以需要的总线程数为 N * 32
// 所以一共启动的block数为 ceil(N * 32 / block_size)

__global__ void softmax_online_kernel2(floatX* in, floatX* out, int N, int C)
{
    namespace cg = cooperative_groups;
    // 这里的block和原本的block一样
    cg::thread_block block = cg::this_thread_block();
    // 将block进行划分，每个tile有32个线程，这里其实就是划分为了warp
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    // 然后根据blockIdx.x来确定当前处理的行数
    // meta_group_size()一个block中warp的个数
    // meta_group_rank()就是当前是在第几个warp中
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N)
    {
        return;
    }
    const floatX* in_ptr = in + idx * C;
    floatX* out_ptr = out + idx * C;

    SumMax sm_partial;
    sm_partial.sum = 0;
    sm_partial.max_val = -INFINITY;
    // 首先当前线程先完成自己的任务，即计算 blockDim.x / warp_size 个元素
    // threadIdx.x 就是当前线程在warp中的位置,即land_id
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        sm_partial = reduce_sum_and_max(sm_partial, {1.0f, in_ptr[i]});
    }
    // then reduction with cg, warp-level
    SumMax sm_total = cg::reduce(warp, sm_partial, reduce_sum_and_max);

    // then divide the sum
    for (int i = warp.thread_rank(); i < C; i += warp.size())
    {
        out_ptr[i] = expf(in_ptr[i] - sm_total.max_val) / sm_total.sum;
    }
}
// 这一版本和上面的基本一致，上面是通过cooperative_groups来实现的，这个是通过warp内的reduction来实现的，比较好理解
__global__ void softmax_online_kernel3(floatX* in, floatX* out, int N, int C)
{
    int idx = blockIdx.x;
    int tid = threadIdx.x;
    // 当前warp_size大于C了
    if (tid >= C)
    {
        return;
    }
    int warp_per_block = blockDim.x / WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    int land_id = tid % WARP_SIZE;
    int row = idx * warp_per_block + warp_id;
    if (row >= N)
    {
        return;
    }
    const floatX* in_ptr = in + row * C;
    floatX* out_ptr = out + row * C;
    // warp内每个线程粗化
    floatX sum = 0;
    float max_val = -INFINITY;
    for (int i = land_id; i < C; i += WARP_SIZE)
    {
        float bigger = fmaxf(max_val, in_ptr[i]);
        sum = sum * expf(max_val - bigger) + expf(in_ptr[i] - bigger);
        max_val = bigger;
    }
    // 然后进行warp内这些线程的reduction，通过shlf_指令来完成
    float tmp_max, tmp_sum;
    for (int i = WARP_SIZE / 2; i >= 1; i >>= 1)
    {
        __syncwarp();
        tmp_max = __shfl_down_sync(0xFFFFFFFF, max_val, i);
        tmp_sum = __shfl_down_sync(0xFFFFFFFF, sum, i);
        if (tmp_max > max_val)
        {
            sum *= expf(max_val - tmp_max);
            max_val = tmp_max;
        }
        else
        {
            tmp_sum *= expf(tmp_max - max_val);
        }
        sum += tmp_sum;
    }
    // warp内进行同步
    sum = __shfl_sync(0xFFFFFFFF, sum, 0);
    max_val = __shfl_sync(0xFFFFFFFF, max_val, 0);
    for (int i = land_id; i < C; i += WARP_SIZE)
    {
        out_ptr[i] = floatX(expf(in_ptr[i] - max_val)) / sum;
    }
}
//___________________KERNLE LUANCHER_________________//

void softmax1(floatX* in, floatX* out, int N, int C, int block_size)
{
    // printf("This is softmax version1, parallel N, loop over C.\n");
    int grid_size = CEIL_DIV(N, block_size);
    softmax_kernel1<<<grid_size, block_size>>>(in, out, N, C);
    cudaCheck(cudaGetLastError());
}
void softmax2(floatX* in, floatX* out, int N, int C, int block_size)
{
    // printf("This is softmax version2, parallel N and C. One block for one row.\n");
    int grid_size = N;
    size_t shm_size = block_size * sizeof(floatX);
    softmax_kernel2<<<grid_size, block_size, shm_size>>>(in, out, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax3(floatX* in, floatX* out, int N, int C, int block_size)
{
    // printf(
    //     "This is softmax version3, same with softmax2, but reduction is implement with warp-level and block level,
    //     and " "save shm use.\n");
    int grid_size = N;
    size_t shm_size = block_size / 32 * sizeof(floatX);
    softmax_kernel3<<<grid_size, block_size, shm_size>>>(in, out, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax4(floatX* in, floatX* out, int N, int C, int block_size)
{
    // printf("ERROR: Not Implement yet.\nSame as version2&3, implement by cooperative groups.\n");
    int grid_size = N;
    softmax_kernel4<<<grid_size, block_size>>>(in, out, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_online1(floatX* in, floatX* out, int N, int C, int block_size)
{
    // printf("This is softmax online version1, parallel N, loop over C.\n");
    int grid_size = CEIL_DIV(N, block_size);
    softmax_online_kernel1<<<grid_size, block_size>>>(in, out, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_online2(floatX* in, floatX* out, int N, int C, int block_size)
{
    // printf("This is softmax online version2, one warp for one row, so N * 32 num_threads.\n");
    int grid_size = CEIL_DIV(N * 32, block_size);
    softmax_online_kernel2<<<grid_size, block_size>>>(in, out, N, C);
    cudaCheck(cudaGetLastError());
}

void softmax_online3(floatX* in, floatX* out, int N, int C, int block_size)
{
    // printf("This is softmax online version3, same with version2, without cooperative-groups.\n");
    int grid_size = CEIL_DIV(N * 32, block_size);
    softmax_online_kernel3<<<grid_size, block_size>>>(in, out, N, C);
    cudaCheck(cudaGetLastError());
}

//___________________KERNEL DISPATCH_________________//
void softmax(int kernel_id, floatX* in, floatX* out, int N, int C, int block_size)
{
    switch (kernel_id)
    {
    case 1: softmax1(in, out, N, C, block_size); break;
    case 2: softmax2(in, out, N, C, block_size); break;
    case 3: softmax3(in, out, N, C, block_size); break;
    case 4: softmax4(in, out, N, C, block_size); break;
    case 5: softmax_online1(in, out, N, C, block_size); break;
    case 6: softmax_online2(in, out, N, C, block_size); break;
    case 7: softmax_online3(in, out, N, C, block_size); break;
    default: printf("Invalid kernel id: %d\n", kernel_id); break;
    }
}
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
    int N = 1000;
    int C = 70000;
    // host memory
    thrust::host_vector<float> h_in(N * C);
    thrust::host_vector<float> h_out(N * C);
    make_random_float(h_in.data(), N * C);
    make_random_float(h_out.data(), N * C);

    // device memory
    thrust::device_vector<floatX> d_in(N * C);
    thrust::device_vector<floatX> d_out(N * C);
    cudaCheck(type_convert_memcpy(d_in.data().get(), h_in.data(), N * C));
    cudaCheck(type_convert_memcpy(d_out.data().get(), h_out.data(), N * C));

    // cpu
    softmax_cpu(h_in.data(), h_out.data(), N, C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        softmax(kernel_id, d_in.data().get(), d_out.data().get(), N, C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out.data().get(), h_out.data(), "out", N * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    //_______________BENCHMARK___________________//
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(
            repeat_times, softmax, kernel_id, d_in.data().get(), d_out.data().get(), N, C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = N * C * 4 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    return 0;
}