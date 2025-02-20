#include <cassert>
#include <cstdio>
#include <stdio.h>
#include <vector>
#include <algorithm>

#include "common.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/**
 * @brief cpu实现的，只是为了验证正确性
 *        input : [N, C]
 *        output: [N, K]
 */
void top_k_cpu(float* in_ptr, int N, int C, int K, float* out_ptr)
{
    assert(C >= K);
    for (int i = 0; i < N; i++)
    {
        std::vector<float> vec(in_ptr + i * C, in_ptr + i * C + C);
        std::sort(vec.begin(), vec.end());
        for (int j = C - 1; j >= C - K; j--)
        {
            out_ptr[i * K + C - 1 - j] = vec[j];
            // std::cout << out_ptr[C - 1 - j] << ' ';
        }
        // std::cout << std::endl;
    }
}

/**
 * 线程内部计算top-k，存储在共享内存中，升序排列
 */
__device__ __forceinline__ void insert_k(float* shm, float value, int K)
{
    // printf("before merge %.5lf, %.5lf, %.5lf \n", shm[0], shm[0 + 1], shm[0 + 2]);
    int pos = K;
    /*
    插入 len = K [x1, x2, x3, ... , xk]   x1 >= x2 >= x3 >= ... >= xk
    */
    // step1 find the first bigger position
    for (int i = 0; i < K; i++)
    {
        if (value > shm[i])
        {
            pos = i;
            break;
        }
    }
    // printf("insert pos = %d, origin value = %.5lf, new value is %.5lf.\n", pos, shm[pos], value);
    if (pos != K)
    {
        for (int i = K - 1; i > pos; i--)
        {
            shm[i] = shm[i - 1];
        }
        shm[pos] = value;
    }
    // printf("after merge %.5lf, %.5lf, %.5lf \n", shm[0], shm[0 + 1], shm[0 + 2]);
}

__device__ __forceinline__ void merge_two_k(float* shm, int first_pos, int second_pos, int K)
{
    // 还没想到原地的算法，只能笨蛋的一个一个往里插
    // 把第二段的每一个值往第一段中插
    // printf("merge two index : %d   %d  \n", first_pos, second_pos);
    // printf("before merge %.5lf, %.5lf, %.5lf,  %.5lf, %.5lf, %.5lf \n", shm[first_pos], shm[first_pos + 1],
    //     shm[first_pos + 2], shm[second_pos], shm[second_pos + 1], shm[second_pos + 2]);
    for (int i = 0; i < K; i++)
    {
        // printf("insert shm idx = %d, value = %.5lf.\n", second_pos + i, shm[second_pos + i]);
        insert_k(shm + first_pos, shm[second_pos + i], K);
    }
    // printf("after merge %.5lf, %.5lf, %.5lf \n", shm[first_pos], shm[first_pos + 1], shm[first_pos + 2]);
}

__global__ void top_k_kernel1(float* in_ptr, int N, int C, int K, float* out_ptr)
{
    extern __shared__ float shm[]; // size is K * blockDim.x
    // 每个线程处理的元素数量为 C / block_size
    int row = blockIdx.x;
    int block_size = blockDim.x;
    int tx = threadIdx.x;
    int shm_ptr = tx * K;

    for (int i = 0; i < K; i++)
    {
        shm[shm_ptr + i] = 0.0;
    }
    for (int i = tx; i < C; i += block_size)
    {
        int glm_addr = row * C + i;
        // printf("before merge %.5lf, %.5lf, %.5lf \n", shm[shm_ptr + 0], shm[shm_ptr + 1], shm[shm_ptr + 2]);
        insert_k(shm + shm_ptr, in_ptr[glm_addr], K);
    }
    __syncthreads(); // 每个线程都处理完了负责的元素
    // block level 两两合并
    for (int stride = block_size / 2; stride >= 1; stride /= 2)
    {
        if (tx < stride)
        {
            // merge shm[tx] , tx[tx + stride]
            int first_pos = tx * K;
            int second_pos = (tx + stride) * K;
            // printf(" id [%d  -- %d]\n", tx, tx + stride);
            // printf(
            //     "merge : [%d - %d]  ---- [%d - %d].\n", tx * K, (tx + 1) * K, (tx + stride) * K, (tx + stride + 1) *
            //     K);
            merge_two_k(shm, first_pos, second_pos, K);
        }
        __syncthreads();
    }
    if (tx == 0)
    {
        for (int i = 0; i < K; i++)
        {
            // printf(" write back %d  %.5lf\n.", row * K + i, shm[i]);
            out_ptr[row * K + i] = shm[i];
        }
    }
}
//_______________________KERNEL LANCHER_________________________//
/**
 * @brief 本机4060 max_shared_memory size = 48 KB
 *        设计划开启的线程数为 thread_nums, 则block内的所有数据被分为thread_nums段，每段计算部分top-K
 *        则 thread_nums * 4 * K / 1024 <= 48
 */
void top_k_forward1(float* in_ptr, int N, int C, int K, float* out_ptr, int block_size)
{
    // block_size是一个block内的线程数
    assert(block_size * K * 4 < 48 * 1024);
    dim3 block_per_grid = N;
    dim3 thread_per_block = block_size;
    size_t shm_size = block_size * K * sizeof(float);
    top_k_kernel1<<<block_per_grid, thread_per_block, shm_size>>>(in_ptr, N, C, K, out_ptr);
    cudaCheck(cudaGetLastError());
}

//_______________________KERNEL DISPATCH_________________________//
void top_k(int kernel_id, float* in_ptr, int N, int C, int K, float* out_ptr, int block_size)
{
    switch (kernel_id)
    {
    case 1: top_k_forward1(in_ptr, N, C, K, out_ptr, block_size); break;
    default: printf("No kernel %d. \n", kernel_id);
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
    int C = 1024;
    int N = 1024;
    int K = 20;
    // host memory
    thrust::host_vector<float> h_in(N * C);
    thrust::host_vector<float> h_out(N * K);
    make_positive_random_float(h_in.data(), N * C);
    make_zeros_float(h_out.data(), N * K);

    // device memory
    thrust::device_vector<floatX> d_in = h_in;
    thrust::device_vector<floatX> d_out(N * K);
    // cpu
    top_k_cpu(h_in.data(), N, C, K, h_out.data());

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        // thrust::fill(d_out.begin(), d_out.end(), 0);

        top_k(kernel_id, d_in.data().get(), N, C, K, d_out.data().get(), block_size);
        // cudaDeviceSynchronize();
        // cudaCheck(cudaGetLastError());
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_out.data().get(), h_out.data(), "out", N * K, tol);
    }
    printf("All results match. Starting benchmarks.\n\n");

    //_______________BENCHMARK___________________//
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(
            repeat_times, top_k, kernel_id, d_in.data().get(), N, C, K, d_out.data().get(), block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = N * C * 4 * 2;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    return 0;
}