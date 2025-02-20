#include <cstdio>
#include <cuda.h>
#include <iostream>

#include <iterator>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "common.h"

/**
 * @brief 二维矩阵转置
 *
 */
void transpose_cpu(float* in_ptr, float* out_ptr, int M, int N)
{
    // 只交换下三角区
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int in_add = i * N + j;
            int out_add = j * M + i;
            out_ptr[out_add] = in_ptr[in_add];
        }
    }
}

/**
 * @brief 将MxN的矩阵划分为 TILE_DIM x TILE_DIM的小块，每个小块内分配 TILE_DIM x 8个线程去完成这片区域元素的交换
 */
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8;
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_kernel1(float* in_ptr, float* out_ptr, int M, int N)
{
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        int in_addr = (row + i) * N + col;
        int out_addr = col * M + row + i;
        out_ptr[out_addr] = in_ptr[in_addr];
    }
}

/**
 * @brief
 * 共享内存区域大小为32x32，每个block中共有8x32个线程，32个为一行，同时也是一个warp中，可以更好的合并访存，提高效率。
 *
 */
template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_kernel2(float* in_ptr, float* out_ptr, int M, int N)
{
    __shared__ float shm[TILE_DIM][TILE_DIM];
    // step 1 load to shared memory
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        shm[threadIdx.y + i][threadIdx.x] = in_ptr[(row + i) * N + col];
    }
    __syncthreads();
    // step 2 store to global memory
    col = blockIdx.y * TILE_DIM + threadIdx.x;
    row = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        out_ptr[(row + i) * M + col] = shm[threadIdx.x][threadIdx.y + i];
    }
}

template <int TILE_DIM, int BLOCK_ROWS>
__global__ void transpose_kernel3(float* in_ptr, float* out_ptr, int M, int N)
{
    __shared__ float shm[TILE_DIM][TILE_DIM + 1];
    // step 1 load to shared memory
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        shm[threadIdx.y + i][threadIdx.x] = in_ptr[(row + i) * N + col];
    }
    __syncthreads();
    // step 2 store to global memory
    col = blockIdx.y * TILE_DIM + threadIdx.x;
    row = blockIdx.x * TILE_DIM + threadIdx.y;
    for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        out_ptr[(row + i) * M + col] = shm[threadIdx.x][threadIdx.y + i];
    }
}

void transpose_forward1(float* in_ptr, float* out_ptr, int M, int N)
{
    dim3 thread_per_block(TILE_DIM, BLOCK_ROWS);
    dim3 block_per_grid(N / TILE_DIM, M / TILE_DIM);
    transpose_kernel1<TILE_DIM, BLOCK_ROWS><<<block_per_grid, thread_per_block>>>(in_ptr, out_ptr, M, N);
    cudaCheck(cudaGetLastError());
}

void transpose_forward2(float* in_ptr, float* out_ptr, int M, int N)
{
    dim3 thread_per_block(TILE_DIM, BLOCK_ROWS);
    dim3 block_per_grid(N / TILE_DIM, M / TILE_DIM);
    transpose_kernel2<TILE_DIM, BLOCK_ROWS><<<block_per_grid, thread_per_block>>>(in_ptr, out_ptr, M, N);
    cudaCheck(cudaGetLastError());
}

void transpose_forward3(float* in_ptr, float* out_ptr, int M, int N)
{
    dim3 thread_per_block(TILE_DIM, BLOCK_ROWS);
    dim3 block_per_grid(N / TILE_DIM, M / TILE_DIM);
    transpose_kernel3<TILE_DIM, BLOCK_ROWS><<<block_per_grid, thread_per_block>>>(in_ptr, out_ptr, M, N);
    cudaCheck(cudaGetLastError());
}

//___________________KERNEL DISPATCH_________________//
void transpose(int kernel_id, float* in_ptr, float* out_ptr, int M, int N)
{
    switch (kernel_id)
    {
    case 1: transpose_forward1(in_ptr, out_ptr, M, N); break;
    case 2: transpose_forward2(in_ptr, out_ptr, M, N); break;
    case 3: transpose_forward3(in_ptr, out_ptr, M, N); break;
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
    int M = 1024;
    int N = 1024;
    // host memory
    thrust::host_vector<float> h_in(M * N);
    thrust::host_vector<float> h_out(M * N);

    make_random_float(h_in.data(), M * N);
    make_zeros_float(h_out.data(), M * N);

    thrust::device_vector<float> d_in = h_in;
    thrust::device_vector<float> d_out = h_out;

    // cpu run
    transpose_cpu(h_in.data(), h_out.data(), M, N);

    printf("Finish cpu execution.\n");

    for (int kernel_id = 1; kernel_id <= 3; kernel_id++)
    {
        printf("Cheking kernel %d \n", kernel_id);
        transpose(kernel_id, d_in.data().get(), d_out.data().get(), M, N);
        float tol = 1e-5;
        validate_result(d_out.data().get(), h_out.data(), "out", N * M, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    //_______________BENCHMARK___________________//
    for (int kernel_id = 1; kernel_id <= 3; kernel_id++)
    {

        int repeat_times = 1;
        float elapsed_time
            = benchmark_kernel(repeat_times, transpose, kernel_id, d_in.data().get(), d_out.data().get(), N, M);

        long memory_ops = M * N * 2 * sizeof(float);
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("kernel id %d | time %.4f ms | bandwidth %.2f GB/s\n", kernel_id, elapsed_time, memory_bandwidth);
    }
}