#include <cstdio>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#define ENABLE_BF16
#include "common.h"

void vector_add_cpu(const float* a, const float* b, float* c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief the first kernel parallelize every element
 *
 */
__global__ void vector_add_kernel1(const floatX* a, const floatX* b, floatX* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief thread coarsening and memory access coalescing, every thread process multiple elements
 *
 */
__global__ void vector_add_kernel2(const floatX* a, const floatX* b, floatX* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
    {
        c[i] = a[i] + b[i];
    }
}

/**
 * @brief the second kernel use Packed128 to load/store data (LDG.128 and STS.128 to speed up)
 *        actually, every thread process 128bit/sizeof(floatX) elements
 */
__global__ void vector_add_kernel3(const floatX* a, const floatX* b, floatX* c, int n)
{
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx < n)
    {
        x128 out;
        x128 a128 = load128(a + idx);
        x128 b128 = load128(b + idx);
        for (int k = 0; k < x128::size; k++)
        {
            out[k] = a128[k] + b128[k];
        }
        store128(c + idx, out);
    }
}

//_______________________KERNEL LANCHER_________________________//

void vector_add1(floatX* a, floatX* b, floatX* c, int n, const int block_size)
{
    int num_blocks = (n + block_size - 1) / block_size;
    vector_add_kernel1<<<num_blocks, block_size>>>(a, b, c, n);
    cudaCheck(cudaGetLastError());
}

void vector_add2(floatX* a, floatX* b, floatX* c, int n, const int block_size)
{
    int num_blocks = (n + block_size - 1) / block_size;
    vector_add_kernel2<<<num_blocks, block_size>>>(a, b, c, n);
    cudaCheck(cudaGetLastError());
}

void vector_add3(floatX* a, floatX* b, floatX* c, int n, const int block_size)
{
    int num_blocks = (n + block_size - 1) / block_size;
    vector_add_kernel3<<<num_blocks, block_size>>>(a, b, c, n);
    cudaCheck(cudaGetLastError());
}

//_______________________KERNEL DISPATCH_________________________//
void vector_add(int kernel_id, floatX* a, floatX* b, floatX* c, int n, const int block_size)
{
    switch (kernel_id)
    {
    case 1: vector_add1(a, b, c, n, block_size); break;
    case 2: vector_add2(a, b, c, n, block_size); break;
    case 3: vector_add3(a, b, c, n, block_size); break;
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
    int B = 8;
    int T = 1024;
    int C = 768;
    // host memory
    thrust::host_vector<float> h_a(B * T * C);
    thrust::host_vector<float> h_b(B * T * C);
    thrust::host_vector<float> h_c(B * T * C);
    make_random_float(h_a.data(), B * T * C);
    make_random_float(h_b.data(), B * T * C);
    make_zeros_float(h_c.data(), B * T * C);

    // device memory
    thrust::device_vector<floatX> d_a(B * T * C);
    thrust::device_vector<floatX> d_b(B * T * C);
    thrust::device_vector<floatX> d_c(B * T * C);
    cudaCheck(type_convert_memcpy(d_a.data().get(), h_a.data(), B * T * C));
    cudaCheck(type_convert_memcpy(d_b.data().get(), h_b.data(), B * T * C));

    // cpu
    vector_add_cpu(h_a.data(), h_b.data(), h_c.data(), B * T * C);

    // time the kernel at different block sizes
    int block_sizes[] = {32, 64, 128, 256, 512, 1024};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];
        printf("Checking block size %d.\n", block_size);
        vector_add(kernel_id, d_a.data().get(), d_b.data().get(), d_c.data().get(), B * T * C, block_size);
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-5;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_c.data().get(), h_c.data(), "out", B * T * C, tol);
    }

    printf("All results match. Starting benchmarks.\n\n");

    //_______________BENCHMARK___________________//
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++)
    {
        int block_size = block_sizes[j];

        int repeat_times = 1000;
        float elapsed_time = benchmark_kernel(repeat_times, vector_add, kernel_id, d_a.data().get(), d_b.data().get(),
            d_c.data().get(), B * T * C, block_size);

        // napkin math: estimate the memory bandwidth achieved
        // for each (B,T,C) output element, we do 3 reads and 1 write, 4 bytes each
        // and e.g. A100 40GB PCIe is advertised at 1,555GB/s
        long memory_ops = B * T * C * 4 * 4;
        float memory_bandwidth = memory_ops / elapsed_time / 1e6;

        printf("block_size %4d | time %.4f ms | bandwidth %.2f GB/s\n", block_size, elapsed_time, memory_bandwidth);
    }

    return 0;
}