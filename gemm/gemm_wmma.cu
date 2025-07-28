#include <cstdio>
#include <stdio.h>
#include <mma.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"

using namespace nvcuda;
void gemm_cpu(float* A, float* B, float* C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < K; k++)
            {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

#define WARP_SIZE 32
#define TILE_K 8

__global__ void gemm_wmma_kernel1(float* A, float* B, float* C, int M, int N, int K)
{

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int warp_id = tid >> 5;
    int lane_id = tid & 0x1f;
    // this version we dont use shared memory, load memory from global memory directlly
    wmma::fragment<wmma::accumulator, 16, 16, 8, float> frag_c;
    wmma::fill_fragment(frag_c, 0.0f);

    wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_a;
    wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> frag_b;

    int k_tiles = K / TILE_K;

    for (int k = 0; k < K; k += TILE_K)
    {

        wmma::load_matrix_sync(frag_a, A + (by * 16 * K) + k, K);
        wmma::load_matrix_sync(frag_b, B + (k * N + bx * 16), N);

        for (int t = 0; t < frag_a.num_elements; t++)
        {
            frag_a.x[t] = wmma::__float_to_tf32(frag_a.x[t]);
        }

        for (int t = 0; t < frag_b.num_elements; t++)
        {
            frag_b.x[t] = wmma::__float_to_tf32(frag_b.x[t]);
        }

        wmma::mma_sync(frag_c, frag_a, frag_b, frag_c);
    }

    wmma::store_matrix_sync(C + (by * 16 * N + bx * 16), frag_c, N, wmma::mem_row_major);
}

void gemm_wmma_forward1(float* A, float* B, float* C, int M, int N, int K)
{
    dim3 threads(WARP_SIZE);
    dim3 blocks(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
    gemm_wmma_kernel1<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaCheck(cudaGetLastError());
}

void gemm(int kernel_id, int M, int N, int K, float* A, float* B, float* C)
{
    switch (kernel_id)
    {
    case 1: gemm_wmma_forward1(A, B, C, M, N, K); break;
    // case 2: gemm_forward2(A, B, C, M, N, K); break;
    // case 3: gemm_forward3(A, B, C, M, N, K); break;
    // case 4: gemm_forward4(A, B, C, M, N, K); break;
    // case 5: gemm_forward5(A, B, C, M, N, K); break;
    // case 6: gemm_forward6(A, B, C, M, N, K); break;
    default: printf("No kernel id %d \n", kernel_id); break;
    }
}

int main(int argc, char* argv[])
{
    // input : M, N, K, default values are 1024, 1024, 1024
    // int M = 40000;
    // int N = 1000;
    // int K = 64;
    int M = 4096;
    int N = 4096;
    int K = 1024;
    if (argc > 1)
    {
        if (argc == 4)
        {
            M = atoi(argv[1]);
            N = atoi(argv[2]);
            K = atoi(argv[3]);
        }
        else
        {
            printf("Usage: %s M N K\n", argv[0]);
            return 1;
        }
    }

    // allocate memory
    thrust::host_vector<float> h_A(M * K);
    thrust::host_vector<float> h_B(K * N);
    thrust::host_vector<float> h_C_blas(M * N);
    thrust::host_vector<float> h_C_out(M * N);

    // initialize A and B
    make_random_float(h_A.data(), M * K);
    make_random_float(h_B.data(), K * N);
    make_zeros_float(h_C_blas.data(), M * N);
    make_zeros_float(h_C_out.data(), M * N);

    // transfer to gpu
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C_blas = h_C_blas;
    thrust::device_vector<float> d_C_out = h_C_out;

    // cublas
    cublasHandle_t handle;
    cublasCreate(&handle);
    float alpha = 1.0f;
    float beta = 0.0f;
    // cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, N, M, K, &alpha, d_B.data().get(), N, d_A.data().get(), K, &beta,
    //     d_C_blas.data().get(), N);
    auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // A 和 B 需要转置
        N, M, K,                                                // 注意维度顺序
        &alpha, d_B.data().get(), N,                            // B^T 的 leading dimension 是 N
        d_A.data().get(), K,                                    // A^T 的 leading dimension 是 K
        &beta, d_C_blas.data().get(), N                         // C^T 的 leading dimension 是 N
    );
    // auto status = cublasSgemm(handle, CUBLAS_OP_T,      //
    //     CUBLAS_OP_T, M, N, K, &alpha, d_A.data().get(), //
    //     K, d_B.data().get(), N, &beta, d_C_blas.data().get(), N);
    // cublasCheck(status);
    h_C_blas = d_C_blas;
    // gemm_cpu(h_A.data(), h_B.data(), h_C_blas.data(), M, N, K);
    // verify the result
    for (int kernel_id = 1; kernel_id <= 1; kernel_id++)
    {
        printf("Validating kernel %d\n", kernel_id);
        gemm(kernel_id, M, N, K, d_A.data().get(), d_B.data().get(), d_C_out.data().get());
        cudaCheck(cudaDeviceSynchronize());
#if !defined(ENABLE_BF16) && !defined(ENABLE_FP16)
        float tol = 1e-3;
#else
        float tol = 1e-2f;
#endif
        validate_result(d_C_out.data().get(), h_C_blas.data(), "gemm_result", M * N, tol);
    }
    printf("All results match. Starting benchmarks.\n\n");
    //_______________BENCHMARK___________________//

    // benchmark cublas
    float elapsed_time = benchmark_cublas(1, handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B.data().get(), N,
        d_A.data().get(), K, &beta, d_C_blas.data().get(), N);
    long long memory_ops = 2ll * M * N * K * sizeof(float);
    float memory_bandwidth_cublas = memory_ops / elapsed_time / 1e9;

    printf("| cublas time %.4f ms | cublas bandwidth %.2f GB/s\n", elapsed_time, memory_bandwidth_cublas);
    for (int kernel_id = 2; kernel_id <= 6; kernel_id++)
    {
        int repeat_times = 1;
        float elapsed_time = benchmark_kernel(
            repeat_times, gemm, kernel_id, M, N, K, d_A.data().get(), d_B.data().get(), d_C_out.data().get());

        float memory_bandwidth = memory_ops / elapsed_time / 1e9;
        printf("|  Kernel id: %d  | time %.4f ms | bandwidth %.2f GB/s  ratios = %.7f \n", kernel_id, elapsed_time,
            memory_bandwidth, memory_bandwidth / memory_bandwidth_cublas);
    }

    return 0;
}
