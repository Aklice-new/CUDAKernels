#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cublas_v2.h>
#include <iostream>

int main()
{
    int M = 2, K = 3, N = 2;

    // 行主序矩阵 A (MxK)
    thrust::host_vector<float> h_A(6);

    // 行主序矩阵 B (KxN)
    thrust::host_vector<float> h_B(6);

    for (int i = 0; i < 6; i++)
    {
        h_A[i] = i + 1;
        h_B[i] = i + 1;
    }

    // 结果矩阵 C (MxN)
    thrust::host_vector<float> h_C_blas(M * N);

    // 设备内存
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C_blas(M * N);

    // 创建 cuBLAS 句柄
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS handle initialization failed!" << std::endl;
        return -1;
    }

    // 标量系数
    float alpha = 1.0f;
    float beta = 0.0f;

    // 行主序矩阵乘法：C = A * B
    // 由于 cuBLAS 是列主序，需要对 A 和 B 进行转置
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, // A 和 B 需要转置
        N, M, K,                                           // 注意维度顺序
        &alpha, d_B.data().get(), N,                       // B^T 的 leading dimension 是 N
        d_A.data().get(), K,                               // A^T 的 leading dimension 是 K
        &beta, d_C_blas.data().get(), N                    // C^T 的 leading dimension 是 N
    );
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Sgemm failed with error: " << status << std::endl;
        return -1;
    }

    // 将结果拷贝回主机
    thrust::copy(d_C_blas.begin(), d_C_blas.end(), h_C_blas.begin());

    // 打印结果
    std::cout << "Result matrix C (MxN):" << std::endl;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            std::cout << h_C_blas[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // 释放资源
    cublasDestroy(handle);

    return 0;
}