#include <cstdio>
#include <stdio.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "common.h"

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

__global__ void gemm_kernel1(float* A, float* B, float* C, int M, int N, int K)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y; // 计算C的行索引
    int j = blockIdx.x * blockDim.x + threadIdx.x; // 计算C的列索引
    if (i < M && j < N)
    {
        float sum = 0;
        for (int k = 0; k < K; k++)
        {
            sum += A[i * K + k] * B[k * N + j]; // 计算C[i][j]的值
        }
        C[i * N + j] = sum;
    }
}

/**
    Feature : Block Level Optimize Shared Memory, Sliced K

    Block : <BM, BN>, BM = 32, BN = 32
    TILE_K : K / BK,  BK = 32
    Grid  : <CEIL(M / BM), CEIL(N / BN)>
    A : M * K
    B : K * N
    Shared memory : As[BM][BK], Bs[BK][BN]
    每个线程块32x32负责计算C的一个32x32的子块,在Sliced K时，每次每个线程负责一个元素的加载。
 */
template <int BM, int BN, int BK>
__global__ void gemm_kernel2(float* A, float* B, float* C, int M, int N, int K)
{
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = blockDim.x * ty + tx;

    int load_a_smem_m = tid / BK;
    int load_a_smem_k = tid % BK;
    int load_b_smem_k = tid / BN;
    int load_b_smem_n = tid % BN;

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // if (load_a_gmem_m >= M || load_b_gmem_n >= N)
    // {
    //     return;
    // }

    float sum = 0;
    for (int tile = 0; tile < K; tile += BK)
    {
        As[load_a_smem_m][load_a_smem_k] = A[load_a_gmem_m * K + tile + load_a_smem_k];
        Bs[load_b_smem_k][load_b_smem_n] = B[(tile + load_b_smem_k) * N + load_b_gmem_n];
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk++)
        {
            sum += As[tx][kk] * Bs[kk][ty];
        }
        __syncthreads();
    }
    int store_c_gmem_m = by * BM + tx;
    int store_c_gmem_n = bx * BN + ty;
    int store_c_gemm_addr = store_c_gmem_m * N + store_c_gmem_n;
    C[store_c_gemm_addr] = sum;
}
/**
    Feature : (Based on kernel2) Thread Tile, Thread Coarsening, Vectorization
    Thread : TM = 8, TN = 8
    Block : <BM / TM, BN / TN>, BM = 128, BN = 128
    TILE_K : K / BK, BK = 8
    Grid  : <CEIL(M / (BM / TM)), CEIL(N / (BN / TN))>
    A : M * K   B : K * N   C : M * N
    Shared memory : As[BM][BK], Bs[BK][BN]
    每个线程块16x16负责计算C的一个128x128的子块
    加载阶段：
    对于A矩阵来说，16x16的线程块，需要加载128x8的数据，每个线程负责加载4个元素
    FLOAT4就会通过LDG128来访问数据，提高访问带宽
    计算阶段：
    每个线程负责一个8x8的子块的数据计算任务
    通过将A的数据按照BKxBM的大小进行转置，提高了L1缓存的命中率。
 */
// 定义FLOAT4宏，将一个地址的数据转换为float4类型，用FLOAT4就会通过LDG128来访问数据，提高访问带宽
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_kernel3(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int N, int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    // 这里划分的是逻辑的位置，一个线程块是16x16
    // 每个线程加载A的子矩阵subA 128x8 的8个元素，B的子矩阵 subB 8x128的8个元素
    // 现在就是要将这16x16个线程分配到这些加载数据的任务上
    int load_smem_a_m = tid / (BK / 4); //  BK / TM = 2, 加载一行需要两个线程
    int load_smem_a_k = (tid % (BK / 4)) * 4;
    int load_smem_b_k = tid / (BN / 4);
    int load_smem_b_n = (tid % (BN / 4)) * 4;

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;

    // WARNING: 这个无用的判断会导致性能下降多达 10%
    // if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    // {
    //     return;
    // }
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    float r_c[TM][TN] = {0}; // 当前线程负责计算的8x8的结果，通过寄存器存储
    for (int tile = 0; tile < K; tile += BK)
    {
        // 加载A和B的数据到共享内存中
        FLOAT4(As[load_smem_a_m][load_smem_a_k]) = FLOAT4(A[load_gmem_a_m * K + tile + load_smem_a_k]);
        FLOAT4(Bs[load_smem_b_k][load_smem_b_n]) = FLOAT4(B[(tile + load_smem_b_k) * N + load_gmem_b_n]);
        __syncthreads();
        // 计算C的一个8x8的子块
        for (int k = 0; k < BK; k++)
        {
            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    int smem_a_m = ty * TM + m;
                    int smem_b_n = tx * TN + n;
                    r_c[m][n] += As[smem_a_m][k] * Bs[k][smem_b_n];
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int m = 0; m < TM; m++)
    {
        for (int n = 0; n < TN; n += 4) // FLOAT4
        {
            int gmem_c_m = by * BM + ty * TM + m;
            int gmem_c_n = bx * BN + tx * TN + n;
            int gemm_c_addr = gmem_c_m * N + gmem_c_n;
            FLOAT4(C[gemm_c_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}
/**
 * Feature : Based on kernel3, Double buffer(by register), Transpose A

 * Thread : TM = 8, TN = 8
 * Block : <BM / TM, BN / TN>, BM = 128, BN = 128
 * TILE_K : K / BK, BK = 8
 * Grid  : <CEIL(M / BM), CEIL(N / BN)>
 * A : M * K   B : K * N   C : M * N
 * Shared memory : As[BK][BM], Bs[BK][BN]
 * 每个线程块16x16负责计算C的一个128x128的子块
 Kernel3中的实现，对于每个TILED_K，首先加载A和B的数据到共享内存中，然后计算C的一个8x8的子块。
 而这两个阶段是可以重叠的，我们可以通过流水优化来使得Copy和Compute阶段overlap。
 具体的实现就是，对于当前计算的TILED_K，我们提前加载它需要的数据，然后计算上一个TILED_K的结果。
 */
template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_kernel4(float* __restrict__ a, float* __restrict__ b, float* __restrict__ c, int M, int N, int K)
{
    //     int tx = threadIdx.x;
    //     int ty = threadIdx.y;
    //     int bx = blockIdx.x;
    //     int by = blockIdx.y;
    //     int tid = ty * blockDim.x + tx;

    //     int load_smem_a_m = tid / (BK / 4); //  BK / TM = 2, 加载一行需要两个线程
    //     int load_smem_a_k = (tid % (BK / 4)) * 4;
    //     int load_smem_b_k = tid / (BN / 4);
    //     int load_smem_b_n = (tid % (BN / 4)) * 4;

    //     int load_gmem_a_m = by * BM + load_smem_a_m;
    //     int load_gmem_b_n = bx * BN + load_smem_b_n;
    //     // if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    //     // {
    //     //     return;
    //     // }
    //     __shared__ float As[2][BK][BM];
    //     __shared__ float Bs[2][BK][BN]; // 2 times space for double buffer

    //     float frag_a[2][TM], frag_b[2][TN];
    //     // float r_load_a[4];
    //     // float r_load_b[4];
    //     // float r_compute_a[TM];
    //     // float r_compute_b[TN];
    //     float r_c[TM][TN] = {0};
    //     // copy data for K = 0, copy data to buffer 0
    //     {
    //         int load_gmem_a_addr = load_gmem_a_m * K + load_smem_a_k;
    //         int load_gmem_b_addr = load_smem_b_k * N + load_gmem_b_n;
    //         FLOAT4(Bs[0][load_smem_b_k][load_smem_b_n]) = FLOAT4(B[load_gmem_b_addr]);
    // #pragma unroll
    //         for (int i = 0; i < 4; i++)
    //         {
    //             As[0][load_smem_a_k + i][load_smem_a_m] = A[load_gmem_a_addr + i];
    //         }
    //     }
    //     __syncthreads();

    //     // 对于tile_i，当前循环进行的是tile_i-1的计算，和tile_i的数据加载
    //     for (int tile = 1; tile < K / BK; tile++)
    //     {
    //         // stage的编号： 0:0, 1:1, 2:0, 3:1, 4:0, 5:1 ... K/BK - 1:?
    //         // copy data for K = tile
    //         int now_stage = (tile - 1) & 1;
    //         int nxt_stage = now_stage ^ 1;
    //         // int now_stage = 0;
    //         // int nxt_stage = 0;
    //         int load_gmem_a_addr = load_gmem_a_m * K + tile * BK + load_smem_a_k;
    //         int load_gmem_b_addr = (tile * BK + load_smem_b_k) * N + load_gmem_b_n;
    //         FLOAT4(Bs[nxt_stage][load_smem_b_k][load_smem_b_n]) = FLOAT4(B[load_gmem_b_addr]);
    // #pragma unroll
    //         for (int i = 0; i < 4; i++)
    //         {
    //             As[nxt_stage][load_smem_a_k + i][load_smem_a_m] = A[load_gmem_a_addr + i];
    //         }
    //         __syncthreads();
    //         // compute for K = tile - 1
    //         for (int k = 0; k < BK; k++)
    //         {
    //             for (int i = 0; i < TM / 4; i++)
    //             {
    //                 FLOAT4(frag_a[now_stage][i * 4]) = FLOAT4(As[now_stage][k][ty * TM + i * 4]);
    //             }
    //             for (int i = 0; i < TN / 4; i++)
    //             {
    //                 FLOAT4(frag_b[now_stage][i * 4]) = FLOAT4(Bs[now_stage][k][tx * TN + i * 4]);
    //             }
    //             // A load
    //             FLOAT4(frag_a[now_stage][0]) = FLOAT4(As[now_stage][k][ty * TM]);
    //             FLOAT4(frag_a[now_stage][4]) = FLOAT4(As[now_stage][k][ty * TM + 4]);
    //             // B load
    //             FLOAT4(frag_b[now_stage][0]) = FLOAT4(Bs[now_stage][k][tx * TN]);
    //             FLOAT4(frag_b[0][4]) = FLOAT4(Bs[0][k][tx * TN + 4]);

    //             for (int m = 0; m < TM; m++)
    //             {
    //                 for (int n = 0; n < TN; n++)
    //                 {
    //                     r_c[m][n] += frag_a[0][m] * frag_b[0][n];
    //                 }
    //             }
    //         }
    //         __syncthreads();
    //     }

    //     // for last tile

    //     {
    //         for (int k = 0; k < BK; k++)
    //         {
    //             // for (int i = 0; i < TM / 4; i++)
    //             // {
    //             //     FLOAT4(frag_a[stage][i * 4]) = FLOAT4(As[stage][k][ty * TM + i * 4]);
    //             // }
    //             // for (int i = 0; i < TN / 4; i++)
    //             // {
    //             //     FLOAT4(frag_b[stage][i * 4]) = FLOAT4(Bs[stage][k][tx * TN + i * 4]);
    //             // }
    //             // A load
    //             FLOAT4(frag_a[1][0]) = FLOAT4(As[1][k][ty * TM]);
    //             FLOAT4(frag_a[1][4]) = FLOAT4(As[1][k][ty * TM + 4]);
    //             // B load
    //             FLOAT4(frag_b[1][0]) = FLOAT4(Bs[1][k][tx * TN]);
    //             FLOAT4(frag_b[1][4]) = FLOAT4(Bs[1][k][tx * TN + 4]);
    //             for (int m = 0; m < TM; m++)
    //             {
    //                 for (int n = 0; n < TN; n++)
    //                 {
    //                     r_c[m][n] += frag_a[1][m] * frag_b[1][n];
    //                 }
    //             }
    //         }
    //     }
    //     // store result to C
    // #pragma unroll
    //     for (int m = 0; m < TM; m++)
    //     {
    //         for (int n = 0; n < TN; n += 4)
    //         {
    //             int gmem_c_m = by * BM + ty * TM + m;
    //             int gmem_c_n = bx * BN + tx * TN + n;
    //             int gemm_c_addr = gmem_c_m * N + gmem_c_n;
    //             FLOAT4(C[gemm_c_addr]) = FLOAT4(r_c[m][n]);
    //         }
    //     }
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;

    __shared__ float s_a[2][BK][BM];
    __shared__ float s_b[2][BK][BN];

    float r_load_a[TM / 2];
    float r_load_b[TN / 2];
    float r_comp_a[TM];
    float r_comp_b[TN];
    float r_c[TM][TN] = {0.0};

    // mapping tid to s_a[BK][BM], for each orginal m-th row, load 4 + 4 K-dim
    // row major values from A matrix, and store it in COL major s_a[BK][BM].
    int load_a_smem_m = tid / 2; // tid / 2，(0,1,2,...,128)
    // (0b00000000 & 0b00000001) << 2 = 0
    // (0b00000001 & 0b00000001) << 2 = 4
    // (0b00000010 & 0b00000001) << 2 = 0
    // (0b00000011 & 0b00000001) << 2 = 4
    int load_a_smem_k = (tid & 1) << 2; // (0,4)
    // mapping tid to s_b[BK][BN], for each orginal k-th row, load 4 + 4 N-dim
    // row major values from B matrix, and store it in ROW major s_b[BK][BN].
    int load_b_smem_k = tid / 32; // 0~8
    // (0b00000000 & 0b00011111) << 2 = 0
    // (0b00000001 & 0b00011111) << 2 = 4
    // (0b00000010 & 0b00011111) << 2 = 8
    // (0b00000011 & 0b00011111) << 2 = 12
    int load_b_smem_n = (tid & 31) << 2; // (0,4,8,12,...,124)

    int load_a_gmem_m = by * BM + load_a_smem_m;
    int load_b_gmem_n = bx * BN + load_b_smem_n;

    // 1）主循环从bk = 1 开始，第一次数据加载在主循环之前，最后一次计算在主循环之后，这是pipeline 的特点决定的；
    // 2）由于计算和下一次访存使用的Shared Memory不同，因此主循环中每次循环只需要一次__syncthreads()即可
    // 3）由于GPU不能向CPU那样支持乱序执行，主循环中需要先将下一次循环计算需要的Gloabal Memory中的数据load
    // 到寄存器，然后进行本次计算，之后再将load到寄存器中的数据写到Shared Memory，这样在LDG指令向Global
    // Memory做load时，不会影响后续FFMA及其它运算指令的 launch 执行，也就达到了Double Buffering的目的。

    // bk = 0 is loading here, buffer 0

    {
        int load_a_gmem_k = load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

        s_a[0][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[0][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[0][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[0][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[0][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);
    }
    // Without this synchronization, accuracy may occasionally be abnormal.
    __syncthreads();

    // bk start from 1，需要注意的是，虽然 bk 从 1 开始，但实际上 bk=1时，使用的是
    // 第0块BK中的数据（已经加载到共享内存s_a[0]和s_b[0]）；bk=2时，实际计算的是第1块
    // BK中的数据。其余以此类推，这个循环结束后，剩下最后一块BK大小的数据需要计算。
    for (int bk = 1; bk < (K + BK - 1) / BK; bk++)
    {

        int smem_sel = (bk - 1) & 1;
        int smem_sel_next = bk & 1;

        int load_a_gmem_k = bk * BK + load_a_smem_k;
        int load_a_gmem_addr = load_a_gmem_m * K + load_a_gmem_k;
        int load_b_gmem_k = bk * BK + load_b_smem_k;
        int load_b_gmem_addr = load_b_gmem_k * N + load_b_gmem_n;
        FLOAT4(r_load_a[0]) = FLOAT4(a[load_a_gmem_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(b[load_b_gmem_addr]);

#pragma unroll
        for (int tk = 0; tk < BK; tk++)
        {
            FLOAT4(r_comp_a[0]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2]);
            FLOAT4(r_comp_a[4]) = FLOAT4(s_a[smem_sel][tk][ty * TM / 2 + BM / 2]);
            FLOAT4(r_comp_b[0]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2]);
            FLOAT4(r_comp_b[4]) = FLOAT4(s_b[smem_sel][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
            for (int tm = 0; tm < TM; tm++)
            {
#pragma unroll
                for (int tn = 0; tn < TN; tn++)
                {
                    // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                    r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
                }
            }
        }

        // 对比非double buffers版本，此处不需要__syncthreads()，总共节省了
        // ((K + BK - 1) / BK) - 1 次block内的同步操作。比如，bk=1时，HFMA计算
        // 使用的是s_a[0]和s_b[0]，因此，和s_a[1]和s_b[1]的加载是没有依赖关系的。
        // 从global内存到s_a[1]和s_b[1]和HFMA计算可以并行。s_a[1]和s_b[1]用于
        // 加载下一块BK需要的数据到共享内存。
        s_a[smem_sel_next][load_a_smem_k + 0][load_a_smem_m] = r_load_a[0];
        s_a[smem_sel_next][load_a_smem_k + 1][load_a_smem_m] = r_load_a[1];
        s_a[smem_sel_next][load_a_smem_k + 2][load_a_smem_m] = r_load_a[2];
        s_a[smem_sel_next][load_a_smem_k + 3][load_a_smem_m] = r_load_a[3];
        FLOAT4(s_b[smem_sel_next][load_b_smem_k][load_b_smem_n]) = FLOAT4(r_load_b[0]);

        __syncthreads();
    }

// 计算剩下最后一块BK
#pragma unroll
    for (int tk = 0; tk < BK; tk++)
    {
        FLOAT4(r_comp_a[0]) = FLOAT4(s_a[1][tk][ty * TM / 2]);
        FLOAT4(r_comp_a[4]) = FLOAT4(s_a[1][tk][ty * TM / 2 + BM / 2]);
        FLOAT4(r_comp_b[0]) = FLOAT4(s_b[1][tk][tx * TN / 2]);
        FLOAT4(r_comp_b[4]) = FLOAT4(s_b[1][tk][tx * TN / 2 + BN / 2]);

#pragma unroll
        for (int tm = 0; tm < TM; tm++)
        {
#pragma unroll
            for (int tn = 0; tn < TN; tn++)
            {
                // r_c[tm][tn] += r_comp_a[tm] * r_comp_b[tn];
                r_c[tm][tn] = __fmaf_rn(r_comp_a[tm], r_comp_b[tn], r_c[tm][tn]);
            }
        }
    }

#pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i][4]);
    }
#pragma unroll
    for (int i = 0; i < TM / 2; i++)
    {
        int store_c_gmem_m = by * BM + BM / 2 + ty * TM / 2 + i;
        int store_c_gmem_n = bx * BN + tx * TN / 2;
        int store_c_gmem_addr = store_c_gmem_m * N + store_c_gmem_n;
        FLOAT4(c[store_c_gmem_addr]) = FLOAT4(r_c[i + TM / 2][0]);
        FLOAT4(c[store_c_gmem_addr + BN / 2]) = FLOAT4(r_c[i + TM / 2][4]);
    }
}

/**
 * @brief 2d tiling
 * 通过让每条线程负责更多的数据的计算，来让计算掩盖访存时候的延迟
 */
template <const int BLOCK_SIZE_M, //
    const int BLOCK_SIZE_K,       //
    const int BLOCK_SIZE_N,       //
    const int THREAD_SIZE_M,      // 对 BM * BK 中的数据进行划分
    const int THREAD_SIZE_N>      // 对 BK * BN 中的数据进行划分
// 此时每个线程负责 TILE_SIZE_M个元素的计算，那么block中的线程就可以减少
#define OFFSET(row, col, ld) ((row) * (ld) + (col))
__global__ void gemm_kernel5(float* __restrict__ A, //
    float* __restrict__ B,                          //
    float* __restrict__ C,
    const int M, // Matrix A : M * K
    const int K, //
    const int N)
{ // Matrix B : K * N
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int row = by * BLOCK_SIZE_M;
    int col = BLOCK_SIZE_N * bx;

    float res[THREAD_SIZE_M][THREAD_SIZE_N] = {0};
    __shared__ float A_s[BLOCK_SIZE_M][BLOCK_SIZE_K]; // 128 * 8 * 4B(float) = 4KB
    __shared__ float B_s[BLOCK_SIZE_K][BLOCK_SIZE_N];
    int num_blocks = CEIL_DIV(K, BLOCK_SIZE_K);
    int THREAD_NUMS = THREAD_SIZE_M * THREAD_SIZE_N;

    int tid = ty * blockDim.x + tx;

    // 除4是因为计算出每个线程至少一次性load4个元素
    const int A_TILE_THREAD_PER_ROW = BLOCK_SIZE_K / 4;
    const int B_TILE_THREAD_PER_ROW = BLOCK_SIZE_N / 4;

    int a_row_start = tid / A_TILE_THREAD_PER_ROW;
    int b_row_start = tid / B_TILE_THREAD_PER_ROW;
    int a_col = (tid % A_TILE_THREAD_PER_ROW) * 4;
    int b_col = (tid % B_TILE_THREAD_PER_ROW) * 4;

    //   int a_stride = 0;
    //   int b_stride = 0;

    A = &A[OFFSET(by * BLOCK_SIZE_M, 0, K)];
    B = &B[OFFSET(0, bx * BLOCK_SIZE_N, N)];
    // 大循环
    for (int i = 0; i < num_blocks; i++)
    {
        // load data from global memory to A_s and B_s
        // 每个线程负责将一部分数据从global memory中加载到shared memory中

        FLOAT4(A_s[a_row_start][a_col]) = FLOAT4(A[OFFSET(a_row_start, i * BLOCK_SIZE_K + a_col, K)]);
        FLOAT4(B_s[b_row_start][b_col]) = FLOAT4(B[OFFSET(i * BLOCK_SIZE_K + b_row_start, b_col, N)]);

        __syncthreads();

        // calculate rm * rn size res
        // 小循环
        for (int k = 0; k < BLOCK_SIZE_K; k++)
        {
            for (int m = 0; m < THREAD_SIZE_M; m++)
            {
                for (int n = 0; n < THREAD_SIZE_N; n++)
                {
                    res[m][n] += A_s[ty * THREAD_SIZE_M + m][k] * B_s[k][tx * THREAD_SIZE_N + n];
                }
            }
        }
        __syncthreads();
    }
    // store to
    for (int m = 0; m < THREAD_SIZE_M; m++)
    {
        for (int n = 0; n < THREAD_SIZE_N; n++)
        {
            if (row + ty * THREAD_SIZE_M + m < M && col + tx * THREAD_SIZE_N + n < N)
            {
                C[OFFSET(row + ty * THREAD_SIZE_M + m, col + tx * THREAD_SIZE_N + n, N)] = res[m][n];
            }
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_kernel6(float* __restrict__ A, float* __restrict__ B, float* __restrict__ C, int M, int N, int K)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tid = ty * blockDim.x + tx;

    int load_smem_a_m = tid / (BK / 4); //  BK / TM = 2, 加载一行需要两个线程
    int load_smem_a_k = (tid % (BK / 4)) * 4;
    int load_smem_b_k = tid / (BN / 4);
    int load_smem_b_n = (tid % (BN / 4)) * 4;

    int load_gmem_a_m = by * BM + load_smem_a_m;
    int load_gmem_b_n = bx * BN + load_smem_b_n;
    // if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    // {
    //     return;
    // }
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN]; // 2 times space for double buffer

    // float frag_a[2][TM], frag_b[2][TN];
    float r_load_a[TM / 2]; // 每个线程负责4个元素的加载
    float r_load_b[TN / 2];
    float r_compute_a[TM]; // 计算的时候是8个元素
    float r_compute_b[TN];
    float r_c[TM][TN] = {0};
    // copy data for K = 0, copy data to buffer 0
    {
        int load_gmem_a_addr = load_gmem_a_m * K + load_smem_a_k;
        int load_gmem_b_addr = load_smem_b_k * N + load_gmem_b_n;
        FLOAT4(r_load_a[0]) = FLOAT4(A[load_gmem_a_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(B[load_gmem_b_addr]);

        FLOAT4(Bs[0][load_smem_b_k][load_smem_b_n]) = FLOAT4(r_load_b[0]);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            As[0][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
    }
    __syncthreads();

    // 对于tile_i，当前循环进行的是tile_i-1的计算，和tile_i的数据加载
    for (int tile = 1; tile < K / BK; tile++)
    {
        // stage的编号： 0:0, 1:1, 2:0, 3:1, 4:0, 5:1 ... K/BK - 1:?
        // copy data for K = tile
        int now_stage = (tile - 1) & 1;
        int nxt_stage = now_stage ^ 1;
        int load_gmem_a_addr = load_gmem_a_m * K + tile * BK + load_smem_a_k;
        int load_gmem_b_addr = (tile * BK + load_smem_b_k) * N + load_gmem_b_n;

        FLOAT4(r_load_a[0]) = FLOAT4(A[load_gmem_a_addr]);
        FLOAT4(r_load_b[0]) = FLOAT4(B[load_gmem_b_addr]);

        // compute for K = tile - 1
        for (int k = 0; k < BK; k++)
        {
            FLOAT4(r_compute_a[0]) = FLOAT4(As[now_stage][k][ty * TM]);
            FLOAT4(r_compute_a[4]) = FLOAT4(As[now_stage][k][ty * TM + 4]);
            FLOAT4(r_compute_b[0]) = FLOAT4(Bs[now_stage][k][tx * TN]);
            FLOAT4(r_compute_b[4]) = FLOAT4(Bs[now_stage][k][tx * TN + 4]);
            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    r_c[m][n] += r_compute_a[m] * r_compute_b[n];
                }
            }
        }

        FLOAT4(Bs[nxt_stage][load_smem_b_k][load_smem_b_n]) = FLOAT4(r_load_b[0]);
#pragma unroll
        for (int i = 0; i < 4; i++)
        {
            As[nxt_stage][load_smem_a_k + i][load_smem_a_m] = r_load_a[i];
        }
        __syncthreads();
    }

    // for last tile

    {
        for (int k = 0; k < BK; k++)
        {
            FLOAT4(r_compute_a[0]) = FLOAT4(As[1][k][ty * TM]);
            FLOAT4(r_compute_a[4]) = FLOAT4(As[1][k][ty * TM + 4]);
            FLOAT4(r_compute_b[0]) = FLOAT4(Bs[1][k][tx * TN]);
            FLOAT4(r_compute_b[4]) = FLOAT4(Bs[1][k][tx * TN + 4]);
            for (int m = 0; m < TM; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    r_c[m][n] += r_compute_a[m] * r_compute_b[n];
                }
            }
        }
    }
    // store result to C
#pragma unroll
    for (int m = 0; m < TM; m++)
    {
        for (int n = 0; n < TN; n += 4)
        {
            int gmem_c_m = by * BM + ty * TM + m;
            int gmem_c_n = bx * BN + tx * TN + n;
            int gemm_c_addr = gmem_c_m * N + gmem_c_n;
            FLOAT4(C[gemm_c_addr]) = FLOAT4(r_c[m][n]);
        }
    }
}

void gemm_forward1(float* A, float* B, float* C, int M, int N, int K)
{
    dim3 threads(16, 16);
    dim3 blocks((M + 15) / 16, (N + 15) / 16);
    gemm_kernel1<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaCheck(cudaGetLastError());
}

void gemm_forward2(float* A, float* B, float* C, int M, int N, int K)
{
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int BK = 32;
    dim3 threads(BM, BN);
    dim3 blocks(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    gemm_kernel2<BM, BN, BK><<<blocks, threads>>>(A, B, C, M, N, K);
    cudaCheck(cudaGetLastError());
}

void gemm_forward3(float* A, float* B, float* C, int M, int N, int K)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    dim3 threads(BM / TM, BN / TN);
    dim3 blocks(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    gemm_kernel3<BM, BN, BK, TM, TN><<<blocks, threads>>>(A, B, C, M, N, K);
    cudaCheck(cudaGetLastError());
}

void gemm_forward4(float* A, float* B, float* C, int M, int N, int K)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    dim3 threads(BM / TM, BN / TN);
    dim3 blocks(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    gemm_kernel6<BM, BN, BK, TM, TN><<<blocks, threads>>>(A, B, C, M, N, K);
    cudaCheck(cudaGetLastError());
}

void gemm_forward5(float* A, float* B, float* C, int M, int N, int K)
{
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;
    dim3 threads(BM / TM, BN / TN);
    dim3 blocks(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
    gemm_kernel5<BM, BK, BN, TM, TN><<<blocks, threads>>>(A, B, C, M, N, K);
    cudaCheck(cudaGetLastError());
}

// void gemm_forward6(float* A, float* B, float* C, int M, int N, int K)
// {
//     constexpr int BM = 128;
//     constexpr int BN = 128;
//     constexpr int BK = 8;
//     constexpr int TM = 8;
//     constexpr int TN = 8;
//     dim3 threads(BM / TM, BN / TN);
//     dim3 blocks(CEIL_DIV(M, BM), CEIL_DIV(N, BN));
//     gemm_kernel6<BM, BK, BN, TM, TN><<<blocks, threads>>>(A, B, C, M, N, K);
//     cudaCheck(cudaGetLastError());
// }

void gemm(int kernel_id, int M, int N, int K, float* A, float* B, float* C)
{
    switch (kernel_id)
    {
    case 1: gemm_forward1(A, B, C, M, N, K); break;
    case 2: gemm_forward2(A, B, C, M, N, K); break;
    case 3: gemm_forward3(A, B, C, M, N, K); break;
    case 4: gemm_forward4(A, B, C, M, N, K); break;
    case 5: gemm_forward5(A, B, C, M, N, K); break;
    // case 6: gemm_forward6(A, B, C, M, N, K); break;
    default: printf("No kernel id %d \n", kernel_id); break;
    }
}

int main(int argc, char* argv[])
{
    // input : M, N, K, default values are 1024, 1024, 1024
    int M = 4096;
    int N = 4096;
    int K = 4096;
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
    for (int kernel_id = 2; kernel_id <= 6; kernel_id++)
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