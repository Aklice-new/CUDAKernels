#include "ATen/core/Formatting.h"
#include "ATen/core/TensorBody.h"
#include "c10/util/ArrayRef.h"
#include <cassert>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>

#define CEIL_DIV(a, b) (((a) + (b - 1)) / b)

__global__ void img2col_kernel(float* output, const float* input, int in_channel, int out_channel, int N, int C, int H,
    int W, int padding_h, int padding_w, int dilation_h, int dilation_w, int kernel_h, int kernel_w, int stride_h,
    int stride_w, int output_h, int output_w)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= output_h * output_w)
        return;
    int row = tid / output_h;
    int col = tid % output_h;
    // int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (row >= H || col >= W)
    {
        return;
    }
    output += row * output_w + col;
    int start_row = row * stride_h - padding_h;
    int start_col = col * stride_w - padding_w;
    int channel_size = W * H;
    // 每个线程负责 in_channel * kernel_w * kernel_h 个元素的生成
    for (int c = 0; c < in_channel; c++)
    {
        for (int i = 0; i < kernel_h; i++)
        {
            for (int j = 0; j < kernel_w; j++)
            {
                int row_im = start_row + i * dilation_h;
                int col_im = start_col + j * dilation_w;
                *output = (row_im >= 0 && col_im >= 0 && row_im < H && col_im < W)
                    ? input[c * channel_size + row_im * W + col_im]
                    : 0;
                output += output_h * output_w;
            }
        }
    }
}
template <int BM = 16, int BN = 16, int BK = 16>
__global__ void gemm_kernel(float* output, const float* weight_col, const float* img_col, int M, int K, int N)
{
    __shared__ float SA[BM][BK];
    __shared__ float SB[BK][BN];
    int row = blockDim.y * blockIdx.y;
    int col = blockDim.x * blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int load_a_gmem_m = row + ty;
    int load_b_gmem_n = col + tx;
    int load_a_smem_m = ty;
    int load_a_smem_k = tx;
    int load_b_smem_k = ty;
    int load_b_smem_n = tx;

    int tiled_k = CEIL_DIV(K, BK);

    float sum = 0;
    for (int i = 0; i < tiled_k; i++)
    {
        if (load_a_gmem_m < M && i * BK + tx < K)
            SA[load_a_smem_m][load_a_smem_k] = weight_col[load_a_gmem_m * K + i * BK + load_a_smem_k];
        else
            SA[load_a_smem_m][load_a_smem_k] = 0;

        if (i * BK + load_b_smem_k < K && load_b_gmem_n < N)
            SB[load_b_smem_k][load_b_smem_n] = img_col[(i * BK + load_b_smem_k) * N + load_b_gmem_n];
        else
            SB[load_b_smem_k][load_b_smem_n] = 0;
        __syncthreads();
        for (int k = 0; k < K; k++)
        {
            sum += SA[ty][k] * SB[k][tx];
        }
        __syncthreads();
    }
    if (load_a_gmem_m < M && load_b_gmem_n < N)
    {
        output[load_a_gmem_m * N + load_b_gmem_n] = sum;
    }
}

// 二维卷积 N C H W

void convolution2d_forward(at::Tensor& output, at::Tensor& img_col, at::Tensor input, at::Tensor weight, int in_channel,
    int out_channel, int N, int C, int H, int W, int padding_h, int padding_w, int dilation_h, int dilation_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w)
{

    auto input_shape = input.sizes();
    auto weight_shape = weight.sizes();
    int output_h = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) / stride_h + 1;
    int output_w = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) / stride_w + 1;
    // torch::ArrayRef<long> img_col_shape = {in_channel * kernel_h * kernel_w, output_h * output_w};

    // at::Tensor img_col = torch::zeros(img_col_shape);
    // torch::ArrayRef<long> weight_col_shape = {out_channel, in_channel * kernel_h * kernel_w};
    // at::Tensor weight_col = weight.reshape(weight_col_shape);

    // torch::ArrayRef<long> output_shape = {N, out_channel, output_h, output_w};

    dim3 thread_per_block(256);
    dim3 block_per_grid(CEIL_DIV(output_h * output_w, 256));
    int intput_per_N = C * H * W; //
    int output_per_N = out_channel * output_h * output_w;
    // 输出的是 N 个 GEMM
    // 每个线程负责output_h * output_w 中一个元素的计算
    dim3 gemm_thread_per_block(16, 16);
    dim3 gemm_block_per_grid(CEIL_DIV(out_channel, 16), CEIL_DIV(output_h * output_w, 16));
    for (int i = 0; i < N; i++)
    {
        img2col_kernel<<<block_per_grid, thread_per_block>>>(img_col.data_ptr<float>(),
            input.data_ptr<float>() + i * intput_per_N, in_channel, out_channel, N, C, H, W, padding_h, padding_w,
            dilation_h, dilation_w, kernel_h, kernel_w, stride_h, stride_w, output_h, output_w);
        cudaDeviceSynchronize();
        gemm_kernel<<<gemm_block_per_grid, gemm_thread_per_block>>>(output.data_ptr<float>() + i * output_per_N,
            weight.data_ptr<float>(), img_col.data_ptr<float>(), out_channel, in_channel * kernel_h * kernel_w,
            output_h * output_w);
        cudaDeviceSynchronize();
    }
    // return output;
}

void gemm_forward(at::Tensor a, at::Tensor b, at::Tensor& out)
{
    auto a_shape = a.sizes();
    auto b_shape = b.sizes();
    int M = a_shape[0];
    int K = a_shape[1];
    int N = b_shape[1];

    assert(a_shape[1] == b_shape[0]);

    dim3 thread_per_block(16, 16);
    dim3 block_per_grid(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
    gemm_kernel<<<block_per_grid, thread_per_block>>>(
        out.data_ptr<float>(), a.data_ptr<float>(), b.data_ptr<float>(), M, K, N);
    cudaDeviceSynchronize();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("convolution2d", &convolution2d_forward, "convolution2d");
    m.def("gemm", &gemm_forward, "gemm");
}
