#include "common.h"
#include <cfloat>

/**
 * input      : [B, T, 3C]. Q, K, V 都是[B, T, C]
 * out        : [B, T, C]
 * preatt&att : [B, NH, T, T]
 * multi-head-attention分为三部分来计算:
 * 1. 首先计算 Q*K^T，在对每个元素除sqrt(d)
 * 2. 然后计算 softmax(Q*K^T)
 * 3. 最后计算 一个矩阵乘法
 */
void attention_cpu(float* input, float* out, float* preatt, float* att, int B, int T, int C, int NH)
{
    int C3 = C * 3;
    int head_size = C / NH;
    float scale = 1.0 / sqrtf(head_size);

    // reslut shape is : [B, NH, T, T]
    for (int b = 0; b < B; b++) // B
    {
        for (int t = 0; t < T; t++) // the first T
        {
            for (int h = 0; h < NH; h++) // NH
            {
                const float* query_t = input + b * T * C3 + t * C3 + h * head_size;
                float* preatt_t = preatt + b * NH * T * T + h * T * T + t * T;
                float* att_t = att + b * NH * T * T + h * T * T + t * T;
                // 1. matmul
                // 下面完成的都是一个[1, T]的块的内容的计算， 对应到结果中就是[t, 0~t]这段
                // 这里为什么不是一个完整的[t, 0~T]呢
                // 因为attention mask的存在，目前仅处理前t个单词，所以不能让看到后面的key
                // 所以顺便记录一下行最大值，方便计算softmax
                float maxval = -FLT_MAX;
                for (int t2 = 0; t2 <= t; t++) // the second T
                {
                    const float* key_t
                        = input + b * T * C3 + t2 * C3 + h * head_size + C; // +C 因为QKV是连在一行里存的，隔了C

                    // 进行点积
                    float sum = 0.f;
                    for (int i = 0; i < head_size; i++)
                    {
                        sum += query_t[i] * key_t[i];
                    }
                    sum *= scale; // div sqrt(D_k)
                    if (sum > maxval)
                    {
                        maxval = sum;
                    }
                    preatt_t[t2] = sum;
                }
                // 完成了这一行的内容的计算，即[b, h, t, 0~T]这段内容的计算
                // 2.计算这一行的 exp 和 exp_sum
                float expsum = 0.f;
                for (int t2 = 0; t2 <= t; t2++)
                {
                    float expv = expf(preatt_t[t2] - maxval);
                    expsum += expv;
                }
                // 3. normalize
                float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;
                for (int t2 = 0; t2 < T; t2++)
                {
                    if (t2 <= t)
                    {
                        att_t[t2] *= expsum_inv;
                    }
                    else
                    {
                        att_t[t2] = 0.f;
                    }
                }
                // 4. dot V
                float* out_t = out + b * NH * T * head_size + h * T * head_size + t * head_size;
                for (int i = 0; i < head_size; i++)
                {
                    out_t[i] = 0.f;
                }
                // 这里进行的是一行[t, T]个元素([T, T]的一行)和[T, hs]个元素的矩阵乘法
                for (int t2 = 0; t2 <= t; t2++)
                {
                    // 这里进行的部分的矩阵乘法，本来是需要一行[t, T]个元素和[T, i]列个元素相乘
                    // 这里先进行了是[t, i]这一个元素和 [t2, hs]个元素相乘
                    const float* val_t = input + b * T * NH * head_size + t2 * NH * head_size + h * head_size + 2 * C;
                    float att_tmp = att_t[t2];
                    for (int i = 0; i < head_size; i++)
                    {
                        out_t[i] += att_tmp * val_t[i];
                    }
                }
            }
        }
    }
}

//_____________________________GPU KERNELS______________________________

__global__ void attention_query_key_kernel1(float* preatt, const float* input, int B, int T, int C, int NH) {}
__global__ void attention_softmax_kernel1(float* att, float* preatt, int B, int T, int NH) {}
__global__ void attention_value_kernel1(float* out, float* att, float* preatt, int B, int T, int C, int NH) {}

//_____________________________KERNEL LAUNCHER__________________________

void attention_forward1(
    float* out, float* preatt, float* att, const float* inp, int B, int T, int C, int NH, const int block_size)
{
    // attention calculation
    int total_threads = B * NH * T * T;
    int num_blocks = CEIL_DIV(total_threads, block_size);
    attention_query_key_kernel1<<<num_blocks, block_size>>>(preatt, inp, B, T, C, NH);
    // softmax and value accumulation
    total_threads = B * T * NH;
    num_blocks = CEIL_DIV(total_threads, block_size);
    attention_softmax_kernel1<<<num_blocks, block_size>>>(att, preatt, B, T, NH);
    attention_value_kernel1<<<num_blocks, block_size>>>(out, att, inp, B, T, C, NH);
}

//______________________________KERNEL DISPATHER________________________

//______________________________MAIN____________________________________