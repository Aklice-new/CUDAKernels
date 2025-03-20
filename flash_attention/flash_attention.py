from setuptools import setup
from torch.utils.cpp_extension import load
import torch



flash_attention = load(
    name='flash_attention',
    sources=['flash_attention.cu'],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        # "--use_fast_math"
    ], 
    extra_cflags=['-std=c++17']
)


def self_attention(Q, K, V):
    B, head_num, N, D = Q.shape
    scale = 1.0 / D ** 0.5
    output = Q @ K.transpose(-2, -1) * scale
    output = torch.softmax(output, dim=1)
    output = output @ V
    return output

B = 1
head_num = 3
N = 512
D = 64

Q = torch.randn(B, head_num, N, D, device='cuda', dtype=torch.float32)
K = torch.randn(B, head_num, N, D, device='cuda', dtype=torch.float32)
V = torch.randn(B, head_num, N, D, device='cuda', dtype=torch.float32)

torch_output = self_attention(Q, K, V)

flash_output = flash_attention.flash_attention(Q, K, V)

print(torch.allclose(torch_output, flash_output))


