from setuptools import setup
from torch.utils.cpp_extension import load
import torch

my_reduce_sum = load(
    name='my_reduce_sum',
    sources=['reduce_sum.cu'],
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

N = 192912
C = 1121
input = torch.arange(N * C, device='cuda', dtype=torch.float32).reshape(N, C)

output = torch.sum(input, dim=1)
my_output = torch.zeros(N, device='cuda', dtype=torch.float32)
my_reduce_sum.reduce_sum2(input, my_output)

print(output)

print(my_output)

print(torch.allclose(output, my_output))