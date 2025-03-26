from setuptools import setup
from torch.utils.cpp_extension import load
import torch
from torch.autograd import profiler
from torch import nn

my_rmsnorm = load(
    name='my_rmsnorm',
    sources=['rmsnorm.cu'],
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

N = 1024
C = 121211

input = torch.arange(N * C, device='cuda', dtype=torch.float32).reshape(N, C)
gamma = torch.randn(C, device='cuda', dtype=torch.float32)

torch_output = gamma * input / torch.sqrt((input ** 2).mean(dim=1, keepdim=True) + 1e-6)

my_output = my_rmsnorm.rmsnorm(input, gamma)

print(torch.allclose(torch_output, my_output))

with profiler.profile(record_shapes=True, use_cuda=True) as prof:
    my_output = my_rmsnorm.rmsnorm(input, gamma)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


with profiler.profile(record_shapes=True, use_cuda=True) as prof:
    torch_output = gamma * input / torch.sqrt((input ** 2).mean(dim=1, keepdim=True) + 1e-6)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))