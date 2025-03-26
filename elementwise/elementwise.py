from setuptools import setup
from torch.utils.cpp_extension import load
import torch
from torch.autograd import profiler
from torch import nn

my_vector_add = load(
    name='my_vector_add',
    sources=['elementwise.cu'],
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

a = torch.randn(N, C, device='cuda', dtype=torch.float32)
b = torch.randn(N, C, device='cuda', dtype=torch.float32)

torch_output = a + b

my_output = my_vector_add.vector_add(a, b)

print(torch.allclose(torch_output, my_output))

with profiler.profile(record_shapes=True, use_cuda=True) as prof:
    my_output = my_vector_add.vector_add(a, b)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


with profiler.profile(record_shapes=True, use_cuda=True) as prof:
    torch_output = a + b

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))