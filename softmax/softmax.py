from setuptools import setup
from torch.utils.cpp_extension import load
import torch
from torch import profiler
from torch.profiler import ProfilerActivity

my_softmax = load(
    name='my_softmax',
    sources=['softmax.cu'],
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

N = 4096
C = 32121
input = torch.arange(N * C, device='cuda', dtype=torch.float32).reshape(N, C)

torch_output = torch.softmax(input, dim=1)
my_output = my_softmax.softmax(input)

print(torch.allclose(torch_output, my_output))


with profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
    my_output = my_softmax.softmax(input)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))


with profiler.profile(activities=[ProfilerActivity.CUDA]) as prof:
    torch_output = torch.softmax(input, dim=1)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))