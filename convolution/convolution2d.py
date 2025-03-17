from setuptools import setup
from torch.utils.cpp_extension import load
import torch

my_convolution2d = load(
    name='my_convolution2d',
    sources=['convolution2d.cu'],
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


# a = torch.arange(12, device='cuda', dtype=torch.float32).reshape(3, 4)
# b = torch.arange(20, device='cuda', dtype=torch.float32).reshape(4, 5)
# print(a, b)
# c = a @ b
# my_c = torch.zeros(3, 5, device='cuda', dtype=torch.float32)
# my_convolution2d.gemm(a, b, my_c)
# print(my_c)
# print(c)

# print(torch.allclose(c, my_c))

N = 4
C = 1
H = 5
W = 5

in_channel = C
out_channel = 1
padding_h = 0
padding_w = 0
dilation_h = 1
dilation_w = 1
kernel_h = 3
kernel_w = 3
stride_h = 1
stride_w = 1

output_h = (H + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
output_w = (W + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1

conv2d = torch.nn.Conv2d(in_channel,out_channel,
                         kernel_size=(kernel_h,kernel_w),
                         stride=(stride_h,stride_w),
                         padding=(padding_h,padding_w),
                         dilation=(dilation_h,dilation_w),
                         bias=False).cuda()
conv2d.weight.data = torch.arange(in_channel*out_channel*kernel_h*kernel_w, device='cuda', dtype=torch.float32).reshape(in_channel, out_channel, kernel_h, kernel_w)

input = torch.arange(N*C*H*W, device='cuda', dtype=torch.float32).reshape(N, C, H, W)
output = conv2d(input)
my_output = torch.zeros_like(output)

print(output.shape)

img_col = torch.zeros(in_channel * kernel_h * kernel_w, output_h * output_w , device='cuda', dtype=torch.float32)
my_conv2d = my_convolution2d.convolution2d(my_output, img_col, input.contiguous(), conv2d.weight.contiguous(),
                                            in_channel, out_channel,  N, C, H, W,
        padding_h, padding_w,  dilation_h, dilation_w, kernel_h, kernel_w, stride_h, stride_w)

print(img_col)

print(output)
print(my_output)

print(torch.allclose(output, my_output))
