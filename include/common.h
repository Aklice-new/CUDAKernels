#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_runtime.h>

constexpr int WARP_SIZE = 32;

//_________________________CHECK ERROR_________________________//

// CUDA ERROR CHECK

void cuda_check(cudaError_t error, const char* file, const int line)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, "CUDA error at %s:%i: %s\n", file, line, cudaGetErrorString(error));
        exit(-1);
    }
}
#define cudaCheck(err) (cuda_check(err, __FILE__, __LINE__))

// CUBLAS ERROR CHECK

void cublas_check(cublasStatus_t status, const char* file, const int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        fprintf(stderr, "CUBLAS error at %s:%i: %i\n", file, line, status);
        exit(-1);
    }
}
#define cublasCheck(err) (cublas_check(err, __FILE__, __LINE__))

#define CEIL_DIV(a, b) (((a) + (b) -1) / (b))

//_____________________________Packed128_____________________________//

template <typename ElementType>
struct alignas(16) Packed128
{
    Packed128() = default;
    __device__ explicit Packed128(int4 bits)
    {
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch");
        memcpy(&payload, &bits, sizeof(bits));
    }

    __device__ static Packed128 constant(ElementType value)
    {
        Packed128 result;
        for (int i = 0; i < size; i++)
            result.payload[i] = value;
        return result;
    }

    __device__ static Packed128 zeros()
    {
        return constant(0);
    }

    __device__ static Packed128 ones()
    {
        return constant(1);
    }
    __device__ ElementType& operator[](int i)
    {
        return payload[i];
    }
    __device__ const ElementType& operator[](int i) const
    {
        return payload[i];
    }
    __device__ int4 get_bits() const
    {
        int4 bits;
        static_assert(sizeof(bits) == sizeof(payload), "Size mismatch");
        memcpy(&bits, &payload, sizeof(bits));
        return bits;
    }
    // size 是实际上该结构体中包含的元素个数，sizeof(int4) = 128bit
    static constexpr int size = sizeof(int4) / sizeof(ElementType);
    ElementType payload[size];
};
// create a type alias for float4
using f128 = Packed128<float>;

// some Packed128 utils
// load 128 bits from address
template <typename ElementType>
__device__ Packed128<ElementType> load128(const ElementType* address)
{
    return Packed128<ElementType>{*reinterpret_cast<const int4*>(address)};
}

// store 128 bits to address
template <typename ElementType>
__device__ void store128(ElementType* target, Packed128<ElementType> value)
{
    *reinterpret_cast<int4*>(target) = value.get_bits();
}

//_____________________________precision setting_____________________________//

#ifdef ENABLE_BF16
using floatX = __nv_bfloat16;
using floatN = __nv_bfloat16;
#elif defined(ENABLE_FP16)
using floatX = half;
using floatN = half;
#else
using floatX = float;
using floatN = float;
#endif

using x128 = Packed128<floatX>;

//_____________________________testing and benchmarking utils_____________________________//

template <class TargetType>
[[nodiscard]] cudaError_t type_convert_memcpy(TargetType* d_ptr, float* h_ptr, size_t count)
{
    // copy from host to device with data type conversion.
    TargetType* converted = (TargetType*) malloc(count * sizeof(TargetType));
    for (int i = 0; i < count; i++)
    {
        converted[i] = (TargetType) h_ptr[i];
    }

    cudaError_t status = cudaMemcpy(d_ptr, converted, count * sizeof(TargetType), cudaMemcpyHostToDevice);
    free(converted);

    // instead of checking the status at cudaMemcpy, we return it from here. This way, we
    // still need to use our checking macro, and get better line info as to where the error
    // happened.
    return status;
}

template <class D, class T>
void validate_result(
    D* device_result, const T* cpu_reference, const char* name, std::size_t elements_num, T tolerance = 1e-4)
{
    D* gpu_out = (D*) malloc(elements_num * sizeof(D));
    cudaCheck(cudaMemcpy(gpu_out, device_result, elements_num * sizeof(D), cudaMemcpyDeviceToHost));
    int nfaults = 0;
#ifndef ENABLE_BP16
    float epsilon = FLT_EPSILON; // FLT_EPSILON 是 float 类型能够表示的最小相对差值，约为1.19209290e-07
#else
    float epsilon = 0.079; // 0.079 是深度学习中bfloat16的有效精度，其机器实际精度约为0.015625
#endif
    for (int i = 0; i < elements_num; i++)
    {
        // 去除无效值 or 跳过mask的值
        if (!std::isfinite(cpu_reference[i]))
            continue;
        // print the first few comparisons
        if (i < 5)
        {
            printf("%f %f\n", cpu_reference[i], (T) gpu_out[i]);
        }
        float t_eff = tolerance + fabs(cpu_reference[i]) * epsilon;
        if (fabs(cpu_reference[i] - (T) gpu_out[i]) > t_eff)
        {
            nfaults++;
            printf("Error: %s[%d] = %f, expected CPU: %f vs GPU: %f\n", name, i, cpu_reference[i], cpu_reference[i],
                (T) gpu_out[i]);
        }
        if (nfaults > 10)
        {
            free(gpu_out);
            exit(EXIT_FAILURE);
        }
    }
    free(gpu_out);
}

template <class Kernel, class... KernelArgs>
float benchmark_kernel(int repeats, Kernel kernel, KernelArgs&&... kerne_args)
{
    cudaEvent_t start, stop;
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaCheck(cudaGetDeviceProperties(&deviceProp, deviceIdx));
    // 刷新一下L2缓存, 以避免干扰测试,stackoverflow上说的方法可以通过cudamemset来刷L2缓存
    void* flush_buffer;
    cudaCheck(cudaMalloc(&flush_buffer, deviceProp.l2CacheSize));

    cudaCheck(cudaEventCreate(&start));
    cudaCheck(cudaEventCreate(&stop));

    float elapsed_time = 0.0f;
    for (int i = 0; i < repeats; i++)
    {
        cudaCheck(cudaMemset(flush_buffer, 0, deviceProp.l2CacheSize));
        cudaCheck(cudaEventRecord(start, 0));
        kernel(std::forward<KernelArgs>(kerne_args)...);
        cudaCheck(cudaEventRecord(stop, 0));
        cudaCheck(cudaEventSynchronize(start));
        cudaCheck(cudaEventSynchronize(stop));
        float time;
        cudaCheck(cudaEventElapsedTime(&time, start, stop));
        elapsed_time += time;
    }

    cudaCheck(cudaFree(flush_buffer));

    return elapsed_time / repeats;
}

//________________________RENDOM UTILS________________________//
// ----------------------------------------------------------------------------
// random utils

void make_random_float_01(float* arr, size_t N)
{
    // float* arr = (float*) malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = ((float) rand() / RAND_MAX); // range 0..1
    }
    // return arr;
}

void make_random_float(float* arr, size_t N)
{
    // float* arr = (float*) malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = ((float) rand() / RAND_MAX) * 2.0 - 1.0; // range -1..1
    }
    // return arr;
}

void make_random_int(int* arr, size_t N, int V)
{
    // int* arr = (int*) malloc(N * sizeof(int));
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = rand() % V; // range 0..V-1
    }
    // return arr;
}

void make_zeros_float(float* arr, size_t N)
{
    // float* arr = (float*) malloc(N * sizeof(float));
    memset(arr, 0, N * sizeof(float)); // all zero
    // return arr;
}

void make_ones_float(float* arr, size_t N)
{
    // float* arr = (float*) malloc(N * sizeof(float));
    for (size_t i = 0; i < N; i++)
    {
        arr[i] = 1.0f;
    }
    // return arr;
}
