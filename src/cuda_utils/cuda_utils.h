#pragma once

//#include "config.h"
#define USE_CUDA

#ifdef USE_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <cstring>
#include <chrono>
#include <iostream>
#include <vector>

#include <curand_kernel.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

#define CUDA_CALL(cuda_function, ...)  { \
    cudaError_t status = cuda_function(__VA_ARGS__); \
    cudaEnsureSuccess(status, #cuda_function, true, __FILE__, __LINE__); \
}


#define CUDA_CHECK_NO_SYNC() \
{\
    cudaError_t status = cudaGetLastError();\
    cudaEnsureSuccess(status, "last check", true, __FILE__, __LINE__); \
}

#define CUDA_CHECK() \
{\
    cudaDeviceSynchronize();\
    cudaError_t status = cudaGetLastError();\
    cudaEnsureSuccess(status, "last check", true, __FILE__, __LINE__); \
}

bool cudaEnsureSuccess(cudaError_t status, const char *status_context_description,
                       bool die_on_error, const char *filename, unsigned line_number);


void run_copy_kernel(void* dst, void* src, size_t bytes, cudaStream_t stream);

void run_copy_free_kernel(void* dst, void* src, size_t bytes, cudaStream_t stream);

#else

using cudaStream_t = void *;
using cudaEvent_t = void *;

#define CUDA_CALL(cuda_function, ...)  { }

#define CUDA_CHECK() { }

#endif
