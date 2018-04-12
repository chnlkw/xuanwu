//
// Created by chnlkw on 11/14/17.
//

#include "cuda_utils.h"

#ifdef USE_CUDA

#define REPORT_CUDA_SUCCESS 0

bool cudaEnsureSuccess(cudaError_t status, const char *status_context_description,
                       bool die_on_error, const char *filename, unsigned line_number) {
    if (status_context_description == NULL)
        status_context_description = "";
    if (status == cudaSuccess) {
#if REPORT_CUDA_SUCCESS
        std::cerr <<  "Succeeded: " << status_context_description << std::endl << std::flush;
#endif
        return true;

    }
    const char *errorString = cudaGetErrorString(status);
    std::cerr << "CUDA Error: ";
    if (status_context_description != NULL) {
        std::cerr << status_context_description << ": ";
    }
    if (errorString != NULL) {
        std::cerr << errorString;
    } else {
        std::cerr << "(Unknown CUDA status code " << status << ")";
    }
    if (filename != NULL) {
        std::cerr << " at " << filename << ":" << line_number;
    }

    std::cerr << std::endl << std::flush;
    if (die_on_error) {
        abort();
        //exit(EXIT_FAILURE);
        // ... or cerr << "FATAL ERROR" << etc. etc.

    }
    return false;

}


template <class T>
__global__
void copy_free_kernel(T* dst, T* src, size_t cnt) {
    for (int i = threadIdx.x; i <cnt; i += blockDim.x)
        dst[i] = src[i];
//        memcpy(dst, src, bytes);
    __syncthreads();
    if (threadIdx.x == 0)
        free(src);
}

void run_copy_free_kernel(void* dst, void* src, size_t bytes, cudaStream_t stream) {
    if (bytes % sizeof(int) == 0) {
        copy_free_kernel<int> << < 1, 1024, 0, stream >> > ((int*)dst, (int*)src, bytes / sizeof(int));
    } else {
        copy_free_kernel<char> << < 1, 1024, 0, stream >> > ((char*)dst, (char*)src, bytes);
    }
}


#endif
