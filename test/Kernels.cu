//
// Created by chnlkw on 1/23/18.
//

#include "Kernels.h"
//#include "Worker.h"
//#include "Task.h"

#if 0
template<class T, class TOff>
__global__ void shuffle_by_idx_kernel(T *dst, const T *src, const TOff *idx, size_t size) {
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        dst[i] = src[idx[i]];
};

template<class T, class TOff>
void shuffle_by_idx_gpu(T *dst, const T *src, const TOff *idx, size_t size, cudaStream_t stream) {

    shuffle_by_idx_kernel << < (size + 31) / 32, 32, 0, stream >> > (dst, src, idx, size);
}

template void
shuffle_by_idx_gpu<float, size_t>(float *dst, const float *src, const size_t *idx, size_t size, cudaStream_t stream);

template void
shuffle_by_idx_gpu<unsigned int, size_t>(unsigned int *dst, const unsigned int *src, const size_t *idx, size_t size,
                                         cudaStream_t stream);

template void shuffle_by_idx_gpu<unsigned int, unsigned int>(unsigned int *, unsigned int const *, unsigned int const *,
                                                             unsigned long, cudaStream_t);
#endif

template<class T>
__global__ void gpu_add_kernel(T *c, const T *a, const T *b, size_t size) { // c[i] = a[i] + b[i]
    size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < size)
        c[i] = a[i] + b[i];
}

template<class T>
void gpu_add<>(T *c, const T *a, const T *b, size_t size, cudaStream_t stream) { // c[i] = a[i] + b[i]
    gpu_add_kernel << < (size + 31) / 32, 32, 0, stream >> > (c, a, b, size);
}

template void gpu_add<int>(int *, const int *, const int *, size_t, cudaStream_t stream);

template<class F, class... Args>
__global__
void kernel(F f, Args... args) {
//    printf("value = %d\n", f(args...));
    f(args...);
}

template<class Lambda>
void launch_gpu(Lambda lambda, int nblock, int nthread, cudaStream_t stream, int shared_memory = 0) {
//    int nblock = 1;
//    int nthread = 1;
    kernel << < nblock, nthread, shared_memory, stream >> > (lambda);

}

#define FOR(i, N)\
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)

TaskPtr create_taskadd(const Data<int> &a, const Data<int> &b, Data<int> &c) {
    auto gputask = std::make_unique<GPUTask>([&](GPUWorker *gpu) {
        auto l = [=, c = c.data(), b = b.data(), a = a.data(), size = a.size()]
        __device__()
        {
//            int beg = blockDim.x * blockIdx.x + threadIdx.x;
//            int step = blockDim.x * gridDim.x;
//            for (int i = beg; i < size; i += step) {
            FOR(i, size) {
                c[i] = a[i] + b[i];
            }
        };
        launch_gpu(l, 1, 1, gpu->Stream());
    }, 2);
    auto cputask = std::make_unique<CPUTask>([&](CPUWorker *cpu) {
        for (int i = 0; i < c.size(); i++) {
            c[i] = a[i] + b[i];
        }
    }, 1);
    TaskPtr task(new TaskBase("Add2", std::move(cputask), std::move(gputask)));
    task->AddInputs({a, b});
    task->AddOutputs({c});
    return task;
}

#if 0

TaskAdd2::TaskAdd2(const Data<int> &a, const Data<int> &b, Data<int> &c) :
        TaskBase("Add2"),
        CPUTask([&](CPUWorker *cpu) {
            for (int i = 0; i < c.size(); i++) {
                c[i] = a[i] + b[i];
            }
        }, 1),
        GPUTask([&](GPUWorker *gpu) {
            auto l = [=, c = c.data(), b = b.data(), a = a.data(), size = a.size()]
            __device__()
            {
                int beg = blockDim.x * blockIdx.x + threadIdx.x;
                int step = blockDim.x * gridDim.x;
                for (int i = beg; i < size; i += step) {
                    c[i] = a[i] + b[i];
                }
            };
            launch_gpu(l);
        }, 2) {
    assert(a.size() == b.size());
    assert(a.size() == c.size());
    AddInputs({a, b});
    AddOutputs({c});
    GPUTask g([&](GPUWorker *gpu) {
        auto l = [=, c = c.data(), b = b.data(), a = a.data(), size = a.size()]
        __device__()
        {
            int beg = blockDim.x * blockIdx.x + threadIdx.x;
            int step = blockDim.x * gridDim.x;
            for (int i = beg; i < size; i += step) {
                c[i] = a[i] + b[i];
            }
        };
        launch_gpu(l);
    }, 2);
}

#endif
