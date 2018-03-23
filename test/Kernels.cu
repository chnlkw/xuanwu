//
// Created by chnlkw on 1/23/18.
//

#include "Kernels.h"

template<class F, class... Args>
__global__
void kernel(F f, Args... args) {
    f(args...);
}

template<class Lambda>
void launch_gpu(Lambda lambda, int nblock, int nthread, cudaStream_t stream, int shared_memory = 0) {
    kernel << < nblock, nthread, shared_memory, stream >> > (lambda);
}

#define FOR(i, N)\
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)

TaskPtr create_taskadd(const Data<int> &a, const Data<int> &b, Data<int> &c) {
    auto gputask = std::make_unique<GPUTask>([&](GPUWorker *gpu) {
        auto l = [=, c = c.data(), b = b.data(), a = a.data(), size = a.size()]
        __device__()
        {
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
