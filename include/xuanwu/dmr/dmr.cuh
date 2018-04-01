//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_KERNELS_H
#define DMR_KERNELS_H

#include "dmr.h"

#include <xuanwu/Xuanwu.h>
#include <xuanwu/Task.h>
#include <xuanwu/Data.h>
#include <xuanwu/Worker.h>

namespace Xuanwu {
    template<class T, class TOff>
    __global__ void shuffle_by_idx_kernel(T *dst, const T *src, const TOff *idx, size_t size) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < size)
            dst[i] = src[idx[i]];
    };

    template<class T, class TOff>
    void ShuffleByIdx(Data<T> dst, Data<T> src, Data<TOff> idx) {
//    LG(INFO) << dst.size() << " ?= " << src.size() << " ?= " << idx.size();
        assert(dst.size() == src.size());
        assert(dst.size() == idx.size());

        TaskPtr task(new TaskBase(
                "Shuffle",
                std::make_unique<CPUTask>([=](CPUWorker *cpu)mutable {
                    for (int i = 0; i < dst.size(); i++) {
                        dst[i] = src[idx[i]];
                    }
                }),
                std::make_unique<GPUTask>([=](GPUWorker *gpu) mutable {
                    size_t size = src.size();
//                shuffle_by_idx_gpu(dst.data(), src.data(), idx.data(), src.size(), gpu->Stream());
                    shuffle_by_idx_kernel << < (size + 31) / 32, 32, 0, gpu->Stream() >> >
                                                                        (dst.data(), src.data(), idx.data(), size);
                })));
        task->AddInputs({src, idx});
        task->AddOutput(dst);
        AddTask(task);
    }

}
#endif //DMR_KERNELS_H
