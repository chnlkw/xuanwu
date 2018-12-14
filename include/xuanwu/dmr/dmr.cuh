//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_KERNELS_H
#define DMR_KERNELS_H

#include "dmr.h"

#include "../Xuanwu.h"
#include "../Task.h"
#include "../Data.h"
#include "../Worker.h"

namespace Xuanwu {
    template<class T, class TOff>
    __global__ void shuffle_by_idx_kernel(T *dst, const T *src, const TOff *idx, size_t size) {
        size_t i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < size)
            dst[i] = src[idx[i]];
    };

    template<class T, class TOff>
    void ShuffleByIdx(Data<T>& dst, const Data<T>& src, const Data<TOff>& idx, std::string name) {
//    LG(INFO) << dst.size() << " ?= " << src.size() << " ?= " << idx.size();
        assert(dst.size() == src.size());
        assert(dst.size() == idx.size());

        TaskPtr task(new TaskBase(
                name,
                std::make_unique<CPUTask>([=](CPUContext cpu)mutable {
                    for (int i = 0; i < dst.size(); i++) {
                        dst[i] = src[idx[i]];
                    }
                }),
                std::make_unique<GPUTask>([=](GPUContext gpu) mutable {
                    size_t size = src.size();
//                shuffle_by_idx_gpu(dst.data(), src.data(), idx.data(), src.size(), gpu->Stream());
                    shuffle_by_idx_kernel << < (size + 31) / 32, 32, 0, gpu.stream >> >
                                                                        (dst.data(), src.data(), idx.data(), size);
                })));
        task->AddInputs({src, idx});
        task->AddOutput(dst);
        task->Type() = TaskBase::Compute;
        AddTask(task);

    }

}
#endif //DMR_KERNELS_H
