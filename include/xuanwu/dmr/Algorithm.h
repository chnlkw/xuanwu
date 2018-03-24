//
// Created by chnlkw on 1/18/18.
//

#ifndef DMR_ALGORITHM_H
#define DMR_ALGORITHM_H

#include "xuanwu/Array.h"
#include "xuanwu/Data.h"
//#include "Kernels.cuh"
#include "xuanwu/DataCopy.h"

//void shuffle_by_idx_gpu_ff(float *dst, const float *src, const size_t *idx, size_t size);

namespace Xuanwu {

template<class V>
V Renew(const V &in, size_t count = 0);

template<class T>
std::vector<T> Renew(const std::vector<T> &in, size_t count) {
    return std::vector<T>(count);
}

template<class T>
Array<T> Renew(const Array<T> &in, size_t count) {
    return in.Renew(count);
}

template<class T>
Data<T> Renew(const Data<T> &in, size_t count) {
    Data<T> ret(count);
    ret->Follow(in);
    return ret;
}

template<class V>
void Copy(const V &src, size_t src_off, V &dst, size_t dst_off, size_t count);

template<class T>
void Copy(const std::vector<T> &src, size_t src_off, std::vector<T> &dst, size_t dst_off, size_t count) {
    std::copy(src.begin() + src_off, src.begin() + src_off + count, dst.begin() + dst_off);
}


template<class T>
void Copy(const Data<T> &src, size_t src_off, Data<T> &dst, size_t dst_off, size_t count) {

    class TaskCopy : public TaskBase {
        const Data<T> src_;
        Data<T> dst_;
        size_t src_off_, dst_off_, count_;
    public:
        TaskCopy(const Data<T> src, size_t src_off, Data<T> dst, size_t dst_off, size_t count) :
                TaskBase("Copy"),
                src_(src),
                src_off_(src_off),
                dst_(dst),
                dst_off_(dst_off),
                count_(count) {
            AddInput(src);
            AddOutput(dst);
        }

        virtual void Run(CPUWorker *cpu) override {
            const T *src = src_.ReadAsync(shared_from_this(), cpu->Device(), 0).data();
            T *dst = dst_.WriteAsync(shared_from_this(), cpu->Device(), 0).data();
//            std::cout << "Run on CPU TaskCopy " << dst + dst_off_ << " <- " << src + src_off_ << std::endl;
            DataCopy(dst + dst_off_, -1, src + src_off_, -1, count_ * sizeof(T));
        }

        virtual void Run(GPUWorker *gpu) override {
//            std::cout << "Run on GPU " << gpu->Device()->Id() << std::endl;
            ArrayPtr<T> src = src_.GetReplicas().at(0);
//            const T *src = src_.ReadAsync(shared_from_this(), gpu->Device(), gpu->Stream()).data();

            T *dst = dst_.WriteAsync(shared_from_this(), gpu->Device(), gpu->Stream(), true).data();
            DataCopyAsync(dst + dst_off_, gpu->Device()->Id(), src->data() + src_off_, src->Device()->Id(),
                          count_ * sizeof(T),
                          gpu->Stream());
        }
    };
//    std::cout << "copy " << src.ToString() << " to " << dst.ToString() << std::endl;
//    DataCopy(dst.begin() + dst_off, dst.DeviceCurrent()->Id(),
//             src.begin() + src_off, src.DeviceCurrent()->Id(),
//             count * sizeof(T));
//    std::copy(src.begin() + src_off, src.begin() + src_off + count, dst.begin() + dst_off);
    Xuanwu::AddTask<TaskCopy>(
            src, src_off,
            dst, dst_off,
            count);
}

}

#endif //DMR_ALGORITHM_H
