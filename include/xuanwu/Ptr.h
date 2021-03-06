//
// Created by chnlkw on 3/30/18.
//

#ifndef XUANWU_PTR_H
#define XUANWU_PTR_H

#include "defs.h"
#include <easylogging++.h>
#include "cuda_utils.h"
#include "Event.h"

namespace Xuanwu {

    class Ptr : public el::Loggable {
    public:
        enum Type {
            CPU, GPU
        };

        explicit Ptr(void *ptr, Type type = Type::CPU) : ptr_(ptr), type_(type) {}

        virtual void *RawPtr() {
            return ptr_;
        }

        Ptr operator+(ssize_t d) {
            return Ptr{(char *) ptr_ + d, type_};
        }

        operator void *() const {
            return ptr_;
        }

        bool isCPU() const { return type_ == Type::CPU; }

        bool isGPU() const { return type_ == Type::GPU; }

        virtual void log(el::base::type::ostream_t &os) const;

    private:
        void *ptr_;
        Type type_;
    };

    Event CPUCopy(Ptr dst, Ptr src, size_t bytes, cudaStream_t stream);

    Event GPUCopy(Ptr dst, Ptr src, size_t bytes, int gpu_id, cudaStream_t stream);

    int CopySpeed(Ptr dst, Ptr src);
}

#endif //XUANWU_PTR_H
