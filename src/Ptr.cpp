//
// Created by chnlkw on 3/30/18.
//

#include "Ptr.h"
#include "Event.h"

namespace Xuanwu {

    void Ptr::log(el::base::type::ostream_t &os) const {
        std::string s;
        switch (type_) {
            case Type::CPU :
                s = "Ptr_CPU";
                break;
            case Type::GPU :
                s = "Ptr_GPU";
                break;
            default:
                s = "Ptr_Unknown";
                break;
        }
        os << s << "[" << ptr_ << "]";
    }

    Event GPUCopy(Ptr dst, Ptr src, size_t bytes, int gpu_id, cudaStream_t stream) {
        CLOG(INFO, "DataCopy") << "GPUCopy " << src << " to " << dst << " stream = " << stream << " bytes = " << bytes;
        CUDA_CALL(cudaSetDevice, gpu_id);
        if (dst.isGPU() && src.isGPU()) {
            CUDA_CALL(cudaMemcpyAsync, dst, src, bytes, cudaMemcpyDefault, stream);
//            run_copy_kernel(dst, src, bytes, stream);
        } else {
            CUDA_CALL(cudaMemcpyAsync, dst, src, bytes, cudaMemcpyDefault, stream);
        }
        return Event(new EventGPU(stream));
    }

    Event CPUCopy(Ptr dst, Ptr src, size_t bytes, cudaStream_t stream) {
        CLOG(INFO, "DataCopy") << "CPUCopy " << src << " to " << dst << " bytes = " << bytes;
        if (dst.isCPU() && src.isCPU()) {
            memcpy(dst, src, bytes);
            return std::make_unique<EventDummy>();
        } else if (dst.isGPU() || src.isGPU()) {
            CUDA_CALL(cudaMemcpyAsync, dst, src, bytes, cudaMemcpyDefault, stream);
            return std::make_unique<EventGPU>(stream);
        } else {
            LOG(FATAL) << "CPUCopy " << src << " to " << dst << " not supported";
            abort();
        }
    }

    int CopySpeed(Ptr dst, Ptr src) {
        if (dst.isCPU() && src.isCPU()) {
            return 50;
        } else if (dst.isGPU() && src.isGPU()) {
            return 100;
        } else if (dst.isCPU() && src.isGPU() || dst.isGPU() && src.isCPU()) {
            return 10;
        } else {
            abort();
            return 0;
        }
    }

}
