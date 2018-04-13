//
// Created by chnlkw on 3/30/18.
//

#include "Ptr.h"

void Xuanwu::Ptr::log(el::base::type::ostream_t &os) const {
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

void Xuanwu::GPUCopy(Xuanwu::Ptr dst, Xuanwu::Ptr src, size_t bytes, int gpu_id, cudaStream_t stream) {
    CLOG(INFO, "DataCopy") << "GPUCopy " << src << " to " << dst << " stream = " << stream;
    CUDA_CALL(cudaSetDevice, gpu_id);
    CUDA_CALL(cudaMemcpyAsync, dst, src, bytes, cudaMemcpyDefault, stream);
}

void Xuanwu::CPUCopy(Xuanwu::Ptr dst, Xuanwu::Ptr src, size_t bytes) {
    CLOG(INFO, "DataCopy") << "CPUCopy " << src << " to " << dst;
    if (dst.isCPU() && src.isCPU())
        memcpy(dst, src, bytes);
    else if (dst.isGPU() || src.isGPU()) {
        CUDA_CALL(cudaMemcpy, dst, src, bytes, cudaMemcpyDefault);
    }
}
