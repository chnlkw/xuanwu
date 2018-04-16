//
// Created by chnlkw on 4/16/18.
//

#include "Event.h"

using namespace Xuanwu;

bool EventDummy::QueryFinished() {
    return true;
}

void EventGPU::clean() {
    if (event) {
        CUDA_CALL(cudaEventDestroy, event);
        event = 0;
    }
}

EventGPU::EventGPU(cudaStream_t stream) : EventBase() {
    CUDA_CALL(cudaEventCreate, &event);
    CUDA_CALL(cudaEventRecord, event, stream);
}

bool EventGPU::QueryFinished() {
    if (event == nullptr)
        return true;
    cudaError_t err = cudaEventQuery(event);
    if (err == cudaSuccess) {
        clean();
        return true;
    } else if (err == cudaErrorNotReady) {
        return false;
    } else {
        CUDA_CHECK();
        return false;
    }
}

EventGPU::~EventGPU() {
    clean();
}
