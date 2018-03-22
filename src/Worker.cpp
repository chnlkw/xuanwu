//
// Created by chnlkw on 3/2/18.
//

#include "Worker.h"
#include "Device.h"
#include "Task.h"
#include "Data.h"
#include "Engine.h"
#include "clock.h"

#define LG(x) CLOG(x, "Worker")

using namespace dmr;

CPUWorker::CPUWorker(CPUDevice *cpu) : WorkerBase(cpu) {}

std::vector<TaskPtr> CPUWorker::GetCompleteTasks() {
    std::vector<TaskPtr> ret;
    for (TaskPtr t : tasks_) {
        CPUTask *cputask = dynamic_cast<CPUTask *>(t.get());
        if (!cputask)
            cputask = t->GetCPUTask();
        CLOG(INFO, "Worker") << *this << " Run " << *t;
        Clock clk;
        if (cputask) {
            t->PrepareData(device_, 0);
            CUDA_CALL(cudaStreamSynchronize, 0);

            (*cputask)(this);
        } else
            t->Run(this);
        CLOG(INFO, "Worker") << *this << " " << *t << " uses " << clk.timeElapsed() << " seconds";
        ret.push_back(t);
    }
    tasks_.clear();
    return ret;
}

GPUWorker::GPUWorker(GPUDevice *gpu) :
        WorkerBase(gpu) {
    CLOG(INFO, "Worker") << "Create GPU Worker with device = " << gpu->Id();
    CUDA_CALL(cudaSetDevice, gpu->Id());
    CUDA_CALL(cudaStreamCreate, &stream_);
}

void GPUWorker::RunTask(TaskPtr t) {
    auto gpu = dynamic_cast<GPUDevice *>(device_);
    assert(gpu);
    CUDA_CALL(cudaSetDevice, gpu->Id());

    Meta meta{GetEvent(), GetEvent(), GetEvent(), t};
    CUDA_CALL(cudaEventRecord, meta.beg_event, stream_);

    auto gputask = dynamic_cast<GPUTask *>(t.get());
    if (!gputask)
        gputask = t->GetGPUTask();
    CLOG(INFO, "Worker") << *this << " Run " << *t;
    if (gputask) {
        for (auto &m : t->GetMetas()) {
            if (m.is_read_only) {
                m.data->ReadAsync(t, device_, stream_);
            } else {
                m.data->WriteAsync(t, device_, stream_);
            }
        }
        CUDA_CALL(cudaEventRecord, meta.transfer_event, stream_);
        (*gputask)(this);
    } else {
        CUDA_CALL(cudaEventRecord, meta.transfer_event, stream_);
        t->Run(this);
    }
    CUDA_CALL(cudaEventRecord, meta.end_event, stream_);
    queue_.push_back(meta);
}

std::vector<TaskPtr> GPUWorker::GetCompleteTasks() {
#ifdef USE_CUDA
    std::vector<TaskPtr> ret;
    if (Empty())
        return ret;

    while (true) {
        Meta meta = queue_.front();
        cudaError_t err = cudaEventQuery(meta.end_event);
        if (err == cudaSuccess) {
            queue_.pop_front();
            float tranfer_ms, calc_ms;
            CUDA_CALL(cudaEventElapsedTime, &tranfer_ms, meta.beg_event, meta.transfer_event);
            CUDA_CALL(cudaEventElapsedTime, &calc_ms, meta.transfer_event, meta.end_event);
            CLOG(INFO, "Worker") << *this << " " << *meta.task << " transfer " << tranfer_ms << " ms, " << " calc "
                                 << calc_ms << " ms";

            ret.push_back(meta.task);
            events_unused_.push_back(meta.end_event);
            events_unused_.push_back(meta.beg_event);
            events_unused_.push_back(meta.transfer_event);
            break;
        } else if (err == cudaErrorNotReady) {
            continue;
        } else {
            CUDA_CHECK();
        }
    }
    return ret;
#else
    return {};
#endif
}

cudaEvent_t GPUWorker::GetEvent() {
    cudaEvent_t e;
    if (!events_unused_.empty()) {
        e = events_unused_.back();
        events_unused_.pop_back();
    } else {
        CUDA_CALL(cudaEventCreate, &e);
    }
    return e;
}
