//
// Created by chnlkw on 3/2/18.
//

#include "Worker.h"
#include "Device.h"
#include "Task.h"
#include "Data.h"
#include "clock.h"

#define LG(x) CLOG(x, "Worker")

namespace {
    bool QueryEvent(cudaEvent_t e) {
        cudaError_t err = cudaEventQuery(e);
        if (err == cudaErrorNotReady)
            return false;
        if (err != cudaSuccess) {
            LOG(ERROR) << " err = " << cudaGetErrorString(cudaGetLastError());
            CUDA_CHECK();
        }
        return true;
    }
}

namespace Xuanwu {

    CPUWorker::CPUWorker(CPUDevice *cpu) : WorkerBase(cpu) {}

    std::vector<TaskPtr> CPUWorker::GetCompleteTasks() {
        auto cpu = dynamic_cast<CPUDevice *>(device_);
        assert(cpu);
        std::vector<TaskPtr> ret;
        for (TaskPtr t : tasks_) {
            CPUTask *cputask = dynamic_cast<CPUTask *>(t.get());
            if (!cputask)
                cputask = t->GetCPUTask();
            CLOG(INFO, "Worker") << *this << " Run " << *t;
            Clock clk;
            if (cputask) {
                for (auto &m : t->Metas()) {
                    if (m.readable) {
                        while (!m.data->ReadAsync(this, device_));
                    }
                    if (m.writable && m.data->Bytes() > 0) {
                        while (!m.data->WriteAsync(this, device_));
                    }
                    CLOG(DEBUG, "Worker") << m;
                }
                (*cputask)(CPUContext(cpu));
            } else
                t->Run(this);
            CLOG(INFO, "Worker") << *this << " " << *t << " uses " << clk.timeElapsed() << " seconds";
            ret.push_back(t);
        }
        tasks_.clear();
        return ret;
    }

    Event CPUWorker::Copy(Ptr dst, Ptr src, size_t bytes) {
        return CPUCopy(dst, src, bytes);
    }

    GPUWorker::GPUWorker(GPUDevice *gpu) :
            WorkerBase(gpu) {
        CLOG(INFO, "Worker") << "Create GPU Worker with device = " << gpu->GPUID();
        CUDA_CALL(cudaSetDevice, gpu->GPUID());
        CUDA_CALL(cudaStreamCreate, &stream_);
    }

    void GPUWorker::RunTask(TaskPtr t) {
        auto gpu = dynamic_cast<GPUDevice *>(device_);
        assert(gpu);
        CUDA_CALL(cudaSetDevice, gpu->GPUID());

        Meta meta{GetEvent(), GetEvent(), GetEvent(), GetEvent(), t, t->Metas()};

        queue_.push_back(meta);
    }

    std::vector<TaskPtr> GPUWorker::GetCompleteTasks() {
        auto gpu = dynamic_cast<GPUDevice *>(device_);
        CUDA_CALL(cudaSetDevice, gpu->GPUID());

        std::vector<TaskPtr> ret;
        while (!Empty()) {
            Meta &meta = queue_.front();
            TaskPtr &t = meta.task;
            CLOG(DEBUG, "Worker") << *this << " Meta step=" << meta.step << " Query " << t << " " << *t;
            if (meta.step == 0) {
                CUDA_CALL(cudaEventRecord, meta.beg_event, stream_);
                meta.step++;
                CLOG(DEBUG, "Worker") << *this << " timer_started " << t << " " << *t;
            }

            if (meta.step == 1) {
                CLOG(DEBUG, "Worker") << *this << " try prepare data " << t << " " << *t;
                auto gputask = dynamic_cast<GPUTask *>(t.get());
                if (!gputask)
                    gputask = t->GetGPUTask();
                if (gputask) {
                    for (auto it = meta.task_metas.begin(); it != meta.task_metas.end();) {
                        auto &m = *it;
                        if (m.readable) {
                            if (!m.data->ReadAsync(this, device_))
                                return ret;
                        }
                        if (m.writable && m.data->Bytes() > 0) {
                            if (!m.data->WriteAsync(this, device_))
                                return ret;
                        }
                        CLOG(DEBUG, "Worker") << "s1 " << m;
                        it = meta.task_metas.erase(it);
                    }
                    // make sure data.currentarray is set to this device
                    for (auto &m : t->Metas()) {
                        if (m.readable) {
                            while (!m.data->ReadAsync(this, device_));
                        }
                        if (m.writable && m.data->Bytes() > 0) {
                            while (!m.data->WriteAsync(this, device_));
                        }
                        CLOG(DEBUG, "Worker") << "s2 " << m;
                    }
                    CLOG(INFO, "Worker") << *this << " Run " << t << " " << *t;
                    CUDA_CALL(cudaEventRecord, meta.transfer_event, stream_);
                    (*gputask)(GPUContext(GetDefaultMM(), gpu, stream_, this, t));
                } else {
                    CLOG(ERROR, "Worker") << *this << " RunOld " << t << " " << *t;
                    CUDA_CALL(cudaEventRecord, meta.transfer_event, stream_);
                    t->Run(this);
                }
                CUDA_CALL(cudaEventRecord, meta.end_event, stream_);
                meta.step++;
            }
            if (meta.step == 2) {
                if (!QueryEvent(meta.end_event))
                    return ret;

                float tranfer_ms, calc_ms;
                CUDA_CALL(cudaEventElapsedTime, &tranfer_ms, meta.beg_event, meta.transfer_event);
                CUDA_CALL(cudaEventElapsedTime, &calc_ms, meta.transfer_event, meta.end_event);
                CLOG(INFO, "Worker") << *this << " " << *meta.task << " transfer " << tranfer_ms << " ms, "
                                     << " calc "
                                     << calc_ms << " ms";

                events_unused_.push_back(meta.end_event);
                events_unused_.push_back(meta.beg_event);
                events_unused_.push_back(meta.transfer_event);
                meta.step++;
            }

            auto &mappings = meta.task->GetTempDataMappings();

            if (!mappings.empty()) {
                if (meta.step == 3) {
                    meta.tmp_arrs.resize(mappings.size());
                    for (size_t i = 0; i < mappings.size(); i++) {
                        auto &tmp_arr = mappings[i].first;
                        auto &data = mappings[i].second;
                        meta.tmp_arrs.emplace_back();
                        CUDA_CALL(cudaMemcpyAsync, &meta.tmp_arrs[i], tmp_arr.GetArrPtr(), sizeof(DeviceArrayBase),
                                  cudaMemcpyDefault,
                                  stream_);
                    }
                    CUDA_CALL(cudaEventRecord, meta.mapping_event, stream_);
                    meta.step++;
                }
                if (meta.step == 4) {
                    if (!QueryEvent(meta.mapping_event))
                        return ret;

                    for (size_t i = 0; i < mappings.size(); i++) {
                        auto &tmp_arr = mappings[i].first;
                        auto &data = mappings[i].second;

                        data->Create(meta.tmp_arrs[i].bytes, device_);
//                    printf("d.data() = %p m.ptr=%p m.bytes=%lu\n", data->data(), h_arr.ptr, h_arr.bytes);
                        run_copy_free_kernel(data->data(), meta.tmp_arrs[i].ptr, meta.tmp_arrs[i].bytes, stream_);
                    }
                    CUDA_CALL(cudaEventRecord, meta.mapping_event, stream_);
                    meta.step++;
                }
                if (meta.step == 5) {
                    if (!QueryEvent(meta.mapping_event))
                        return ret;
                    events_unused_.push_back(meta.mapping_event);
                    meta.step++;
                }
            }
            ret.push_back(meta.task);
            queue_.pop_front();
        }
        return ret;
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

    Event GPUWorker::Copy(Ptr dst, Ptr src, size_t bytes) {
        auto gpu = dynamic_cast<GPUDevice *>(device_);
        assert(gpu);
        return GPUCopy(dst, src, bytes, gpu->GPUID(), stream_);
    }

}
