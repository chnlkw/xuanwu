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

    void CPUWorker::RunTask(TaskPtr t) {
//        {
//            std::unique_lock<std::mutex> lk(m_);
        tasks_.push_back(t);
//        }
//        cv_.notify_one();
    }

    CPUWorker::CPUWorker(CPUDevice &cpu) :
            WorkerBase(&cpu),
            worker_thread_([this]() { start_worker(); }) {
        CUDA_CALL(cudaStreamCreate, &stream_);
    }

    std::vector<TaskPtr> CPUWorker::GetCompleteTasks() {

        auto prepare = [&](TaskPtr t) -> bool {
            auto cputask = dynamic_cast<CPUTask *>(t.get());
            if (!cputask)
                cputask = t->GetCPUTask();
            if (cputask) {
                for (auto &m : t->Metas()) {
                    if (m.readable) {
                        if (!m.data->ReadAsync(this, device_))
                            return false;

                    }
                    if (m.writable && m.data->Bytes() > 0) {
                        if (!m.data->WriteAsync(this, device_))
                            return false;
                    }
                    CLOG(DEBUG, "CPUWorker prepare OK ") << m;
                }
                {
                    std::unique_lock<std::mutex> lk(m_);
                    running_tasks_.push_back(t);
                }
                cv_.notify_one();
            } else
                t->Run(this);
            return true;
        };

        for (auto it = tasks_.begin(); it != tasks_.end();) {
            if (prepare(*it)) {
                it = tasks_.erase(it);
            } else {
                ++it;
            }
        }

        std::unique_lock<std::mutex> lk(m_);
        std::vector<TaskPtr> ret = std::move(finished_tasks_);
        finished_tasks_.clear();
        return ret;
    }

    Event CPUWorker::Copy(Ptr dst, Ptr src, size_t bytes) {
        return CPUCopy(dst, src, bytes, stream_);
    }

    void CPUWorker::start_worker() {
        auto cpu = dynamic_cast<CPUDevice *>(device_);
        assert(cpu);

        std::unique_lock<std::mutex> lk(m_);
        while (true) {
            cv_.wait(lk, [this] { return finished_ || !running_tasks_.empty(); });
            if (finished_)
                return;
            auto tasks = std::move(running_tasks_);
            running_tasks_.clear();
            lk.unlock();

            for (TaskPtr t : tasks) {
                Clock clk;

                CLOG(INFO, "Worker") << *this << " Run " << *t;

                auto cputask = dynamic_cast<CPUTask *>(t.get());
                if (!cputask)
                    cputask = t->GetCPUTask();
                assert(cputask);

                (*cputask)(CPUContext(cpu, this));

                CLOG(INFO, "Worker") << *this << " " << *t << " uses "
                                     << clk.timeElapsed() << " seconds";
            }

            lk.lock();
            for (TaskPtr t : tasks) {
                finished_tasks_.push_back(t);
            }
        }
    }

    CPUWorker::~CPUWorker() {
        {
            std::unique_lock<std::mutex> lk(m_);
            finished_ = true;
        }
        cv_.notify_all();
        worker_thread_.join();
    }

    GPUWorker::GPUWorker(GPUDevice *gpu) :
            WorkerBase(gpu) {
        CLOG(INFO, "Worker") << "Create GPU Worker with device = " << gpu->GPUID();
        CUDA_CALL(cudaSetDevice, gpu->GPUID());
        CUDA_CALL(cudaStreamCreate, &stream_);
    }

    void GPUWorker::RunTask(TaskPtr t) {
        CLOG(INFO, "Worker") << *this << " Run " << *t;

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
                        CLOG(DEBUG, "Worker") << "s11 " << m;
                        if (m.readable) {
                            if (m.data->ReadAsync(this, device_)->Busy())
                                return ret;
                        }
                        if (m.writable && m.data->Bytes() > 0) {
                            if (m.data->WriteAsync(this, device_)->Busy())
                                return ret;
                        }
                        CLOG(DEBUG, "Worker") << "s12 " << m;
                        it = meta.task_metas.erase(it);
                    }
                    // make sure data.currentarray is set to this device
                    for (auto &m : t->Metas()) {
                        CLOG(DEBUG, "Worker") << "s21 " << m;
                        if (m.readable) {
                            while (m.data->ReadAsync(this, device_)->Busy()) {}
                        }
                        if (m.writable && m.data->Bytes() > 0) {
                            while (m.data->WriteAsync(this, device_)->Busy()) {}
                        }
                        CLOG(DEBUG, "Worker") << "s22 " << m;
                    }
                    CLOG(INFO, "Worker") << *this << " prepared ok " << *t;
                    CUDA_CALL(cudaEventRecord, meta.transfer_event, stream_);
                    (*gputask)(GPUContext(GetDefaultMM(), gpu, stream_, this, t));
                } else {
                    CLOG(ERROR, "Worker") << *this << " RunOld " << *t;
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
            for (auto &m : meta.task->Metas()) {
                if (m.data->GetPinnedDevice().first) {
                    bool finished = m.data->ReadAsync(this, m.data->GetPinnedDevice().first);
                    LG(DEBUG) << *this << " write back " << *m.data << " to dev " << m.data->GetPinnedDevice().first
                              << " finished=" << finished;
                    m.data->GetMM()->GetCache(device_).Push(m.data->CurrentArray());
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
