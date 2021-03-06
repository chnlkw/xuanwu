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

    CPUWorker::CPUWorker(CPUDevice &cpu) :
            WorkerBase(&cpu),
            worker_thread_([this]() { start_worker(); }) {
        CUDA_CALL(cudaSetDevice, 0);
        CUDA_CALL(cudaStreamCreate, &stream_);
    }

    Event CPUWorker::Copy(Ptr dst, Ptr src, size_t bytes) {
        CUDA_CALL(cudaSetDevice, 0);
        LG(INFO) << *this << " copy " << src << " to " << dst << " bytes=" << bytes;
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

                LG(INFO) << *this << *t << " Run";

                auto cputask = dynamic_cast<CPUTask *>(t);
                if (!cputask)
                    cputask = t->GetCPUTask();
                assert(cputask);

                (*cputask)(CPUContext(cpu, this));

                LG(INFO) << *this << " " << *t << " uses "
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

    std::vector<TaskPtr> CPUWorker::RunTasks(std::vector<TaskPtr> tasks) {
        std::vector<TaskPtr> ret;
        for (auto &t : tasks)
            LG(INFO) << *this << " Run Task " << *t;
        for (auto t : tasks) {
            task_metas_.emplace_back(t, PrepareMeta(t));
        }

        auto prepare = [&](TaskPtr t, PrepareMeta &prepare) -> bool {
            if (start_time_.first != t) {
                start_time_.first = t;
                start_time_.second = std::chrono::high_resolution_clock::now();
            }
            auto cputask = dynamic_cast<CPUTask *>(t);
            if (!cputask)
                cputask = t->GetCPUTask();
            if (cputask) {
                if (!prepare.Prepare(this, device_)) {
                    return false;
                }
                /*
                for (auto &m : t->Metas())
                    if (!m.remote) {
                        if (m.readable && !m.remote) {
                            if (!m.data->ReadAsync(this, dev_gpu_))
                                return false;

                        }
                        if (m.writable && m.data->Bytes() > 0 && !m.remote) {
                            if (!m.data->WriteAsync(this, dev_gpu_))
                                return false;
                        }
                    }
                    */
                std::chrono::duration<double> elapsed = std::chrono::high_resolution_clock::now() - start_time_.second;
                LG(INFO) << *this << " Prepare OK " << *t << " " << elapsed.count() * 1000 << " ms";
                Timeline::Add(id_, elapsed.count() * 1000, 0);
                {
                    std::unique_lock<std::mutex> lk(m_);
                    running_tasks_.push_back(t);
                }
                cv_.notify_one();
            } else
                t->Run(this);
            return true;
        };

        while (!task_metas_.empty()) {
            auto &tm = task_metas_.front();
            if (prepare(tm.first, tm.second)) {
                task_metas_.pop_front();
            } else {
                break;
            }
        }

        std::unique_lock<std::mutex> lk(m_);
        ret = std::move(finished_tasks_);
        finished_tasks_.clear();
        return ret;
    }

    GPUWorker::GPUWorker(GPUDevice *gpu) :
            WorkerBase(gpu) {
        CUDA_CALL(cudaSetDevice, gpu->GPUID());
        CUDA_CALL(cudaStreamCreate, &stream_);
        LG(INFO) << "Create GPU Worker with device = " << gpu->GPUID() << " stream = " << stream_;
    }

    std::vector<TaskPtr> GPUWorker::RunTasks(std::vector<TaskPtr> tasks) {
        auto gpu = dynamic_cast<GPUDevice *>(device_);
        assert(gpu);
        CUDA_CALL(cudaSetDevice, gpu->GPUID());

        bool query_complete = tasks.empty();

        for (auto &t : tasks) {
            LG(INFO) << *this << *t << " Run ";
            Meta meta{GetEvent(), GetEvent(), GetEvent(), GetEvent(), t, PrepareMeta(t)};
            preparing_queue_.push_back(meta);
        }
        auto GetCompleteTasks = [&]() {
            std::vector<TaskPtr> ret;
            if (!query_complete)
                return ret;
            while (!running_queue_.empty()) {
                Meta &meta = running_queue_.front();
                TaskPtr &t = meta.task;
                assert (meta.step == 2);
                auto err = cudaEventQuery(meta.end_event);
                if (err == cudaErrorNotReady)
                    return ret;
                if (err != cudaSuccess) {
                    LOG(ERROR) << "run " << *t << " Error code = " << cudaGetErrorString(cudaGetLastError());
                    CUDA_CHECK();
                    abort();
                }

                float tranfer_ms, calc_ms;
                CUDA_CALL(cudaEventElapsedTime, &tranfer_ms, meta.beg_event, meta.transfer_event);
                CUDA_CALL(cudaEventElapsedTime, &calc_ms, meta.transfer_event, meta.end_event);
                CLOG(INFO, "Worker") << *this << " " << *meta.task << " transfer " << tranfer_ms << " ms, "
                                     << " calc "
                                     << calc_ms << " ms";
                Timeline::Add(id_, tranfer_ms, calc_ms);

                events_unused_.push_back(meta.end_event);
                events_unused_.push_back(meta.beg_event);
                events_unused_.push_back(meta.transfer_event);
                meta.step++;
                ret.push_back(std::move(t));
                running_queue_.pop_front();
            }
            return ret;
        };
        while (!preparing_queue_.empty()) {
            Meta &meta = preparing_queue_.front();
            TaskPtr &t = meta.task;
//            CLOG(DEBUG, "Worker") << *this << " Meta step=" << meta.step << " Query " << t << " " << *t;
            if (meta.step == 0) {
                CUDA_CALL(cudaEventRecord, meta.beg_event, stream_);
                meta.step++;
                CLOG(DEBUG, "Worker") << *this << " timer_started " << " " << *t;
            }

            if (meta.step == 1) {
                CLOG(DEBUG, "Worker") << *this << " try prepare data " << " " << *t;
                auto gputask = dynamic_cast<GPUTask *>(t);
                if (!gputask)
                    gputask = t->GetGPUTask();
                if (gputask) {
                    if (!meta.prepare.Prepare(this, device_)) {
                        return GetCompleteTasks();
                    }

#ifndef NDEBUG
                    // make sure data.currentarray is set to this device
                    for (auto &m : t->Metas()) {
                        if (m.readable && !m.remote) {
                            while (!m.data->ReadAsync(this, device_)) {
                                CLOG(INFO, "Worker") << " unexpected wait" << *m.data;

                            }
                        }
                        if (m.writable && m.data->Bytes() > 0 && !m.remote) {
                            while (!m.data->WriteAsync(this, device_)) {
                                CLOG(INFO, "Worker") << " unexpected wait" << *m.data;
                            }
                        }
                    }
#endif
                    CLOG(INFO, "Worker") << " " << *this << *t << " Prepare OK ";
                    CUDA_CALL(cudaEventRecord, meta.transfer_event, stream_);
                    (*gputask)(GPUContext(GetDefaultMM(), gpu, stream_, this, t));
                } else {
                    CLOG(ERROR, "Worker") << " " << *this << *t << " RunOld ";
                    CUDA_CALL(cudaEventRecord, meta.transfer_event, stream_);
                    t->Run(this);
                }
                CUDA_CALL(cudaEventRecord, meta.end_event, stream_);
                meta.step++;
            }
            running_queue_.push_back(std::move(preparing_queue_.front()));
            preparing_queue_.pop_front();

        }
        return GetCompleteTasks();
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
        LG(INFO) << *this << " copy " << src << " to " << dst << " bytes=" << bytes;
        return GPUCopy(dst, src, bytes, gpu->GPUID(), stream_);
    }

    size_t GPUWorker::NumRunningTasks() const {
        return preparing_queue_.size() + running_queue_.size();
    }

    bool PrepareMeta::Prepare(WorkerPtr worker, DevicePtr dev) {
        for (auto it = task_metas.begin(); it != task_metas.end();) {
            auto &m = *it;
            if (m.readable && !m.remote) {
                if (!m.data->ReadAsync(worker, dev)) {
                    ++it;
                    CLOG(DEBUG, "Worker") << *worker << " NOT prepared " << *m.data;
                    continue;
                }
            }
            if (m.writable && m.data->Bytes() > 0 && !m.remote) {
                if (!m.data->WriteAsync(worker, dev)) {
                    ++it;
                    CLOG(DEBUG, "Worker") << *worker << " NOT prepared " << *m.data;
                    continue;
                }
            }
            CLOG(DEBUG, "Worker") << *worker << " prepared " << *m.data;
            it = task_metas.erase(it);
        }
        return task_metas.empty();
    }

    PrepareMeta::PrepareMeta(TaskPtr t) : task_metas(t->Metas()) {
    }
}
