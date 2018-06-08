//
// Created by chnlkw on 2/1/18.
//

#ifndef DMR_WORKER_H
#define DMR_WORKER_H

#include "defs.h"
#include "cuda_utils.h"
#include "Runnable.h"
#include "Ptr.h"
#include "Task.h"
#include <deque>
#include <condition_variable>
#include <thread>

namespace Xuanwu {
    class WorkerBase : public Runnable {
    protected:
        DevicePtr device_;
        size_t id_;
    public:
        explicit WorkerBase(DevicePtr d) : device_(d) {
            static size_t s_id = 0;
            id_ = s_id++;
        }

        DevicePtr Device() const {
            return device_;
        }

        virtual Event Copy(Ptr dst, Ptr src, size_t bytes) = 0;

        virtual void log(el::base::type::ostream_t &os) const;
    };


    class CPUWorker : public WorkerBase {
        bool finished_ = false;
        cudaStream_t stream_;
        std::deque<TaskPtr> tasks_;
        std::vector<TaskPtr> running_tasks_, finished_tasks_;
        std::mutex m_;
        std::condition_variable cv_;
        std::thread worker_thread_;
        std::pair<TaskPtr, std::chrono::time_point<std::chrono::high_resolution_clock> > start_time_;

        void start_worker();
    public:
        explicit CPUWorker(CPUDevice &cpu);

        ~CPUWorker();

        std::vector<TaskPtr> RunTasks(std::vector<TaskPtr> tasks) override;

        bool Empty() const override { return tasks_.empty(); }

        size_t NumRunningTasks() const override { return tasks_.size(); }

        Event Copy(Ptr dst, Ptr src, size_t bytes) override;
    };

    class GPUWorker : public WorkerBase {
        cudaStream_t stream_;

        std::vector<cudaEvent_t> events_unused_;

        struct Meta {
            cudaEvent_t beg_event, transfer_event, end_event, mapping_event;
            TaskPtr task;
            std::vector<::Xuanwu::TaskBase::Meta> task_metas;
            std::vector<DeviceArrayBase> tmp_arrs;
            int step = 0;
        };
        std::deque<Meta> preparing_queue_, running_queue_;

    public:
        explicit GPUWorker(GPUDevice *gpu);

        bool Empty() const override {
            return preparing_queue_.empty() && running_queue_.empty();
        }

        const cudaStream_t &Stream() const {
            return stream_;
        }

    private:

        std::vector<TaskPtr> RunTasks(std::vector<TaskPtr> tasks) override;

        size_t NumRunningTasks() const override;

        cudaEvent_t GetEvent();

        Event Copy(Ptr dst, Ptr src, size_t bytes) override;

    };

}

#endif //DMR_WORKER_H
