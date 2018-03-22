//
// Created by chnlkw on 2/1/18.
//

#ifndef DMR_WORKER_H
#define DMR_WORKER_H

#include "defs.h"
#include "cuda_utils.h"
#include "Runnable.h"
#include <deque>

class WorkerBase : public el::Loggable, public Runnable {
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

    virtual void log(el::base::type::ostream_t &os) const;
};


class CPUWorker : public WorkerBase {
    std::deque<TaskPtr> tasks_;
public:
    explicit CPUWorker(CPUDevice *cpu);

    void RunTask(TaskPtr t) override {
        tasks_.push_back(t);
    }

    bool Empty() const override { return tasks_.empty(); }

    std::vector<TaskPtr> GetCompleteTasks() override;

    size_t NumRunningTasks() const override { return tasks_.size(); }
};

class GPUWorker : public WorkerBase {
    cudaStream_t stream_;

    std::vector<cudaEvent_t> events_unused_;

    struct Meta {
        cudaEvent_t beg_event, transfer_event, end_event;
        TaskPtr task;
    };
    std::deque<Meta> queue_;

public:
    explicit GPUWorker(GPUDevice *gpu);

    bool Empty() const override {
        return queue_.empty();
    }

    const cudaStream_t &Stream() const {
        return stream_;
    }

private:

    void RunTask(TaskPtr t) override;

    size_t NumRunningTasks() const override {
        return queue_.size();
    }

    cudaEvent_t GetEvent();

    std::vector<TaskPtr> GetCompleteTasks() override;

};


#endif //DMR_WORKER_H
