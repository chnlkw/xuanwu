//
// Created by chnlkw on 1/25/18.
//

#include "Device.h"
#include "Worker.h"
#include "Allocator.h"
#include "Task.h"

#define LG(x) CLOG(x, "Device")

namespace Xuanwu {
    GPUDevice::GPUDevice(std::unique_ptr<CudaAllocator> allocator, const Config &cfg) :
            DeviceBase(std::move(allocator)) {
        LG(INFO) << "Create GPUDevice " << this << " with allocator " << GetAllocator() << " num_workers = "
                 << cfg.num_workers_per_gpu << " ID = " << this->Id();
        for (int i = 0; i < cfg.num_workers_per_gpu; i++) {
            workers_.emplace_back(new GPUWorker(this));
        }
    }

    std::vector<TaskPtr> DeviceBase::GetCompleteTasks() {
        std::vector<TaskPtr> ret;
        for (auto &w : workers_) {
            auto r = w->GetCompleteTasks();
            ret.insert(ret.end(), r.begin(), r.end());
        }
        return ret;
    }

    void GPUDevice::RunTask(TaskPtr t) {
        running_tasks_++;
        ChooseRunnable(workers_.begin(), workers_.end())->RunTask(t);
    }

    DeviceBase::DeviceBase(std::unique_ptr<AllocatorBase> allocator) :
            allocator_(std::move(allocator)) {
    }

    DeviceBase::~DeviceBase() {}

    int DeviceBase::Id() const { return allocator_->Id(); }

    CPUDevice::CPUDevice() :
            DeviceBase(std::make_unique<CudaPreAllocator>(-1, 8LU << 30)) {
        workers_.emplace_back(new CPUWorker(this));
        LG(INFO) << "Create CPUDevice " << this << " with allocator " << GetAllocator() << " num_workers = "
                 << workers_.size() << " ID = " << this->Id();
    }

    void CPUDevice::RunTask(TaskPtr t) {
        workers_.at(0)->RunTask(t);
        LG(INFO) << "CPUDevice run task " << *t;
    }

    int DeviceBase::ScoreRunTask(TaskPtr t) {
        return 0;
    }

    int CPUDevice::ScoreRunTask(TaskPtr t) {
        CPUTask *c = dynamic_cast<CPUTask *>(t.get());
        if (!c)
            c = t->GetCPUTask();
        return c ? c->score : 0;
    }

    int GPUDevice::ScoreRunTask(TaskPtr t) {
        GPUTask *c = dynamic_cast<GPUTask *>(t.get());
        if (!c)
            c = t->GetGPUTask();
        return c ? c->score : 0;
    }
}
