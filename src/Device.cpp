//
// Created by chnlkw on 1/25/18.
//

#include "Device.h"
#include "Worker.h"
#include "Allocator.h"
#include "Task.h"

#define LG(x) CLOG(x, "Device")

namespace Xuanwu {
    GPUDevice::GPUDevice(NumWorkers workers_per_gpu) :
            gpu_id_(GetGPUId()) {
        LG(INFO) << "Create GPUDevice " << this << " num_workers = " << workers_per_gpu << " ID = " << gpu_id_;
        for (int i = 0; i < workers_per_gpu; i++) {
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

    DeviceBase::DeviceBase() {
    }

    DeviceBase::~DeviceBase() {}

    CPUDevice::CPUDevice() :
            DeviceBase() {
        workers_.emplace_back(new CPUWorker(this));
        LG(INFO) << "Create CPUDevice " << this << " num_workers = "
                 << workers_.size() << " ID = ";
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
