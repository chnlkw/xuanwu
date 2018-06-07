//
// Created by chnlkw on 1/25/18.
//

#include "Device.h"
#include "Worker.h"
#include "Allocator.h"
#include "Task.h"

#define LG(x) CLOG(x, "Device")

namespace Xuanwu {
    GPUDevice::GPUDevice(NumWorkers workers_per_gpu, HeapLimit heap_limit) :
            gpu_id_(GetGPUId()) {
        LG(INFO) << "Create GPUDevice " << this << " num_workers = " << workers_per_gpu << " ID = " << gpu_id_;
        for (int i = 0; i < workers_per_gpu; i++) {
            workers_.emplace_back(new GPUWorker(this));
        }
        CUDA_CALL(cudaSetDevice, gpu_id_);
        CUDA_CALL(cudaDeviceSetLimit, cudaLimitMallocHeapSize, heap_limit);
        CUDA_CALL(cudaGetDeviceProperties, &deviceProp, gpu_id_);
    }

    DeviceBase::DeviceBase() {
        scheduler_ = std::make_unique<Scheduler>();
        scheduler_->SetSelector([this](TaskPtr task) -> Runnable * {
            return (Runnable *) this->ChooseRunnable(workers_.begin(), workers_.end()).get();
        });
    }

    DeviceBase::~DeviceBase() {}

    CPUDevice::CPUDevice() :
            DeviceBase() {
        for (int i = 0; i < 4; i++)
            workers_.emplace_back(new CPUWorker(*this));
        LG(INFO) << "Create CPUDevice " << this << " num_workers = " << workers_.size();
    }

    int DeviceBase::ScoreRunTask(TaskPtr t) {
        return 0;
    }

    size_t DeviceBase::NumRunningTasks() const {
        size_t ret = 0;
        for (auto &w : workers_)
            ret += w->NumRunningTasks();
        return ret;
    }

    std::vector<TaskPtr> DeviceBase::RunTasks(std::vector<TaskPtr> tasks) {
        std::vector<TaskPtr> ret;

        for (auto &t : tasks) {
            LG(INFO) << *this << " Run " << *t;
            scheduler_->AddTask(t);
        }

        bool first = true;
        auto ready_tasks = scheduler_->FetchReadyTasks();
        while (first || !ready_tasks.empty()) {
            first = false;
            std::map<Runnable *, std::vector<TaskPtr>> r;
            for (auto &w : workers_)
                r[w.get()] = {};
            for (auto &p : ready_tasks) {
                r[p.second].push_back(p.first);
                scheduler_->RunTask(p.first);
            }
            for (auto &r_ts : r) {
                auto complete_tasks = r_ts.first->RunTasks(std::move(r_ts.second));
                scheduler_->FinishTasks(complete_tasks);
                Append(ret, std::move(complete_tasks));
            }
            ready_tasks = scheduler_->FetchReadyTasks();
        }
        return ret;
    }

    int CPUDevice::ScoreRunTask(TaskPtr t) {
        CPUTask *c = dynamic_cast<CPUTask *>(t.get());
        if (!c)
            c = t->GetCPUTask();
        return c ? c->score : -10;
    }

    int GPUDevice::ScoreRunTask(TaskPtr t) {
        GPUTask *c = dynamic_cast<GPUTask *>(t.get());
        if (!c)
            c = t->GetGPUTask();
        return c ? c->score : -10;
    }
}
