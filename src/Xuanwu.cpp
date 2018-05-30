//
// Created by chnlkw on 3/13/18.
//

#include "Xuanwu.h"
#include "Engine.h"
#include "Worker.h"
#include <boost/di.hpp>
#include <memory>
#include <utility>

namespace Xuanwu {

    Engine *Xuanwu::GetEngine() {
        return e.get();
    }

    Xuanwu *Xuanwu::xw;

    Xuanwu *Xuanwu::GetXuanwu() {
        return xw;
    }

    MMBase *Xuanwu::GetMM() {
        return mm.get();
    }

    AllocatorPtr Xuanwu::GetAllocator() {
        return allocator.get();
    }

    WorkerPtr Xuanwu::GetWorker() {
        return worker;
    }

    DevicePtr Xuanwu::GetDevice() {
        return device.get();
    }

    Xuanwu::Xuanwu(std::shared_ptr<MMBase> mm, std::unique_ptr<Engine> engine) :
            mm(std::move(mm)),
            e(std::move(engine)) {
        allocator.reset(new CPUAllocator());
        xw = this;
        all_devices_ = e->GetDevices();
        for (auto &d : all_devices_) {
            if (auto p = std::dynamic_pointer_cast<CPUDevice>(d)) {
                device = p;
                break;
            }
        }
        if (!device)
            device = std::make_shared<CPUDevice>();
        worker = &device->GetWorker();
        add_task_func_ = [this](TaskPtr t) {
            GetEngine()->AddTask(std::move(t));
        };
    }

    void Xuanwu::AddTask(TaskPtr t) {
        add_task_func_(std::move(t));
    }

    add_task_func_t Xuanwu::SetAddTaskFunc(add_task_func_t f) {
        auto ret = add_task_func_;
        add_task_func_ = f;
        return ret;
    }

    DevicePtr GetDefaultDevice() {
        return Xuanwu::GetXuanwu()->GetDevice();
    }

    WorkerPtr GetDefaultWorker() {
        return Xuanwu::GetXuanwu()->GetWorker();

    }

    AllocatorPtr GetDefaultAllocator() {
        return Xuanwu::GetXuanwu()->GetAllocator();
    }

    MMBase *GetDefaultMM() {
        return Xuanwu::GetXuanwu()->GetMM();
    }

    void AddTask(TaskPtr task) {
        Xuanwu::GetXuanwu()->AddTask(std::move(task));
    }

    bool Tick() {
        return Xuanwu::GetXuanwu()->GetEngine()->Tick();
    }
}
