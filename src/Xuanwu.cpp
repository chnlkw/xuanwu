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

    TaskBase &AddTask(TaskPtr task) {
        return Xuanwu::GetXuanwu()->GetEngine()->AddTask(task);
    }

    bool Tick() {
        return Xuanwu::GetXuanwu()->GetEngine()->Tick();
    }
}
