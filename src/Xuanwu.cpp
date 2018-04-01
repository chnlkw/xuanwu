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

    Xuanwu* Xuanwu::xw;

    Xuanwu* Xuanwu::GetXuanwu() {
        return xw;
    }

    MMBase *Xuanwu::GetMM() {
        return mm.get();
    }

    AllocatorPtr Xuanwu::GetAllocator() {
        return allocator.get();
    }

    WorkerPtr Xuanwu::GetWorker() {
        return worker.get();
    }

    DevicePtr Xuanwu::GetDevice() {
        return device.get();
    }

    Xuanwu::Xuanwu(std::shared_ptr<MMBase> mm, std::unique_ptr<Engine> e) :
            mm(std::move(mm)),
            e(std::move(e)) {
        allocator.reset(new CPUAllocator());
        device = std::make_unique<CPUDevice>();
        worker = std::make_unique<CPUWorker>((CPUDevice *) device.get());
        xw = this;
    }

    DevicePtr GetDefaultDevice() {
        Xuanwu::GetXuanwu()->GetDevice();
    }

    WorkerPtr GetDefaultWorker() {
        Xuanwu::GetXuanwu()->GetWorker();

    }

    AllocatorPtr GetDefaultAllocator() {
        Xuanwu::GetXuanwu()->GetAllocator();
    }

    MMBase *GetDefaultMM() {
        Xuanwu::GetXuanwu()->GetMM();
    }

    TaskBase &AddTask(TaskPtr task) {
        return Xuanwu::GetXuanwu()->GetEngine()->AddTask(task);
    }

    bool Tick() {
        return Xuanwu::GetXuanwu()->GetEngine()->Tick();
    }
}
