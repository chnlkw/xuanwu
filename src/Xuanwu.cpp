//
// Created by chnlkw on 3/13/18.
//

#include "Xuanwu.h"
#include "Engine.h"

namespace Xuanwu {

    DevicePtr GetCPUDevice() { return engine->CpuDevice(); }

    TaskBase &AddTask(TaskPtr task) {
        return engine->AddTask(task);
    }

    bool Tick() {
        return engine->Tick();
    }

    void Set(std::shared_ptr<Engine> e) { engine = e; }

    void Finish() { engine.reset(); }

    std::shared_ptr<Engine> engine;
}
