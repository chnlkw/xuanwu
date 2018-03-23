//
// Created by chnlkw on 3/13/18.
//

#include <xuanwu/Xuanwu.h>
#include <xuanwu/Engine.h>

DevicePtr Xuanwu::GetCPUDevice() { return engine->CpuDevice(); }

Engine &Xuanwu::Get() { return *engine; }

TaskBase &Xuanwu::AddTask(TaskPtr task) {
    return engine->AddTask(task);
}

bool Xuanwu::Tick() {
    return engine->Tick();
}

std::shared_ptr<Engine> Xuanwu::engine;
