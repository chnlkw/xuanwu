//
// Created by chnlkw on 3/13/18.
//

#include "Xuanwu.h"
#include "Engine.h"

DevicePtr Xuanwu::GetCPUDevice() { return engine->CpuDevice(); }

TaskBase &Xuanwu::AddTask(TaskPtr task) {
    return engine->AddTask(task);
}

bool Xuanwu::Tick() {
    return engine->Tick();
}

std::shared_ptr<Engine> Xuanwu::engine;
