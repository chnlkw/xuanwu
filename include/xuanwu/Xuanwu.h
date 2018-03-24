//
// Created by chnlkw on 3/13/18.
//

#ifndef XUANWU_XUANWU_H
#define XUANWU_XUANWU_H

#include "defs.h"

namespace Xuanwu {

    extern std::shared_ptr<Engine> engine;

    void Set(std::shared_ptr<Engine> e);

    bool Tick();

    DevicePtr GetCPUDevice();

    void Finish();

    template<class Task, class... Args>
    TaskBase &AddTask(Args &&... args) {
        auto t = std::make_shared<Task>(std::forward<Args>(args)...);
        return AddTask(t);
    }

    TaskBase &AddTask(TaskPtr task);

}

#endif //XUANWU_XUANWU_H
