//
// Created by chnlkw on 3/13/18.
//

#ifndef DMR_CAR_H
#define DMR_CAR_H

#include "defs.h"

class Xuanwu {
private:
    static std::shared_ptr<Engine> engine;

public:
    static void Set(std::shared_ptr<Engine> e) { engine = e; }

    static bool Tick();

    static DevicePtr GetCPUDevice();

    static void Finish() { engine.reset(); }

    template<class Task, class... Args>
    static TaskBase &AddTask(Args &&... args) {
        auto t = std::make_shared<Task>(std::forward<Args>(args)...);
        return AddTask(t);
    }

    static TaskBase &AddTask(TaskPtr task);
};


#endif //DMR_CAR_H
