//
// Created by chnlkw on 3/13/18.
//

#ifndef XUANWU_XUANWU_H
#define XUANWU_XUANWU_H

#include "defs.h"

namespace Xuanwu {

    bool Tick();

    DevicePtr GetDefaultDevice();

    WorkerPtr GetDefaultWorker();

    AllocatorPtr GetDefaultAllocator();

    MMBase *GetDefaultMM();

    void Finish();

    template<class Task, class... Args>
    TaskBase &AddTask(Args &&... args) {
        auto t = std::make_shared<Task>(std::forward<Args>(args)...);
        return AddTask(t);
    }

    TaskBase &AddTask(TaskPtr task);

    class Xuanwu {
        std::unique_ptr<DeviceBase> device;
        std::unique_ptr<WorkerBase> worker;
        std::shared_ptr<AllocatorBase> allocator;
        std::shared_ptr<MMBase> mm;
        std::unique_ptr<Engine> e;
        static Xuanwu *xw;
    public:
        Xuanwu(std::shared_ptr<MMBase> mm, std::unique_ptr<Engine> e);

        Engine *GetEngine();

        DevicePtr GetDevice();

        WorkerPtr GetWorker();

        AllocatorPtr GetAllocator();

        MMBase *GetMM();

        static Xuanwu* GetXuanwu();

    };

}

#endif //XUANWU_XUANWU_H
