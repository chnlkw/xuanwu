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
        std::vector<std::shared_ptr<DeviceBase>> all_devices_;
        std::shared_ptr<CPUDevice> device;
        WorkerPtr worker;
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

        template<class Dev>
        Dev *GetDevice(int order) {
            std::vector<Dev*> devs;
            for (auto & dev : all_devices_) {
                Dev* d = dynamic_cast<Dev*>(dev.get());
                if (d) {
                    devs.push_back(d);
                }
            }
            if (devs.size())
                return devs[order % devs.size()];
            else
                return nullptr;
        }

    };

    template<class Dev>
    Dev *GetDevice(int order) {
        return Xuanwu::GetXuanwu()->GetDevice<Dev>(order);
    }
}

#endif //XUANWU_XUANWU_H
