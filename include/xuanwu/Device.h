//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_DEVICE_H
#define DMR_DEVICE_H

#include "DeviceBase.h"
#include "Config.h"
namespace Xuanwu {
    class CPUDevice : public DeviceBase {
    public:

        CPUDevice();

        void RunTask(TaskPtr t) override;

        size_t NumRunningTasks() const override { return 0; }

        int ScoreRunTask(TaskPtr t) override;
    };

    class GPUDevice : public DeviceBase {
        size_t running_tasks_ = 0;
    public:
        GPUDevice(std::unique_ptr<CudaAllocator> allocator, const Config &cfg);

        void RunTask(TaskPtr t) override;

        size_t NumRunningTasks() const override { return running_tasks_; }

        int ScoreRunTask(TaskPtr t) override;
    };

}
#endif //DMR_DEVICE_H
