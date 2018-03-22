//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_DEVICE_H
#define DMR_DEVICE_H

#include "DeviceBase.h"
#include <boost/di.hpp>

class CPUDevice : public DeviceBase {
public:
#ifdef USE_CUDA

    CPUDevice();

    void RunTask(TaskPtr t) override;

    size_t NumRunningTasks() const override { return 0; }

#else
    CPUDevice() : DeviceBase(-1, AllocatorPtr(new CPUAllocator)) { }
#endif

    int ScoreRunTask(TaskPtr t) override;
};

auto NumWorkersOfGPUDevices = [] {};

class GPUDevice : public DeviceBase {
    size_t running_tasks_ = 0;
public:
    BOOST_DI_INJECT (GPUDevice, std::unique_ptr<CudaAllocator> allocator, (named = NumWorkersOfGPUDevices)
            int num_workers = 1);

    void RunTask(TaskPtr t) override;

    size_t NumRunningTasks() const override { return running_tasks_; }

    int ScoreRunTask(TaskPtr t) override;
};

//class Device {
//    static DevicePtr current;
//    static DevicePtr cpu;
//public:
//    static DevicePtr Current() {
//        printf("current device = %d\n", current->Id());
//        return current;
//    }

//    static DevicePtr UseCPU() { return current = cpu; }

//    static DevicePtr CpuDevice() { return cpu; }

//    static void Use(DevicePtr dev) {
//        current = dev;
//    }

//    static int NumGPUs();

//};

#endif //DMR_DEVICE_H
