//
// Created by chnlkw on 11/21/17.
//

#ifndef XUANWU_DEFS_H
#define XUANWU_DEFS_H

#include <memory>
#include "easylogging++.h"

namespace Xuanwu {
    class ArrayBase;

    template<class T>
    class Array;

    class AllocatorBase;

    class DeviceBase;

    class Node;

    class DataBase;

    template<class T>
    class Data;

    class TaskBase;

    class WorkerBase;

    class CPUTask;

    class GPUTask;

    class CPUWorker;

    class GPUWorker;

    class Engine;

    class GPUDevice;

    class CPUDevice;

    using DevicePtr = DeviceBase *;
    using WorkerPtr = WorkerBase *;

    class CudaAllocator;

    class AllocatorFactoryBase;

    class PtrBase;

    class MMBase;

    using AllocatorFactoryPtr = std::shared_ptr<AllocatorFactoryBase>;
    using ArrayBasePtr = std::unique_ptr<ArrayBase>;
    template<class T>
    using ArrayPtr = std::unique_ptr<Array<T>>;
    using AllocatorPtr = AllocatorBase *;
    using NodePtr = std::shared_ptr<Node>;
    using DataBasePtr = std::shared_ptr<DataBase>;
    using TaskPtr = std::shared_ptr<TaskBase>;

    namespace detail {
        template<class U, class V>
        struct As {
            using type = V;
        };
    }
    template<class Device>
    class Allocator;

    template<class Device>
    class AllocatorFactory;

    class Devices; // bind to tuple of Devices

    class DeviceArrayBase;

    class LocalArrayGPU;

}
#endif //XUANWU_DEFS_H
