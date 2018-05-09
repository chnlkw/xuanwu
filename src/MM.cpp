//
// Created by chnlkw on 3/27/18.
//

#include "MM.h"

#include <utility>
#include "Allocator.h"
#include "Array.h"
#include "Device.h"
#include "Data.h"
#include "Worker.h"

using namespace Xuanwu;

AllocatorPtr MMImpl::GetFrom(AllocatorFactoryBase *factory, DevicePtr dev) {
    auto it = allocator_pool_.find(dev);
    if (it == allocator_pool_.end()) {
        it = allocator_pool_.emplace(dev, factory->Create(dev)).first;
    }
    return it->second.get();
}

DataBasePtr MMImpl::MakeDataBase(size_t size) {
    return std::make_shared<DataImpl>(this, size);
}

ArrayBasePtr MMImpl::MakeArrayBase(size_t bytes, DevicePtr device) {
    auto &cache = caches_[device];
    auto allocator = GetAllocatorByDevice(device);
    ArrayBasePtr arr = std::make_shared<ArrayBase>(bytes, GetAllocatorByDevice(device));
    if (arr->data()) return arr;
    //try release
    for (auto node = cache.array_lru.Last(); node; node = node->left) {
        if (auto p = node->val.lock()) {
            if (!p->Busy()) {
                p->Free();
                arr = std::make_shared<ArrayBase>(bytes, GetAllocatorByDevice(device));
                if (arr->data())
                    return arr;
            }
        }
    }
    return arr;
}

#if 0
AllocatorPtr Xuanwu::MyMM::GetAllocatorByDevice(DevicePtr device) {
    if (dynamic_cast<CPUDevice *>(device)) {
        return GetFrom(cpu_factory_.get(), device);
    } else if (dynamic_cast<GPUDevice *>(device)) {
        return GetFrom(gpu_factory_.get(), device);
    }
    return {};

}

MyMM::MyMM(AllocatorFactoryPtr cpu_factory, AllocatorFactoryPtr gpu_factory) :
        cpu_factory_(std::move(cpu_factory)),
        gpu_factory_(std::move(gpu_factory)) {}

DataBasePtr MyMM::MakeDataBase(size_t size) {
    return std::make_shared<DataImpl>(this, size);
}

#endif

//ArrayBasePtr MMBase::ReadData(DataBasePtr data, WorkerPtr worker) {
//    ReadData(data, worker, worker->Device());
//}

//ArrayBasePtr MMBase::WriteData(DataBasePtr data, WorkerPtr worker) {
//    WriteData(data, worker, worker->Device());
//}
