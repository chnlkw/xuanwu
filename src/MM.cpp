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
#include "easylogging++.h"

#define LG(x) CLOG(x, "MM")

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

#if 0
ArrayBasePtr MMImpl::MakeArrayBase(size_t bytes, DevicePtr device) {
    auto &cache = caches_[device];
    auto allocator = GetAllocatorByDevice(device);
    ArrayBasePtr arr = std::make_shared<ArrayBase>(bytes, allocator);
    if (arr->data()) return arr;
    //try release
    for (auto node = cache.array_lru.Last(); node; node = node->left) {
        if (auto p = node->val.lock()) {
            if (!p->Busy()) {
                p->Free();
                arr = std::make_shared<ArrayBase>(bytes, allocator);
                if (arr->data())
                    return arr;
            }
        }
    }
    return arr;
}
#endif

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
ArrayBasePtr MMBase::MakeArrayBase(size_t bytes, DevicePtr device) {
    auto &cache = caches_[device];
    auto allocator = GetAllocatorByDevice(device);
    auto ret = std::make_shared<ArrayBase>(bytes, allocator);
    if (ret->data())
        return ret;

    bool has_busy = false;
    LG(DEBUG) << "MakeArrayBase ret = " << *ret << ". need cache kicking";
    for (auto node = cache.array_lru.Last(); node; node = node->left) {
        if (auto arr = node->val.lock()) {
            LG(DEBUG) << "found " << *arr;
            if (!arr->Busy()) {
                LG(INFO) << "Kick " << *arr;
                arr->Free();
                ret = std::make_shared<ArrayBase>(bytes, allocator);
                if (ret->data()) {
                    LG(INFO) << "New Allocated " << *arr;
                    return ret;
                }
            } else
                has_busy = true;
        }
    }
    if (!has_busy) {
        LOG(FATAL) << "no array can be kicked";
    }
    return ret;
}

void MMBase::TryPop(DevicePtr dev, std::weak_ptr<ArrayBase> p) {
    caches_[dev].TryPop(std::move(p));
}

void MMBase::Push(DevicePtr dev, std::weak_ptr<ArrayBase> p) {
    LG(INFO) << "Push " << *dev << " : " << *p.lock();
    caches_[dev].Push(std::move(p));
}

void MMBase::Cache::TryPop(std::weak_ptr<ArrayBase> p) {
    auto it = mapping.find(p);
    if (it != mapping.end()) {
        LG(INFO) << "Pop " << *p.lock();
        array_lru.Delete(it->second);
    }
}

void MMBase::Cache::Push(std::weak_ptr<ArrayBase> p) {
    auto it = mapping.find(p);
    assert (it == mapping.end());
    mapping[p] = array_lru.Insert(std::move(p));
}
