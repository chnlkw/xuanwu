//
// Created by chnlkw on 3/27/18.
//

#ifndef XUANWU_MM_H
#define XUANWU_MM_H

#include "defs.h"
#include <map>
#include "lru.h"

namespace Xuanwu {
    class MMBase {
        std::map<AllocatorPtr, DataBasePtr> need_write_back_;
    public:
        virtual AllocatorPtr GetAllocatorByDevice(DevicePtr device) = 0;

        virtual DataBasePtr MakeDataBase(size_t size) = 0;

        template<class T>
        Data <T> MakeData(size_t count) {
            return Data<T>(count, this);
        }

        virtual ArrayBasePtr MakeArrayBase(size_t bytes, DevicePtr device);

        struct Cache {
            using Lru = LRU<std::weak_ptr<ArrayBase>>;
            Lru array_lru;
            std::map<std::weak_ptr<ArrayBase>, Lru::Node *, std::owner_less<std::weak_ptr<ArrayBase>>> mapping;

            void TryPop(std::weak_ptr<ArrayBase> p);

            void Push(std::weak_ptr<ArrayBase> p);
        };

        std::map<DevicePtr, Cache> caches_;

        void TryPop(DevicePtr dev, std::weak_ptr<ArrayBase> p);
        void Push(DevicePtr dev, std::weak_ptr<ArrayBase> p);
    };

    class MMImpl : public MMBase {
        std::map<DevicePtr, std::unique_ptr<AllocatorBase>> allocator_pool_;

    public:
        AllocatorPtr GetFrom(AllocatorFactoryBase *factory, DevicePtr dev);

        DataBasePtr MakeDataBase(size_t size) override;

//        ArrayBasePtr MakeArrayBase(size_t bytes, DevicePtr device) override;

    };

    template<class ...Devices>
    class MMMultiDevice;

    template<>
    class MMMultiDevice<> : public MMImpl {
    public:
        AllocatorPtr GetAllocatorByDevice(DevicePtr device) override {
            return nullptr;
        }
    };

    template<class Device, class ...Devices>
    class MMMultiDevice<Device, Devices...> : public MMMultiDevice<Devices...> {
    public:

        AllocatorFactoryPtr allocator_factory_;

        MMMultiDevice(std::shared_ptr<AllocatorFactory < Device>>

        allocator_factory,
        std::shared_ptr<AllocatorFactory < Devices>>... args) :

        MMMultiDevice<Devices...>(args...),
        allocator_factory_(allocator_factory) {
        }

        AllocatorPtr GetAllocatorByDevice(DevicePtr device) override {
            if (auto dev = dynamic_cast<Device *>(device)) {
                return this->GetFrom(allocator_factory_.get(), device);
            }
            return MMMultiDevice<Devices...>::GetAllocatorByDevice(device);
        }

    };
}

#endif //XUANWU_MM_H
