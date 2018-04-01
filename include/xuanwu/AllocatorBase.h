//
// Created by chnlkw on 3/14/18.
//

#ifndef GRIDLDA_ALLOCATORBASE_H
#define GRIDLDA_ALLOCATORBASE_H

#include "defs.h"
#include "Ptr.h"

namespace Xuanwu {

    class AllocatorBase {
        DevicePtr device_;
    public:

        AllocatorBase(DevicePtr device = nullptr);

        virtual ~AllocatorBase() = default;

        virtual void *Alloc(size_t size) = 0;

        virtual void Free(void *ptr) = 0;

        DevicePtr GetDevice();

        virtual Ptr MakePtr(void* ptr) const = 0;

    };

    class AllocatorFactoryBase {
    public:

        AllocatorFactoryBase() = default;

        virtual ~AllocatorFactoryBase() = default;

        virtual std::unique_ptr<AllocatorBase> Create(DevicePtr device) = 0;

    };

}
#endif //GRIDLDA_ALLOCATORBASE_H
