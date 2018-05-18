//
// Created by chnlkw on 18-5-17.
//

#ifndef XUANWU_NUMAS_H
#define XUANWU_NUMAS_H

#include <numa.h>
#include <xuanwu/Allocator.h>
#include <xuanwu/Device.h>


namespace Xuanwu {
    template<class Allocator>
    class NumaAllocator : public Allocator {
    public:
        NumaAllocator(int numa_id) : Allocator(), numa_id_(numa_id) {
            int n = numa_max_node() + 1;
            assert(numa_id < n);
        }

        void *Alloc(size_t size) {
            int ret = numa_run_on_node(numa_id_);
            assert(ret == 0);
            return Allocator::Alloc(size);
        }

    private:
        int numa_id_;
    };

    template<class Allocator>
    class NumaAllocatorFactory : public AllocatorFactory<CPUDevice> {
    public:
        std::unique_ptr<AllocatorBase> Create(DevicePtr device) override {
            auto d = dynamic_cast<CPUDevice *>(device);
            if (d)
                return std::make_unique<NumaAllocator<Allocator>>(d->ID());
            else
                return {};
        }
    };
}

#endif //XUANWU_NUMAS_H
