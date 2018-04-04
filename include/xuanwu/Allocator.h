//
// Created by chnlkw on 11/20/17.
//

#ifndef LDA_ALLOCATOR_H
#define LDA_ALLOCATOR_H

#include "AllocatorBase.h"
#include "cuda_utils.h"
#include "Config.h"
#include "Ptr.h"

namespace Xuanwu {

    template<class Device>
    class Allocator : public AllocatorBase {
    };

    template<class Device>
    class AllocatorFactory : public AllocatorFactoryBase {
    };

    class CPUAllocator : public Allocator<CPUDevice> {
    public:
        void *Alloc(size_t size) override;

        void Free(void *ptr) override;

        Ptr MakePtr(void *ptr) const override;
    };

    class CPUAllocatorFactory : public AllocatorFactory<CPUDevice> {
    public:
        CPUAllocatorFactory() = default;

        std::unique_ptr<AllocatorBase> Create(DevicePtr device);
    };
//
//    class MultiDeviceAllocator {
//        struct Meta {
//            size_t size;
//            int device;
//        };
//        std::map<int, AllocatorPtr> device_allocator_;
//        std::map<void *, Meta> meta_;
//
//    public:
//        using F_AllocatorFactory = std::function<AllocatorPtr(int)>;
//    private:
//        F_AllocatorFactory allocator_factory_;
//    public:
//
//        MultiDeviceAllocator(F_AllocatorFactory allocator_factory) :
//                allocator_factory_(allocator_factory) {}
//
//        AllocatorPtr GetAllocatorByDevice(int device) {
//            auto it = device_allocator_.find(device);
//            if (it == device_allocator_.end()) {
//                it = device_allocator_.emplace(device, AllocatorPtr(allocator_factory_(device))).first;
//            }
//            return it->second;
//        }
//
//        void *Alloc(size_t size, int device = 0) {
//            void *ptr = GetAllocatorByDevice(device)->Alloc(size);
//            meta_.emplace(ptr, Meta{size, device});
//            return ptr;
//        }
//
//        void Free(void *ptr) {
//            auto it = meta_.find(ptr);
//            if (it == meta_.end()) {
//                std::cerr << "free error : " << ptr << std::endl;
//                abort();
//            }
//            int device = it->second.device;
//            GetAllocatorByDevice(device)->Free(ptr);
//            meta_.erase(it);
//        }
//
//    };
//
//    class UnifiedAllocator : public AllocatorBase {
//    private:
//        int gpu_id;
//        MultiDeviceAllocator allocator_;
//    public:
//        UnifiedAllocator();
//
//        void Init();
//
//        void SetDevice(int device_id);
//
//        int GetDevice();
//
//        void *Alloc(size_t size) override;
//
//        void Free(void *ptr) override;
//    };

    template<class Impl>
    class AllocatorMetric {
        struct Meta {
            size_t size;
        };
        int device_;
        size_t maxsize_;
        size_t allocated_;
        std::map<void *, Meta> meta_;
        Impl impl_;

    public:
        AllocatorMetric(int device, size_t maxsize) :
                device_(device),
                maxsize_(maxsize),
                allocated_(0),
                impl_(device) {}

        void *Alloc(size_t size) {
            allocated_ += size;
            void *ptr = impl_.Alloc(size);
            meta_.emplace(ptr, Meta{size});
            return ptr;
        }

        void Free(void *ptr) {
            auto it = meta_.find(ptr);
            if (it == meta_.end()) {
                return;
            }
            allocated_ -= it->second.size;
            impl_.Free(ptr);
        }

        int GetDevice() {
            return device_;
        }

    };

    class DummyAllocator : public AllocatorBase {
        char *ptr;

    public:
        DummyAllocator(int device) : ptr((char *) 0x1) {

        }

        virtual void *Alloc(size_t size) override {
            std::cout << "Alloc " << size << " = " << (void *) ptr << std::endl;
            return ptr++;
        }

        virtual void Free(void *ptr) override {
            std::cout << "Free " << ptr << std::endl;
        }

        Ptr MakePtr(void *ptr) const override;
    };

    auto myDeviceId = [] {};

    class CudaAllocator : public Allocator<GPUDevice> {
    protected:
        int gpu_id_;
    public:
        explicit CudaAllocator(int gpu_id) : gpu_id_(gpu_id) {}

        virtual void *Alloc(size_t size) override {
            void *ptr;
            CUDA_CALL(cudaSetDevice, gpu_id_);
            CUDA_CALL(cudaMalloc, &ptr, size);
            return ptr;
        }

        virtual void Free(void *ptr) override {
            CUDA_CALL(cudaSetDevice, gpu_id_);
            CUDA_CALL(cudaFree, ptr);
        }

        Ptr MakePtr(void *ptr) const override;

    };

    class CudaAllocatorFactory : public AllocatorFactory<GPUDevice> {
    public:
        CudaAllocatorFactory() = default;

        std::unique_ptr<AllocatorBase> Create(DevicePtr d) override;
    };

    class CudaHostAllocator : public Allocator<CPUDevice> {
    public:
        CudaHostAllocator() = default;

        void *Alloc(size_t size) override;

        void Free(void *ptr) override;

        Ptr MakePtr(void *ptr) const override;
    };

    class CudaHostAllocatorFactory : public AllocatorFactory<CPUDevice> {
    public:
        CudaHostAllocatorFactory() = default;

        std::unique_ptr<AllocatorBase> Create(DevicePtr d) override;
    };

    class PreAllocator : public AllocatorBase {
        size_t size_;
        size_t allocated_;
        size_t align_;
        void *ptr_;
        std::map<off_t, size_t> m_;
        std::shared_ptr<AllocatorBase> base_allocator_;
    public:
        PreAllocator(std::shared_ptr<AllocatorBase> base_allocator, size_t pre_alloc_size);

        ~PreAllocator();

        void *Alloc(size_t size) override;

        void Free(void *ptr) override;

        Ptr MakePtr(void *ptr) const override;
    };

    template<class BaseFactory>
    class PreAllocatorFactory : public BaseFactory {
    public:
        using Space = Strong<size_t, 1 << 20>;

        PreAllocatorFactory(Space pre_alloc_Size) :
                BaseFactory(),
                pre_alloc_size_(pre_alloc_Size) {
            LOG(INFO) << "create PreAllocatorFactory with pre_alloc_size = " << pre_alloc_Size;
        }

        std::unique_ptr<AllocatorBase> Create(DevicePtr d) override {
            return std::make_unique<PreAllocator>(BaseFactory::Create(d), pre_alloc_size_);
        }

    private:
        size_t pre_alloc_size_;
    };

}
#endif //LDA_ALLOCATOR_H
