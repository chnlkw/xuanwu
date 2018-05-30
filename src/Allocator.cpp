//
// Created by chnlkw on 1/16/18.
//

#include "Allocator.h"
#include "Device.h"
#include "Ptr.h"

#define LG(x) CLOG(x, "Allocator")

namespace Xuanwu {

    std::string bytes_to_str(size_t bytes) {
        double b = bytes;
        std::vector<std::string> suffix = {"B", "KB", "MB", "GB", "TB"};
        for (auto s : suffix) {
            if (b < 1024)
                return std::to_string(b) + " " + s;
            b /= 1024;
        }
        return std::to_string(bytes);
    }

//    int UnifiedAllocator::GetDevice() {
//        return gpu_id;
//    }
//
//    void UnifiedAllocator::SetDevice(int device_id) {
//        gpu_id = device_id;
//    }
//
//    UnifiedAllocator::UnifiedAllocator() :
//            gpu_id(-1),
//            allocator_([](int device) { return AllocatorPtr(new Allocator(device)); }) {
//    }
//
//    void *UnifiedAllocator::Alloc(size_t size) {
//        return allocator_.Alloc(size, gpu_id);
//    }
//
//    void UnifiedAllocator::Free(void *ptr) {
//        allocator_.Free(ptr);
//    }

    PreAllocator::PreAllocator(std::shared_ptr<AllocatorBase> base_allocator, size_t pre_alloc_size) :
            AllocatorBase(),
            base_allocator_(base_allocator),
            size_(pre_alloc_size),
            allocated_(0),
            align_(4096) {
        LG(INFO) << "PreAllocator with " << "  pre_alloc_bytes = "
                 << bytes_to_str(size_)
                 << " align = " << align_;
        assert(align_ > 0);
        ptr_ = base_allocator_->Alloc(size_);
    }

    PreAllocator::~PreAllocator() {
        if (allocated_ > 0) {
            fprintf(stderr, "[WARN] dangling pointer to PreAllocater, allocated size = %lu\n", allocated_);
            for (auto it = m_.begin(); it != m_.end(); ++it) {
                fprintf(stderr, "      not freed pointer %p, %lu\n", ptr_ + it->first, it->second);
            }
        }
        base_allocator_->Free(ptr_);
    }

    void *PreAllocator::Alloc(size_t size) {
        auto align_up = [this](size_t off) {
            off += align_ - 1;
            off &= ~(align_ - 1);
            return off;
        };
        off_t off = 0;
        for (auto p : m_) {
            auto beg = p.first;
            auto end = p.first + p.second;
            if (off + size > beg) {
                off = align_up(end);
            } else {
                break;
            }
        }
        if (off + size > size_) {
            std::ostringstream os;
            os << "PreAllocator :: not enough memory when allocating " << bytes_to_str(size) << " remain "
               << bytes_to_str(size_ - allocated_) << " total " << bytes_to_str(size_) << '\n';
            os << "\tallocated are:\n";
            for (auto p : m_) {
                os << "\t\t" << ptr_ + p.first << " " << p.second << '\n';
            }

            throw std::runtime_error(os.str().c_str());
        }
        m_.emplace(off, size);
        allocated_ += size;
        LG(DEBUG) << "PreAllocator: " << " Alloc=" << bytes_to_str(size) << " ptr_ = " << ptr_ << " off = "
                  << off;
        LG(INFO) << "PreAllocator: " << "Total=" << bytes_to_str(size_) << " Alloc=" << bytes_to_str(size)
                 << " allocated=" << bytes_to_str(allocated_) << " remain=" << bytes_to_str(size_ - allocated_);
        return (char *) ptr_ + off;
    }

    void PreAllocator::Free(void *ptr) {
        off_t off = (char *) ptr - (char *) ptr_;
        auto it = m_.find(off);
        if (it == m_.end()) {
            std::ostringstream os;
            os << "PreAllocator :: " << " Free pointer not found ptr=" << ptr << " off = " << off;
            LG(FATAL) << os.str();
            throw std::runtime_error(os.str().c_str());
        }
        allocated_ -= it->second;
        LG(INFO) << "PreAllocator: " << "Total=" << bytes_to_str(size_) << " Free=" << bytes_to_str(it->second)
                 << " allocated=" << bytes_to_str(allocated_) << " remain=" << bytes_to_str(size_ - allocated_);
        m_.erase(it);
    }

//    Allocator::Allocator(int gpu_id) : gpu_id(gpu_id) {
//        LG(INFO) << "Allocator with gpu_id = " << gpu_id;
//    }
//    HostAllocator::HostAllocator()  {
//        LG(INFO) << "HostAllocator ";
//    }
    std::unique_ptr<AllocatorBase> CudaAllocatorFactory::Create(DevicePtr d) {
        auto gpu = dynamic_cast<GPUDevice *>(d);
        if (gpu)
            return std::make_unique<CudaAllocator>(gpu->GPUID());
        else
            return {};
    }

    std::unique_ptr<AllocatorBase> CudaHostAllocatorFactory::Create(DevicePtr d) {
        return std::make_unique<CudaHostAllocator>();
    }

    void CudaHostAllocator::Free(void *ptr) {
        CUDA_CALL(cudaFreeHost, ptr);
    }

    void *CudaHostAllocator::Alloc(size_t size) {
        void *ptr;
        CUDA_CALL(cudaMallocHost, &ptr, size);
        return ptr;
    }

    void CPUAllocator::Free(void *ptr) {
        free(ptr);
    }

    void *CPUAllocator::Alloc(size_t size) {
        return malloc(size);
    }

    std::unique_ptr<AllocatorBase> CPUAllocatorFactory::Create(DevicePtr device) {
        return std::make_unique<CPUAllocator>();
    }

    Ptr CPUAllocator::MakePtr(void *ptr) const {
        return Ptr(ptr, Ptr::Type::CPU);
    }

    Ptr DummyAllocator::MakePtr(void *ptr) const {
        return Ptr(ptr, Ptr::Type::CPU);
    }

    Ptr CudaAllocator::MakePtr(void *ptr) const {
        return Ptr(ptr, Ptr::Type::GPU);
    }

    Ptr CudaHostAllocator::MakePtr(void *ptr) const {
        return Ptr(ptr, Ptr::Type::CPU);
    }

    Ptr PreAllocator::MakePtr(void *ptr) const {
        return base_allocator_->MakePtr(ptr);
    }
}
