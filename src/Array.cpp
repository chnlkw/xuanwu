//
// Created by chnlkw on 1/16/18.
//

#include "Array.h"
#include "DataCopy.h"
#include "Device.h"
#include "Allocator.h"
#include "Xuanwu.h"
#include "Ptr.h"

namespace Xuanwu {
    ArrayBase::ArrayBase(size_t bytes, AllocatorPtr allocator)
            : allocator_(allocator), bytes_(bytes), ptr_(allocator->Alloc(bytes)) {
    }

//    ArrayBase::ArrayBase(const ArrayBase &that) :
//            allocator_(Xuanwu::GetDefaultDevice()->GetDefaultAllocator()) {
//        Allocate(that.bytes_);
//        ArrayCopy(allocator_->GetDevice(), ptr_, that.allocator_->GetDevice(), that.ptr_, that.bytes_);
//    }

//    ArrayBase::ArrayBase(void *ptr, size_t bytes) : //alias from cpu array
//            allocator_(nullptr),
//            ptr_(ptr),
//            bytes_(bytes) {
//    }

    ArrayBase::ArrayBase(ArrayBase &&that) :
            allocator_(that.allocator_),
            bytes_(that.bytes_),
            ptr_(that.ptr_) {
        that.ptr_ = nullptr;
    }

    ArrayBase::ArrayBase(const ArrayBase &that, size_t off, size_t bytes)
            : allocator_(nullptr),
              bytes_(bytes),
              ptr_((char *) that.ptr_ + off) {
    }

    ArrayBase::~ArrayBase() {
        Free();
    }

    void ArrayBase::Free() {
        if (allocator_ && ptr_)
            allocator_->Free(ptr_);
        ptr_ = nullptr;
    }

    void ArrayBase::Allocate() {
        if (bytes_ > 0 && allocator_) {
            ptr_ = allocator_->Alloc(bytes_);
        } else {
            ptr_ = nullptr;
        }
//    printf("reallocate ptr %p bytes = %lu\n", ptr_, bytes);
    }

//    void ArrayBase::CopyFrom(const ArrayBase &that) {
//        CopyFromAsync(that, GetDefaultWorker());
//    }
//
//    void ArrayBase::CopyFromAsync(const ArrayBase &that, WorkerPtr worker) {
//        size_t bytes = std::min(this->bytes_, that.bytes_);
//        assert(this->bytes_ == that.bytes_);
//        worker->Copy(GetPtr(), that.GetPtr(), bytes);
//    }

//    DevicePtr ArrayBase::GetDevice() const { return allocator_->GetDevice(); }

//    AllocatorPtr ArrayBase::GetAllocator() const { return allocator_; }

    Ptr ArrayBase::GetPtr() const {
        return allocator_->MakePtr(ptr_);
    }

}
