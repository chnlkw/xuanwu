//
// Created by chnlkw on 11/21/17.
//

#ifndef LDA_ARRAY_H_H
#define LDA_ARRAY_H_H

#include "defs.h"
#include "cuda_utils.h"
#include "Xuanwu.h"
#include "Ptr.h"

namespace Xuanwu {
    class ArrayBase {
    protected:
        AllocatorPtr allocator_;
//    int gpu_id;
        size_t bytes_;
        void *ptr_;

        void Allocate(size_t bytes);

        void Free();

    public:

        ArrayBase(size_t bytes, AllocatorPtr allocator);

//        ArrayBase(const ArrayBase &that);

//        ArrayBase(void *ptr, size_t bytes);

        ArrayBase(ArrayBase &&that);

        ArrayBase(const ArrayBase &that, size_t off, size_t bytes);

        ~ArrayBase();

//    ArrayBase Renew(size_t bytes) const {
//        return {allocator_, gpu_id, bytes};
//    }

        void CopyFrom(const ArrayBase &that);

        void CopyFromAsync(const ArrayBase &that, WorkerPtr worker);

        size_t GetBytes() const { return bytes_; }

        AllocatorPtr GetAllocator() const;

        DevicePtr GetDevice() const;

        void ResizeBytes(size_t bytes) {
            if (bytes > bytes_)
                abort();
            bytes_ = bytes;
        }

        void *data() const { return ptr_; }

        Ptr GetPtr() const;
    };

    template<class T>
    class Array : public ArrayBase {
//    size_t count_;
    public:

        using value_type = T;

        explicit Array(size_t count = 0);

        Array(AllocatorPtr allocator,  size_t count = 0)
                : ArrayBase(count * sizeof(T), allocator) {
        }

//    Array(AllocatorBase *allocator, size_t count) : // need allocated
//            ArrayBase(allocator, count * sizeof(T)),
//            count_(count) {
//    }

//    Array(MultiDeviceAllocator &allocator, T *ptr, size_t count, int device) : // not allocated
//            ArrayBase(allocator, ptr, count * sizeof(T), device),
//            count_(count) {
//    }

        Array(Array<T> &&that) :
                ArrayBase(std::move(that)) {
        }

//    Array &operator=(Array<T> &&that) {
//        count_ = that.count_;
//        that.count_ = 0;
//        ArrayBase::operator=(std::move(that));
//        return *this;
//    }

//        Array(const std::vector<T> &that) :
//                ArrayBase((void *) that.data(), that.size() * sizeof(T)) {
//        }

        Array(const Array<T> &that) :
                ArrayBase(that) {
        }

        T *data() {
            return reinterpret_cast<T *>(ptr_);
        }

        const T *data() const {
            return reinterpret_cast<const T *>(ptr_);
        }

        T *begin() {
            return reinterpret_cast<T *>(ptr_);
        }

        const T *begin() const {
            return reinterpret_cast<const T *>(ptr_);
        }

        T *end() {
            return begin() + Count();
        }

        const T *end() const {
            return begin() + Count();
        }

        size_t Count() const {
            return GetBytes() / sizeof(T);
        }

        size_t size() const {
            return Count();
        }

        void resize(size_t size) {
            ResizeBytes(size * sizeof(T));
        }

        Array<T> CopyTo(int device) {
            Array<T> that(allocator_, Count(), device);
            that.CopyFrom(*this);
            return that;
        }

        Array<T> Renew(size_t count) const {
            return {allocator_, count};
        }

        Array<T> Slice(size_t beg, size_t end) {
            assert(beg < end && end <= Count());
            T *ptr = this->data() + beg;
            size_t count = end - beg;
            return Array<T>(allocator_, ptr, count);
        }

//    Array<T> &operator=(const Array<T> &that) {
//        CopyFrom(that);
//    }

        const T &operator[](ssize_t idx) const {
            return data()[idx];
        }

        T &operator[](ssize_t idx) {
            return data()[idx];
        }

        Array<T> operator[](std::pair<ssize_t, ssize_t> range) {
            return Slice(range.first, range.second);
        }
    };

    template<class T>
    Array<T>::Array(size_t count) :
            ArrayBase(count * sizeof(T), GetDefaultAllocator()) {
    }

    struct array_constructor_t {
        template<class T, class ...Args>
        static Array<T> Construct(Args &&... args) {
            return {std::forward<Args>(args)...};
        }
    };

}
#endif //LDA_ARRAY_H_H
