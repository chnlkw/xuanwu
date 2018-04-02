//
// Created by chnlkw on 2/6/18.
//

#include "Data.h"

#include <cassert>
#include "Array.h"
#include "Device.h"
#include "Worker.h"

#define LG(x) CLOG(x, "Data")
namespace Xuanwu {

//ArrayBasePtr DataBase::ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
//    last_state_.ReadAt(dev, stream);
//    ArrayBasePtr arr = last_state_.WriteAt(dev, stream, true, last_state_.bytes);
//    return arr;
//}

//    ArrayBasePtr DataBase::ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
//    tasks_writing_.clear();
//    tasks_reading_.push_back(task);
//        ArrayBasePtr arr = last_state_.ReadAt(dev, stream);
//        data_ = arr->data();
//        return arr;
//    }

//ArrayBasePtr DataBase::WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes) {
//    tasks_reading_.clear();
//    tasks_writing_.push_back(task);
//    ArrayBasePtr arr = last_state_.WriteAt(dev, stream, false, bytes);
//    return arr;
//}

//    ArrayBasePtr DataBase::WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, bool keep_old) {
//        ArrayBasePtr arr = last_state_.WriteAt(dev, stream, keep_old);
//        data_ = arr->data();
//        return arr;
////    return WriteAsync(task, dev, stream, last_state_.bytes);
//    }

//    ArrayBasePtr DataBase::Read(DevicePtr dev) const {
//        Wait();
//        ArrayBasePtr ret = last_state_.ReadAt(dev, 0);
//        CUDA_CALL(cudaStreamSynchronize, 0);
//        data_ = ret->data();
//        return ret;
//    }
//
//    ArrayBasePtr DataBase::Write(DevicePtr dev, bool keep_old) {
//        Wait();
//        ArrayBasePtr ret = last_state_.WriteAt(dev, 0, keep_old);
//        CUDA_CALL(cudaStreamSynchronize, 0);
//        data_ = ret->data();
//        return ret;
//    }

//ArrayBasePtr DataBase::ReadWrite(DevicePtr dev) {
//    ArrayBasePtr ret = last_state_.WriteAt(dev, 0, true, last_state_.bytes);
//    CUDA_CALL(cudaStreamSynchronize, 0);
//    return ret;
//}

    const std::vector<std::weak_ptr<TaskBase>> &DataBase::RegisterTask(const TaskPtr &t) {
        tasks_scheduled_.push_back(t);

    }

//    std::vector<ArrayBasePtr> DataBase::GetReplicas() const {
//        std::vector<ArrayBasePtr> ret;
//        ret.reserve(last_state_.replicas.size());
//        for (auto &e : last_state_.replicas)
//            ret.push_back(e.second);
//        return std::move(ret);
//    }

//    std::vector<DevicePtr> DataBase::DevicesPrefered() const {
//        std::vector<DevicePtr> ret;
//        for (auto &r : GetReplicas()) {
//            ret.push_back(r->Device());
//        }
//        for (auto &f : follows_) {
//            if (f.expired())
//                continue;
//            for (auto &r : f.lock()->GetReplicas())
//                ret.push_back(r->Device());
//        }
//        return ret;
//    }

    ArrayBasePtr DataBase::ReadAsync(WorkerPtr worker) {
        return ReadAsync(worker, worker->Device());
    }

    ArrayBasePtr DataBase::WriteAsync(WorkerPtr worker) {
        return WriteAsync(worker, worker->Device());
    }

    DataImpl::DataImpl(MMBase *mm, size_t size) : DataBase(size), mm_(mm) {}

    void DataImpl::ResizeBytes(size_t bytes) {
        Wait();
        if (bytes_ > 0 && bytes > bytes_) {
            LOG(FATAL) << "Data increasing size not supported yet";
        }
        if (bytes == 0 && !replicas.empty()) {
            LOG(FATAL) << "Data size is 0, but has replicas";
        }
//        LG(INFO) << "resize " << last_state_.bytes << " to " << bytes;
        bytes_ = bytes;
    }

    ArrayBasePtr DataImpl::ReadAsync(WorkerPtr worker, DevicePtr dev) {
        if (replicas.count(dev) == 0) {
            ArrayBasePtr arr;
            assert(!replicas.empty());
            ArrayBasePtr from = replicas.begin()->second;
            if (invalids.count(dev)) {
                arr = invalids[dev];
                invalids.erase(dev);
            } else {
                arr = std::make_shared<ArrayBase>(bytes_, mm_->GetAllocatorByDevice(dev));
            }
            assert(from->GetBytes() >= bytes_);
            arr->CopyFromAsync(*from, worker);
            replicas[dev] = arr;
        }
        if (replicas[dev]->GetBytes() > bytes_)
            replicas[dev]->ResizeBytes(bytes_);

        assert(replicas[dev]->GetBytes() == bytes_);
        current_array_ = replicas[dev];
        return current_array_;
    }

    ArrayBasePtr DataImpl::WriteAsync(WorkerPtr worker, DevicePtr dev) {
        assert(bytes_ > 0);
//    Invalid others
        for (auto it = replicas.begin(); it != replicas.end();) {
            if (it->first != dev) {
                invalids.emplace(*it);
                it = replicas.erase(it);
            } else {
                ++it;
            }
        }

        if (!replicas.count(dev)) {
            auto arr = std::make_shared<ArrayBase>(bytes_, mm_->GetAllocatorByDevice(dev));
            if (!replicas.empty())
                arr->CopyFromAsync(*replicas.begin()->second, worker);
            replicas[dev] = arr;
        }
        current_array_ = replicas[dev];
        return current_array_;
    }

    float DataImpl::ReadOverhead(DevicePtr dev) {
        float ret;
        if (replicas.count(dev)) {
            ret =  0;
        } else {
            ret = 1;
        }
        LG(INFO)<< "Data " << this << " " << *dev << " Read overhead = " << ret;
        return ret;
    }

    void *DataImpl::data() const { return current_array_->data(); }

    void *DataImpl::data() { return current_array_->data(); }

    Ptr DataImpl::GetPtr() {
        return current_array_->GetPtr();
    }
}
