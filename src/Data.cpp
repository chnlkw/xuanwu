//
// Created by chnlkw on 2/6/18.
//

#include "Data.h"

#include <cassert>
#include "Array.h"
#include "Device.h"
#include "Worker.h"

#define LG(x) CLOG(x, "Data")
#define LG_IF(c, x) CLOG_IF(c, x, "Data")

namespace Xuanwu {

    size_t DataBase::s_uid = 0;

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

    void DataBase::RegisterTask(const TaskPtr &t) {
        tasks_scheduled_.push_back(t);
    }

    void DataBase::SetName(std::string name) {
        name_ = std::move(name);
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

    bool DataBase::ReadAsync(WorkerPtr worker) {
        return ReadAsync(worker, worker->Device());
    }

    bool DataBase::WriteAsync(WorkerPtr worker) {
        return WriteAsync(worker, worker->Device());
    }

    void DataBase::PinnedToDevice(DevicePtr dev, bool is_strict) {
        device_pinned_ = dev;
        device_pinned_strict_ = is_strict;
    }

    DataImpl::DataImpl(MMBase *mm, size_t size) : DataBase(mm, size) {}

    void DataImpl::ResizeBytes(size_t bytes) {
//        Wait();
        if (bytes_ > 0 && bytes > bytes_) {
            LOG(FATAL) << "Data increasing size not supported yet";
        }
        if (bytes == 0 && !replicas.empty()) {
            LOG(FATAL) << "Data size is 0, but has replicas";
        }
//        LG(INFO) << "resize " << last_state_.bytes << " to " << bytes;
        bytes_ = bytes;
    }

    bool DataImpl::ReadAsync(WorkerPtr worker, DevicePtr dev) {
        LG(DEBUG) << "DataImpl ReadAsync :: Start" << *this << " at " << *dev;
        if (replicas.count(dev) == 0) {
            ArrayBasePtr arr;
            LOG_IF(replicas.empty(), FATAL) << *this << " calls ReadAsync() with no replicas";
            assert(!replicas.empty());
//            if (invalids.count(dev)) {
//                arr = invalids[dev];
//                invalids.erase(dev);
//            } else {
            arr = std::make_unique<ArrayBase>(bytes_, mm_->GetAllocatorByDevice(dev));
//            }
            decltype(replicas.begin()) from;
            int max_copy_speed = 0;
            for (auto it = replicas.begin(); it != replicas.end(); ++it) {
//            for (auto &r : replicas) {
                int copy_speed = CopySpeed(arr->GetPtr(), it->second.first->GetPtr());
                if (max_copy_speed < copy_speed) {
                    max_copy_speed = copy_speed;
                    from = it;
                }
            }
            if (!from->second.second->QueryFinished()) {
                return false;
            }
            assert(from->second.first->GetBytes() >= bytes_);
            LG(INFO) << "DataImpl ReadAsync Copy :: " << *this << " -- " << from->second.first->GetPtr() << " to "
                      << arr->GetPtr() << " bytes=" << bytes_;
            Event event = worker->Copy(arr->GetPtr(), from->second.first->GetPtr(), bytes_);
            replicas.emplace(std::make_pair(dev, std::make_pair(std::move(arr), std::move(event))));
        }
        if (replicas[dev].first->GetBytes() > bytes_)
            replicas[dev].first->ResizeBytes(bytes_);

        current_array_ = replicas[dev].first.get();
        assert(current_array_->GetBytes() >= bytes_);
        LG(DEBUG) << "DataImpl ReadAsync Finish :: " << *this << " at " << *dev;
        return replicas[dev].second->QueryFinished();
    }

    bool DataImpl::WriteAsync(WorkerPtr worker, DevicePtr dev) {
        assert(bytes_ > 0);
        LG(DEBUG) << "DataImpl WriteAsync :: Start" << *this << " at " << *dev;
//    Invalid others
        for (auto it = replicas.begin(); it != replicas.end();) {
            if (it->first != dev) {
//                invalids[it->first] = it->second.first;
                LG(DEBUG) << "DataImpl WriteAsync :: erase " << *this << " at " << *it->first;
                it = replicas.erase(it);
            } else {
                ++it;
            }
        }
        if (!replicas.count(dev)) {
            auto arr = std::make_unique<ArrayBase>(bytes_, mm_->GetAllocatorByDevice(dev));
//            if (!replicas.empty())
//                worker->Copy(arr->GetPtr(), replicas.begin()->second->GetPtr(), bytes_);
//                arr->CopyFromAsync(*replicas.begin()->second, worker);
            replicas.emplace(std::make_pair(dev, std::make_pair(std::move(arr), std::make_unique<EventDummy>())));
            LG(INFO) << "DataImpl WriteAsync :: Create " << *this << " at " << dev;
        }
        current_array_ = replicas[dev].first.get();
        assert(current_array_->GetBytes() >= bytes_);
        LG(DEBUG) << "DataImpl WriteAsync :: Finish" << *this << " at " << *dev;
        return true;
    }

    float DataImpl::ReadOverhead(DevicePtr dev) {
        float ret;
        if (replicas.count(dev)) {
            ret = 0;
        } else {
            ret = 1;
        }
        LG(INFO) << "Data " << this << " " << *dev << " Read overhead = " << ret;
        return ret;
    }

    void *DataImpl::data() const {
        LOG_IF(!current_array_, FATAL) << *this << " Data::current_array_ is null";
        return current_array_->data();
    }

    void *DataImpl::data() {
        LOG_IF(!current_array_, FATAL) << *this << " Data::current_array_ is null";
        return current_array_->data();
    }

    Ptr DataImpl::GetPtr() {
        LOG_IF(!current_array_, FATAL) << *this << " Data::current_array_ is null";
        return current_array_->GetPtr();
    }

    void DataImpl::clear() {
        replicas.clear();
//        invalids.clear();
        bytes_ = 0;
        current_array_ = nullptr;
    }

    void DataImpl::Create(size_t bytes, DevicePtr dev) {
        clear();
        bytes_ = bytes;
        auto arr = std::make_unique<ArrayBase>(bytes_, mm_->GetAllocatorByDevice(dev));
        replicas.emplace(std::make_pair(dev, std::make_pair(std::move(arr), std::make_unique<EventDummy>())));
        current_array_ = replicas[dev].first.get();
//        replicas[device] = arr;
        LG(DEBUG) << "DataImpl Create : " << *this << " " << *dev << " bytes=" << bytes;
    }

    ArrayBase* DataImpl::CurrentArray() const {
        return current_array_;
    }
}
