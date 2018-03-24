//
// Created by chnlkw on 2/6/18.
//

#include <cassert>
#include "Array.h"
#include "Data.h"

#define LG(x) CLOG(x, "Data")
namespace Xuanwu {
    ArrayBasePtr DataBase::State::ReadAt(const DevicePtr &dev, cudaStream_t stream) {
        if (replicas.count(dev) == 0) {
            ArrayBasePtr arr;
            assert(!replicas.empty());
            ArrayBasePtr from = replicas.begin()->second;
            if (invalids.count(dev)) {
                arr = invalids[dev];
                invalids.erase(dev);
            } else {
                arr = std::make_shared<ArrayBase>(dev->GetAllocator(), dev, bytes);
            }
            assert(from->GetBytes() >= bytes);
            arr->CopyFromAsync(*from, stream, false);
            replicas[dev] = arr;
        }
        if (replicas[dev]->GetBytes() > bytes)
            replicas[dev]->ResizeBytes(bytes);

        assert(replicas[dev]->GetBytes() == bytes);
        return replicas[dev];
    }

    ArrayBasePtr DataBase::State::WriteAt(const DevicePtr &dev, cudaStream_t stream, bool keep_old) {
        assert(bytes > 0);
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
            auto arr = std::make_shared<ArrayBase>(dev->GetAllocator(), dev, bytes);
            if (!replicas.empty() && keep_old)
                arr->CopyFromAsync(*replicas.begin()->second, stream, false);
            replicas[dev] = arr;
        }
        return replicas[dev];
    }

//ArrayBasePtr DataBase::ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
//    last_state_.ReadAt(dev, stream);
//    ArrayBasePtr arr = last_state_.WriteAt(dev, stream, true, last_state_.bytes);
//    return arr;
//}

    ArrayBasePtr DataBase::ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
//    tasks_writing_.clear();
//    tasks_reading_.push_back(task);
        ArrayBasePtr arr = last_state_.ReadAt(dev, stream);
        data_ = arr->data();
        return arr;
    }

//ArrayBasePtr DataBase::WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes) {
//    tasks_reading_.clear();
//    tasks_writing_.push_back(task);
//    ArrayBasePtr arr = last_state_.WriteAt(dev, stream, false, bytes);
//    return arr;
//}

    ArrayBasePtr DataBase::WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, bool keep_old) {
        ArrayBasePtr arr = last_state_.WriteAt(dev, stream, keep_old);
        data_ = arr->data();
        return arr;
//    return WriteAsync(task, dev, stream, last_state_.bytes);
    }

    ArrayBasePtr DataBase::Read(DevicePtr dev) const {
        Wait();
        ArrayBasePtr ret = last_state_.ReadAt(dev, 0);
        CUDA_CALL(cudaStreamSynchronize, 0);
        data_ = ret->data();
        return ret;
    }

    ArrayBasePtr DataBase::Write(DevicePtr dev, bool keep_old) {
        Wait();
        ArrayBasePtr ret = last_state_.WriteAt(dev, 0, keep_old);
        CUDA_CALL(cudaStreamSynchronize, 0);
        data_ = ret->data();
        return ret;
    }

//ArrayBasePtr DataBase::ReadWrite(DevicePtr dev) {
//    ArrayBasePtr ret = last_state_.WriteAt(dev, 0, true, last_state_.bytes);
//    CUDA_CALL(cudaStreamSynchronize, 0);
//    return ret;
//}

    const std::vector<std::weak_ptr<TaskBase>> &DataBase::RegisterTask(const TaskPtr &t, bool read_only) {
        tasks_scheduled_.push_back(t);
        if (read_only) {
            if (writing) {
                writing = false;
                last_reading_.clear();
            }
            last_reading_.push_back(t);
            return last_writing_;
        } else {
            if (!writing) {
                writing = true;
                last_writing_.clear();
            }
            last_writing_.push_back(t);
            return last_reading_;
        }
    }

    std::vector<ArrayBasePtr> DataBase::GetReplicas() const {
        std::vector<ArrayBasePtr> ret;
        ret.reserve(last_state_.replicas.size());
        for (auto &e : last_state_.replicas)
            ret.push_back(e.second);
        return std::move(ret);
    }

    std::vector<DevicePtr> DataBase::DevicesPrefered() const {
        std::vector<DevicePtr> ret;
        for (auto &r : GetReplicas()) {
            ret.push_back(r->Device());
        }
        for (auto &f : follows_) {
            if (f.expired())
                continue;
            for (auto &r : f.lock()->GetReplicas())
                ret.push_back(r->Device());
        }
        return ret;
    }

    void DataBase::ResizeBytes(size_t bytes) {
        Wait();
        if (last_state_.bytes > 0 && bytes > last_state_.bytes) {
            LOG(FATAL) << "Data increasing size not supported yet";
        }
        if (last_state_.bytes == 0 && !last_state_.replicas.empty()) {
            LOG(FATAL) << "Data size is 0, but has replicas";
        }
        LG(INFO) << "resize " << last_state_.bytes << " to " << bytes;
        last_state_.bytes = bytes;
    }

}
