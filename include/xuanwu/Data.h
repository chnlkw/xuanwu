//
// Created by chnlkw on 1/22/18.
//

#ifndef DMR_DATA_H
#define DMR_DATA_H

#include <deque>
#include "DeviceBase.h"
#include "DataCopy.h"
#include "Car.h"

class DataBase {
protected:
    struct State {
        size_t bytes = 0;
        std::map<DevicePtr, ArrayBasePtr> replicas;
        std::map<DevicePtr, ArrayBasePtr> invalids;

        ArrayBasePtr ReadAt(const DevicePtr &dev, cudaStream_t stream);

        ArrayBasePtr WriteAt(const DevicePtr &dev, cudaStream_t stream, bool keep_old);
    };

    mutable State last_state_;
    mutable std::deque<std::weak_ptr<TaskBase>> tasks_scheduled_;

    mutable std::vector<std::weak_ptr<TaskBase>> last_reading_, last_writing_;

    friend class Engine;

    bool writing = false;

    const std::vector<std::weak_ptr<TaskBase>> &RegisterTask(const TaskPtr &t, bool read_only);

    std::vector<std::weak_ptr<DataBase>> follows_;

    mutable void *data_ = nullptr;

public:

    explicit DataBase(size_t bytes = 0) {
        last_state_.bytes = bytes;
    }

    DataBase(const DataBase &) = delete;

    size_t Bytes() const {
        return last_state_.bytes;
    }

    size_t NumTasks() const {
        return tasks_scheduled_.size();
    }

    ArrayBasePtr ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

    ArrayBasePtr Read(DevicePtr dev) const;

//    ArrayBasePtr Write(DevicePtr dev, size_t bytes);

    ArrayBasePtr Write(DevicePtr dev, bool keep_old = false);

//    ArrayBasePtr WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes);

    ArrayBasePtr WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, bool keep_old = false);

//    ArrayBasePtr ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream);

//    ArrayBasePtr ReadWrite(DevicePtr dev);

    std::vector<ArrayBasePtr> GetReplicas() const;

    void Wait() const;

    void ResizeBytes(size_t bytes);

    void Follow(DataBasePtr that) {
        follows_.push_back(that);
    }

    void *data() const { return data_; }

    void *data() { return data_; }

    std::vector<DevicePtr> DevicesPrefered() const;
};

template<class T>
class Data : public std::shared_ptr<DataBase> {
private:

    //add policy

public:
    Data() : std::shared_ptr<DataBase>(new DataBase()) {
    }

    void swap(Data<T> &that) {
        std::shared_ptr<DataBase>::swap(that);
    }

    explicit Data(size_t count) : std::shared_ptr<DataBase>(new DataBase(count * sizeof(T))) {
    }

    Data(size_t count, DevicePtr device) : std::shared_ptr<DataBase>(new DataBase(count * sizeof(T))) {
        Write(device);
    }

    explicit Data(const std::vector<T> &vec, DevicePtr device = Car::GetCPUDevice()) :
            std::shared_ptr<DataBase>(new DataBase(vec.size() * sizeof(T))) {
        Write(device);
        size_t bytes = vec.size() * sizeof(T);
        DataCopy(data(), device->Id(), vec.data(), -1, bytes);
    }

    using value_type = T;

    const Array<T> &Read(DevicePtr dev = Car::GetCPUDevice()) const {
        const Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->Read(dev));
        return ret;
    }

    const Array<T> &ReadAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) const {
        const Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->ReadAsync(task, dev, stream));
        return ret;
    }

//    Array<T> &WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, size_t bytes) {
//        data_ = nullptr;
//        return *std::static_pointer_cast<Array<T>>(get()->WriteAsync(task, dev, stream, bytes));
//    }

    Array<T> &WriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream, bool keep_old = false) {
        Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->WriteAsync(task, dev, stream, keep_old));
        return ret;
    }

//    Array<T> &ReadWriteAsync(TaskPtr task, DevicePtr dev, cudaStream_t stream) {
//        data_ = nullptr;
//        return *std::static_pointer_cast<Array<T>>(get()->ReadWriteAsync(task, dev, stream));
//    }

//    Array<T> &Write(DevicePtr dev) {
//        Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->Write(dev));
//        data_ = ret.data();
//        return ret;
//    }

    Array<T> &Write(DevicePtr dev = Car::GetCPUDevice(), bool keep_old = true) {
        Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->Write(dev, keep_old));
        return ret;
    }

    std::vector<ArrayPtr<T>> GetReplicas() const {
        std::vector<ArrayBasePtr> replicas = get()->GetReplicas();
        std::vector<ArrayPtr<T>> ret;
        ret.reserve(replicas.size());
        for (auto &e : replicas)
            ret.push_back(std::static_pointer_cast<Array<T>>(e));
        return ret;
    }

    size_t size() const {
        return get()->Bytes() / sizeof(T);
    }

    void resize(size_t count) {
        get()->ResizeBytes(count * sizeof(T));
    }

    T *data() { return (T *) get()->data(); }

    const T *data() const { return (const T *) get()->data(); }

    const T &operator[](ssize_t idx) const { return data()[idx]; }

    T &operator[](ssize_t idx) { return data()[idx]; }

    std::string ToString() const {
        std::ostringstream os;
        const T *a = Read(Car::GetCPUDevice()).data();
        os << "Data(" << "ptr=" << a << " count=" << size() << ": ";
        for (size_t i = 0; i < size(); i++)
            os << a[i] << ',';
        os << ")";

        return os.str();
    }

private:
};

namespace std {
template<class T>
std::string to_string(const Data<T> &v) { return v.ToString(); }

template<class T>
std::string to_string(const std::vector<T> &v) {
    std::ostringstream os;
    os << "vector(" << v.size() << " : ";
    for (auto x : v) os << x << ",";
    os << ") ";
    return os.str();
}
}

struct data_constructor_t {
    template<class T, class ...Args>
    static Data<T> Construct(Args &&... args) {
        return {std::forward<Args>(args)...};
    }
};

#endif //DMR_DATA_H
