//
// Created by chnlkw on 1/22/18.
//

#ifndef DMR_DATA_H
#define DMR_DATA_H

#include <deque>
#include "DeviceBase.h"
#include "Array.h"
#include "DataCopy.h"
#include "Xuanwu.h"
#include "MM.h"

namespace Xuanwu {
    class DataBase : public el::Loggable {
    protected:
        size_t bytes_ = 0;

        mutable std::deque<std::weak_ptr<TaskBase>> tasks_scheduled_;

        const std::vector<std::weak_ptr<TaskBase>> &RegisterTask(const TaskPtr &t);

        friend class Engine;

        MMBase *mm_;

        std::string name_ = "Noname";
    public:

        explicit DataBase(MMBase *mm, size_t bytes) : mm_(mm), bytes_(bytes) {}

        DataBase(const DataBase &) = delete;

        size_t Bytes() const { return bytes_; }

        size_t NumTasks() const {
            return tasks_scheduled_.size();
        }

        virtual Ptr GetPtr() = 0;

        virtual ArrayBasePtr ReadAsync(WorkerPtr worker, DevicePtr device) = 0;

        ArrayBasePtr ReadAsync(WorkerPtr worker);

        virtual ArrayBasePtr WriteAsync(WorkerPtr worker, DevicePtr dev) = 0;

        ArrayBasePtr WriteAsync(WorkerPtr worker);

        virtual float ReadOverhead(DevicePtr device) = 0;

        virtual float WriteOverhead(DevicePtr device) = 0;

        void Wait() const;

        virtual void ResizeBytes(size_t bytes) = 0;

//        void Follow(DataBasePtr that) {
//            follows_.push_back(that);
//        }

        virtual void *data() const = 0;

        virtual void *data() = 0;

        MMBase *GetMM() const { return mm_; }

        void SetName(std::string name);

        void log(el::base::type::ostream_t &os) const override;

        std::string Name() const { return name_; }
    };

    template<class T>
    class Data : public std::shared_ptr<DataBase> {
    private:

    public:
        explicit Data(size_t count = 0, MMBase *mm = GetDefaultMM()) : std::shared_ptr<DataBase>(
                mm->MakeDataBase(count * sizeof(T))) {}

        void swap(Data<T> &that) {
            std::shared_ptr<DataBase>::swap(that);
        }

        explicit Data(const std::vector<T> &vec, MMBase *mm = GetDefaultMM()) :
                std::shared_ptr<DataBase>(mm->MakeData<T>(vec.size())) {
            Array<T> &arr = Write();
            size_t bytes = vec.size() * sizeof(T);
            ArrayCopyAsyncPtr(GetDefaultWorker(), arr.GetPtr(), Ptr((void *) vec.data()), bytes);
        }

        using value_type = T;

        const Array<T> &Read() const {
            get()->Wait();
            const Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->ReadAsync(GetDefaultWorker()));
            return ret;
        }

        Array<T> &Write() {
            get()->Wait();
            Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->WriteAsync(GetDefaultWorker()));
            return ret;
        }

        const Array<T> &ReadAsync(WorkerPtr worker, DevicePtr device) const {
            const Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->ReadAsync(worker, device));
            return ret;
        }

        Array<T> &WriteAsync(WorkerPtr worker, DevicePtr device) const {
            Array<T> &ret = *std::static_pointer_cast<Array<T>>(get()->WriteAsync(worker, device));
            return ret;
        }

//        std::vector<ArrayPtr<T>> GetReplicas() const {
//            std::vector<ArrayBasePtr> replicas = get()->GetReplicas();
//            std::vector<ArrayPtr<T>> ret;
//            ret.reserve(replicas.size());
//            for (auto &e : replicas)
//                ret.push_back(std::static_pointer_cast<Array<T>>(e));
//            return ret;
//        }

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
            const T *a = Read(GetDefaultDevice()).data();
            os << "Data(" << "ptr=" << a << " count=" << size() << ": ";
            for (size_t i = 0; i < size(); i++)
                os << a[i] << ',';
            os << ")";

            return os.str();
        }

    private:
    };

    class DataImpl : public DataBase {
        std::map<DevicePtr, ArrayBasePtr> replicas;
        std::map<DevicePtr, ArrayBasePtr> invalids;

        mutable ArrayBasePtr current_array_ = nullptr;

    public:
        DataImpl(MMBase *mm, size_t size);

        void ResizeBytes(size_t bytes) override;

        ArrayBasePtr ReadAsync(WorkerPtr worker, DevicePtr dev) override;

        ArrayBasePtr WriteAsync(WorkerPtr worker, DevicePtr dev) override;

        void *data() const override;

        void *data() override;

        float ReadOverhead(DevicePtr dev) override;

        float WriteOverhead(DevicePtr dev) override {
            return 0;
        }

        Ptr GetPtr() override;
    };

    struct data_constructor_t {
        template<class T, class ...Args>
        static Data<T> Construct(Args &&... args) {
            return {std::forward<Args>(args)...};
        }
    };

}

namespace std {
    template<class T>
    std::string to_string(const Xuanwu::Data<T> &v) { return v.ToString(); }

    template<class T>
    std::string to_string(const std::vector<T> &v) {
        std::ostringstream os;
        os << "vector(" << v.size() << " : ";
        for (auto x : v) os << x << ",";
        os << ") ";
        return os.str();
    }
}

#endif //DMR_DATA_H
