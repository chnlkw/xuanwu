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

        MMBase *mm_;

        size_t bytes_ = 0;

        mutable std::deque<std::weak_ptr<TaskBase>> tasks_scheduled_;

        void RegisterTask(const TaskPtr &t);

        friend class Engine;

        std::string name_ = "Noname";

        static size_t s_uid;

        size_t uid;

        DevicePtr device_pinned_ = nullptr;
        bool device_pinned_strict_ = true;

    public:
        virtual ~DataBase() = default;

        explicit DataBase(MMBase *mm, size_t bytes) : mm_(mm), bytes_(bytes), uid(s_uid++) {}

        DataBase(const DataBase &) = delete;

        size_t Bytes() const { return bytes_; }

        size_t NumTasks() const {
            return tasks_scheduled_.size();
        }

        virtual Ptr GetPtr() = 0;

        using FnCopy = std::function<void(Ptr, Ptr, size_t)>; // copy(dst, src, bytes)

        virtual bool ReadAsync(WorkerPtr worker, DevicePtr dev) = 0;

        bool ReadAsync(WorkerPtr worker);

        virtual bool WriteAsync(WorkerPtr worker, DevicePtr dev) = 0;

        bool WriteAsync(WorkerPtr worker);

        virtual float ReadOverhead(DevicePtr device) = 0;

        virtual float WriteOverhead(DevicePtr device) = 0;

        void Wait() const;

        virtual void ResizeBytes(size_t bytes) = 0;

//        void Follow(DataBasePtr that) {
//            follows_.push_back(that);
//        }

        virtual void *data() const = 0;

        virtual void *data() = 0;

        virtual void clear() = 0;

        virtual void Create(size_t bytes, DevicePtr device) = 0;

        virtual ArrayBasePtr CurrentArray() const = 0;

        MMBase *GetMM() const { return mm_; }

        size_t GetUID() const { return uid; }

        void SetName(std::string name);

        void log(el::base::type::ostream_t &os) const override;

        std::string Name() const { return name_; }

        void PinnedToDevice(DevicePtr dev, bool is_strict = true);

        std::pair<DevicePtr, bool> GetPinnedDevice() {
            return {device_pinned_, device_pinned_strict_};
        };
    };

    template<class T>
    class Data : public std::shared_ptr<DataBase> {
    private:

    public:
        explicit Data(size_t count = 0, MMBase *mm = GetDefaultMM()) : std::shared_ptr<DataBase>(
                mm->MakeDataBase(count * sizeof(T))) {}

        Data(const Data &that) : std::shared_ptr<DataBase>(that) {
            if (get()->Name().find("cdk_keys_") != std::string::npos) {
                CLOG(INFO, "Data") << "copy data<T> for " << *get();
            }
        }

        void swap(Data<T> &that) {
            std::shared_ptr<DataBase>::swap(that);
        }

        explicit Data(const std::vector<T> &vec, MMBase *mm = GetDefaultMM()) :
                std::shared_ptr<DataBase>(mm->MakeData<T>(vec.size())) {
            Write();
            size_t bytes = vec.size() * sizeof(T);
            CPUCopy(CurrentArray().GetPtr(), Ptr((void *) vec.data()), bytes, 0);
        }

        using value_type = T;

        void Wait() const {
            get()->Wait();
        }

        void clear() {
            get()->clear();
        }

        Array<T> &Create(size_t count, DevicePtr device = GetDefaultDevice()) {
            get()->Create(count * sizeof(T), device);
            return CurrentArray();
        }

        Array<T> &CurrentArray() const {
            return *std::static_pointer_cast<Array<T>>(get()->CurrentArray());
        }

        Array<T> &Read() const {
            get()->Wait();
            while (!get()->ReadAsync(GetDefaultWorker()));
            return CurrentArray();
        }

        Array<T> &Write() {
            get()->Wait();
            while (!get()->WriteAsync(GetDefaultWorker()));
            return CurrentArray();
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
        std::map<DevicePtr, std::pair<ArrayBasePtr, Event>> replicas;
//        std::map<DevicePtr, ArrayBasePtr> invalids;

        mutable std::weak_ptr<ArrayBase> current_array_;

    public:
        DataImpl(MMBase *mm, size_t size);

        void ResizeBytes(size_t bytes) override;

        bool ReadAsync(WorkerPtr worker, DevicePtr dev) override;

        bool WriteAsync(WorkerPtr worker, DevicePtr dev) override;

        void *data() const override;

        void *data() override;

        void clear() override;

        void Create(size_t bytes, DevicePtr device) override;

        ArrayBasePtr CurrentArray() const override;

        float ReadOverhead(DevicePtr dev) override;

        float WriteOverhead(DevicePtr dev) override {
            return 0;
        }

        Ptr GetPtr() override;

        void log(el::base::type::ostream_t &os) const override;
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
