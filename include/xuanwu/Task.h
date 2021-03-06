//
// Created by chnlkw on 1/22/18.
//

#ifndef DMR_TASK_H
#define DMR_TASK_H

#include <memory>
#include <set>
#include <list>
#include <queue>
#include <utility>
#include "defs.h"
#include "cuda_utils.h"
#include "Xuanwu.h"
#include "MM.h"
#include "Array.h"
#include "Device.h"


//enum Flag {
//    Default = 0,
//    Shared = 1,
//    Exclusive = 2
//};

namespace Xuanwu {

    template<class V1, class V2>
    void Append(V1 &a, V2 b) {
        a.insert(a.end(), std::make_move_iterator(b.begin()), std::make_move_iterator(b.end()));
    }

    class TaskBase : public std::enable_shared_from_this<TaskBase>, public el::Loggable {
    public:
        struct Meta : public el::Loggable {
            DataBasePtr data;
            bool readable = false;
            bool writable = false;
            bool remote = false; // can remote access without migration

            Meta(DataBasePtr d, bool readable, bool writable, bool remote) :
                    data(std::move(d)),
                    readable(readable),
                    writable(writable),
                    remote(remote) {}

//            bool operator<(const Meta &that) const {
//                return priority > that.priority;
//            }

            void log(el::base::type::ostream_t &os) const override;
        };

        enum TaskType {
            Default = -1,
            Compute = 0,
            H2D = 1,
            D2D = 2,
            D2H = 3
        };
        TaskType type_ = Default;

    private:

        std::vector<Meta> metas_;
        std::vector<DataBasePtr> tmp_datas_;
//        std::vector<std::pair<LocalArrayGPU, DataBasePtr>> tmp_data_mapping_;
        bool finished = false;

        friend class Engine;

        std::string name_;
        int seq;

        std::unique_ptr<CPUTask> cputask_;
        std::unique_ptr<GPUTask> gputask_;

        std::vector<TaskPtr> depend_tasks_; // Strong : must start after previous task finish
        std::vector<TaskPtr> run_after_tasks_; // Weak : start after previous task start

    public:
        ~TaskBase() override;

        TaskBase(const TaskBase &) = delete;

        int Seq() const { return seq; }

        const auto &Metas() {
//            std::sort(metas_.begin(), metas_.end());
            return metas_;
        }

        virtual void Run(CPUWorker *) { LOG(FATAL) << "not implemented in CPUWorker : " << *this; };

        virtual void Run(GPUWorker *) { LOG(FATAL) << "not implemented in GPUWorker : " << *this; };

        void WaitFinish();

        virtual std::string Name() const { return name_; }

        void log(el::base::type::ostream_t &os) const override;

        bool IsFinished() const;

        explicit TaskBase(std::string name = "nonamed task");

        TaskBase(std::string name, std::unique_ptr<CPUTask> cputask, std::unique_ptr<GPUTask> gputask);

        void AddInputRemote(DataBasePtr data);

        void AddRemoteInputs(std::vector<DataBasePtr> data);

        void AddOutputRemote(DataBasePtr data);

        void AddRemoteOutputs(std::vector<DataBasePtr> data);

        void AddInput(DataBasePtr data);

        void AddInputs(std::vector<DataBasePtr> data);

        void AddOutput(DataBasePtr data);

        void AddOutputs(std::vector<DataBasePtr> data);

        void AddInOutput(DataBasePtr data);

        void AddInOutputs(std::vector<DataBasePtr> data);

        void AddTempData(DataBasePtr data);

//        void AddTempDataMapping(LocalArrayGPU, DataBasePtr);

//        auto &GetTempDataMappings() { return tmp_data_mapping_; }

        void AddDependency(std::vector<TaskPtr> tasks);

        const auto &DependTasks() const {
            return depend_tasks_;
        }

        const auto &RunAfterTasks() const {
            return run_after_tasks_;
        }

        void RunAfter(std::vector<TaskPtr> tasks);

        virtual void Finish() final;

        CPUTask *GetCPUTask() const;

        GPUTask *GetGPUTask() const;

        std::string &Name() {
            return name_;
        }

        auto& Type() {
            return type_;
        }

    };

    struct Context {
        virtual void Copy(Ptr dst, Ptr src, size_t bytes) = 0;

    };

    struct CPUContext : Context {
        CPUDevice *dev;
        WorkerPtr worker;

        CPUContext(CPUDevice *dev, WorkerPtr worker) : dev(dev), worker(worker) {}

        void Copy(Ptr dst, Ptr src, size_t bytes) override;
    };

    struct DeviceArrayBase {
        void *ptr;
        size_t bytes;

        __device__
        void Malloc(size_t b) {
            bytes = b;
            ptr = malloc(bytes);
            if (ptr == nullptr)
                printf("DeviceArray alloc failed ptr = %p bytes = %lu\n", ptr, bytes);
        }
    };

    template<class T>
    struct DeviceArray : DeviceArrayBase {
        __device__
        void Alloc(size_t n) {
            Malloc(n * sizeof(T));
        }

        __device__
                T
        *

        data() { return (T *) ptr; }

        __device__
                T
        *

        data() const { return (T *) ptr; }

        __device__
                T
        &

        operator[](size_t idx) {
            return data()[idx];
        }

        __device__
                T

        operator[](size_t idx) const {
            return data()[idx];
        }
    };

    struct LocalArrayGPU {

        DeviceArrayBase *d_arr = nullptr;

        LocalArrayGPU() = default;

        void Create() {
            CUDA_CALL(cudaMalloc, &d_arr, sizeof(DeviceArrayBase));
        }

        DeviceArrayBase *GetArrPtr() {
            return d_arr;
//        return thrust::raw_pointer_cast(arr.data());
        }

    };

    template<class T>
    struct LocalArray : LocalArrayGPU {

        DeviceArray<T> *GetArrPtr() {
            return static_cast<DeviceArray<T> *>(d_arr);
//        return thrust::raw_pointer_cast(arr.data());
        }

    };

    struct GPUContext : Context {
        MMBase *mm;
        GPUDevice *dev;
        cudaStream_t stream;
//        GPUWorker *worker;
        WorkerPtr worker;
        TaskPtr task;

        GPUContext(MMBase *mm, GPUDevice *dev, cudaStream_t stream, WorkerPtr worker, TaskPtr task) :
                mm(mm),
                dev(dev),
                stream(stream),
                worker(worker),
                task(task) {
        }

        ArrayBase *MakeArrayBase(size_t bytes);

        /*
        template<class T>
        DeviceArray<T> *MakeLocalMapping(Data<T> &d) {
            LocalArray<T> g;
            g.Create();
            task->AddTempDataMapping(g, d);
            return g.GetArrPtr();
        }
         */

        template<class T>
        T *Alloc(size_t count) {
            return static_cast<T *>(MakeArrayBase(count * sizeof(T))->data());
        }

        void Copy(Ptr dst, Ptr src, size_t bytes) override {
            GPUCopy(dst, src, bytes, dev->GPUID(), stream);
        }

    };

    struct GPUTask : public std::function<void(GPUContext)> {
        int score;

        explicit GPUTask(std::function<void(GPUContext)> f, int score = 2) :
                std::function<void(GPUContext)>(f),
                score(score) {
        }
    };

    struct CPUTask : public std::function<void(CPUContext)> {
        int score;

        explicit CPUTask(std::function<void(CPUContext)> f, int score = 1) :
                std::function<void(CPUContext)>(f), score(score) {}
    };

}

#endif //DMR_TASK_H
