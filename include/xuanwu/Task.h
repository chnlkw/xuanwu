//
// Created by chnlkw on 1/22/18.
//

#ifndef DMR_TASK_H
#define DMR_TASK_H

#include <memory>
#include <set>
#include <list>
#include <queue>

#include "defs.h"
#include "cuda_utils.h"
#include "Car.h"

//enum Flag {
//    Default = 0,
//    Shared = 1,
//    Exclusive = 2
//};

struct GPUTask : public std::function<void(GPUWorker *)> {
    int score;

    explicit GPUTask(std::function<void(GPUWorker *)> f, int score = 2) :
            std::function<void(GPUWorker *)>(f),
            score(score) {
    }
};

struct CPUTask : public std::function<void(CPUWorker *)> {
    int score;

    explicit CPUTask(std::function<void(CPUWorker *)> f, int score = 1) :
            std::function<void(CPUWorker *)>(f), score(score) {}
};

class TaskBase : public std::enable_shared_from_this<TaskBase>, public el::Loggable {
    TaskBase(const TaskBase &) = delete;

    struct Meta : public el::Loggable {
        DataBasePtr data;
        bool is_read_only = true;
        int priority = 0;

        Meta(DataBasePtr d, bool b, int p) :
                data(d),
                is_read_only(b),
                priority(p) {}

        bool operator<(const Meta &that) const {
            return priority > that.priority;
        }

        void log(el::base::type::ostream_t &os) const override;
    };

    std::vector<Meta> metas_;
    bool finished = false;
    Engine &engine_;

    friend class Engine;

    friend class WithOutputs;

    friend class WithInputs;

    std::string name_;

    std::unique_ptr<CPUTask> cputask_;
    std::unique_ptr<GPUTask> gputask_;

public:
    ~TaskBase() override;

//    template<class Worker>
//    void Run(Worker *t) {
//        RunWorker(t);
//    }

    void PrepareData(DevicePtr dev, cudaStream_t stream);

    const auto &GetMetas() {
        std::sort(metas_.begin(), metas_.end());
        return metas_;
    }
//    const std::vector<DataBasePtr> &GetInputs() const {
//        return inputs_;
//    }
//
//    const std::vector<DataBasePtr> &GetOutputs() const {
//        return outputs_;
//    }

//    TaskBase &Prefer(WorkerPtr w) {
//        worker_prefered_.insert(w);
//        return *this;
//    }

    virtual void Run(CPUWorker *) { LOG(FATAL) << "not implemented in CPUWorker : " << *this;};

    virtual void Run(GPUWorker *) { LOG(FATAL) << "not implemented in GPUWorker : " << *this;};

    void WaitFinish();

    virtual std::string Name() const { return name_; }

    void log(el::base::type::ostream_t &os) const override;

    bool IsFinished() const {
        return finished;
    }

public:
    explicit TaskBase(std::string name = "nonamed task");

    TaskBase(std::string name, std::unique_ptr<CPUTask> cputask, std::unique_ptr<GPUTask> gputask);

    void AddInput(DataBasePtr data, int priority = 1) {
        metas_.push_back(Meta{data, true, priority});
    }

    void AddInputs(std::vector<DataBasePtr> data, int priority = 1) {
        for (auto &d : data)
            AddInput(d, priority);
    }

    void AddOutput(DataBasePtr data, int priority = 2) {
        metas_.push_back(Meta{data, false, priority});
    }

    void AddOutputs(std::vector<DataBasePtr> data, int priority = 2) {
        for (auto &d : data)
            AddOutput(d, priority);
    }

    void Finish() {
        finished = true;
    }

    CPUTask *GetCPUTask() const { return cputask_.get(); }

    GPUTask *GetGPUTask() const { return gputask_.get(); }
};


#endif //DMR_TASK_H
