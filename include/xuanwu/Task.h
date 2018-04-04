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

//enum Flag {
//    Default = 0,
//    Shared = 1,
//    Exclusive = 2
//};

namespace Xuanwu {
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
        struct Meta : public el::Loggable {
            DataBasePtr data;
            bool readable = false;
            bool writable = false;

            Meta(DataBasePtr d, bool readable, bool writable) :
                    data(std::move(d)),
                    readable(readable),
                    writable(writable) {}

//            bool operator<(const Meta &that) const {
//                return priority > that.priority;
//            }

            void log(el::base::type::ostream_t &os) const override;
        };

        std::vector<Meta> metas_;
        bool finished = false;

        friend class Engine;

        friend class WithOutputs;

        friend class WithInputs;

        std::string name_;

        std::unique_ptr<CPUTask> cputask_;
        std::unique_ptr<GPUTask> gputask_;

    public:
        ~TaskBase() override;

        TaskBase(const TaskBase &) = delete;

        const auto &GetMetas() const {
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

        void AddInput(DataBasePtr data);

        void AddInputs(std::vector<DataBasePtr> data);

        void AddOutput(DataBasePtr data);

        void AddOutputs(std::vector<DataBasePtr> data);

        void AddInOutput(DataBasePtr data);

        void AddInOutputs(std::vector<DataBasePtr> data);

        void Finish();

        CPUTask *GetCPUTask() const;

        GPUTask *GetGPUTask() const;
    };

}

#endif //DMR_TASK_H
