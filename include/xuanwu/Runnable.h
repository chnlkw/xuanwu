//
// Created by chnlkw on 3/5/18.
//

#ifndef DMR_RUNNABLE_H
#define DMR_RUNNABLE_H

#include "defs.h"

namespace Xuanwu {
    class Runnable : public el::Loggable {
    public:
        virtual void RunTask(TaskPtr) = 0;

        virtual size_t NumRunningTasks() const = 0;

        virtual std::vector<TaskPtr> GetCompleteTasks() = 0;

        virtual bool Empty() const {
            return NumRunningTasks() == 0;
        }

        template<class It>
        static auto &ChooseRunnable(It beg, It end) {
            assert(beg != end);
            It ret;
            size_t best = std::numeric_limits<size_t>::max();
            while (beg != end) {
                if (best > (*beg)->NumRunningTasks()) {
                    best = (*beg)->NumRunningTasks();
                    ret = beg;
                }
                ++beg;
            }
            return *ret;
        }

        void log(el::base::type::ostream_t &os) const override {
            os << "Runnable[]";
        }
    };

    class SchedulerBase {
    protected:
        using Member = Runnable*;
    public:
        virtual void AddTask(const TaskPtr &t) = 0;

        virtual void RunTask(const TaskPtr &t) = 0;

        virtual void FinishTask(const TaskPtr &t) = 0;

        virtual std::vector<std::pair<TaskPtr, Member>> FetchReadyTasks() = 0;

        std::function<Member(TaskPtr)> f_selector_;

        void SetSelector(std::function<Member(TaskPtr)> f_selector) {
            f_selector_ = f_selector;
        }
    };

    class Scheduler : public SchedulerBase {
        struct Node {
            size_t unfinished_depend_tasks_ = 0;
            size_t nostart_depend_tasks_ = 0;
            std::set<Member> devices_of_depend_tasks_;
            Member member_chosen_ = nullptr;
            std::vector<TaskPtr> next_tasks_;
        };
        std::map<TaskPtr, Node> tasks_;
        std::vector<std::pair<TaskPtr, Member>> ready_tasks_;

        void CheckTaskReady(const TaskPtr &task);

    public:
        void AddTask(const TaskPtr &t) override;

        void RunTask(const TaskPtr &t) override;

        void FinishTask(const TaskPtr &t) override;

        std::vector<std::pair<TaskPtr, Member>> FetchReadyTasks() override;
    };

};
#endif //DMR_RUNNABLE_H
