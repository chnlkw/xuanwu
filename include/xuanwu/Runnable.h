//
// Created by chnlkw on 3/5/18.
//

#ifndef DMR_RUNNABLE_H
#define DMR_RUNNABLE_H

#include "defs.h"
#include <cassert>

namespace Xuanwu {
    class Runnable : public el::Loggable {
    public:
        /// @param tasks [in] Tasks to run
        /// @returns Completed tasks
        virtual std::vector<TaskPtr> RunTasks(std::vector<TaskPtr> tasks) = 0;

        virtual size_t NumRunningTasks() const = 0;

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
        using Member = Runnable *;
    public:
        virtual void AddTask(const TaskPtr &t) = 0;

        virtual void RunTask(const TaskPtr &t) = 0;

        virtual void FinishTask(const TaskPtr &t) = 0;

        void RunTasks(const std::vector<TaskPtr> &ts);

        void FinishTasks(const std::vector<TaskPtr> &ts);

        virtual std::vector<std::pair<TaskPtr, Member>> FetchReadyTasks() = 0;

        std::function<Member(TaskPtr)> f_selector_;

        void SetSelector(std::function<Member(TaskPtr)> f_selector) {
            f_selector_ = f_selector;
        }
    };

    class Scheduler : public SchedulerBase {
        struct Node : el::Loggable {
            struct Depend : el::Loggable {
                struct TaskSet : std::set<TaskPtr>, el::Loggable {
                    void log(el::base::type::ostream_t &os) const override;
                };
                std::map<Member, TaskSet> depends_;
                TaskSet nomember_;

                void Add(Member m, const TaskPtr &t) {
                    if (m) {
                        if (nomember_.count(t))
                            nomember_.erase(t);
                        depends_[m].insert(t);
                    } else {
                        nomember_.insert(t);
                    }
                }

                void Del(Member m, const TaskPtr &t) {
                    assert(depends_.find(m) != depends_.end());
                    assert(depends_[m].find(t) != depends_[m].end());
                    depends_[m].erase(t);
                    if (depends_[m].empty()) {
                        depends_.erase(m);
                    }
                }

                size_t NumMembers() const {
                    return depends_.size() + nomember_.size() * 100;
                }

                Member GetMember() const {
                    assert(depends_.size() > 0);
                    return depends_.begin()->first;
                }

                void log(el::base::type::ostream_t &os) const override;
            };

            Member member_chosen_ = nullptr;
            Depend start_depend_, finish_depend_;
            std::set<TaskPtr> next_tasks_when_finish_, next_tasks_when_start_;
            bool started = false;
            bool finished = false;

            void log(el::base::type::ostream_t &os) const override;
        };

        std::map<TaskPtr, Node> tasks_;
        std::vector<std::pair<TaskPtr, Member>> ready_tasks_;
        const char *log_name_;

        bool CheckTaskReady(const TaskPtr &task);

    public:
        Scheduler(const char *log_name);

        void AddTask(const TaskPtr &t) override;

        void RunTask(const TaskPtr &task) override;

        void FinishTask(const TaskPtr &t) override;

        std::vector<std::pair<TaskPtr, Member>> FetchReadyTasks() override;
    };

};
#endif //DMR_RUNNABLE_H
