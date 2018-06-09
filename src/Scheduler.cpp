//
// Created by chnlkw on 6/4/18.
//

#include "Runnable.h"
#include "Task.h"

#define LG(x) CLOG(x, log_name_)

namespace Xuanwu {

    void Scheduler::AddTask(const TaskPtr &task) {
        for (const auto &src_w : task->DependTasks()) {
            if (auto src = src_w.lock()) {
                if (src->IsFinished())
                    continue;
                if (tasks_.find(src) == tasks_.end())
                    continue;
                if (tasks_[src].finished)
                    continue;
                LG(INFO) << *src << " Finishs before start " << *task;
                tasks_[src].next_tasks_when_finish_.insert(task);
                auto dev = tasks_[src].member_chosen_;
                tasks_[task].finish_depend_.Add(dev, src);
            }
        }
        for (const auto &src_w : task->RunAfterTasks()) {
            if (auto src = src_w.lock()) {
                if (src->IsFinished())
                    continue;
                if (tasks_.find(src) == tasks_.end())
                    continue;
                if (tasks_[src].started)
                    continue;
                LG(INFO) << *src << " Starts before start " << *task;
                tasks_[src].next_tasks_when_start_.insert(task);
                auto dev = tasks_[src].member_chosen_;
                tasks_[task].start_depend_.Add(dev, src);
            }

        }
        if (CheckTaskReady(task))
            LG(INFO) << *task << " becomes ready just added";
    }

    std::vector<std::pair<TaskPtr, Scheduler::Member>> Scheduler::FetchReadyTasks() {
        auto ret = std::move(ready_tasks_);
        ready_tasks_.clear();
        return std::move(ret);
    }

    void Scheduler::RunTask(const TaskPtr &task) {
        auto &node = tasks_[task];
        assert(node.member_chosen_);
        node.started = true;
        for (auto &t : node.next_tasks_when_start_) {
            tasks_[t].start_depend_.Del(node.member_chosen_, task);
            if (CheckTaskReady(t))
                LG(INFO) << *t << " becomes ready when RunTask " << *task;
        }
    }

    void Scheduler::FinishTask(const TaskPtr &task) {
        auto &node = tasks_[task];
        assert(node.member_chosen_);
        node.finished = true;
        for (auto &t : node.next_tasks_when_finish_) {
            tasks_[t].finish_depend_.Del(node.member_chosen_, task);
            if (CheckTaskReady(t))
                LG(INFO) << *t << " becomes ready when Finish " << *task;
        }
        tasks_.erase(task);
    }

    bool Scheduler::CheckTaskReady(const TaskPtr &task) {
        auto &node = tasks_[task];

        if (node.member_chosen_)
            return false;

        LG(DEBUG) << "CheckTaskReady " << *task;
        LG(DEBUG) << "\tstart_depends " << node;
        LG(DEBUG) << "\tfinish_depends " << node;

        if (node.start_depend_.NumMembers() > 1 || node.finish_depend_.NumMembers() > 1)
            return false;

        std::set<Member> members;
        if (node.start_depend_.NumMembers() == 1)
            members.insert(node.start_depend_.GetMember());
        if (node.finish_depend_.NumMembers() == 1)
            members.insert(node.finish_depend_.GetMember());

        if (members.size() > 1)
            return false;
        members.insert(f_selector_(task));
        if (members.size() > 1)
            return false;
        node.member_chosen_ = *members.begin();
        ready_tasks_.emplace_back(task, node.member_chosen_);
        LG(INFO) << "\t ready to run at " << *node.member_chosen_;

        for (auto &nxt : node.next_tasks_when_finish_) {
            tasks_[nxt].finish_depend_.Add(node.member_chosen_, task);
            LG(DEBUG) << "\t" << *nxt << " finish_depends " << node;
        }
        for (auto &nxt : node.next_tasks_when_start_) {
            tasks_[nxt].start_depend_.Add(node.member_chosen_, task);
            LG(DEBUG) << "\t" << *nxt << " start_depends " << node;
        }

        return true;
    }

    Scheduler::Scheduler(const char *log_name) : SchedulerBase(), log_name_(log_name) {}

    void SchedulerBase::RunTasks(const std::vector<TaskPtr> &ts) {
        for (auto &t : ts)
            RunTask(t);
    }

    void SchedulerBase::FinishTasks(const std::vector<TaskPtr> &ts) {
        for (auto &t : ts)
            FinishTask(t);
    }

    void Scheduler::Node::Depend::TaskSet::log(el::base::type::ostream_t &os) const {
        os << "(";
        for (auto &t : *this)
            os << *t << ", ";
        os << ")";
    }

    void Scheduler::Node::Depend::log(el::base::type::ostream_t &os) const {
        os << "[ depends: ";
        for (auto &p : depends_)
            os << *p.first << ":" << p.second << " , ";
        os << " nomenber: " << nomember_ << "]";
    }

    void Scheduler::Node::log(el::base::type::ostream_t &os) const {
        os << "Node[ start_depend: " << start_depend_ << " finish_depend: " << finish_depend_ << "]";
    }
}
