//
// Created by chnlkw on 6/4/18.
//

#include "Runnable.h"
#include "Task.h"

#define LG(x) CLOG(x, "Engine")

namespace Xuanwu {

    void Scheduler::AddTask(const TaskPtr &task) {
        for (const auto &depend_task_w : task->DependTasks()) {
            if (auto src = depend_task_w.lock()) {
                LG(INFO) << "AddEdge " << *src << " -> " << *task;
                if (src->IsFinished())
                    continue;
                tasks_[src].next_tasks_.push_back(task);
                tasks_[task].unfinished_depend_tasks_++;
                tasks_[task].nostart_depend_tasks_++;
                if (auto dev = tasks_[src].member_chosen_)
                    tasks_[task].devices_of_depend_tasks_.insert(dev);
            }
        }
        CheckTaskReady(task);
    }

    std::vector<std::pair<TaskPtr, Scheduler::Member>> Scheduler::FetchReadyTasks() {
        auto ret = std::move(ready_tasks_);
        ready_tasks_.clear();
        return std::move(ret);
    }

    void Scheduler::RunTask(const TaskPtr &t) {
        auto &node = tasks_[t];
        for (auto &t : node.next_tasks_) {
            --tasks_[t].nostart_depend_tasks_;
            CheckTaskReady(t);
        }
    }

    void Scheduler::FinishTask(const TaskPtr &task) {
        for (const auto &t : tasks_[task].next_tasks_) {
            --tasks_[t].unfinished_depend_tasks_;
            CheckTaskReady(t);
        }
        tasks_.erase(task);
    }

    void Scheduler::CheckTaskReady(const TaskPtr &task) {
        auto &node = tasks_[task];
        auto Choose = [&](Member mem_chosen) {
            for (auto &nxt : node.next_tasks_) {
                LG(DEBUG) << *task << " --> " << *nxt;
                tasks_[nxt].devices_of_depend_tasks_.insert(mem_chosen);
                for (auto &devdep : tasks_[nxt].devices_of_depend_tasks_) {
                    LG(DEBUG) << *nxt << " 's depend_tasks : " << *devdep;
                }
            }
            node.member_chosen_ = f_selector_(task);
            ready_tasks_.push_back(std::make_pair(task, node.member_chosen_));

        };
        if (node.member_chosen_)
            return;
        if (node.nostart_depend_tasks_ == 0 && node.devices_of_depend_tasks_.size() <= 1) {
            Member dev = f_selector_(task);
            LG(DEBUG) << *task << " 's dependent tasks = " << node.devices_of_depend_tasks_.size();
            for (auto &devdep : node.devices_of_depend_tasks_) {
                LG(DEBUG) << *task << " 's dependent task run at " << *devdep;
            }
            node.devices_of_depend_tasks_.insert(dev);
            if (node.devices_of_depend_tasks_.size() <= 1) {
                Choose(dev);
                LG(INFO) << *task << " is ready to run at " << *node.member_chosen_ << " because of locality";
                return;
            } else {
                LG(DEBUG) << *task << " choosed " << *dev << " but not ready";
            }
        }
        if (node.unfinished_depend_tasks_ == 0) {
            Choose(f_selector_(task));
            LG(INFO) << *task << " is ready to run at " << *node.member_chosen_;
        }
    }

}
