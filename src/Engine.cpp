//
// Created by chnlkw on 11/28/17.
//

#include <easylogging++.h>
#include <atomic>
#include "Engine.h"
#include "Task.h"
#include "Data.h"

#define LG(x) CLOG(x, "Engine")
namespace Xuanwu {
    void TaskBase::WaitFinish() {
        while (!IsFinished() && Tick());
        if (!IsFinished())
            throw std::runtime_error("Task unfinished, while engine ends");
    }

    void DataBase::Wait() const {
        // Wait all tasks finish
        for (auto &s : tasks_scheduled_) {
            if (auto t = s.lock()) {
                t->WaitFinish();
            }
        }
        tasks_scheduled_.clear();
    }

    Engine::Engine(std::unique_ptr<MyDeviceGroup> g) :
            devices_(std::make_move_iterator(g->begin()), std::make_move_iterator(g->end())) {
        LG(INFO) << "engine created with and devices.size() = "
                 << devices_.size();
        for (auto &d : devices_)
            device_entries_.insert(d.get());
    }

//Engine::Engine(std::unique_ptr<CPUDevice> cpu_device, std::unique_ptr<DevicesGroup> devices_group) :
//        cpu_device_(std::move(cpu_device)),
//        devices_(std::move(devices_group->FetchDevices())) {
//    LG(INFO) << "engine created with cpudevice=" << cpu_device_.get() << " and devices.size() = " << devices_.size();
//}

//Engine::Engine( const di::extension::ifactory<DeviceBase, int>& device_factory) {
//    auto d1 = device_factory.create(-1);
//    auto d2 = device_factory.create(-1);
//}

    void Engine::AddEdge(TaskPtr src, TaskPtr dst) {
        if (src->finished)
            return;
        tasks_[src].next_tasks_.push_back(dst);
        tasks_[dst].in_degree++;
    }

    bool Engine::Tick() {
        static std::atomic<bool> ticking;
        assert(!ticking);
        ticking = true;
        LG(DEBUG) << "Tick";

        GetCompleteTasks();

        if (Empty()) {
            ticking = false;
            return false;
        }

        std::sort(ready_tasks_.begin(), ready_tasks_.end(), [](auto &a, auto &b) { return a->Seq() < b->Seq(); });
        for (auto it = ready_tasks_.begin(); it != ready_tasks_.end();) {
            RunTask(*it);
            it = ready_tasks_.erase(it);
        }
        ticking = false;
        return true;
    }

    DevicePtr Engine::ChooseDevice(TaskPtr t) {
        LG(DEBUG) << " Choose Device of " << *t;
        std::map<DevicePtr, float> dev_score;
        for (auto &dev : devices_) {
            float score_overhead = 0;
            for (auto &m : t->Metas()) {
                if (m.readable)
                    score_overhead += 1.0 / (1 + m.data->ReadOverhead(dev.get()));
                else
                    score_overhead += 1.0 / (1 + m.data->WriteOverhead(dev.get()));
            }
            float score_overload = 10.0f / (1 + dev->NumRunningTasks());
            float score_dev_run_task = 1000 * dev->ScoreRunTask(t);
            dev_score[dev.get()] = score_overhead + score_overload + score_dev_run_task;
            LG(DEBUG) << *dev << " has score " << dev_score[dev.get()] <<
                      "=" << score_overhead << "+" << score_overload << "+" << score_dev_run_task;
//            LG(DEBUG) << *t << "is runnable on " << *dev;
        }
        assert(!dev_score.empty());
        for (auto &m : t->Metas()) {
            if (!m.readable) {
                if (auto dev = data_steps_[m.data->GetUID()].DeviceChosen()) {
                    LG(DEBUG) << *dev << " has been chosen by data " << m.data;
                    for (auto it = dev_score.begin(); it != dev_score.end();) {
                        if (it->first != dev) {
                            LG(DEBUG) << *it->first << " has been erased by data " << m.data;
                            it = dev_score.erase(it);
                        } else
                            ++it;
                    }
                }
            }
        }
        assert(!dev_score.empty());
        DevicePtr dev_chosen = std::max_element(dev_score.begin(), dev_score.end(),
                                                [](auto a, auto b) { return a.second < b.second; })->first;
//    return ChooseRunnable(devices_.begin(), devices_.end()).get();
        LG(INFO) << "Choose " << *dev_chosen << " to run " << *t;
        for (auto &m : t->Metas()) {
            if (!m.readable) {
                data_steps_[m.data->GetUID()].ChooseDevice(dev_chosen);
            }
        }
        return dev_chosen;
    }

    void Engine::CheckTaskReady(TaskPtr task) {
        if (tasks_[task].in_degree == 0) {
            ready_tasks_.push_back(task);
            LG(INFO) << *task << " is ready";
        }
    }

    void Engine::FinishTask(TaskPtr task) {
        LG(INFO) << "Finish task " << *task;
        for (auto t : tasks_[task].next_tasks_) {
            --tasks_[t].in_degree;
            CheckTaskReady(t);
        }
        for (auto &m : task->Metas()) {
            data_steps_[m.data->GetUID()].UnregisterTask(task);
        }
        task->Finish();
        tasks_.erase(task);
        num_running_tasks_--;
    }

    TaskBase &Engine::AddTask(TaskPtr task) {
        LG(INFO) << "AddTask " << *task;
        num_running_tasks_++;
        for (auto &m : task->Metas()) {
            m.data->RegisterTask(task);
            LG(DEBUG) << "RegisterTask " << *task << " " << m;
            Flag f;
            if (m.readable && m.writable)
                f = Flag::ReadWrite;
            else if (m.readable)
                f = Flag::Read;
            else if (m.writable)
                f = Flag::Write;

            const auto &depend_tasks = data_steps_[m.data->GetUID()].RegisterTask(task, f);
            for (const auto &depend_task : depend_tasks) {
//                if (!depend_task.expired())
//                    AddEdge(depend_task.lock(), task);
                LG(INFO) << "AddEdge " << *depend_task << " -> " << *task << " " << *m.data;
                AddEdge(depend_task, task);
            }
        }
        CheckTaskReady(task);
        return *task;
    }

    const std::vector<std::shared_ptr<DeviceBase>> &Engine::GetDevices() const {
        return devices_;
    }

    std::vector<TaskPtr> Engine::GetCompleteTasks() {
        std::vector<TaskPtr> complete_tasks;
        for (auto &d : devices_) {
            for (auto &t : d->GetCompleteTasks()) {
                FinishTask(t);
                complete_tasks.push_back(t);
            }
        }
        return complete_tasks;
    }

    void Engine::RunTask(TaskPtr task) {
        LG(INFO) << "Engine Run " << *task;
        DevicePtr d = ChooseDevice(task);
        d->RunTask(task);
    }

    std::vector<TaskPtr> Engine::DataStep::RegisterTask(TaskPtr task, Flag f) {
        if (steps_.empty() || f == Flag::ReadWrite || steps_.back().flag != f) {
            steps_.emplace_back(f);
        }
        steps_.back().tasks.insert(task);
        if (steps_.size() >= 2) {
            auto &tasks = steps_[steps_.size() - 2].tasks;
            return {tasks.begin(), tasks.end()};
        } else
            return {};
    }

    void Engine::DataStep::UnregisterTask(TaskPtr task) {
        assert(!steps_.empty());
        assert(steps_[0].tasks.count(task));
        steps_[0].tasks.erase(task);
        if (steps_[0].tasks.empty())
            steps_.pop_front();
    }

    void Engine::DataStep::ChooseDevice(DevicePtr device) {
        assert(DeviceChosen() == nullptr || DeviceChosen() == device);
        assert(steps_.at(0).flag == Flag::Write);
        steps_.at(0).device_chosen_ = device;
    }

    DevicePtr Engine::DataStep::DeviceChosen() const {
        assert(steps_.at(0).flag == Flag::Write);
        return steps_.at(0).device_chosen_;
    }
}
