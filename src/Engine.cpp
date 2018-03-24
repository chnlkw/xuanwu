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
        while (!IsFinished() && Xuanwu::Tick());
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

    Engine::Engine(std::shared_ptr<CPUDevice> cpu_device, std::unique_ptr<MyDeviceGroup> g) :
            cpu_device_(std::move(cpu_device)),
            devices_(std::make_move_iterator(g->begin()), std::make_move_iterator(g->end())) {
        LG(INFO) << "engine created with cpu_device=" << cpu_device_.get() << " and devices.size() = "
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
        LG(DEBUG) << "AddEdge " << *src << " -> " << *dst;
        tasks_[src].next_tasks_.push_back(dst);
        tasks_[dst].in_degree++;
    }

    bool Engine::Tick() {
        static std::atomic<bool> ticking;
        assert(!ticking);
        ticking = true;
        LG(INFO) << "Tick";


        GetCompleteTasks();

        if (Empty()) {
            ticking = false;
            return false;
        }

        for (auto t : ready_tasks_) {
            DevicePtr d = ChooseDevice(t);
            d->RunTask(t);
        }
        ready_tasks_.clear();
        ticking = false;
        return true;
    }

    DevicePtr Engine::ChooseDevice(TaskPtr t) {
        std::map<DevicePtr, float> data_score;
        for (auto &m : t->GetMetas()) {
            LG(DEBUG) << m << " replica count = " << m.data->last_state_.replicas.size();
            for (DevicePtr dev : m.data->DevicesPrefered()) {
                data_score[dev] += m.priority;
            }
        }
        std::map<DevicePtr, float> dev_score;
        for (auto &dev : devices_) {
            dev_score[dev.get()] = data_score[dev.get()] + 1.0f / (1 + dev->NumRunningTasks());
            dev_score[dev.get()] += 1000 * dev->ScoreRunTask(t);
            LG(DEBUG) << *dev << " has score " << dev_score[dev.get()];
//            LG(DEBUG) << *t << "is runnable on " << *dev;
        }
        assert(!dev_score.empty());
        DevicePtr dev_choosed = std::max_element(dev_score.begin(), dev_score.end(),
                                                 [](auto a, auto b) { return a.second < b.second; })->first;
//    return ChooseRunnable(devices_.begin(), devices_.end()).get();
        LG(INFO) << "Choose " << *dev_choosed << " to run " << *t;
        return dev_choosed;
    }

    void Engine::CheckTaskReady(TaskPtr task) {
        if (tasks_[task].in_degree == 0)
            ready_tasks_.push_back(task);
    }

    void Engine::FinishTask(TaskPtr task) {
        LG(INFO) << "Finish task " << *task;
        for (auto t : tasks_[task].next_tasks_) {
            --tasks_[t].in_degree;
            CheckTaskReady(t);
        }
        task->Finish();
        tasks_.erase(task);
        num_running_tasks_--;
    }

    TaskBase &Engine::AddTask(TaskPtr task) {
        RunTask(task);
        LG(INFO) << "AddTask " << *task;

        return *task;
    }

    const std::vector<std::shared_ptr<DeviceBase>> &Engine::GetDevices() const {
        return devices_;
    }

    const DevicePtr Engine::CpuDevice() const { return cpu_device_.get(); }

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
        num_running_tasks_++;
        for (auto &m : task->GetMetas()) {
            const auto &depend_tasks = m.data->RegisterTask(task, m.is_read_only);
            for (const auto &depend_task : depend_tasks) {
                if (!depend_task.expired())
                    AddEdge(depend_task.lock(), task);
            }
        }
        CheckTaskReady(task);
    }


}
