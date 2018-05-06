//
// Created by chnlkw on 11/28/17.
//

#ifndef LDA_ENGINE_H
#define LDA_ENGINE_H

#include "cuda_utils.h"
#include <deque>
#include <unordered_map>
#include "defs.h"
#include "Device.h"
#include "DevicesGroup.h"
#include "Runnable.h"

namespace Xuanwu {
    class Engine : public Runnable {
        struct Node {
            size_t in_degree = 0;
            std::vector<TaskPtr> next_tasks_;
        };
        std::map<TaskPtr, Node> tasks_;

//    std::vector<WorkerPtr> workers_;
        std::vector<TaskPtr> ready_tasks_;

        std::vector<std::shared_ptr<DeviceBase>> devices_;
        std::set<DevicePtr> device_entries_;

        size_t num_running_tasks_ = 0;

        enum class Flag {
            Read = 1,
            Write = 2,
            ReadWrite = 3
        };

        class DataStep {
            struct Step {
                Flag flag;
                std::set<TaskPtr> tasks;

                DevicePtr device_chosen_ = nullptr;

                Step(Flag f) : flag(f) {}
            };

            std::deque<Step> steps_;
        public:

            DataStep() = default;

            std::vector<TaskPtr> RegisterTask(TaskPtr task, Flag flag);

            void UnregisterTask(TaskPtr task);

            void ChooseDevice(DevicePtr device);

            DevicePtr DeviceChosen() const;

        };

        std::unordered_map<size_t, DataStep> data_steps_;

    public:

        Engine(std::unique_ptr<MyDeviceGroup> g);

        size_t NumRunningTasks() const override {
            return num_running_tasks_;
        }

        const std::vector<std::shared_ptr<DeviceBase>> &GetDevices() const;

        TaskBase &AddTask(TaskPtr task);

        template<class Task, class... Args>
        TaskBase &AddTask(Args &&... args) {
            auto t = std::make_shared<Task>(std::forward<Args>(args)...);
            return AddTask(t);
        }

        void RunTask(TaskPtr t) override;

        bool Tick();

        std::vector<TaskPtr> GetCompleteTasks() override;

        DevicePtr ChooseDevice(TaskPtr t);

    private:
        void AddEdge(TaskPtr src, TaskPtr dst);

        void CheckTaskReady(TaskPtr task);

        void FinishTask(TaskPtr task);
    };

}
#endif //LDA_ENGINE_H
