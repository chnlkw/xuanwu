//
// Created by chnlkw on 2/28/18.
//

#include <functional>
#include "Task.h"
#include "Worker.h"
#include "Data.h"
#include "Device.h"

namespace Xuanwu {
#define LG(x) CLOG(x, "Task")

    TaskBase::~TaskBase() {
        LG(INFO) << "Destory " << Name();
    }

    TaskBase::TaskBase(std::string name, std::unique_ptr<CPUTask> cputask, std::unique_ptr<GPUTask> gputask) :
            name_(std::move(name)), cputask_(std::move(cputask)), gputask_(std::move(gputask)) {
    }

    TaskBase::TaskBase(std::string name) :
            name_(std::move(name)) {
        LG(INFO) << "Create " << Name();
    }

    void TaskBase::AddInput(DataBasePtr data) {
        LG(INFO) << "AddInput " << *data;
        metas_.emplace_back(data, true, false);
    }

    void TaskBase::AddInputs(std::vector<DataBasePtr> data) {
        for (auto &d : data)
            AddInput(d);
    }

    void TaskBase::AddOutput(DataBasePtr data) {
        LG(INFO) << "AddOutput " << *data;
        metas_.emplace_back(data, false, true);
    }

    void TaskBase::AddOutputs(std::vector<DataBasePtr> data) {
        for (auto &d : data)
            AddOutput(d);
    }

    void TaskBase::AddInOutput(DataBasePtr data) {
        LG(INFO) << "AddInOutput " << *data;
        metas_.emplace_back(data, true, true);
    }

    void TaskBase::AddInOutputs(std::vector<DataBasePtr> data) {
        for (auto &d : data)
            AddInOutput(d);
    }

    void TaskBase::Finish() {
        tmp_arrays_.clear();
        finished = true;
    }

    CPUTask *TaskBase::GetCPUTask() const { return cputask_.get(); }

    GPUTask *TaskBase::GetGPUTask() const { return gputask_.get(); }

    bool TaskBase::IsFinished() const {
        return finished;
    }

    void TaskBase::AddTempArray(ArrayBasePtr arr) {
        tmp_arrays_.push_back(arr);
    }

    void TaskBase::AddTempDataMapping(LocalArrayGPU arr, DataBasePtr d) {
        tmp_data_mapping_.emplace_back(arr, d);
    }

    ArrayBasePtr GPUContext::MakeArrayBase(size_t bytes) {
        auto arr = std::make_shared<ArrayBase>(bytes, mm->GetAllocatorByDevice(dev));
        task->AddTempArray(arr);
        return arr;
    }

}
