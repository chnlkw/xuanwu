//
// Created by chnlkw on 2/28/18.
//

#include <functional>
#include "Task.h"
#include "Worker.h"
#include "Data.h"

#define LG(x) CLOG(x, "Task")

TaskBase::~TaskBase() {
    LG(INFO) << "Destory " << Name();
}

void TaskBase::PrepareData(DevicePtr dev, cudaStream_t stream) {
    LG(INFO) << " PrepareData for " << *this;
    for (auto &m : GetMetas()) {
        if (m.is_read_only) {
            m.data->ReadAsync(shared_from_this(), dev, stream);
        } else {
            m.data->WriteAsync(shared_from_this(), dev, stream);
        }
    }
}

TaskBase::TaskBase(std::string name, std::unique_ptr<CPUTask> cputask, std::unique_ptr<GPUTask> gputask) :
        name_(std::move(name)), cputask_(std::move(cputask)), gputask_(std::move(gputask)) {
}

TaskBase::TaskBase(std::string name) :
        name_(std::move(name)) {
}

