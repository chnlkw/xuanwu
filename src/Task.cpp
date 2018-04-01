//
// Created by chnlkw on 2/28/18.
//

#include <functional>
#include "Task.h"
#include "Worker.h"
#include "Data.h"
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
    }

}
