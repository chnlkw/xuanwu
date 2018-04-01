//
// Created by chnlkw on 3/2/18.
//

#include "Worker.h"
#include "Device.h"
#include "Task.h"
namespace Xuanwu {
    void TaskBase::log(el::base::type::ostream_t &os) const {
        os << "Task[" << Name() << "]";
        if (finished) os << " Finished";
    }

    void WorkerBase::log(el::base::type::ostream_t &os) const {
        os << "Worker[" << id_ << ", " << *device_ << "]";
    }

    void CPUDevice::log(el::base::type::ostream_t &os) const {
        os << "CPUDevice[]";
    }

    void GPUDevice::log(el::base::type::ostream_t &os) const {
        os << "GPUDevice[" << GPUID() << "]";
    }

    void TaskBase::Meta::log(el::base::type::ostream_t &os) const {
        os << "[Meta] "
           << data << " "
           << (is_read_only ? "R " : "W ")
           << priority << ". ";
    }

}
