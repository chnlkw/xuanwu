//
// Created by chnlkw on 3/2/18.
//

#include "Worker.h"
#include "Device.h"
#include "Task.h"

void TaskBase::log(el::base::type::ostream_t &os) const {
    os << "Task[" << Name() << "]";
    if (finished) os << " Finished";
}

void WorkerBase::log(el::base::type::ostream_t &os) const {
    os << "Worker["<< id_ << ", " << *device_ << "]";
}

void DeviceBase::log(el::base::type::ostream_t &os) const {
    os << "Device[" << Id() << "]";
}

void TaskBase::Meta::log(el::base::type::ostream_t &os) const {
    os << "[Meta] "
       << data << " "
       << (is_read_only ? "R " : "W ")
       << priority << ". ";
}

