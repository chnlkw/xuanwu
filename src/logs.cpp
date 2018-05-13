//
// Created by chnlkw on 3/2/18.
//

#include "Worker.h"
#include "Device.h"
#include "Task.h"
#include "Data.h"

namespace Xuanwu {
    void DataBase::log(el::base::type::ostream_t &os) const {
        os << "Data[" << Name() << "]";
        if (device_pinned_)
            os << "Pin[" << *device_pinned_ << "]";
    }

    void DataImpl::log(el::base::type::ostream_t &os) const {
        os << "Data[#" << GetUID() << ":" << Name() << "]";
        os << " size=" << this->Bytes();
        os << " (r:";
        for (auto &r : replicas) os << " " << *r.first << " " << r.second.first->data();
        os << ")";
//        os << "(i:";
//        for (auto &i : invalids) os << " " << *i.first << " " << i.second->data();
        os << ")";
    }

    void TaskBase::log(el::base::type::ostream_t &os) const {
        os << "Task[" << seq << "_" << Name() << "]";
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
           << *data << " "
           << (readable ? "R" : "")
           << (writable ? "W" : "");
    }

}
