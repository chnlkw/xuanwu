#ifndef NUMA_DEVICE_H
#define NUMA_DEVICE_H

#include <xuanwu/DeviceBase.h>

namespace Xuanwu {
    class NumaDevice : public DeviceBase {
    public:

        NumaDevice();

        void RunTask(TaskPtr t) override;

        int ScoreRunTask(TaskPtr t) override;

        void log(el::base::type::ostream_t &os) const override;

    private:
        int numa_id_;

    };


}

#endif
