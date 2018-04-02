//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_DEVICE_H
#define DMR_DEVICE_H

#include "DeviceBase.h"
#include "Config.h"

namespace Xuanwu {
    class CPUDevice : public DeviceBase {
    public:

        CPUDevice();

        void RunTask(TaskPtr t) override;

        int ScoreRunTask(TaskPtr t) override;

        void log(el::base::type::ostream_t &os) const override;
    };

    class GPUDevice : public DeviceBase {
        int gpu_id_;

        static int GetGPUId() {
            static int id = 0;
            return id++;
        }

    public:
        using NumWorkers = Strong<unsigned, 2>;
        GPUDevice(NumWorkers);

        void RunTask(TaskPtr t) override;

        int ScoreRunTask(TaskPtr t) override;

        int GPUID() const {
            return gpu_id_;
        }

        void log(el::base::type::ostream_t &os) const override;

    };

}
#endif //DMR_DEVICE_H
