//
// Created by chnlkw on 3/13/18.
//

#ifndef DMR_DEVICEBASE_H
#define DMR_DEVICEBASE_H

#include <queue>
#include "defs.h"
#include "cuda_utils.h"
#include "Runnable.h"

namespace Xuanwu {
    class DeviceBase : public std::enable_shared_from_this<DeviceBase>, public Runnable {
    protected:
        std::vector<std::unique_ptr<WorkerBase>> workers_;

    public:
        explicit DeviceBase();

        virtual ~DeviceBase();

        virtual int ScoreRunTask(TaskPtr t);

        std::vector<TaskPtr> GetCompleteTasks() override;

        const auto &Workers() const {
            return workers_;
        }

        size_t NumRunningTasks() const override;

    };

}
#endif //DMR_DEVICEBASE_H
