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
    class DeviceBase : public el::Loggable, public std::enable_shared_from_this<DeviceBase>, public Runnable {
        std::unique_ptr<AllocatorBase> allocator_;
    protected:
        std::vector<std::unique_ptr<WorkerBase>> workers_;

    public:
        explicit DeviceBase(std::unique_ptr<AllocatorBase> allocator);

        virtual ~DeviceBase();

        AllocatorPtr GetAllocator() {
            return allocator_.get();
        }

        virtual int ScoreRunTask(TaskPtr t);

        std::vector<TaskPtr> GetCompleteTasks() override;

        const auto &Workers() const {
            return workers_;
        }

        int Id() const;

        void log(el::base::type::ostream_t &os) const override;
    };

}
#endif //DMR_DEVICEBASE_H
