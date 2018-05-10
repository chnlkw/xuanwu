//
// Created by chnlkw on 4/15/18.
//

#ifndef XUANWU_EVENT_H
#define XUANWU_EVENT_H

#include "cuda_utils.h"

namespace Xuanwu {

    class EventBase {
    public:
        EventBase() = default;

        EventBase(const EventBase &) = delete;

        virtual ~EventBase() = default;

        virtual bool QueryFinished() = 0;

    };

    using Event = std::shared_ptr<EventBase>;

    class EventDummy : public EventBase {
        bool QueryFinished() override;
    };

    class EventGPU : public EventBase {
        cudaEvent_t event;

        void clean();

    public:
        EventGPU(cudaStream_t stream);

        ~EventGPU() override;

        bool QueryFinished() override;

        cudaEvent_t GetEvent() const {
            return event;
        }
    };

}

#endif //XUANWU_EVENT_H

