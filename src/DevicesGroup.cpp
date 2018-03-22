//
// Created by chnlkw on 3/2/18.
//

#include "DevicesGroup.h"

GPUGroup::GPUGroup(size_t n, const boost::di::extension::ifactory<DeviceBase> &device_factory) {
    LOG(INFO) << "GPUGroupSize " << n;
    for (size_t i = 0; i < n; i++) {
        LOG(INFO) << "devices group " << i;
        devices_.push_back(
                device_factory.create()
        );
    }
}
