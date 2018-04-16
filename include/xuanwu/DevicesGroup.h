//
// Created by chnlkw on 3/2/18.
//

#ifndef DMR_DEVICESGROUP_H
#define DMR_DEVICESGROUP_H

#include "defs.h"
#include <xuanwu/Allocator.h>
#include "Device.h"

namespace Xuanwu {
    template<class T>
    struct Generator {
        std::vector<std::unique_ptr<T>> data;

        Generator(size_t n) {
            for (size_t i = 0; i < n; i++) {
                data.push_back(std::make_unique<T>(i));
            }
        }

        auto &&operator()() {
            return std::move(data);
        }

    };

    struct MyDeviceGroup : std::vector<std::shared_ptr<DeviceBase>> {
    };

    template<class ...Devices>
    struct MultipleDevicesGroup;

    template<>
    struct MultipleDevicesGroup<> {

        template<class TInjector, class TDependency>
        auto operator()(const TInjector &injector, const TDependency &) const {
            return std::make_unique<MyDeviceGroup>();
        }
    };

    template<class Device, class ...OtherDevices>
    struct MultipleDevicesGroup<Device, OtherDevices...> : public MultipleDevicesGroup<OtherDevices...> {

        int num_;

        MultipleDevicesGroup(int num, typename detail::As<OtherDevices, int>::type ... other_num)
                : MultipleDevicesGroup<OtherDevices...>(other_num...),
                  num_(num) {
            CLOG(DEBUG, "Device") << "MultipleDeviceGroup create With " << num << " " << typeid(Device).name();

        }

        template<class TInjector, class TDependency>
        auto operator()(const TInjector &injector, const TDependency &dep) const {
            CLOG(DEBUG, "Device") << "MultipleDeviceGroup run injector With " << num_ << " " << typeid(Device).name();
            auto ret = static_cast<const MultipleDevicesGroup<OtherDevices...> *>(this)->operator()(injector, dep);
            for (int i = 0; i < num_; i++) {
                CLOG(DEBUG, "Device") << "MyDeviceGroup creating CPUDevice " << i;
                ret->push_back(injector.template create<std::unique_ptr<Device>>());
                CLOG(DEBUG, "Device") << "MyDeviceGroup creating CPUDevice " << i << " OK";
            }
            return std::move(ret);
        }
    };

    struct CPUGroupFactory {
        size_t num_cpus;

        explicit CPUGroupFactory(size_t num_cpus) : num_cpus(num_cpus) {}

        template<class TInjector, class TDependency>
        auto operator()(const TInjector &injector, const TDependency &) const {
            auto g = std::make_unique<MyDeviceGroup>();
            for (size_t i = 0; i < num_cpus; i++) {
                CLOG(INFO, "Device") << "MyDeviceGroup creating CPUDevice " << i;
                g->push_back(injector.template create<std::shared_ptr<CPUDevice>>());
                CLOG(INFO, "Device") << "MyDeviceGroup creating CPUDevice " << i << " OK";
            }
            return std::move(g);
        }

    };

#if 0
    struct GPUGroupFactory {
        size_t num_gpus;

        explicit GPUGroupFactory(size_t num_gpus) : num_gpus(num_gpus) {}

        template<class TInjector, class TDependency>
        auto operator()(const TInjector &injector, const TDependency &) const {
            TInjector &injector_(const_cast<TInjector &>(injector));
            auto inj = boost::di::make_injector(std::move(injector_),
                                                boost::di::bind<int>().named(myDeviceId).to([]() {
                                                    static int seq = 0;
                                                    return seq++;
                                                })
            );
            auto g = std::make_unique<MyDeviceGroup>();
            for (size_t i = 0; i < num_gpus; i++) {
                CLOG(INFO, "Device") << "MyDeviceGroup creating i = " << i;
                g->emplace_back(inj.template create<GPUDevice *>()); //singleton
            }
            return std::move(g);
        }

    };

    struct CPUGPUGroupFactory {
        size_t num_cpus;
        size_t num_gpus;

        explicit CPUGPUGroupFactory(size_t num_cpus, size_t num_gpus) :
                num_cpus(num_cpus),
                num_gpus(num_gpus) {}

        template<class TInjector, class TDependency>
        auto operator()(const TInjector &injector, const TDependency &) const {
            TInjector &injector_(const_cast<TInjector &>(injector));
            auto g = std::make_unique<MyDeviceGroup>();

            for (size_t i = 0; i < num_cpus; i++) {
                CLOG(INFO, "Device") << "MyDeviceGroup creating CPU device " << i;
                g->emplace_back(injector_.template create<std::shared_ptr<CPUDevice>>()); // sharing
                CLOG(INFO, "Device") << "MyDeviceGroup creating CPU device " << i << " OK";
            }
            auto inj = boost::di::make_injector(std::move(injector_),
                                                boost::di::bind<int>().named(myDeviceId).to([]() {
                                                    static int seq = 0;
                                                    return seq++;
                                                })
            );
            for (size_t i = 0; i < num_gpus; i++) {
                CLOG(INFO, "Device") << "MyDeviceGroup creating GPU device " << i;
                g->emplace_back(inj.template create<GPUDevice *>()); //singleton
            }
            return std::move(g);
        }

    };
#endif

}
#endif //DMR_DEVICESGROUP_H
