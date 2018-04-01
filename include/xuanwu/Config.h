//
// Created by chnlkw on 3/24/18.
//

#ifndef XUANWU_CONFIG_H
#define XUANWU_CONFIG_H

namespace Xuanwu {
    struct Config {
        int num_cpus = 1;
        int num_gpus = 1;
        size_t gpu_pre_alloc_memory_mb = 1000;
        size_t cpu_pre_alloc_memory_mb = 1000;
        int num_workers_per_gpu = 2;
    };

    template<class T, T init>
    struct Strong {
        T value;

        Strong(T value = init) : value(value) {
        }

        operator T() const {
            return value;
        }
    };


}
#endif //XUANWU_CONFIG_H
