//
// Created by chnlkw on 3/24/18.
//

#ifndef XUANWU_CONFIG_H
#define XUANWU_CONFIG_H

namespace Xuanwu {
    struct Config {
        int num_cpus =1;
        int num_gpus = 1;
        int gpu_pre_alloc_memory_mb = 1000;
        int cpu_pre_alloc_memory_mb = 1000;
        int num_workers_per_gpu = 2;
    };
}
#endif //XUANWU_CONFIG_H
