//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_KERNELS_H
#define DMR_KERNELS_H

#include <cstdio>

#include <xuanwu.cuh>

TaskPtr create_taskadd(const Data<int> &a, const Data<int> &b, Data<int> &c);

#endif //DMR_KERNELS_H
