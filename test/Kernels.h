//
// Created by chnlkw on 1/23/18.
//

#ifndef DMR_KERNELS_H
#define DMR_KERNELS_H

#include <cstdio>

#include <xuanwu.cuh>

Xuanwu::TaskPtr create_taskadd(const Xuanwu::Data<int> &a, const Xuanwu::Data<int> &b, Xuanwu::Data<int> &c);

#endif //DMR_KERNELS_H
