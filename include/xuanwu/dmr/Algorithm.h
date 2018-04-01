//
// Created by chnlkw on 1/18/18.
//

#ifndef DMR_ALGORITHM_H
#define DMR_ALGORITHM_H

#include "xuanwu/Array.h"
#include "xuanwu/Data.h"
//#include "Kernels.cuh"
#include "xuanwu/DataCopy.h"

//void shuffle_by_idx_gpu_ff(float *dst, const float *src, const size_t *idx, size_t size);

namespace Xuanwu {

    /*
template<class V>
V Renew(const V &in, size_t count = 0);

template<class T>
std::vector<T> Renew(const std::vector<T> &in, size_t count) {
    return std::vector<T>(count);
}

template<class T>
Array<T> Renew(const Array<T> &in, size_t count) {
    return in.Renew(count);
}

template<class T>
Data<T> Renew(const Data<T> &in, size_t count) {
    Data<T> ret(count);
    ret->Follow(in);
    return ret;
}
 */

}

#endif //DMR_ALGORITHM_H
