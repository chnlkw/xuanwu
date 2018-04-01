//
// Created by chnlkw on 11/21/17.
//

#ifndef LDA_DATACOPY_H
#define LDA_DATACOPY_H

#include "defs.h"
#include <map>
#include "cuda_utils.h"
#include "Ptr.h"

namespace Xuanwu {
    extern std::map<int, std::map<int, bool>> data_copy_p2p;

//    void DataCopy(void *dst_ptr, int dst_device, const void *src_ptr, int src_device, size_t bytes);

//    void DataCopyAsync(void *dst_ptr, int dst_device, const void *src_ptr, int src_device, size_t bytes,
//                       cudaStream_t stream);

    int DataCopyInitP2P();

//    template<class DstDevice, class SrcDevice>
//    void
//    ArrayCopy(const DstDevice &dst_device, void *dst_ptr, const SrcDevice &src_device, void *src_ptr, size_t bytes);

//    template<class Worker, class DstDevice, class SrcDevice>
//    void
//    ArrayCopyAsync(Worker& worker, const DstDevice &dst_device, const SrcDevice &src_device, void *dst_ptr,
//                   const void *src_ptr, size_t bytes);


//    void ArrayCopy(DevicePtr dst_device, void *dst_ptr, DevicePtr src_device, void *src_ptr, size_t bytes);

    void ArrayCopyAsyncPtr(WorkerPtr worker, Ptr dst, Ptr src, size_t bytes);

    template<class V>
    void Copy(const V &src, size_t src_off, V &dst, size_t dst_off, size_t count);

    template<class T>
    void Copy(const std::vector<T> &src, size_t src_off, std::vector<T> &dst, size_t dst_off, size_t count) {
        std::copy(src.begin() + src_off, src.begin() + src_off + count, dst.begin() + dst_off);
    }

    void DataCopy(DataBasePtr src, size_t src_off, DataBasePtr dst, size_t dst_off, size_t count);

    template<class T>
    void Copy(const Data <T> &src, size_t src_off, Data <T> &dst, size_t dst_off, size_t count) {
        DataCopy(src, src_off * sizeof(T), dst, dst_off * sizeof(T), count * sizeof(T));
    }

}
#endif //LDA_DATACOPY_H
