//
// Created by chnlkw on 11/28/17.
//


#include "DataCopy.h"
#include "Task.h"
#include "Data.h"
#include "Array.h"
#include "Worker.h"
#include "Allocator.h"

#include <cassert>
#include <cstring>
#include <iostream>
#include <easylogging++.h>

#define LG(x) CLOG(x, "DataCopy")

namespace Xuanwu {

#if 1

    using AllWorkers = std::tuple<CPUWorker, GPUWorker>;

    template<class Impl, class Base, class F>
    bool CastAndRun(Base *base, F f) {
        Impl *impl = dynamic_cast<Impl *>(base);
        if (!impl)
            return false;
        f(impl);
        return true;
    };


//    template<class... Impl, class Base, class F>
//    bool MultiCastAndRun(Base *base, F f) {
//        return ( ... || CastAndRun<Impl>(base, f));
//    };

//    template<class... Impl, class Base, class F>
//    bool MultiCastAndRun(std::tuple<Impl...> *, Base *base, F f) {
//        return ( ... || CastAndRun<Impl>(base, f));
//    };


    template<class F>
    void CastWorkerAndRun(WorkerPtr worker, F f) {
        if (!MultiCastAndRun((AllWorkers *) (0), worker, f))
            abort();
    }

//    template<class F>
//    void CastDeviceAndRun(DevicePtr device, F f) {
//        if (!MultiCastAndRun((AllDevices *) (0), device, f))
//            abort();
//    }
//
//    template<class F>
//    void CastDevicesAndRun(DevicePtr dev, F f) {
//        CastDeviceAndRun(dev, f);
//    }
//
//    template<class F>
//    void CastAllocatorAndRun(AllocatorPtr allocator, F f) {
//        if (!MultiCastAndRun((AllAllocators *) (0), allocator, f))
//            abort();
//    }
//
//    template<class ...Args, class F>
//    void CastDevicesAndRun(DevicePtr dev1, Args... devs, F f) {
//        CastDeviceAndRun(dev1, [&](auto d1) {
//            auto f2 = std::bind(f, d1);
//            CastDevicesAndRun(devs..., f2);
//        });
//    }
//
//    template<class F>
//    void CastAllocatorAndRun(AllocatorPtr allocator1, AllocatorPtr allocator2, F f) {
//        CastAllocatorAndRun(allocator1, [&](auto a1) {
//            CastAllocatorAndRun(allocator2, [&](auto a2) {
//                f(a1, a2);
//            });
//        });
//    }

#endif

    std::map<int, std::map<int, bool>> data_copy_p2p;

#if 0
    void DataCopy(void *dst_ptr, int dst_device, const void *src_ptr, int src_device, size_t bytes) {
        if (bytes == 0)
            return;
        assert(bytes > 0);
        CLOG(INFO, "DataCopy") << "CopySync " << src_device << " -> " << dst_device << " bytes " << bytes;
        if (src_device < 0) {
            if (dst_device < 0) { //src CPU dst CPU
                memcpy(dst_ptr, src_ptr, bytes);
            } else { // src CPU dst GPU
                CUDA_CALL(cudaMemcpy, dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice);
            }
        } else { // src GPU dst CPU
            if (dst_device < 0) {
                CUDA_CALL(cudaMemcpy, dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost);
            } else { // src GPU dst GPU
                CUDA_CALL(cudaMemcpy, dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice);
            }
        }
    }

    void
    DataCopyAsync(void *dst_ptr, int dst_device, const void *src_ptr, int src_device, size_t bytes,
                  cudaStream_t stream) {
        CLOG(INFO, "DataCopy") << "CopyAsync " << src_device << " -> " << dst_device << " bytes = " << bytes
                               << " stream = "
                               << stream;
        if (src_device < 0) {
            if (dst_device < 0) { //src CPU dst CPU
                CUDA_CALL(cudaStreamSynchronize, stream);
                memcpy(dst_ptr, src_ptr, bytes);
            } else { // src CPU dst GPU
                CUDA_CALL(cudaMemcpyAsync, dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, stream);
            }
        } else { // src GPU dst CPU
            if (dst_device < 0) {
                CUDA_CALL(cudaMemcpyAsync, dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, stream);
            } else { // src GPU dst GPU
                if (data_copy_p2p[src_device][dst_device]) {
                    CUDA_CALL(cudaMemcpyPeerAsync, dst_ptr, dst_device, src_ptr, src_device, bytes, stream);
//                std::cout << dst_device << " <- " << src_device << std::endl;
                    CLOG(DEBUG, "DataCopy") << "use P2P " << src_device << " to " << dst_device;
                } else {
                    CUDA_CALL(cudaMemcpyAsync, dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToDevice, stream);
                    if (src_device != dst_device)
                        CLOG(WARNING, "DataCopy") << "use Origin " << src_device << " to " << dst_device;
                }
            }
        }
    }
#endif

    int DataCopyInitP2P() {
        int num_gpus;
        CUDA_CALL(cudaGetDeviceCount, &num_gpus);
        for (int i = 0; i < num_gpus; i++) {
            CUDA_CALL(cudaSetDevice, i);
            for (int j = 0; j < num_gpus; j++) {
                int access;
                CUDA_CALL(cudaDeviceCanAccessPeer, &access, i, j);
                if (access) {
                    CUDA_CALL(cudaDeviceEnablePeerAccess, j, 0);
                    data_copy_p2p[i][j] = true;
                    CUDA_CHECK();
                    CLOG(INFO, "DataCopy") << "P2P " << i << " to " << j;
                }
            }
        }
        return num_gpus;
    }

//    void ArrayCopy(DevicePtr dst_device, void *dst_ptr, DevicePtr src_device, void *src_ptr, size_t bytes) {
//        CastDeviceAndRun(dst_device, src_device, [&](auto dst, auto src) {
//            ArrayCopy(*dst, dst_ptr, src, src_ptr, bytes);
//        });
//    }

//    void ArrayCopyAsyncPtr(WorkerPtr worker, AllocatorPtr dst_allocator, void *dst_ptr, AllocatorPtr src_allocator,
//                           const void *src_ptr,
//                           size_t bytes) {
//        CastWorkerAndRun(worker, [&](auto worker) {
//            CastAllocatorAndRun(dst_allocator, src_allocator, [&](auto dst, auto src) {
//                worker->ArrayCopyAsync(*dst, *src, dst_ptr, src_ptr, bytes);
//            });
//        });
//    }

//    template<class Worker>
//    void ArrayCopyAsync<Worker, CPUDevice, CPUDevice>(Worker &worker, const CPUDevice &dst_device, const CPUDevice &src_device,
//                                              void *dst_ptr, const void *src_ptr, size_t bytes) {
//        memcpy(dst_ptr, src_ptr, bytes);
//    };

    void DataCopy(DataBasePtr src, size_t src_off, DataBasePtr dst, size_t dst_off, size_t size) {

        auto cputask = std::make_unique<CPUTask>([=](CPUContext cpu) {
            cpu.Copy(dst->GetPtr() + dst_off, src->GetPtr() + src_off, size);
        });

        auto gputask = std::make_unique<GPUTask>([=](GPUContext gpu) {
//            std::cout << "Run on GPU " << gpu->Device()->Id() << std::endl;
//            const T *src = src_.ReadAsync(shared_from_this(), gpu->Device(), gpu->Stream()).data();
            gpu.Copy(dst->GetPtr() + dst_off, src->GetPtr() + src_off, size);
        });

//    std::cout << "copy " << src.ToString() << " to " << dst.ToString() << std::endl;
//    DataCopy(dst.begin() + dst_off, dst.DeviceCurrent()->Id(),
//             src.begin() + src_off, src.DeviceCurrent()->Id(),
//             count * sizeof(T));
//    std::copy(src.begin() + src_off, src.begin() + src_off + count, dst.begin() + dst_off);

        TaskPtr task(new TaskBase("Copy", std::move(cputask), std::move(gputask)));
        task->AddInputRemote(src);
//        task->AddInput(src);
        task->AddOutput(dst);
        AddTask(task);

    }

}
