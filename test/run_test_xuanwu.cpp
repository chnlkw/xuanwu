#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <functional>
#include <set>

#include <xuanwu.hpp>
#include <gtest/gtest.h>
#include "Kernels.h"
#include <boost/di.hpp>

namespace di = boost::di;
using namespace Xuanwu;

namespace std {
    template<class K, class V>
    std::ostream &operator<<(std::ostream &os, const std::pair<K, V> &p) {
        os << "(" << p.first << "," << p.second << ")";
        return os;
    };
}

TEST(Xuanwu, AddTask) {
    auto print = [](const auto &arr) {
        printf("%p : ", &arr[0]);
        for (unsigned i = 0; i < arr.size(); i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    };

    auto d1 = Data<int>(10);
    d1.Write();
    auto d2 = Data<int>(d1.size());
    d2.Write();
    for (unsigned i = 0; i < d1.size(); i++) {
        d1[i] = i;
        d2[i] = i * i;
    }
    auto d3 = Data<int>(d1.size());

    print(d1);
    print(d2);
    auto t1 = create_taskadd(d1, d2, d3);
    Xuanwu::AddTask(t1);

    auto d4 = Data<int>(d1.size());
    auto t2 = create_taskadd(d2, d3, d4);


    Xuanwu::AddTask(t2);

    t2->WaitFinish();
//    LOG(INFO) << "After resize";
    d1.Read();
    d2.Read();
    d3.Read();
    d4.Read();
//    d1.resize(2);
    print(d1);
    print(d2);
    d3.Read();
    CUDA_CHECK();
    print(d3);
    d4.Read();
    CUDA_CHECK();
    print(d4);
}

INITIALIZE_EASYLOGGINGPP

size_t gpu_memory = 100LU<<20;

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    el::Loggers::configureFromGlobal("logging.conf");

    int num_gpu = DataCopyInitP2P();

    auto injector = di::make_injector(
            di::bind<>.to(GPUDevice::NumWorkers{2}),
            di::bind<AllocatorFactory<CPUDevice>>().to<CudaHostAllocatorFactory>(),
//            di::bind<AllocatorFactory<GPUDevice>>().to<CudaAllocatorFactory>(),
            di::bind<AllocatorFactory<GPUDevice>>().to<PreAllocatorFactory<CudaAllocatorFactory>>(),
            di::bind<>.to(PreAllocatorFactory<CudaAllocatorFactory>::Space{gpu_memory}),
            di::bind<MMBase>().to<MMMultiDevice<CPUDevice, GPUDevice>>(),
            di::bind<MyDeviceGroup>().to(MultipleDevicesGroup<CPUDevice, GPUDevice>(1, num_gpu))
    );
//    auto m = injector.create<std::unique_ptr<MMBase>>();
//    auto e = injector.create<std::unique_ptr<Engine>>();
    auto xw = injector.create<std::shared_ptr<Xuanwu::Xuanwu>>();

    int ret = RUN_ALL_TESTS();

    return ret;
}