#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <functional>
#include <set>

#include <xuanwu.hpp>
#include <gtest/gtest.h>
#include "Kernels.h"

namespace di = boost::di;

auto print = [](auto &x) { std::cout << " " << x; };
auto self = [](auto x) { return x; };

namespace std {
template<class K, class V>
std::ostream &operator<<(std::ostream &os, const std::pair<K, V> &p) {
    os << "(" << p.first << "," << p.second << ")";
    return os;
};
}

template<class T>
class TaskAdd : public TaskBase {
    Data<T> a_, b_, c_;
public:
    TaskAdd(Data<T> a, Data<T> b, Data<T> c) :
            TaskBase("Add"),
            a_(a), b_(b), c_(c) {
        assert(a.size() == b.size());
        assert(a.size() == c.size());
        AddInput(a);
        AddInput(b);
        AddOutput(c);
    }

    virtual void Run(CPUWorker *cpu) override {
        const T *a = a_.ReadAsync(shared_from_this(), cpu->Device(), 0).data();
        const T *b = b_.ReadAsync(shared_from_this(), cpu->Device(), 0).data();
        T *c = c_.WriteAsync(shared_from_this(), cpu->Device(), 0).data();
        for (int i = 0; i < c_.size(); i++) {
            c[i] = a[i] + b[i];
        }
    }

    virtual void Run(GPUWorker *gpu) override {
        const T *a = a_.ReadAsync(shared_from_this(), gpu->Device(), gpu->Stream()).data();
        const T *b = b_.ReadAsync(shared_from_this(), gpu->Device(), gpu->Stream()).data();
        T *c = c_.WriteAsync(shared_from_this(), gpu->Device(), gpu->Stream()).data();
        gpu_add(c, a, b, c_.size(), gpu->Stream());
    }
};

TEST(Xuanwu, AddTask) {
    auto &engine = Xuanwu::Get();

    auto print = [](const auto &arr) {
        printf("%p : ", &arr[0]);
        for (int i = 0; i < arr.size(); i++) {
            printf("%d ", arr[i]);
        }
        printf("\n");
    };

    auto d1 = Data<int>(10);
    d1.Write();
    auto d2 = Data<int>(d1.size());
    d2.Write();
    for (int i = 0; i < d1.size(); i++) {
        d1[i] = i;
        d2[i] = i * i;
    }
    auto d3 = Data<int>(d1.size());

    print(d1);
    print(d2);
//    auto t1 = std::make_shared<TaskAdd<int>>(d1, d2, d3);
//    engine.RegisterTask(t1);
    engine.AddTask<TaskAdd<int>>(d1, d2, d3);

    auto d4 = Data<int>(d1.size());
//    auto t2 = std::make_shared<TaskAdd2<int>>(engine, d2, d3, d4);
    auto t2 = create_taskadd(d2, d3, d4);


    engine.AddTask(t2);

//    while (engine.Tick());
    t2->WaitFinish();
//    LOG(INFO) << "After resize";
    d1.Read();
    d2.Read();
    d3.Read();
    d4.Read();
    d1.resize(2);
    print(d1);
    print(d2);
    d3.Read(Xuanwu::GetCPUDevice());
    CUDA_CHECK();
    print(d3);
    d4.Read(Xuanwu::GetCPUDevice());
    CUDA_CHECK();
    print(d4);
}

INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    el::Loggers::configureFromGlobal("logging.conf");

//    LOG(INFO) << "start";

    int num_gpu = DataCopyInitP2P();
    auto injector = di::make_injector(
            di::bind<CudaAllocator>().to<CudaPreAllocator>(),
            di::bind<MyDeviceGroup>().to(CPUGPUGroupFactory(1, num_gpu)),
//            di::bind<MyDeviceGroup>().to(CPUGroupFactory(2)),
            di::bind<int>().named(NumWorkersOfGPUDevices).to(num_gpu),
            di::bind<size_t>().named(PreAllocBytes).to(2LU << 30)
    );
    Xuanwu::Set(injector.create<std::shared_ptr<Engine>>());

    int ret = RUN_ALL_TESTS();

    Xuanwu::Finish();

    return ret;
}