#include <iostream>
#include <vector>
#include <random>
#include <map>
#include <functional>
#include <set>

#include <xuanwu/xuanwu.hpp>
#include "Kernels.h"
#include <boost/di.hpp>
#include <gtest/gtest.h>

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

//void test_dmr(size_t npar, size_t num_element, int repeat) {
//    LOG(INFO) << "start test dmr npar=" << npar << " num_element=" << num_element << " repeat=" << repeat;
//
//    LOG(INFO) << "Initializing Key Value";
//    num_element /= npar;
//    std::vector<std::vector<uint32_t>> keys(npar), values(npar);
//    for (int pid = 0; pid < npar; pid++) {
//        keys[pid].resize(num_element);
//        values[pid].resize(num_element);
//    }
//    size_t N = 1000;
//    size_t sum_keys = 0, sum_values = 0;
//#pragma omp parallel reduction(+:sum_keys, sum_values)
//    {
//        std::random_device rd;  //Will be used to obtain a seed for the random number engine
//        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
//        std::uniform_int_distribution<uint32_t> dis(1, N);
//        for (int pid = 0; pid < npar; pid++) {
//            auto &k = keys[pid];
//            auto &v = values[pid];
//#pragma omp for
//            for (int i = 0; i < k.size(); i++) {
//                k[i] = dis(gen);
//                v[i] = dis(gen);
//                sum_keys += k[i];
//                sum_values += v[i];
//            }
//        }
//    }
//
//    LOG(INFO) << "Initializing DMR keys";
//    size_t block = (N + npar) / npar;
//    auto partitioner = [block](uint32_t k) { return k / block; };
//    PartitionedDMR<uint32_t, uint32_t, data_constructor_t> dmr2(keys, partitioner);
//
//    LOG(INFO) << "Initializing Input values";
//    std::vector<Data<uint32_t>> d_values;
//    for (int i = 0; i < npar; i++) {
//        d_values.emplace_back(values[i]);
//    }
//
//    LOG(INFO) << "Shufflevalues";
//    auto result = dmr2.ShuffleValues<uint32_t>(d_values);
//    while (Car::Get().Tick());
//    LOG(INFO) << "Shufflevalues OK";
//
////    for (auto &v : result) {
////        LOG(DEBUG) << "result " << v.ToString();
////    }
//    size_t sum_keys_2 = 0, sum_values_2 = 0;
//
//    LOG(INFO) << "Checking results";
//
//    for (size_t par_id = 0; par_id < dmr2.Size(); par_id++) {
//        auto keys = dmr2.Keys(par_id).Read().data();
//        auto offs = dmr2.Offs(par_id).Read().data();
//        auto values = result[par_id].Read().data();
//
//#pragma omp parallel for reduction(+:sum_keys_2, sum_values_2)
//        for (size_t i = 0; i < dmr2.Keys(par_id).size(); i++) {
//            auto k = keys[i];
//            for (int j = offs[i]; j < offs[i + 1]; j++) {
//                auto v = values[j];
//                sum_keys_2 += k;
//                sum_values_2 += v;
//            }
//        }
//    }
//    if (sum_keys != sum_keys_2 || sum_values != sum_values_2) {
//        LOG(FATAL) << "sum not match" << sum_keys << ' ' << sum_keys_2 << ' ' << sum_values << ' ' << sum_keys_2;
//    }
//    LOG(INFO) << "Result OK";
//
//    LOG(INFO) << "Run benchmark ";
//    for (int i = 0; i < repeat; i++) {
////        Clock clk;
//        for (auto &d : d_values) {
//            d.Write();
//        }
//        auto r = dmr2.ShuffleValues<uint32_t>(d_values);
//        size_t sum = 0;
//        for (auto &x : r) {
//            x.Read();
//            sum += x.size() * sizeof(int);
//        }
////        double t = clk.timeElapsed();
//        double speed = sum / t / (1LU << 30);
//        while (Car::Get().Tick());
//        printf("sum %lu bytes, time %lf seconds, speed %lf GB/s\n", sum, t, speed);
//    }
//
//    LOG(INFO) << "Run benchmark device only";
//    for (int i = 0; i < repeat; i++) {
//        Clock clk;
//        for (auto &d : d_values) {
//        }
//        auto r = dmr2.ShuffleValues<uint32_t>(d_values);
//        size_t sum = 0;
//        for (auto &x : r) {
//            sum += x.size() * sizeof(int);
//        }
//        while (Car::Get().Tick());
//        double t = clk.timeElapsed();
//        double speed = sum / t / (1LU << 30);
//        printf("sum %lu bytes, time %lf seconds, speed %lf GB/s\n", sum, t, speed);
//    }
//}

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
    auto &engine = Car::Get();

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
    LOG(INFO) << "After resize";
    d1.Read();
    d2.Read();
    d3.Read();
    d4.Read();
    d1.resize(2);
    print(d1);
    print(d2);
    d3.Read(Car::GetCPUDevice());
    CUDA_CHECK();
    print(d3);
    d4.Read(Car::GetCPUDevice());
    CUDA_CHECK();
    print(d4);
}

INITIALIZE_EASYLOGGINGPP

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    el::Loggers::configureFromGlobal("logging.conf");

    LOG(INFO) << "start";

#ifdef USE_CUDA
    int num_gpu = DataCopyInitP2P();
    auto injector = di::make_injector(
            di::bind<CudaAllocator>().to<CudaPreAllocator>(),
            di::bind<MyDeviceGroup>().to(CPUGPUGroupFactory(1, num_gpu)),
//            di::bind<MyDeviceGroup>().to(CPUGroupFactory(2)),
            di::bind<int>().named(NumWorkersOfGPUDevices).to(num_gpu),
            di::bind<size_t>().named(PreAllocBytes).to(2LU << 30)
    );
    Car::Set(injector.create<std::shared_ptr<Engine>>());
#else
#error "CUDA not defined"
    std::vector<WorkerPtr> cpu_workers;
    cpu_workers.emplace_back(new CPUWorker());
    Car::Create({cpu_workers.begin(), cpu_workers.end()});
#endif
//    test_engine();

//    gpu_devices.clear();
//    gpu_workers.clear();
    int ret = RUN_ALL_TESTS();
    Car::Finish();
    return ret;
}