#include <xuanwu.cuh>
#include <xuanwu/dmr/PartitionedDMR.h>
#include <gtest/gtest.h>
#include <random>

namespace std {
    template<class K, class V>
    std::ostream &operator<<(std::ostream &os, const std::pair<K, V> &p) {
        os << "(" << p.first << "," << p.second << ")";
        return os;
    };
}

using namespace Xuanwu;

void test_dmr(size_t npar, size_t num_element, int repeat) {
    LOG(INFO) << "start test dmr npar=" << npar << " num_element=" << num_element << " repeat=" << repeat;

    LOG(INFO) << "Initializing Key Value";
    num_element /= npar;
    std::vector<std::vector<uint32_t>> keys(npar), values(npar);
    for (int pid = 0; pid < npar; pid++) {
        keys[pid].resize(num_element);
        values[pid].resize(num_element);
    }
    size_t N = 1000;
    size_t sum_keys = 0, sum_values = 0;
#pragma omp parallel reduction(+:sum_keys, sum_values)
    {
        std::random_device rd;  //Will be used to obtain a seed for the random number engine
        std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
        std::uniform_int_distribution<uint32_t> dis(1, N);
        for (int pid = 0; pid < npar; pid++) {
            auto &k = keys[pid];
            auto &v = values[pid];
#pragma omp for
            for (int i = 0; i < k.size(); i++) {
                k[i] = dis(gen);
                v[i] = dis(gen);
                sum_keys += k[i];
                sum_values += v[i];
            }
        }
    }

    LOG(INFO) << "Initializing DMR keys";
    size_t block = (N + npar) / npar;
    auto partitioner = [block](uint32_t k) { return k / block; };
    PartitionedDMR<uint32_t, uint32_t, data_constructor_t> dmr2(keys, partitioner);

    LOG(INFO) << "Initializing Input values";
    std::vector<Data<uint32_t>> d_values;
    for (int i = 0; i < npar; i++) {
        d_values.emplace_back(values[i]);
    }

    LOG(INFO) << "Shufflevalues";
    auto result = dmr2.ShuffleValues<uint32_t>(d_values);
    while (Xuanwu::Tick());
    LOG(INFO) << "Shufflevalues OK";

//    for (auto &v : result) {
//        LOG(DEBUG) << "result " << v.ToString();
//    }
    size_t sum_keys_2 = 0, sum_values_2 = 0;

    LOG(INFO) << "Checking results";

    for (size_t par_id = 0; par_id < dmr2.Size(); par_id++) {
        auto &keys = dmr2.Keys(par_id).Read();
        auto &offs = dmr2.Offs(par_id).Read();
        auto &values = result[par_id].Read();

#pragma omp parallel for reduction(+:sum_keys_2, sum_values_2)
        for (size_t i = 0; i < dmr2.Keys(par_id).size(); i++) {
            auto k = keys[i];
            for (int j = offs[i]; j < offs[i + 1]; j++) {
                auto v = values[j];
                sum_keys_2 += k;
                sum_values_2 += v;
            }
        }
    }
    if (sum_keys != sum_keys_2 || sum_values != sum_values_2) {
        LOG(FATAL) << "sum not match " << sum_keys << "!=" << sum_keys_2 << " || " << sum_values << "!="
                   << sum_values_2;
    }
    LOG(INFO) << "Result OK";

    LOG(INFO) << "Run benchmark ";
    for (int i = 0; i < repeat; i++) {
//        Clock clk;
        for (auto &d : d_values) {
            d.Write();
        }
        auto r = dmr2.ShuffleValues<uint32_t>(d_values);
//        while (Xuanwu::Tick());
        size_t sum = 0;
        for (auto &x : r) {
            x.Read();
            sum += x.size() * sizeof(int);
        }
//        double t = clk.timeElapsed();
//        double speed = sum / t / (1LU << 30);
        while (Xuanwu::Tick());
//        printf("sum %lu bytes, time %lf seconds, speed %lf GB/s\n", sum, t, speed);
    }

    LOG(INFO) << "Run benchmark device only";
    for (int i = 0; i < repeat; i++) {
//        Clock clk;
        for (auto &d : d_values) {
        }
        auto r = dmr2.ShuffleValues<uint32_t>(d_values);
        size_t sum = 0;
        for (auto &x : r) {
            sum += x.size() * sizeof(int);
        }
        while (Xuanwu::Tick());
//        double t = clk.timeElapsed();
//        double speed = sum / t / (1LU << 30);
//        printf("sum %lu bytes, time %lf seconds, speed %lf GB/s\n", sum, t, speed);
    }
}

TEST(Xuanwu, PartitionedDMR) {
    test_dmr(2, 1000, 1);
}

__global__ void arr_init(int *a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        a[idx] = idx;
}

TEST(Xuanwu, DataCreateInTask) {
    int n = 10;
    Data<int> d;
    auto gpu_task = std::make_unique<GPUTask>([=](GPUContext gpu) mutable {
        d.Create(n, gpu.dev);
        arr_init << < 1, n >> > (d.data(), n);
    });
    TaskPtr task(new TaskBase(
            "testcreate",
            {},
            std::move(gpu_task)));
    task->AddOutput(d);
    Xuanwu::AddTask(task);
    d.Read();
    ASSERT_EQ(n, d.size());
    for (int i = 0; i < d.size(); i++) {
        ASSERT_EQ(i, d[i]);
    }
}

__global__ void local_create_kernel(DeviceArray<int> *arr_ptr) {
    auto &arr = *arr_ptr;
    int n = 10;
    arr.Alloc(n);
    for (int i = 0; i < n; i++)
        arr[i] = i;
}

/*
TEST(Xuanwu, LocalCreateInTask) {
    Data<int> sz(1);
    Data<int *> ptr(1);
    Data<int> d;
    auto gpu_task = std::make_unique<GPUTask>([=](GPUContext gpu) mutable {
        DeviceArray<int> *arr = gpu.MakeLocalMapping(d);
        local_create_kernel << < 1, 1, 0, gpu.stream >> > (arr);
    });
    TaskPtr task(new TaskBase("testlocalCreate", {}, std::move(gpu_task)));
    task->AddOutputs({d, ptr});
    Xuanwu::AddTask(task);
    d.Read();
    ASSERT_EQ(10, d.size());
    for (int i = 0; i < d.size(); i++) {
        ASSERT_EQ(i, d[i]);
    }
    while (Xuanwu::Tick());
}
*/

__global__
void accumulate_kernel(int *arr, int *sum) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    atomicAdd(sum, arr[idx]);
}

TEST(Xuanwu, ScatterData) {
    size_t nblocks = 50 * 1024 / 4;
    int nthreads = 1024;
    size_t sz = nblocks * nthreads;
    Data<int> arr(sz);
    arr->SetName("array");
    auto work = [&](int val) {
        arr.Write();
        for (int i = 0; i < sz; i++) arr[i] = val;

        int ntasks = 4;
        std::vector<Data<int>> outs;
        for (int i = 0; i < ntasks; i++) {
            Data<int> sum(1);
            sum.Write();
            sum->SetName(std::string("out") + std::to_string(i));
            sum[0] = 0;

            auto gpu_task = std::make_unique<GPUTask>([=](GPUContext gpu) mutable {
                accumulate_kernel << < nblocks, nthreads, 0, gpu.stream >> > (arr.data(), sum.data());
            });
            TaskPtr task(new TaskBase("accumulate", {}, std::move(gpu_task)));
            task->AddInput(arr);
            task->AddInOutput(sum);
            Xuanwu::AddTask(task);
            outs.emplace_back(std::move(sum));
        }

        ASSERT_EQ(ntasks, outs.size());
        for (auto &out : outs) {
            out.Read();
            ASSERT_EQ(out[0], sz * val);
        }
    };

    work(1);
    work(2);

}

__global__ void add_kernel(int *a, int delta, size_t n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] += delta;
    }
}

extern size_t gpu_memory;

TEST(Xuanwu, Streaming) {
    int p = 3;
    std::vector<Data<bool>> syncer;
    for (int i = 0; i < p; i++) {
        syncer.emplace_back(1);
        syncer.back().Write();
    }
    int n = 10, m = ::gpu_memory / (p+1) / sizeof(int);


    std::vector<Data<int>> arrs;
    for (int i = 0; i < n; i++) {
        Data<int> d(m);
        d->PinnedToDevice(Xuanwu::GetDefaultDevice(), false);
        d.Write();
        memset(d.data(), 0, d->Bytes());
        arrs.push_back(std::move(d));
    }
    int nthread = 1024;
    int nblock = (m + nthread - 1) / nthread;
    for (int round = 1; round <= 3; round++) {
        for (int i = 0; i < n; i++) {
            auto gpu_task = std::make_unique<GPUTask>([=](GPUContext gpu) mutable {
                add_kernel << < nblock, nthread >> > (arrs[i].data(), i, m);
            });
            TaskPtr task(new TaskBase("StreamingAdd", {}, std::move(gpu_task)));
            task->AddInOutputs({arrs[i], syncer[i%p]});
            Xuanwu::AddTask(task);

            auto cpu_task = std::make_unique<CPUTask>([=](CPUContext cpu) mutable {
                auto arr = arrs[i].data();
            });
            TaskPtr task2(new TaskBase("ClearGPU", std::move(cpu_task), {}));
            task2->AddInOutputs({arrs[i], syncer[i%p]});
            Xuanwu::AddTask(task2);
        }
        for (int i = 0; i < n; i++) {
            auto arr = arrs[i].Read().data();
            for (int j = 0; j < m; j++)
                ASSERT_EQ(i * round, arr[j]);
        }
    }
}