//
// Created by chnlkw on 1/5/18.
//

#ifndef DMR_PARTITIONEDDMR_H
#define DMR_PARTITIONEDDMR_H

#include "dmr.h"
#include "array_constructor.h"
#include "AlltoAllDMR.h"

#include <algorithm>

#define LG(x) CLOG(x, "DMR")

namespace Xuanwu {
    template<class TKey, class TOff, class ArrayConstructor = vector_constructor_t>
    class PartitionedDMR {
    public:
        template<class T>
        using Vector = decltype(ArrayConstructor::template Construct<T>());
        using TPar = uint32_t;

    private:
        size_t size_;
        TKey max_key_;
        std::vector<DMR<TPar, TOff, ArrayConstructor>> dmr1_;
        AlltoAllDMR alltoall_;
        std::vector<DMR<TKey, TOff, ArrayConstructor>> dmr3_;

        using Partitioner = std::function<int(TKey)>;
        Partitioner partitioner_;

        std::vector<Vector<TKey>> keys_;
        std::vector<Vector<TOff>> offs_;
    public:

        explicit PartitionedDMR(const std::vector<std::vector<TKey>> &mapper_keys, Partitioner partitioner) :
                size_(mapper_keys.size()),
                dmr1_(size_),
                dmr3_(size_),
                max_key_(0),
                partitioner_(partitioner) {
            LG(INFO) << "create PartitionedDMR num_par = " << mapper_keys.size();
            Prepare(mapper_keys);
            for (size_t i = 0; i < size_; i++) {
                keys_.push_back(dmr3_[i].Keys());
                offs_.push_back(dmr3_[i].Offs());
            }
            for (size_t i = 0; i < size_; i++) {
                dmr1_[i].name = std::string("shuffle1.") + std::to_string(i);
                dmr3_[i].name = std::string("shuffle3.") + std::to_string(i);
            }
        }

        void Prepare(const std::vector<std::vector<TKey>> &mapper_keys) {

            static_assert(std::is_integral<TKey>::value, "TKey must be integral");
            static_assert(std::is_integral<TOff>::value, "TOff must be integral");

            std::vector<std::vector<TKey>> parted_keys;
            // local partition
            for (size_t mapper_id = 0; mapper_id < size_; mapper_id++) {
                auto keys = mapper_keys[mapper_id];
                std::vector<TPar> par_id(keys.size());
                std::transform(keys.begin(), keys.end(), par_id.begin(), partitioner_);
                DMR<TPar, TOff> dmr(par_id);
                auto parted_key = dmr.template ShuffleValues<TKey>(keys);
                parted_keys.push_back(parted_key);
                dmr1_[mapper_id] = std::move(dmr);
            }

            // global partition
            std::vector<std::vector<size_t>> send_counts(size_);
            for (size_t mapper_id = 0; mapper_id < size_; mapper_id++) {
                auto &dmr = dmr1_[mapper_id];
                std::vector<size_t> counts(size_);
                for (size_t i = 0; i < dmr.Keys().size(); i++) {
                    TKey k = dmr.Keys()[i];
                    counts.at(k) = dmr.Offs()[i + 1] - dmr.Offs()[i];
                }
                send_counts[mapper_id] = std::move(counts);
            }
            alltoall_.Prepare(send_counts);
            auto results = alltoall_.ShuffleValues(parted_keys);

            // local sort
            for (size_t mapper_id = 0; mapper_id < size_; mapper_id++) {
                dmr3_[mapper_id].Prepare(results[mapper_id]);
            }
        }

        template<class Cons>
        explicit PartitionedDMR(const PartitionedDMR<TKey, Cons> &that) :
                size_(that.Size()),
                max_key_(that.MaxKey()),
                alltoall_(that.GetAlltoallDMR()) {
            static_assert(std::is_integral<TKey>::value, "TKey must be integral");
            static_assert(std::is_integral<TOff>::value, "TOff must be integral");
            for (auto &d : that.GetDMR1())
                dmr1_.push_back(d);
            for (auto &d : that.GetDMR3())
                dmr3_.push_back(d);
        }

        template<class TValue>
        std::vector<Vector<TValue>> ShuffleValues(const std::vector<Vector<TValue>> &value_in) const;
//    {
//        assert(value_in.size() == pre_alloc_size_);
//        std::vector<Vector<TValue>> parted_values;
//        for (size_t i = 0; i < pre_alloc_size_; i++) {
//            parted_values.push_back(dmr1_[i].template ShuffleValues<TValue>(value_in[i]));
//        }
//
//        auto shufed = alltoall_.ShuffleValues(parted_values);
//
//        std::vector<Vector<TValue>> ret;
//        for (size_t i = 0; i < pre_alloc_size_; i++) {
//            ret.push_back(dmr3_[i].template ShuffleValues<TValue>(shufed[i]));
//        }
//
//        return std::move(ret);
//    }

        const Vector<TKey> &Keys(size_t i) const { return dmr3_[i].Keys(); }

        const auto &Keys() const { return keys_; }

        const Vector<TOff> &Offs(size_t i) const { return dmr3_[i].Offs(); }

        const auto &Offs() const { return offs_; }

        size_t Size() const {
            return size_;
        }

        TKey MaxKey() const {
            return max_key_;
        }

        auto &GetDMR1() { return dmr1_; }

        auto &GetAlltoallDMR() { return alltoall_; }

        auto &GetDMR3() { return dmr3_; }

        const auto &GetDMR1() const { return dmr1_; }

        const auto &GetAlltoallDMR() const { return alltoall_; }

        const auto &GetDMR3() const { return dmr3_; }
    };

    template<class TKey, class TOff, class ArrayConstructor>
    template<class TValue>
    std::vector<decltype(ArrayConstructor::template Construct<TValue>())>
//PartitionedDMR<TKey, TOff, ArrayConstructor>::Vector<TValue>
    PartitionedDMR<TKey, TOff, ArrayConstructor>::ShuffleValues(const std::vector<Vector<TValue>> &value_in) const {
        assert(value_in.size() == size_);
        std::vector<Vector<TValue>> parted_values;
        for (size_t i = 0; i < size_; i++) {
            parted_values.push_back(dmr1_[i].template ShuffleValues<TValue>(value_in[i]));
        }

        auto shufed = alltoall_.ShuffleValues(parted_values);

        std::vector<Vector<TValue>> ret;
        for (size_t i = 0; i < size_; i++) {
            ret.push_back(dmr3_[i].template ShuffleValues<TValue>(shufed[i]));
        }

        return std::move(ret);
    }
}

#undef LG

#endif //DMR_PARTITIONEDDMR_H
