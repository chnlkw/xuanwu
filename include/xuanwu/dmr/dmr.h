//
// Created by chnlkw on 12/29/17.
//

#ifndef DMR_DMR_H
#define DMR_DMR_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cassert>
#include <xuanwu/Data.h>

#include "array_constructor.h"
#include "xuanwu/AllocatorBase.h"
#include "xuanwu/Task.h"
#include "xuanwu/Worker.h"
#include "xuanwu/Xuanwu.h"
#include "Algorithm.h"

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/unique.h>


#define LG(x) CLOG(x, "DMR")

namespace Xuanwu {
//    template<class VData, class VOff>
//    void ShuffleByIdx(VData &p_dst, const VData &p_src, const VOff &p_idx, std::string);

    template<class T, class TOff>
    void ShuffleByIdx(Data<T>& dst, const Data<T>& src, const Data<TOff>& idx, std::string name);

    template<class T, class TOff>
    void ShuffleByIdx(std::vector<T> &p_dst, const std::vector<T> &p_src, const std::vector<TOff> &p_idx, std::string) {
        size_t size = p_dst.size();
        assert(size == p_src.size());
        assert(size == p_idx.size());
        auto dst = p_dst.data();
        auto src = p_src.data();
        auto idx = p_idx.data();

        for (size_t i = 0; i < p_src.size(); i++) {
            dst[i] = src[idx[i]];
        }
    }

    template<class TKey, class TOff, class ArrayConstructor = vector_constructor_t>
    class DMR {
        template<class T>
        using Vector = decltype(ArrayConstructor::template Construct<T>());

    private:
        size_t size_;
        Vector<TKey> reducer_keys_;
        Vector<TOff> reducer_offs_;
        Vector<TOff> gather_indices_;

    public:
        std::string name = "dmr";

        DMR() {}

        DMR(const std::vector<TKey> &keys) {
            static_assert(std::is_integral<TKey>::value, "TKey must be integral");
            static_assert(std::is_integral<TOff>::value, "TOff must be integral");

            LG(INFO) << "create DMR num_keys = " << keys.size();
            Prepare(keys);
        }

        void Prepare(const std::vector<TKey> &keys) {
            size_ = keys.size();

            thrust::host_vector<TKey> h_keys = keys;
            thrust::device_vector<TKey> d_keys = h_keys;
            thrust::device_vector<TOff> d_gather_indices(Size());
            thrust::device_vector<TOff> d_off(Size());

            thrust::sequence(d_gather_indices.begin(), d_gather_indices.end());
            thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_gather_indices.begin());

            thrust::sequence(d_off.begin(), d_off.end());
            auto new_end = thrust::unique_by_key(d_keys.begin(), d_keys.end(), d_off.begin());
            auto off_size = new_end.first - d_keys.begin();
            d_keys.resize(off_size);
            d_off.resize(off_size);
            d_off.push_back(Size());

            h_keys = d_keys;
            thrust::host_vector<TOff> h_off = d_off;
            thrust::host_vector<TOff> h_gather_indices = d_gather_indices;
//            std::vector<std::pair<TKey, TOff>> metas(Size());
//            for (size_t i = 0; i < Size(); i++) {
//                metas[i] = {keys[i], i};
//            }
//
//            std::sort(metas.begin(), metas.end());

//            std::vector<TOff> gather_indices(Size());
//            std::vector<TKey> reducer_keys;
//            std::vector<TOff> reducer_offs;
//
//            gather_indices.resize(Size());

//            for (size_t i = 0; i < metas.size(); i++) {
//                TKey k = metas[i].first;
//                gather_indices[i] = metas[i].second;
//
//                if (i == 0 || metas[i].first != metas[i - 1].first) {
//                    reducer_keys.push_back(k);
//                    reducer_offs.push_back(i);
//                }
//            }
//            reducer_offs.push_back(Size());
//            reducer_keys.shrink_to_fit();
//            reducer_offs.shrink_to_fit();

            reducer_keys_ = Vector<TKey>({h_keys.begin(), h_keys.end()});
            reducer_offs_ = Vector<TOff>({h_off.begin(), h_off.end()});
            gather_indices_ = Vector<TOff>({h_gather_indices.begin(), h_gather_indices.end()});
        }

        template<class Cons>
        DMR(const DMR<TKey, Cons> &that) :
                size_(that.Size()),
                reducer_keys_(that.Keys()),
                reducer_offs_(that.Offs()),
                gather_indices_(that.GatherIndices()),
                name(that.name) {
        }

        size_t Size() const {
            return size_;
        }

        Vector<TOff> &GatherIndices() {
            return gather_indices_;
        }

        Vector<TKey> &Keys() {
            return reducer_keys_;
        }

        Vector<TOff> &Offs() {
            return reducer_offs_;
        }

        const Vector<TOff> &GatherIndices() const {
            return gather_indices_;
        }

        const Vector<TKey> &Keys() const {
            return reducer_keys_;
        }

        const Vector<TOff> &Offs() const {
            return reducer_offs_;
        }

        template<class TValue>
        Vector<TValue> ShuffleValues(const Vector<TValue> &value_in) const {
            Vector<TValue> value_out = Renew(value_in, value_in.size());
            ShuffleByIdx(value_out, value_in, gather_indices_, name);
            return std::move(value_out);
        }

    };

}
#undef LG

#endif //DMR_DMR_H
