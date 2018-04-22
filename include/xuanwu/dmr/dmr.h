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

#define LG(x) CLOG(x, "DMR")

namespace Xuanwu {
    template<class VData, class VOff>
    void ShuffleByIdx(VData &p_dst, const VData &p_src, const VOff &p_idx, std::string);

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

            std::vector<std::pair<TKey, TOff>> metas(Size());
            for (size_t i = 0; i < Size(); i++) {
                metas[i] = {keys[i], i};
            }

            std::sort(metas.begin(), metas.end());

            std::vector<TOff> gather_indices(Size());
            std::vector<TKey> reducer_keys;
            std::vector<TOff> reducer_offs;

            gather_indices.resize(Size());

            for (size_t i = 0; i < metas.size(); i++) {
                TKey k = metas[i].first;
                gather_indices[i] = metas[i].second;

                if (i == 0 || metas[i].first != metas[i - 1].first) {
                    reducer_keys.push_back(k);
                    reducer_offs.push_back(i);
                }
            }
            reducer_offs.push_back(Size());
            reducer_keys.shrink_to_fit();
            reducer_offs.shrink_to_fit();

            reducer_keys_ = Vector<TKey>(std::move(reducer_keys));
            reducer_offs_ = Vector<TOff>(std::move(reducer_offs));
            gather_indices_ = Vector<TOff>(std::move(gather_indices));
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
            Vector<TValue> value_out(value_in.size());
            ShuffleByIdx(value_out, value_in, gather_indices_, name);
            return std::move(value_out);
        }

    };

}
#undef LG

#endif //DMR_DMR_H
