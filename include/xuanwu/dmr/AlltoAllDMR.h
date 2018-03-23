//
// Created by chnlkw on 1/8/18.
//

#ifndef DMR_ALLTOALL_H
#define DMR_ALLTOALL_H

#include <xuanwu/Worker.h>
#include "Algorithm.h"

class AlltoAllDMR {
    size_t size_;
    std::vector<std::vector<size_t>> send_counts_;
    std::vector<std::vector<size_t>> recv_counts_;
    std::vector<std::vector<size_t>> send_offs_;
    std::vector<std::vector<size_t>> recv_offs_;
    std::vector<size_t> send_sum_;
    std::vector<size_t> recv_sum_;

public:
    AlltoAllDMR() : size_(0) {}

    size_t Size() const { return size_; }

    AlltoAllDMR(const std::vector<std::vector<size_t>> &counts) {
        Prepare(counts);
    }

    void Prepare(const std::vector<std::vector<size_t>> &counts) {
        size_ = counts.size();
        for (auto &c : counts)
            assert(c.size() == size_);
        send_sum_.resize(size_);
        recv_sum_.resize(size_);
        send_counts_ = counts;

        for (size_t i = 0; i < size_; i++) {
            send_sum_[i] = std::accumulate(send_counts_[i].begin(), send_counts_[i].end(), 0LU);
        }

        recv_counts_.resize(size_);
        for (size_t i = 0; i < size_; i++) {
            recv_counts_[i].resize(size_);
            for (size_t j = 0; j < size_; j++) {
                recv_counts_[i][j] = send_counts_[j][i];
                recv_sum_[i] += recv_counts_[i][j];
            }
        }

        recv_offs_.resize(size_);
        send_offs_.resize(size_);
        for (size_t i = 0; i < size_; i++) {
            recv_offs_[i].resize(size_);
            std::partial_sum(recv_counts_[i].begin(), recv_counts_[i].end() - 1, recv_offs_[i].begin() + 1);
            send_offs_[i].resize(size_);
            std::partial_sum(send_counts_[i].begin(), send_counts_[i].end() - 1, send_offs_[i].begin() + 1);
        }
    }

    template<class Vec>
    std::vector<Vec> ShuffleValues(const std::vector<Vec> &value_in) const {
        std::vector<Vec> value_out;
        for (size_t i = 0; i < size_; i++) {
            assert(value_in[i].size() == send_sum_[i]);
            value_out.push_back(Algorithm::Renew(value_in[i], recv_sum_[i]));
            assert(value_out.back().size() == recv_sum_[i]);
        }
        for (size_t i = 0; i < size_; i++) {
            for (size_t k = 0; k < size_; k++) {
                size_t j = (i + k) % size_;
                Algorithm::Copy(value_in[j], send_offs_[j][i], value_out[i], recv_offs_[i][j], send_counts_[j][i]);
            }
        }
        return std::move(value_out);
    }
};

#endif //DMR_ALLTOALL_H
