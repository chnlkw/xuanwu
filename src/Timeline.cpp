//
// Created by chnlkw on 6/9/18.
//

#include "Timeline.h"

std::unique_ptr<Timeline> Timeline::s_time_line;

void Timeline::Create() {
    s_time_line = std::make_unique<Timeline>();
}

void Timeline::Add(size_t worker_id, float transfer_ms, float calc_ms) {
    float ms = std::chrono::duration<double>(GetTime() - s_time_line->start_time).count() * 1000;
    s_time_line->metas.push_back(Meta{ms, worker_id, transfer_ms, calc_ms});
}

void Timeline::Save(const std::string &fname) {
    std::ofstream f(fname);
    if (!f)
        abort();
    for (auto &m : s_time_line->metas) {
        f << m.worker_id << ',' << m.end_time << ',' << m.transfer_ms << ',' << m.calc_ms << std::endl;
    }
    f.close();
}

Timeline::Timeline() {
    start_time = GetTime();
}
