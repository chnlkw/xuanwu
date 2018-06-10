//
// Created by chnlkw on 6/9/18.
//

#ifndef XUANWU_TIMELINE_H
#define XUANWU_TIMELINE_H

#include <chrono>
#include <list>
#include <string>
#include <fstream>
#include <memory>

class Timeline {
    using Clock = std::chrono::time_point<std::chrono::high_resolution_clock>;

    static Clock GetTime() {
        return std::chrono::high_resolution_clock::now();
    }

    Clock start_time;

    struct Meta {

        float end_time;
        size_t worker_id;
        float transfer_ms;
        float calc_ms;
    };

    std::list<Meta> metas;

public:
    Timeline();

    static std::unique_ptr<Timeline> s_time_line;

    static void Create();


    static void Add(size_t worker_id, float transfer_ms, float calc_ms);

    static void Save(const std::string &fname);
};

#endif //XUANWU_TIMELINE_H
