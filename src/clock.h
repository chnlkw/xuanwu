#ifndef __CLOCK
#define __CLOCK

#include <cstdio>
#include <chrono>

namespace Xuanwu {
    class Clock {
    public:
        void start();

        double timeElapsed();

        double restart();

        void pause();

        void resume();

        Clock();

        template<class Function>
        static double CalcTime(Function f) {
            auto start = std::chrono::high_resolution_clock::now();
            f();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            return diff.count();
        }


    private:
        double _elapsed;
        double _last;
        bool _started;
    };
}
#endif
