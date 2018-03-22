//
// Created by chnlkw on 3/5/18.
//

#ifndef DMR_RUNNABLE_H
#define DMR_RUNNABLE_H

class Runnable {
public:
    virtual void RunTask(TaskPtr) = 0;

    virtual size_t NumRunningTasks() const = 0;

    virtual std::vector<TaskPtr> GetCompleteTasks() = 0;

    virtual bool Empty() const {
        return NumRunningTasks() == 0;
    }

    template<class It>
    auto &ChooseRunnable(It beg, It end) {
        assert(beg != end);
        It ret;
        size_t best = std::numeric_limits<size_t>::max();
        while (beg != end) {
            if (best > (*beg)->NumRunningTasks()) {
                best = (*beg)->NumRunningTasks();
                ret = beg;
            }
            ++beg;
        }
        return *ret;
    }
};

#endif //DMR_RUNNABLE_H
