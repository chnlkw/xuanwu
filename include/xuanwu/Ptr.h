//
// Created by chnlkw on 3/30/18.
//

#ifndef XUANWU_PTR_H
#define XUANWU_PTR_H

#include "defs.h"

class Ptr {
public:
    enum Type {
        CPU, GPU
    };

    Ptr(void *ptr, Type type = Type::CPU) : ptr_(ptr), type_(type) {}

    virtual void *RawPtr() {
        return ptr_;
    }

    Ptr operator+(ssize_t d) {
        return {(char *) ptr_ + d, type_};
    }

    operator void *() const {
        return ptr_;
    }

    bool isCPU() const { return type_ == Type::CPU; }
    bool isGPU() const { return type_ == Type::GPU; }


private:
    void *ptr_;
    Type type_;
};

#endif //XUANWU_PTR_H
