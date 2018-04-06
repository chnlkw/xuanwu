//
// Created by chnlkw on 3/30/18.
//

#include "Ptr.h"

void Ptr::log(el::base::type::ostream_t &os) const {
    std::string s;
    switch (type_) {
        case Type::CPU :
            s = "Ptr_CPU";
            break;
        case Type::GPU :
            s = "Ptr_GPU";
            break;
        default:
            s = "Ptr_Unknown";
            break;
    }
    os << s << "[" << ptr_ << "]";
}
