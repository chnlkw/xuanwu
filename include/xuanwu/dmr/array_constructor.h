//
// Created by chnlkw on 1/12/18.
//

#ifndef DMR_ARRAY_CONSTRUCTOR_H
#define DMR_ARRAY_CONSTRUCTOR_H

struct vector_constructor_t {
    template<class T, class ...Args>
    static std::vector<T> Construct(Args &&... args) {
        return {std::forward<Args>(args)...};
    }
};

#endif //DMR_ARRAY_CONSTRUCTOR_H
