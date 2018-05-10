//
// Created by chnlkw on 18-5-8.
//

#ifndef XUANWU_LRU_H
#define XUANWU_LRU_H

#include "defs.h"
#include <cassert>

template<class T>
class LRU {
public:
    struct Node {
        T val;
        Node *left = nullptr;
        Node *right = nullptr;
    };

    LRU() = default;

    ~LRU() {
        while (head)
            Delete(head);
    }

    Node *Insert(T &&v) {
        assert((head != nullptr) == (tail != nullptr));
        auto *node = new Node{std::move(v), nullptr, head};
        if (head)
            head->left = node;
        head = node;
        if (!tail)
            tail = node;
        return head;
    }

    void Delete(Node *node) {
        auto l = node->left;
        auto r = node->right;
        if (l) l->right = r; else head = r;
        if (r) r->left = l; else tail = l;
        delete node;
    }

    Node *Last() {
        return tail;
    }

private:
    Node *head = nullptr;
    Node *tail = nullptr;
};

#endif //XUANWU_LRU_H
