//
// Created by chnlkw on 3/14/18.
//

#ifndef GRIDLDA_ALLOCATORBASE_H
#define GRIDLDA_ALLOCATORBASE_H
namespace Xuanwu {
    class AllocatorBase {
    public:

        AllocatorBase() {}

        virtual ~AllocatorBase() {}

        virtual void *Alloc(size_t size) = 0;

        virtual void Free(void *ptr) = 0;

        virtual int Id() const { return -1; }

    };

}
#endif //GRIDLDA_ALLOCATORBASE_H
