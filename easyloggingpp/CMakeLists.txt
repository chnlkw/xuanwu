
add_library(easyloggingpp easylogging++.cc ../include/xuanwu/Config.h ../include/xuanwu/lru.h)
target_include_directories(easyloggingpp
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
