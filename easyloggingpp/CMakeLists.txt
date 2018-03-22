
add_library(easyloggingpp easylogging++.cc)
target_include_directories(easyloggingpp
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
