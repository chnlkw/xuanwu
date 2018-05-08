cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(xuanwu LANGUAGES CXX CUDA)

find_package(CUDA)

#enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(easyloggingpp)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wreturn-type")

add_subdirectory(src)

option(TEST "Build Tests" OFF)

if (${TEST})

    enable_testing()

    add_subdirectory(googletest)

    add_subdirectory(test)

endif ()
