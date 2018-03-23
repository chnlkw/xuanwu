cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(xuanwu LANGUAGES CXX CUDA)

find_package(CUDA)

#enable_language(CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_subdirectory(easyloggingpp)

add_subdirectory(src)

enable_testing()

add_subdirectory(gtest)

add_subdirectory(test)
