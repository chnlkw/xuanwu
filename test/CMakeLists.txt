set(CMAKE_CUDA_FLAGS ${CMAKE_CUDA_FLAGS} --expt-extended-lambda)
add_executable(run_test_xuanwu run_test_xuanwu.cpp Kernels.cu)
target_link_libraries(run_test_xuanwu xuanwu gtest)

add_test(run_test_xuanwu run_test_xuanwu)
