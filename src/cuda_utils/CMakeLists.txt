
add_library(cuda_utils cuda_utils.cu)
target_include_directories(cuda_utils
        PUBLIC .
        PUBLIC ${CUDA_INCLUDE_DIRS})