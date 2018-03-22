add_subdirectory(cuda_utils)

aux_source_directory(. SRC_FILES)
add_library(xuanwu ${SRC_FILES})
target_link_libraries(xuanwu cuda_utils easyloggingpp)
#target_link_libraries(xuanwu cuda_utils logging ${CUDA_LIBRARIES})

target_include_directories(xuanwu
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include/xuanwu
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
