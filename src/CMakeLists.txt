add_subdirectory(cuda_utils)

aux_source_directory(. SRC_FILES)

if (NUMA_FOUND)
    aux_source_directory(numas SRC_FILES)
    message("SRC_FILES ${SRC_FILES}")
endif ()

add_library(xuanwu ${SRC_FILES} numas/NumaWorker.cpp ../include/xuanwu/numas/NumaWorker.h)
target_link_libraries(xuanwu cuda_utils easyloggingpp)

if (NUMA_FOUND)
    message("link NUMA")
    target_link_libraries(xuanwu numa)
endif ()


target_include_directories(xuanwu
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include/xuanwu
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../include)
