add_executable(run_mapping_online run_mapping_online.cc)
target_link_libraries(run_mapping_online
        ${PROJECT_NAME} gflags
        ${LIBDW}
        )
#add_executable(run_mapping_offline run_mapping_offline.cc)
#target_link_libraries(run_mapping_offline
#        ${PROJECT_NAME} gflags
#        )
add_executable(run_loop_online run_loop_online.cpp)
target_link_libraries(run_loop_online
        ${PROJECT_NAME} gflags
        )