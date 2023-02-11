add_library(${PROJECT_NAME}
        laser_mapping.cc
        pointcloud_preprocess.cc
        options.cc
        utils.cc
        ../include/tools.hpp ../include/inital_aligment.hpp ../include/lidar_selection.h ../include/frame.h ../include/point.h ../include/feature.h ../include/map.h lidar_selection.cpp frame.cpp point.cpp map.cpp ../include/voxel_octree_map/voxel_map_util.hpp)

add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}_gencpp livox_ros_driver_gencpp)

target_link_libraries(${PROJECT_NAME}
        ${catkin_LIBRARIES}
        ${PCL_LIBRARIES}
        ${PYTHON_LIBRARIES}
        tbb
        glog
        yaml-cpp
        )

target_include_directories(${PROJECT_NAME} PRIVATE ${PYTHON_INCLUDE_DIRS})
