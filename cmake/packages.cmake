list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# glog
find_package(Glog REQUIRED)
include_directories(${Glog_INCLUDE_DIRS})

# for ubuntu 18.04, update gcc/g++ to 9, and download tbb2018 from
# https://github.com/oneapi-src/oneTBB/releases/download/2018/tbb2018_20170726oss_lin.tgz,
# extract it into CUSTOM_TBB_DIR 
# specifiy tbb2018, e.g. CUSTOM_TBB_DIR=/home/idriver/Documents/tbb2018_20170726oss
if (CUSTOM_TBB_DIR)
    set(TBB2018_INCLUDE_DIR "${CUSTOM_TBB_DIR}/include")
    set(TBB2018_LIBRARY_DIR "${CUSTOM_TBB_DIR}/lib/intel64/gcc4.7")
    include_directories(${TBB2018_INCLUDE_DIR})
    link_directories(${TBB2018_LIBRARY_DIR})
endif ()

message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)" )
    include(ProcessorCount)
    ProcessorCount(N)
    message("Processer number:  ${N}")
    if(N GREATER 5)
        add_definitions(-DMP_EN)
        add_definitions(-DMP_PROC_NUM=4)
        message("core for MP:  4")
    elseif(N GREATER 3)
        math(EXPR PROC_NUM "${N} - 2")
        add_definitions(-DMP_EN)
        add_definitions(-DMP_PROC_NUM="${PROC_NUM}")
        message("core for MP:  ${PROC_NUM}")
    else()
        add_definitions(-DMP_PROC_NUM=1)
    endif()
else()
    add_definitions(-DMP_PROC_NUM=1)
endif()

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

find_package(PythonLibs REQUIRED)
find_path(MATPLOTLIB_CPP_INCLUDE_DIRS "matplotlibcpp.h")
#set(cv_bridge_DIR /home/hr/third_party/ws/devel/share/cv_bridge/cmake)

find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        nav_msgs
        sensor_msgs
        roscpp
        rospy
        std_msgs
        pcl_ros
        tf
        livox_ros_driver
        message_generation
        eigen_conversions
        vikit_common
        vikit_ros
        cv_bridge
        image_transport
        )


add_message_files(
        FILES
        Pose6D.msg
        Euler.msg
        States.msg
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
)
catkin_package(
        CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime cv_bridge image_transport vikit_common vikit_ros
        DEPENDS EIGEN3 PCL OpenCV Sophus
        INCLUDE_DIRS
)


find_package(Eigen3 REQUIRED)
find_package(PCL 1.8 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Ceres REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)
FIND_PACKAGE(Sophus REQUIRED)
FIND_PACKAGE(Boost REQUIRED COMPONENTS thread)
FIND_PACKAGE(GTSAM REQUIRED)
set(Sophus_LIBRARIES libSophus.so)

include_directories(
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${PYTHON_INCLUDE_DIRS}
        ${yaml-cpp_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${Sophus_INCLUDE_DIRS}
        include
)

