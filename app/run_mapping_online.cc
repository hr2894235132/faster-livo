//
// Created by xiang on 2021/10/8.
//
#include <gflags/gflags.h>
#include <unistd.h>
#include <csignal>

#include "laser_mapping.h"

#ifdef USE_BACKWARD
#define BACKWARD_HAS_DW 1
#include "backward.hpp"
namespace backward
{
    backward::SignalHandling sh;
}
#endif

/// run the lidar mapping in online mode

DEFINE_string(traj_log_file, "/home/hr/workspace/faster_lio_ws/src/faster-lio/Log/traj.txt", "path to traj log file");

void SigHandle(int sig) {
    faster_lio::options::FLAG_EXIT = true;
    ROS_WARN("catch sig %d", sig);
}

int main(int argc, char **argv) {
    FLAGS_stderrthreshold = google::INFO;
    FLAGS_colorlogtostderr = true;
    google::InitGoogleLogging(argv[0]);
    fLS::FLAGS_log_dir = "/home/hr/workspace/faster_lio_ws/src/faster-lio/Log";

    ros::init(argc, argv, "faster_lio");
    ros::NodeHandle nh;

    auto laser_mapping = std::make_shared<faster_lio::LaserMapping>();
    // 初始化ROS相关组件，包括参数导入以及订阅、发布器的定义
    laser_mapping->InitROS(nh);

    signal(SIGINT, SigHandle);
    ros::Rate rate(5000); // 0.0002s/次

    // online, almost same with offline, just receive the messages from ros
    while (ros::ok()) {
        if (faster_lio::options::FLAG_EXIT) {
            break;
        }
        ros::spinOnce();
        laser_mapping->Run(); // main function
        rate.sleep();
    }

    LOG(INFO) << "finishing mapping";
    laser_mapping->Finish();

    faster_lio::Timer::PrintAll();
    LOG(INFO) << "save trajectory to: " << FLAGS_traj_log_file;
    laser_mapping->Savetrajectory(FLAGS_traj_log_file);

    return 0;
}
