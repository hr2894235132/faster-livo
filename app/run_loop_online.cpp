//
// Created by hr on 23-4-25.
//
#include "../src/models/STDesc.cpp"
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/PointCloud2.h>
#include <queue>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

using PointType = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<PointType>;

std::mutex laser_mtx;
std::mutex odom_mtx;

std::queue<sensor_msgs::PointCloud2::ConstPtr> laser_buffer;
std::queue<nav_msgs::Odometry::ConstPtr> odom_buffer;
int flag = 0;

void laserCloudHandler(const sensor_msgs::PointCloud2::ConstPtr &msg) {
    /* 在unique_lock对象的声明周期内，它所管理的锁对象会一直保持上锁状态；而unique_lock的生命周期结束之后，它所管理的锁对象会被解锁
     * 用unique_lock管理互斥对象，可以作为函数的返回值，也可以放到STL的容器中 */
    std::unique_lock<std::mutex> lock(laser_mtx);
    laser_buffer.push(msg);
}

void OdomHandler(const nav_msgs::Odometry::ConstPtr &msg) {
    std::unique_lock<std::mutex> lock(odom_mtx);
    odom_buffer.push(msg);
}

bool syncPackages(PointCloud::Ptr &cloud, Eigen::Affine3d &transform) {
    if (laser_buffer.empty() || odom_buffer.empty())
        return false;
    auto laser_msg = laser_buffer.front();
    double laser_timestamp = laser_msg->header.stamp.toSec();
    auto odom_msg = odom_buffer.front();
    double odom_timestamp = odom_msg->header.stamp.toSec();


    /* 检查时间戳是否对齐 */
    if (abs(laser_timestamp - odom_timestamp) < 1e-3) {
//        if(flag % 20 == 0 && flag != 0){
//            key_frame_times.push_back(odom_timestamp);
//            cout << "flag: " << flag << "time: " << odom_timestamp << endl;
//        }
//        flag++;
        pcl::fromROSMsg(*laser_msg, *cloud);
        Eigen::Quaterniond r(odom_msg->pose.pose.orientation.w, odom_msg->pose.pose.orientation.x,
                             odom_msg->pose.pose.orientation.y, odom_msg->pose.pose.orientation.z);
        Eigen::Vector3d t(odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y,
                          odom_msg->pose.pose.position.z);
        transform = Eigen::Affine3d::Identity();
        transform.translate(t);
        transform.rotate(r);

        std::unique_lock<std::mutex> l_lock(laser_mtx);
        std::unique_lock<std::mutex> o_lock(odom_mtx);
        laser_buffer.pop();
        odom_buffer.pop();
        return true;
    } else if (odom_timestamp < laser_timestamp) {
        ROS_WARN("Current odometry is earlier than laser scan, discard one odometry data.");
        std::unique_lock<std::mutex> o_lock(odom_mtx);
        odom_buffer.pop();
        return false;
    } else {
        ROS_WARN("Current laser scan is earlier than odometry, discard one laser scan.");
        std::unique_lock<std::mutex> l_lock(laser_mtx);
        laser_buffer.pop();
        return false;
    }


}

void update_poses(const gtsam::Values &estimates, std::vector<Eigen::Affine3d> &poses) {
    assert(estimates.size() == poses.size());
    poses.clear();
    for (int i = 0; i < estimates.size(); ++i) {
        auto est = estimates.at<gtsam::Pose3>(i);
        Eigen::Affine3d est_affine3d(est.matrix());
        poses.push_back(est_affine3d);
    }
}

void visualizeLoopClosure(const ros::Publisher &publisher, const std::vector<std::pair<int, int>> &loop_container,
                          const std::vector<Eigen::Affine3d> &key_pose_vec) {
    if (loop_container.empty())
        return;

    visualization_msgs::MarkerArray markerArray;
    // 闭环顶点
    visualization_msgs::Marker markerNode;
    markerNode.header.frame_id = "camera_init"; // camera_init
    // markerNode.header.stamp = ros::Time().fromSec( keyframeTimes.back() );
    markerNode.action = visualization_msgs::Marker::ADD;
    markerNode.type = visualization_msgs::Marker::SPHERE_LIST;
    markerNode.ns = "loop_nodes";
    markerNode.id = 0;
    markerNode.pose.orientation.w = 1;
    markerNode.scale.x = 0.3;
    markerNode.scale.y = 0.3;
    markerNode.scale.z = 0.3;
    markerNode.color.r = 0;
    markerNode.color.g = 0.8;
    markerNode.color.b = 1;
    markerNode.color.a = 1;
    // 闭环边
    visualization_msgs::Marker markerEdge;
    markerEdge.header.frame_id = "camera_init";
    // markerEdge.header.stamp = ros::Time().fromSec( keyframeTimes.back() );
    markerEdge.action = visualization_msgs::Marker::ADD;
    markerEdge.type = visualization_msgs::Marker::LINE_LIST;
    markerEdge.ns = "loop_edges";
    markerEdge.id = 1;
    markerEdge.pose.orientation.w = 1;
    markerEdge.scale.x = 0.1;
    markerEdge.color.r = 0.9;
    markerEdge.color.g = 0.9;
    markerEdge.color.b = 0;
    markerEdge.color.a = 1;

    // 遍历闭环
    for (const auto & it : loop_container) {
        int key_cur = it.first;
        int key_pre = it.second;
        geometry_msgs::Point p;
        p.x = key_pose_vec[key_cur].translation().x();
        p.y = key_pose_vec[key_cur].translation().y();
        p.z = key_pose_vec[key_cur].translation().z();
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
        p.x = key_pose_vec[key_pre].translation().x();
        p.y = key_pose_vec[key_pre].translation().y();
        p.z = key_pose_vec[key_pre].translation().z();
        markerNode.points.push_back(p);
        markerEdge.points.push_back(p);
    }

    markerArray.markers.push_back(markerNode);
    markerArray.markers.push_back(markerEdge);
    publisher.publish(markerArray);
}



int main(int argc, char **argv) {
    ros::init(argc, argv, "online_loop");
    ros::NodeHandle nh;
    ConfigSetting config_setting;
    read_parameters(nh, config_setting);
    /* publisher and subscriber */
    ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
    ros::Publisher pubCurrentCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_current", 100);
    ros::Publisher pubCurrentCorner = nh.advertise<sensor_msgs::PointCloud2>("/cloud_key_points", 100);
    ros::Publisher pubMatchedCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched", 100);
    ros::Publisher pubMatchedCorner = nh.advertise<sensor_msgs::PointCloud2>("/cloud_matched_key_points", 100);
    ros::Publisher pubSTD = nh.advertise<visualization_msgs::MarkerArray>("descriptor_line", 10);

    ros::Publisher pubOriginCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_origin", 10000);
    ros::Publisher pubCorrectCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_correct", 10000);
    ros::Publisher pubCorrectPath = nh.advertise<nav_msgs::Path>("/correct_path", 100000);

    ros::Publisher pubOdomOrigin = nh.advertise<nav_msgs::Odometry>("/odom_origin", 10);
    ros::Publisher pubOdomCorrected = nh.advertise<nav_msgs::Odometry>("/odom_corrected", 10);
    ros::Publisher pubLoopConstraintEdge = nh.advertise<visualization_msgs::MarkerArray>("/loop_closure_constraints",
                                                                                         10);
    ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/cloud_registered_body", 100,
                                                                           laserCloudHandler);
    ros::Subscriber subOdom = nh.subscribe<nav_msgs::Odometry>("/Odometry", 100, OdomHandler);

    auto *std_manager = new STDescManager(config_setting);

    /* gtsam图优化初始化 */
    gtsam::Values initial;
    gtsam::NonlinearFactorGraph graph;
    gtsam::noiseModel::Diagonal::shared_ptr odomNoise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4).finished());
    gtsam::noiseModel::Diagonal::shared_ptr priorNoise = gtsam::noiseModel::Diagonal::Variances(
            (gtsam::Vector(6) << 1e-2, 1e-2, M_PI * M_PI, 1e8, 1e8, 1e8).finished()); // rad*rad, meter*meter
    double loopNoiseScore = 1e-1;
    gtsam::Vector robustNoiseVector6(6); // gtsam::Pose3 factor has 6 elements (6D)
    robustNoiseVector6
            << loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore, loopNoiseScore;
    // 用Cauchy估计器创建了稳健的噪声模型，其尺度为1，Variances设置在robustNoiseVector6
    gtsam::noiseModel::Base::shared_ptr robustLoopNoise = gtsam::noiseModel::Robust::Create(
            gtsam::noiseModel::mEstimator::Cauchy::Create(1),
            gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6));

    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01; // 重新线性化阈值
    parameters.relinearizeSkip = 1; // 步长
    gtsam::ISAM2 isam(parameters);

    size_t cloudInd = 0;
    size_t keyCloudInd = 0;

    std::vector<PointCloud::Ptr> cloud_vec;
    std::vector<Eigen::Affine3d> pose_vec;
    std::vector<Eigen::Affine3d> origin_pose_vec;
    std::vector<Eigen::Affine3d> key_pose_vec;
    std::vector<std::pair<int, int>> loop_container;

    std::vector<double> descriptor_time;
    std::vector<double> querying_time;
    std::vector<double> update_time;

    PointCloud::Ptr key_cloud(new PointCloud);

    bool has_loop_flag = false;

    gtsam::Values curr_estimate;
    Eigen::Affine3d last_pose;
    last_pose.setIdentity();

    while (ros::ok()) {
        ros::spinOnce();
        PointCloud::Ptr current_cloud_body(new PointCloud);
        PointCloud::Ptr current_cloud_world(new PointCloud);
        Eigen::Affine3d transform;

        if (syncPackages(current_cloud_body, transform)) {
            auto origin_estimate_affine3d = transform;
            pcl::transformPointCloud(*current_cloud_body, *current_cloud_world, transform);
            down_sampling_voxel(*current_cloud_world, config_setting.ds_size_);
            // down sample body cloud TODO:用处？
            down_sampling_voxel(*current_cloud_body, 0.5);
            cloud_vec.push_back(current_cloud_body);
            pose_vec.push_back(transform);
            origin_pose_vec.push_back(transform);

            PointCloud origin_cloud;
            pcl::transformPointCloud(*current_cloud_body, origin_cloud, origin_estimate_affine3d);
            /* 发布origin cloud */
            sensor_msgs::PointCloud2 pub_cloud;
            pcl::toROSMsg(origin_cloud, pub_cloud);
            pub_cloud.header.frame_id = "camera_init";
            pubOriginCloud.publish(pub_cloud);

            /* 发布origin odometry */
            Eigen::Quaterniond _r(origin_estimate_affine3d.rotation());
            nav_msgs::Odometry odom;
            odom.header.frame_id = "camera_init";
            odom.pose.pose.position.x = origin_estimate_affine3d.translation().x();
            odom.pose.pose.position.y = origin_estimate_affine3d.translation().y();
            odom.pose.pose.position.z = origin_estimate_affine3d.translation().z();
            odom.pose.pose.orientation.w = _r.w();
            odom.pose.pose.orientation.x = _r.x();
            odom.pose.pose.orientation.y = _r.y();
            odom.pose.pose.orientation.z = _r.z();
            pubOdomOrigin.publish(odom);

            *key_cloud += *current_cloud_world; // 累积连续扫描点云形成子帧

            /* pose graph construction */
            initial.insert(cloudInd, gtsam::Pose3(transform.matrix()));
            if (!cloudInd) {
                graph.add(gtsam::PriorFactor<gtsam::Pose3>(0, gtsam::Pose3(transform.matrix()), priorNoise));
            } else {
                auto pre_pose = gtsam::Pose3(origin_pose_vec[cloudInd - 1].matrix());
                auto curr_pose = gtsam::Pose3(transform.matrix());
                graph.add(gtsam::BetweenFactor<gtsam::Pose3>(cloudInd - 1, cloudInd, pre_pose.between(curr_pose),
                                                             odomNoise));
            }
            /* printf the current graph */
//            if(cloudInd % 100 == 0){
//                cout << "posegraph odom node " << cloudInd << " added." << endl;
//                graph.print("\nFactor Graph: \n");
//            }

            // check if keyframe
            /* STD detector */
            if (cloudInd % config_setting.sub_frame_num_ == 0 && cloudInd != 0) {

                ROS_INFO("key frame idx: [%d], key cloud size: [%d]", (int) keyCloudInd, (int) key_cloud->size());
                /* step1. Descriptor Extraction */
                auto t_descriptor_begin = std::chrono::high_resolution_clock::now();
                std::vector<STDesc> std_vec;
                cout << "cloudInd: " << cloudInd << "current_frame_id: " << std_manager->current_frame_id_ << endl;
                std_manager->GenerateSTDescs(key_cloud, std_vec);
                auto t_descriptor_end = std::chrono::high_resolution_clock::now();
                descriptor_time.push_back(time_inc(t_descriptor_end, t_descriptor_begin));
                /* step2. Searching Loop */
                auto t_query_begin = std::chrono::high_resolution_clock::now();
                std::pair<int, double> search_result(-1, 0);
                std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
                loop_transform.first << 0, 0, 0;
                loop_transform.second = Eigen::Matrix3d::Identity();
                std::vector<std::pair<STDesc, STDesc>> loop_std_pair;
                if (keyCloudInd > config_setting.skip_near_num_) {
                    std_manager->SearchLoop(std_vec, search_result, loop_transform, loop_std_pair);
                }
                auto t_query_end = std::chrono::high_resolution_clock::now();
                querying_time.push_back(time_inc(t_query_end, t_query_begin));
                /* step3. Add descriptors to the database */
                auto t_std_update_begin = std::chrono::high_resolution_clock::now();
                std_manager->AddSTDescs(std_vec);
                auto t_std_update_end = std::chrono::high_resolution_clock::now();
                update_time.push_back(time_inc(t_std_update_end, t_std_update_begin));
                std::cout << "[Time] descriptor extraction: "
                          << time_inc(t_descriptor_end, t_descriptor_begin) << "ms, "
                          << "query: " << time_inc(t_query_end, t_query_begin)
                          << "ms, "
                          << "update map:"
                          << time_inc(t_std_update_end, t_std_update_begin) << "ms"
                          << std::endl;
                std::cout << std::endl;

                /* 发布当前点云 && 当前提取的特征点 */
                sensor_msgs::PointCloud2 pub_cloud_c;
                pcl::toROSMsg(*key_cloud, pub_cloud_c);
                pub_cloud_c.header.frame_id = "camera_init";
                pubCurrentCloud.publish(pub_cloud_c);
                sensor_msgs::PointCloud2 pub_corner_c;
                pcl::toROSMsg(*std_manager->corner_cloud_vec_.back(), pub_corner_c);
                pub_corner_c.header.frame_id = "camera_init";
                pubCurrentCorner.publish(pub_corner_c);

                std_manager->key_cloud_vec_.push_back(key_cloud->makeShared());
                if (search_result.first > 0) {
                    ROS_INFO("[Loop Detection] triggle loop: [%d], -- [%d], , score: [%f]",
                             static_cast<int>(keyCloudInd), search_result.first, search_result.second);
                    has_loop_flag = true;
                    int match_frame = search_result.first;
                    /* 进一步优化点面距离，以获得更精确的循环校正变换，通过ceres-solver实现 */
                    std_manager->PlaneGeomrtricIcp(std_manager->plane_cloud_vec_.back(),
                                                   std_manager->plane_cloud_vec_[match_frame], loop_transform);
                    /* debug */
                    // std::cout << "delta transform:" << std::endl;
                    // std::cout << "translation: " << loop_transform.first.transpose() << std::endl;
                    // auto euler = loop_transform.second.eulerAngles(2, 1, 0) * 57.3;
                    // std::cout << "rotation(ypr): " << euler[0] << ' ' << euler[1] << ' ' << euler[2] << std::endl;

                    /*
                      add connection between loop frame.
                      e.g. if src_key_frame_id 5 with sub frames 51~60 triggle loop with
                      tar_key_frame_id 1 with sub frames 11~20, add connection between
                      each sub frame, 51-11, 52-12,...,60-20.
                    */
                    int sub_frame_num = config_setting.sub_frame_num_;
                    for (size_t j = 1; j <= sub_frame_num; ++j) {
                        int src_frame = static_cast<int>(cloudInd) + static_cast<int>(j) - sub_frame_num;
                        auto delta_T = Eigen::Affine3d::Identity();
                        delta_T.translate(loop_transform.first);
                        delta_T.rotate(loop_transform.second);
                        Eigen::Affine3d src_pose_refined = delta_T * pose_vec[src_frame];

                        int tar_frame = static_cast<int>(match_frame) * sub_frame_num + static_cast<int>(j);
                        /* old */
                        Eigen::Affine3d tar_pose = origin_pose_vec[tar_frame];
                        loop_container.emplace_back(tar_frame, src_frame);
                        graph.add(gtsam::BetweenFactor<gtsam::Pose3>(tar_frame, src_frame,
                                                                     gtsam::Pose3(tar_pose.matrix()).between(
                                                                             gtsam::Pose3(src_pose_refined.matrix())),
                                                                     robustLoopNoise));
                    }
                    /* 发布匹配成功点云 && 匹配成功的关键点 && STD匹配对 */
                    pcl::toROSMsg(*std_manager->key_cloud_vec_[search_result.first], pub_cloud);
                    pub_cloud.header.frame_id = "camera_init";
                    pubMatchedCloud.publish(pub_cloud);

                    pcl::toROSMsg(*std_manager->corner_cloud_vec_[search_result.first], pub_cloud);
                    pub_cloud.header.frame_id = "camera_init";
                    pubMatchedCorner.publish(pub_cloud);
                    /* 发布检测到的描述子匹配对 */
                    publish_std_pairs(loop_std_pair, pubSTD);
                }
                key_cloud->clear();
                ++keyCloudInd;
            }
            isam.update(graph, initial);
            isam.update(); // 执行非线性优化

            if (has_loop_flag) {
                isam.update();
                isam.update();
                isam.update();
                isam.update();
                isam.update();
            }
            graph.resize(0);
            initial.clear();

            curr_estimate = isam.calculateEstimate();
            update_poses(curr_estimate, pose_vec);

            auto latest_estimate_affine3d = pose_vec.back();
            if (has_loop_flag){
                /* 发布矫正后的地图 */
                PointCloud full_map;
                for (int i = 0; i < pose_vec.size(); ++i) {
                    PointCloud correct_cloud;
                    pcl::transformPointCloud(*cloud_vec[i], correct_cloud, pose_vec[i]);
                    full_map += correct_cloud;
                }
                sensor_msgs::PointCloud2 pub_correct_cloud;
                pcl::toROSMsg(full_map, pub_correct_cloud);
                pub_correct_cloud.header.frame_id = "camera_init";
                pubCorrectCloud.publish(pub_correct_cloud);
                /* 发布矫正后的路径 */
                nav_msgs::Path correct_path;
                for (auto & pose : pose_vec) {
                    geometry_msgs::PoseStamped msg_pose;
                    msg_pose.header.frame_id = "camera_init";
                    msg_pose.pose.position.x = pose.translation()[0];
                    msg_pose.pose.position.y = pose.translation()[1];
                    msg_pose.pose.position.z = pose.translation()[2];
                    Eigen::Quaterniond pose_q(pose.rotation());
                    msg_pose.pose.orientation.x = pose_q.x();
                    msg_pose.pose.orientation.y = pose_q.y();
                    msg_pose.pose.orientation.z = pose_q.z();
                    msg_pose.pose.orientation.w = pose_q.w();
                    correct_path.poses.push_back(msg_pose);
                }
                correct_path.header.stamp = ros::Time::now();
                correct_path.header.frame_id = "camera_init";
                pubCorrectPath.publish(correct_path);
            }

            visualizeLoopClosure(pubLoopConstraintEdge, loop_container, pose_vec);

            has_loop_flag = false;
            ++cloudInd;
        }
    }
    // You can save full map with refined pose
    // assert(cloud_vec.size() == pose_vec.size());
    // PointCloud full_map;
    // for (int i = 0; i < pose_vec.size(); ++i) {
    //     PointCloud correct_cloud;
    //     pcl::transformPointCloud(*cloud_vec[i], correct_cloud, pose_vec[i]);
    //     full_map += correct_cloud;
    // }
    // down_sampling_voxel(full_map, 0.05);

    // std::cout << "saving map..." << std::endl;
    // pcl::io::savePCDFileBinary("/home/dustier/data/map.pcd", full_map);
    return 0;
}




