#include <tf/transform_broadcaster.h>
#include <yaml-cpp/yaml.h>
#include <execution>
#include <fstream>
#include <iostream>

#include "laser_mapping.h"
#include "utils.h"
#include "tools.hpp"
#include <vikit/camera_loader.h>


#define DEBUG_FILE_DIR(name)     (std::string(std::string(ROOT_DIR) + "Log/"+ name))
#define USE_VOXEL_OCTREE
#define CALIB_ANGLE_COV (0.01)

//inline void kitti_log(FILE *fp) {}

namespace faster_lio {
    /**
     * @brief ROS初始化
     * @param nh
     * @return true / false
     */
    bool LaserMapping::InitROS(ros::NodeHandle &nh) {
        LoadParams(nh); // load parameters from .yaml
        SubAndPubToROS(nh); // ROS subscribe and publish initialization

        // localmap init (after LoadParams)
        ivox_ = std::make_shared<IVoxType>(ivox_options_);

#ifndef USE_IKFOM
        G.setZero();
        H_T_H.setZero();
        I_STATE.setIdentity();
#endif

#ifdef USE_IKFOM
        // esekf init
        std::vector<double> epsi(23, 0.001);
        /* 初始化，传入几个函数
         * 1. get_f: 用于根据IMU数据向前推算
         * 2. df_dx: 误差状态模型，（连续时间下）
         * 3. df_dw: 误差状态模型，误差状态对过程噪声求导
         * 4. lambda: 函数类型 std::function<void(state &, dyn_share_datastruct<scalar_type> &)>
         * 函数作用：计算点面残差，然后计算测量的雅可比矩阵
        */
        kf_.init_dyn_share(
                get_f, df_dx, df_dw,
                [this](state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) { ObsModel(s, ekfom_data); },
                options::NUM_MAX_ITERATIONS, epsi.data());
        // std::vector::data():该函数返回一个指向数组中第一个元素的指针，该指针在向量内部使用。
        // get_f: return Eigen::Matrix<double, 24, 1> (imu data)
        // df_dx: return Eigen::Matrix<double, 24, 23>
        // df_dw: return Eigen::Matrix<double, 24, 12>
        state_ikfom init_state = kf_.get_x();
#endif
        return true;
    }

    /**
     * @brief 读取配置文件，初始化IMUProcessor
     * @param nh
     * @return true / false
     */
    bool LaserMapping::LoadParams(ros::NodeHandle &nh) {
        // get params from param server
        int lidar_type, ivox_nearby_type;
        double gyr_cov, acc_cov, b_gyr_cov, b_acc_cov;
        double filter_size_surf_min, filter_size_corner_min = 0;
        std::vector<int> layer_point_size;

        // TODO：12.1 hr 打印状态信息文件
        fout_pre.open(DEBUG_FILE_DIR("mat_pre.txt"), std::ios::out);
        fout_out.open(DEBUG_FILE_DIR("mat_out.txt"), std::ios::out);
        if (fout_pre && fout_out)
            std::cout << "~~~~" << ROOT_DIR << " file opened" << std::endl;
        else
            std::cout << "~~~~" << ROOT_DIR << " doesn't exist" << std::endl;

        // todo:12.8 新增视觉相关参数
        nh.param<int>("dense_map_enable", dense_map_en, 1);
        nh.param<int>("img_enable", img_en, 1);
        nh.param<int>("lidar_enable", lidar_en, 1);
        nh.param<int>("debug", debug, 0);
        nh.param<bool>("ncc_en", ncc_en, false);
        nh.param<int>("depth_img_enable", depth_img_en_, 1);
        nh.param<int>("min_img_count", MIN_IMG_COUNT, 1000);
        nh.param<double>("cam_fx", cam_fx, 453.483063); // 相机内参
        nh.param<double>("cam_fy", cam_fy, 453.254913);
        nh.param<double>("cam_cx", cam_cx, 318.908851);
        nh.param<double>("cam_cy", cam_cy, 234.238189);
        nh.param<double>("laser_point_cov", LASER_POINT_COV, 0.001);
        nh.param<double>("img_point_cov", IMG_POINT_COV, 10);
        nh.param<string>("map_file_path", map_file_path_, "");

        nh.param<bool>("path_save_en", path_save_en_, true);
        nh.param<bool>("publish/path_publish_en", path_pub_en_, true);
        nh.param<bool>("publish/scan_publish_en", scan_pub_en_, true);
        nh.param<bool>("publish/dense_publish_en", dense_pub_en_, false);
        nh.param<bool>("publish/scan_bodyframe_pub_en", scan_body_pub_en_, true);
        nh.param<bool>("publish/scan_effect_pub_en", scan_effect_pub_en_, false);
        nh.param<int>("max_iteration", options::NUM_MAX_ITERATIONS, 4);
        nh.param<float>("esti_plane_threshold", options::ESTI_PLANE_THRESHOLD, 0.1);
        nh.param<std::string>("map_file_path", map_file_path_, "");
        nh.param<bool>("common/time_sync_en", time_sync_en_, false);
        nh.param<double>("filter_size_corner", filter_size_corner_min, 0.5);
        nh.param<double>("filter_size_surf", filter_size_surf_min, 0.5);
        nh.param<double>("filter_size_map", filter_size_map_min_, 0.5);
        nh.param<double>("cube_side_length", cube_len_, 200);
        nh.param<double>("mapping/fov_degree", fov_deg, 180); // FOV
        nh.param<float>("mapping/det_range", det_range_, 300.f);
        nh.param<double>("mapping/gyr_cov", gyr_cov, 0.1);
        nh.param<double>("mapping/acc_cov", acc_cov, 0.1);
        nh.param<double>("mapping/b_gyr_cov", b_gyr_cov, 0.0001);
        nh.param<double>("mapping/b_acc_cov", b_acc_cov, 0.0001);
        nh.param<double>("preprocess/blind", preprocess_->Blind(), 0.01);
        nh.param<float>("preprocess/time_scale", preprocess_->TimeScale(), 1e-3);
        nh.param<int>("preprocess/lidar_type", lidar_type, 1);
        nh.param<int>("preprocess/scan_line", preprocess_->NumScans(), 16);
        nh.param<int>("point_filter_num", preprocess_->PointFilterNum(), 2);
        nh.param<bool>("feature_extract_enable", preprocess_->FeatureEnabled(), false);
        nh.param<bool>("runtime_pos_log_enable", runtime_pos_log_, true);
        nh.param<bool>("mapping/extrinsic_est_en", extrinsic_est_en_, true);
        nh.param<bool>("pcd_save/pcd_save_en", pcd_save_en_, false);
        nh.param<int>("pcd_save/interval", pcd_save_interval_, -1);
        nh.param<std::vector<double>>("mapping/extrinsic_T", extrinT_, std::vector<double>());
        nh.param<std::vector<double>>("mapping/extrinsic_R", extrinR_, std::vector<double>());
        // todo:12.8 new param
        nh.param<double>("mapping/gyr_cov_scale", gyr_cov_scale, 1.0);
        nh.param<double>("mapping/acc_cov_scale", acc_cov_scale, 1.0);
        nh.param<vector<double>>("camera/Pcl", cameraextrinT, vector<double>()); // 相机雷达外参
        nh.param<vector<double>>("camera/Rcl", cameraextrinR, vector<double>());
        nh.param<int>("grid_size", grid_size, 40);
        nh.param<int>("patch_size", patch_size, 4);
        nh.param<double>("outlier_threshold", outlier_threshold, 100);
        nh.param<double>("ncc_thre", ncc_thre, 100);


        // 新数据结构里用到的参数
        nh.param<float>("ivox_grid_resolution", ivox_options_.resolution_, 0.2);
        nh.param<int>("ivox_nearby_type", ivox_nearby_type, 18);

#ifdef USE_VOXEL_OCTREE
        // TODO:0216 自适应体素结构参数 mapping algorithm params
        // noise model params
        nh.param<double>("noise_model/ranging_cov", ranging_cov, 0.02);
        nh.param<double>("noise_model/angle_cov", angle_cov, 0.05);

        nh.param<int>("voxel/max_points_size", max_points_size, 100);
        nh.param<int>("voxel/max_cov_points_size", max_cov_points_size, 100);
        nh.param<vector<int>>("voxel/layer_point_size", layer_point_size, vector<int>()); // TODO:0216原代码中使用double
        nh.param<int>("voxel/max_layer", max_layer, 2);
        nh.param<double>("voxel/voxel_size", max_voxel_size, 1.0);
        std::cout << "voxel_size:" << max_voxel_size << std::endl;
        nh.param<double>("voxel/plannar_threshold", min_eigen_value, 0.01);

        for (double i: layer_point_size) {
            layer_size.push_back(i);
        }
        // visualization params
        nh.param<bool>("publish/pub_voxel_map", publish_voxel_map, false);
        nh.param<int>("publish/publish_max_voxel_layer", publish_max_voxel_layer, 0);
        nh.param<int>("publish/publish_only_voxel_layer", publish_only_voxel_layer, 0);
        nh.param<bool>("publish/pub_point_cloud", publish_point_cloud, true);
        nh.param<int>("publish/pub_point_cloud_skip", pub_point_cloud_skip, 1);
#endif

        LOG(INFO) << "lidar_type " << lidar_type;
        if (lidar_type == 1) {
            preprocess_->SetLidarType(LidarType::AVIA);
            LOG(INFO) << "Using AVIA Lidar";
        } else if (lidar_type == 2) {
            preprocess_->SetLidarType(LidarType::VELO32);
            LOG(INFO) << "Using Velodyne 32 Lidar";
        } else if (lidar_type == 3) {
            preprocess_->SetLidarType(LidarType::OUST64);
            LOG(INFO) << "Using OUST 64 Lidar";
        } else {
            LOG(WARNING) << "unknown lidar_type";
            return false;
        }

        if (ivox_nearby_type == 0) {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::CENTER;
        } else if (ivox_nearby_type == 6) {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY6;
        } else if (ivox_nearby_type == 18) {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        } else if (ivox_nearby_type == 26) {
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY26;
        } else {
            LOG(WARNING) << "unknown ivox_nearby_type, use NEARBY18";
            ivox_options_.nearby_type_ = IVoxType::NearbyType::NEARBY18;
        }

        path_.header.stamp = ros::Time::now();
        path_.header.frame_id = "camera_init";

        voxel_scan_.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);

//        // todo:12.8 降采样系数
//        downSizeFilterSurf.setLeafSize(filter_size_surf_min, filter_size_surf_min, filter_size_surf_min);
//        downSizeFilterMap.setLeafSize(filter_size_map_min_, filter_size_map_min_, filter_size_map_min_);

        lidar_T_wrt_IMU = VecFromArray<double>(extrinT_);
        lidar_R_wrt_IMU = MatFromArray<double>(extrinR_);
        if (!vk::camera_loader::loadFromRosNs("laserMapping", lidar_selector->cam))
            throw std::runtime_error("Camera model not correctly specified.");

        // todo:12.8 初始化lidar_selection和imu的一些参数
        lidar_selector->MIN_IMG_COUNT = MIN_IMG_COUNT; // 1000
        lidar_selector->debug = debug; // 0 是否显示debug信息
        lidar_selector->grid_size = grid_size;
        lidar_selector->patch_size = patch_size; // 8
        lidar_selector->outlier_threshold = outlier_threshold; // 300
        lidar_selector->ncc_thre = ncc_thre; // 0 ncc 的阈值
        lidar_selector->sparse_map->set_camera2lidar(cameraextrinR, cameraextrinT);
        lidar_selector->set_extrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
        lidar_selector->state = &state;
        lidar_selector->state_propagat = &state_propagat;
        lidar_selector->NUM_MAX_ITERATIONS = options::NUM_MAX_ITERATIONS; // 4
        lidar_selector->img_point_cov = IMG_POINT_COV; // 100
        lidar_selector->fx = cam_fx;
        lidar_selector->fy = cam_fy;
        lidar_selector->cx = cam_cx;
        lidar_selector->cy = cam_cy;
        lidar_selector->ncc_en = ncc_en; // 0
        lidar_selector->init();

        p_imu_->SetExtrinsic(lidar_T_wrt_IMU, lidar_R_wrt_IMU);
        p_imu_->SetGyrCovScale(V3D(gyr_cov_scale, gyr_cov_scale, gyr_cov_scale));
        p_imu_->SetAccCovScale(V3D(acc_cov_scale, acc_cov_scale, acc_cov_scale));
        p_imu_->SetGyrBiasCov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
        p_imu_->SetAccBiasCov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));
//        p_imu_->SetGyrBiasCov(V3D(0.00003, 0.00003, 0.00003));
//        p_imu_->SetAccBiasCov(V3D(0.01, 0.01, 0.01));
        return true;
    }

    /**
    * @brief ROS subscribe and publish initialization
    * @param nh
    * @return
    */
    void LaserMapping::SubAndPubToROS(ros::NodeHandle &nh) {
        // ROS subscribe initialization
        image_transport::ImageTransport it(nh);
        std::string lidar_topic, imu_topic, img_topic;
        nh.param<std::string>("common/lid_topic", lidar_topic, "/livox/lidar");
        nh.param<std::string>("common/imu_topic", imu_topic, "/livox/imu");
        // todo:12.8 新增图像话题
        nh.param<std::string>("camera/img_topic", img_topic, "/usb_cam/image_raw");
        cout << "debug:" << debug << " MIN_IMG_COUNT: " << MIN_IMG_COUNT << endl;
        pcl_wait_pub_->clear(); // world frame

//        // result params
//        nh.param<bool>("Result/write_kitti_log", write_kitti_log, 'false');
//        nh.param<string>("Result/result_path", result_path, "");

        // 订阅点云：根据雷达类型选择对应的点云回调函数（自定义类型：LivoxPCLCallBack；pcl标准点云类型：StandardPCLCallBack）
        if (preprocess_->GetLidarType() == LidarType::AVIA) {
            sub_pcl_ = nh.subscribe<livox_ros_driver::CustomMsg>(
                    lidar_topic, 200000,
                    [this](const livox_ros_driver::CustomMsg::ConstPtr &msg) { LivoxPCLCallBack(msg); });
        } else {
            sub_pcl_ = nh.subscribe<sensor_msgs::PointCloud2>(
                    lidar_topic, 200000,
                    [this](const sensor_msgs::PointCloud2::ConstPtr &msg) { StandardPCLCallBack(msg); });
        }
        // 订阅imu
        sub_imu_ = nh.subscribe<sensor_msgs::Imu>(imu_topic, 200000,
                                                  [this](const sensor_msgs::Imu::ConstPtr &msg) { IMUCallBack(msg); });
        // todo:12.8 订阅img
        sub_img_ = nh.subscribe<sensor_msgs::Image>(img_topic, 200000,
                                                    [this](const sensor_msgs::ImageConstPtr &msg) {
                                                        IMGCallBack(msg);
                                                    });
//        sub_img_ = nh.subscribe(img_topic, 200000, IMGCallBack);

        // ROS publisher init
        path_.header.stamp = ros::Time::now();
        path_.header.frame_id = "camera_init";


        // todo：12.8 新增publisher
        img_pub_ = it.advertise("/rgb_img", 1);
        pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
        pubLaserCloudFullResRgb = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_rgb", 100);
        pubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_map", 100);
        pubSubVisualCloud = nh.advertise<sensor_msgs::PointCloud2>("/cloud_visual_sub_map", 100);
        pubLaserCloudEffect = nh.advertise<sensor_msgs::PointCloud2>("/cloud_effected", 100);
        pubLaserCloudMap = nh.advertise<sensor_msgs::PointCloud2>("/Laser_map", 100);
#ifdef USE_VOXEL_OCTREE
        voxel_map_pub = nh.advertise<visualization_msgs::MarkerArray>("/planes", 10000);
#endif

        // TODO：更新成livo pubLaserCloudFullRes
        pub_laser_cloud_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 100);
        pub_laser_cloud_body_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_body", 100);
        // TODO：更新成livo pubLaserCloudEffect
        pub_laser_cloud_effect_world_ = nh.advertise<sensor_msgs::PointCloud2>("/cloud_registered_effect_world",
                                                                               100);
        pub_odom_aft_mapped_ = nh.advertise<nav_msgs::Odometry>("/Odometry", 10);
//        ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 10);
        pub_path_ = nh.advertise<nav_msgs::Path>("/path", 10);
//        ros::Publisher pubPath = nh.advertise<nav_msgs::Path>("/path", 10);
        pub_Euler_ = nh.advertise<faster_lio::Euler>("/odom_euler", 100000);
    }

    LaserMapping::LaserMapping() {
        preprocess_.reset(new PointCloudPreprocess());
        p_imu_.reset(new ImuProcess());
    }

    void LaserMapping::Run() {
        if (!SyncPackages(LidarMeasures_)) {
            cv::waitKey(1);
            return;
        }

        /*** Packaged got ***/
        if (flg_reset_) {
            ROS_WARN("reset when rosbag play back");
            p_imu_->Reset();
            flg_reset_ = false;
            return;
        }
        std::cout << "scan_Idx: " << scanIdx << std::endl;
        double t0, t1, t2, t3, t4, t5, match_start, solve_start;
        match_time = solve_time = solve_const_H_time = 0;
        t0 = omp_get_wtime();
        /// IMU process, kf prediction, undistortion
        // IMU数据初始化，初始化完成后进入去畸变函数
        // 去畸变函数中，利用IMU数据对kf_状态进行前向传播，同时对点云进行去畸变，得到scan_undistort_（lidar系）
#ifdef USE_IKFOM
        p_imu_->Process(measures_, kf_, scan_undistort_);
        // TODO:12.1 hr 获取kf预测的全局状态（imu）
        state_point_ = kf_.get_x(); // 前向传播后body的状态预测值 body:imu
        pos_lidar_ = state_point_.pos + state_point_.rot * state_point_.offset_T_L_I; // global系下 lidar的位置
#else
        auto undistort_start = std::chrono::high_resolution_clock::now();
        Timer::Evaluate(
                [&, this]() {
                    p_imu_->Process2(LidarMeasures_, state, scan_undistort_);
                },
                "Undistort PointCloud");
        auto undistort_end = std::chrono::high_resolution_clock::now();
        auto undistort_time =
                std::chrono::duration_cast<std::chrono::duration<double >>(undistort_end - undistort_start).count() *
                1000;
        state_propagat = state;
#endif
//        //TODO:重力对齐
//        Eigen::Vector3d g_b = state_point_.grav;
////                      cout << "11111111" << g_b << endl;
//        Eigen::Matrix3d R_wb = Tools::g2R(g_b);
////                      R_wb = Eigen::Quaterniond::FromTwoVectors(g_b.normalized(), g_w).toRotationMatrix();
//        double yaw = Tools::R2ypr(R_wb).x();
//        R_wb = Tools::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R_wb;
//        Eigen::Vector3d g_alig = R_wb * g_b;
//        state_point_.grav = S2(g_alig);
////                    std::cout << "g_aligment: " << g_alig[0]  << ","<< g_alig[1] << "," << g_alig[2] << std::endl;
//
//        state_point_.vel = R_wb * state_point_.vel;
//        state_point_.pos = R_wb * state_point_.pos;
//        state_point_.rot = R_wb * state_point_.rot;
//
//        std::cout << "g0     " << g_alig.transpose() << std::endl;
////        std::cout << "my R0  "  << Tools::R2ypr(R_wb).transpose() << std::endl;

        if (lidar_selector->debug) {
            LidarMeasures_.debug_show();
        }

        if (scan_undistort_->empty() || (scan_undistort_ == nullptr)) {
            LOG(WARNING) << "No point, skip this scan!";
            if (!fast_lio_is_ready) {
                first_lidar_time_ = LidarMeasures_.lidar_beg_time_;
                p_imu_->first_lidar_time = first_lidar_time_;
                LidarMeasures_.measures.clear();
                printf("FAST_LIO not ready\n");
                return;
            }
        } else {
            int size = scan_undistort_->points.size();
        }
        fast_lio_is_ready = true;

        // the first scan
        if (flg_first_scan_) {
#ifndef USE_VOXEL_OCTREE
            ivox_->AddPoints(scan_undistort_->points);
#endif
            first_lidar_time_ = LidarMeasures_.lidar_beg_time_;
            flg_first_scan_ = false;
            return;
        }
        // 检查当前lidar数据时间，与最早lidar数据时间是否足够
        flg_EKF_inited_ = (LidarMeasures_.lidar_beg_time_ - first_lidar_time_) >= options::INIT_TIME; // 0.1

#ifdef USE_VOXEL_OCTREE
        // TODO:2023.02.09 初始化体素-八叉树表征地图
        if (flg_EKF_inited_ && !init_map) {
            pcl::PointCloud<pcl::PointXYZINormal>::Ptr world_lidar_(new pcl::PointCloud<pcl::PointXYZINormal>);
            Eigen::Quaterniond q(state.rot_end);
            transformLidar(state, p_imu_, scan_undistort_, world_lidar_);
            std::vector<pointWithCov> pv_list;
            /* 计算每个点的协方差 */
            calcPointcov(world_lidar_, scan_undistort_, pv_list);
            buildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size, max_points_size, max_points_size,
                          min_eigen_value, voxel_map);
            scanIdx++;
            if (publish_voxel_map) {
                pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
            }
            init_map = true;
            return;
        }
#endif

        /* TODO:12.14 视觉部分迭代优化 */
        auto t_visual_start = std::chrono::high_resolution_clock::now();
        Timer::Evaluate(
                [&, this]() {
                    if (!LidarMeasures_.is_lidar_end) {
                        cout << "[ VIO ]: Raw feature num: " << pcl_wait_pub_->points.size() << "." << endl;
                        if (first_lidar_time_ < 10) return; // 累积足够多的雷达点
                        if (img_en) {
                            euler_cur_ = RotMtoEuler(state.rot_end);
                            fout_pre << setw(20) << LidarMeasures_.last_update_time_ - first_lidar_time_ << " "
                                     << euler_cur_.transpose() * 57.3 << " " << state.pos_end.transpose() << " "
                                     << state.vel_end.transpose() << " " << state.bias_g.transpose() << " "
                                     << state.bias_a.transpose() << " " << state.gravity.transpose() << endl;
                            /* visual main */
                            lidar_selector->detect(LidarMeasures_.measures.back().img,
                                                   pcl_wait_pub_); // 传入 最新的图像，世界坐标系点云
                            int size_sub = lidar_selector->sub_map_cur_frame_.size();
                            sub_map_cur_frame_point->clear();
                            for (int i = 0; i < size_sub; i++) {
                                PointType temp_map;
                                temp_map.x = lidar_selector->sub_map_cur_frame_[i]->pos_[0];
                                temp_map.y = lidar_selector->sub_map_cur_frame_[i]->pos_[1];
                                temp_map.z = lidar_selector->sub_map_cur_frame_[i]->pos_[2];
                                temp_map.intensity = 0.;
                                sub_map_cur_frame_point->push_back(temp_map); // save sub_map_cur_frame_points
                            }
                            cv::Mat img_rgb = lidar_selector->img_cp;
                            cv_bridge::CvImage out_msg;
                            out_msg.header.stamp = ros::Time::now();
                            // out_msg.header.frame_id = "camera_init";
                            out_msg.encoding = sensor_msgs::image_encodings::BGR8;
                            out_msg.image = img_rgb;
                            img_pub_.publish(out_msg.toImageMsg());

                            publish_frame_world_rgb(pubLaserCloudFullResRgb, lidar_selector); // 发布带有rgb信息的点云信息
                            publish_visual_world_sub_map(pubSubVisualCloud); // 发布sub_map_cur_frame_point

                            geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur_(0), euler_cur_(1),
                                                                              euler_cur_(2));
                            PublishOdometry(pub_odom_aft_mapped_);
                            euler_cur_ = RotMtoEuler(state.rot_end);
                            fout_out << setw(20) << LidarMeasures_.last_update_time_ - first_lidar_time_ << " "
                                     << euler_cur_.transpose() * 57.3 << " " << state.pos_end.transpose() << " "
                                     << state.vel_end.transpose() \
 << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() << " " << state.gravity.transpose() << " "
                                     << scan_undistort_->points.size() << endl;
                        }
                        return;
                    }
                },
                "visual fusion with imu");
        auto t_visual_end = std::chrono::high_resolution_clock::now();
        auto t_visual =
                std::chrono::duration_cast<std::chrono::duration<double>>(t_visual_end - t_visual_start).count() * 1000;

        /// downsample the feature points in a scan
        auto t_downsample_start = std::chrono::high_resolution_clock::now();
        Timer::Evaluate(
                [&, this]() {
                    voxel_scan_.setInputCloud(scan_undistort_);
                    voxel_scan_.filter(*scan_down_body_); // 这个时候的点云还是在激光雷达坐标系
                },
                "Downsample PointCloud");
        auto t_downsample_end = std::chrono::high_resolution_clock::now();
        auto t_downsample = std::chrono::duration_cast<std::chrono::duration<double>>(
                t_downsample_end - t_downsample_start).count() * 1000;
        int cur_pts = scan_down_body_->points.size();
        cout << "[ LIO ]: Raw feature num: " << scan_undistort_->points.size() << " downsamp num " << cur_pts << "."
             << endl;
        // TODO: 按曲率排序点云
        sort(scan_down_body_->points.begin(), scan_down_body_->points.end(), time_list);

        if (cur_pts < 5) {
            LOG(WARNING) << "Too few points, skip this scan!" << scan_undistort_->points.size() << ", "
                         << scan_down_body_->points.size();
            return;
        }
#ifndef USE_VOXEL_OCTREE
        scan_down_world_->resize(cur_pts);
        nearest_points_.resize(cur_pts);
        pointSearchInd_surf_.resize(cur_pts);
        residuals_.resize(cur_pts, 0);
        point_selected_surf_.resize(cur_pts, true);
        plane_coef_.resize(cur_pts, V4F::Zero());
        normvec->resize(cur_pts);
#endif

        t1 = omp_get_wtime();
        if (lidar_en) {
            // TODO：12.1 hr 打印状态信息
            Eigen::Vector3d ext_cur_ = RotMtoEuler(state.rot_end);
            fout_pre << std::setw(20) << LidarMeasures_.last_update_time_ - first_lidar_time_ << " " << "!"
                     << euler_cur_.transpose() << "!" << state.pos_end.transpose() << " "
                     << state.vel_end.transpose() << " " << " " << state.bias_g.transpose() << " "
                     << state.bias_a.transpose() << " ! " << state.gravity.transpose() << std::endl;

        }

        int rematch_num = 0;
        bool nearest_search_en = true;
        t2 = omp_get_wtime();
        double t_update_start_ = omp_get_wtime();
#ifdef USE_IKFOM
        // ICP and iterated Kalman filter update
        Timer::Evaluate(
                [&, this]() {

                    // iterated state estimation
                    double solve_H_time = 0;
                    // update the observation model, will call nn and point-to-plane residual computation
                    // 这里内部会执行LaserMapping::ObsModel函数，因为在初始化的时候通过lambda传进kf_了
                    // 迭代误差状态的EKF更新
                    // 求解得到K， P
                    kf_.update_iterated_dyn_share_modified(options::LASER_POINT_COV,
                                                           solve_H_time); // (0.001, solve_H_time)
                    // save the state
                    state_point_ = kf_.get_x();

                    euler_cur_ = SO3ToEuler(state_point_.rot); // state_point_.rot: imu to world
                    std::cout << "euler:" << euler_cur_.transpose() << std::endl;

                    // TODO：12.1 hr 打印状态信息
                    ext_euler = SO3ToEuler(state_point_.offset_R_L_I);
                    fout_out << std::setw(20) << measures_.lidar_bag_time_ - first_lidar_time_ << " " << "!"
                             << euler_cur_.transpose() << "!"
                             << state_point_.pos.transpose() << " " << ext_euler.transpose() << " "
                             << state_point_.offset_T_L_I.transpose() << " " << state_point_.vel.transpose() \
 << " " << state_point_.bg.transpose() << " " << state_point_.ba.transpose() << " ! " << state_point_.grav << std::endl;
                },
                "IEKF Solve and Update");
#else

        double calc_point_cov_time;
        std::vector<M3D> body_var; // 存点的cov
        std::vector<M3D> crossmat_list; // 存协方差矩阵

        Timer::Evaluate(
                [&, this]() {
                    if (lidar_en) {
                        /** lidar IEKF **/
#ifdef USE_VOXEL_OCTREE
                        /* TODO:0214 加入计算雷达点的协方差的过程 */
                        auto calc_point_cov_start = std::chrono::high_resolution_clock::now();
                        Timer::Evaluate(
                                [&, this]() {
                                    for (auto &point: scan_down_body_->points) {
                                        V3D point_this(point.x, point.y, point.z);
                                        if (point_this[2] == 0)
                                            point_this[2] = 0.001;

                                        M3D point_crossmat;
                                        point_this = lidar_R_wrt_IMU * point_this + lidar_T_wrt_IMU;
                                        M3D cov;
                                        calcBodyCov(point_this, ranging_cov, angle_cov, cov);
                                        point_crossmat << SKEW_SYM_MATRIX(point_this);
                                        crossmat_list.push_back(point_crossmat);
                                        body_var.push_back(cov);
                                    }
                                },
                                "calculate points' cov");
                        auto calc_point_cov_end = std::chrono::high_resolution_clock::now();
                        calc_point_cov_time = std::chrono::duration_cast<std::chrono::duration<double >>(
                                calc_point_cov_end - calc_point_cov_start).count() * 1000; // 成员函数count(),用来表示这一段时间的长度
#endif
                        for (iterCount = -1; iterCount < options::NUM_MAX_ITERATIONS && flg_EKF_inited_; iterCount++) {
                            cout << "------------------" << "第" << iterCount << "次迭代：" << endl;
                            match_start = omp_get_wtime();
                            PointCloudType().swap(*laserCloudOri);
                            PointCloudType().swap(*corr_normvect);
                            total_residual_ = 0.0;

                            /** closest surface search and residual computation **/
#ifdef USE_VOXEL_OCTREE
                            std::vector<double> r_list;
                            std::vector<pTpl> ptpl_list;
                            vector<pointWithCov> pv_list;
                            std::vector<M3D> var_list;
                            pcl::PointCloud<pcl::PointXYZINormal>::Ptr world_lidar_(
                                    new pcl::PointCloud<pcl::PointXYZINormal>);
                            transformLidar(state, p_imu_, scan_down_body_, world_lidar_);
                            /** LiDAR match based on 3 sigma criterion **/
                            /* 更新pv_list(pv),var_list(cov) */
                            for (size_t i = 0; i < scan_down_body_->points.size(); ++i) {
                                pointWithCov pv;
                                pv.point << scan_down_body_->points[i].x, scan_down_body_->points[i].y,
                                        scan_down_body_->points[i].z;
                                pv.point_world << world_lidar_->points[i].x, world_lidar_->points[i].y,
                                        world_lidar_->points[i].z;
                                M3D cov = body_var[i];
                                M3D point_crossmat = crossmat_list[i];
                                M3D rot_var = state.cov.block<3, 3>(0, 0);
                                M3D t_var = state.cov.block<3, 3>(3, 3);
                                cov = state.rot_end * cov * state.rot_end.transpose() +
                                      state.rot_end * (-point_crossmat) * rot_var * (-point_crossmat.transpose()) *
                                      state.rot_end.transpose() + t_var; // 公式3
                                pv.cov = cov;
                                pv_list.push_back(pv);
                                var_list.push_back(cov);
                            }
                            /* point-to-plane匹配 */
                            auto scan_match_time_start = std::chrono::high_resolution_clock::now();
                            Timer::Evaluate(
                                    [&, this]() {
                                        std::vector<V3D> non_match_list;
                                        BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list,
                                                             ptpl_list,
                                                             non_match_list);
                                    },
                                    "Scan Match");
                            auto scan_match_time_end = std::chrono::high_resolution_clock::now();

                            effect_feat_num_ = 0;
                            for (int i = 0; i < ptpl_list.size(); ++i) {
                                PointType pi_body;
                                PointType pi_world;
                                PointType pl;
                                pi_body.x = ptpl_list[i].point(0);
                                pi_body.y = ptpl_list[i].point(1);
                                pi_body.z = ptpl_list[i].point(2);
                                PointBodyToWorld(&pi_body, &pi_world);
                                // pl: 平面法向量
                                pl.x = ptpl_list[i].normal(0);
                                pl.y = ptpl_list[i].normal(1);
                                pl.z = ptpl_list[i].normal(2);
                                effect_feat_num_++;
                                float dis = (pi_world.x * pl.x + pi_world.y * pl.y + pi_world.z * pl.z +
                                             ptpl_list[i].d); // 公式9
                                pl.intensity = dis;
                                laserCloudOri->push_back(pi_body);
                                corr_normvect->push_back(pl);
                                total_residual_ += fabs(dis);
                            }
                            res_mean_last = total_residual_ / effect_feat_num_;
                            match_time += std::chrono::duration_cast<std::chrono::duration<double >>(
                                    scan_match_time_end - scan_match_time_start).count() * 1000;
                            cout << "[ Matching ]: Time:" << std::chrono::duration_cast<std::chrono::duration<double>>(
                                    scan_match_time_end - scan_match_time_start).count() * 1000
                                 << " ms  Effective feature num: " << effect_feat_num_
                                 << " All num:" << scan_down_body_->size() << "  res_mean_last "
                                 << res_mean_last << endl;
#else
#ifdef MP_EN
                            omp_set_num_threads(MP_PROC_NUM);
#pragma omp parallel for
#endif
                            for (int i = 0; i < cur_pts; i++) {
                                PointType &point_body = scan_down_body_->points[i];
                                PointType &point_world = scan_down_world_->points[i];
                                V3D p_body(point_body.x, point_body.y, point_body.z);
                                /* transform to world frame */
                                PointBodyToWorld(&point_body, &point_world); // to world frame
                                point_world.intensity = point_body.intensity;
                                vector<float> pointSearchSqDis(options::NUM_MATCH_POINTS); // #define 5
                                auto &points_near = nearest_points_[i];
//                    auto &points_near = pointSearchInd_surf[i];
                                VF(4) pabcd;
                                uint8_t search_flag = 0;
                                double search_start = omp_get_wtime();
                                if (nearest_search_en) {
                                    /** Find the closest surfaces in the map **/
                                    // 使用ivox_获取地图坐标系中与point_world最近的5个点（N=options::NUM_MATCH_POINTS 5）
                                    ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                                    // 设置标志位，表明是否找到足够的接近点(3个)
                                    point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                                    // 判断近邻点是否形成平面（找到最近的点>=5个，说明可以形成平面），平面法向量存储于plane_coef[i](pabcd)
                                    if (point_selected_surf_[i]) {
                                        // 平面拟合并且获取对应的单位平面法向量plane_coef[i](4*1)
//                            point_selected_surf_[i] =
//                                    esti_plane(plane_coef_[i], points_near, options::ESTI_PLANE_THRESHOLD);
                                        point_selected_surf_[i] =
                                                esti_plane(pabcd, points_near, options::ESTI_PLANE_THRESHOLD);
                                        // ESTI_PLANE_THRESHOLD:0.1
                                    }
                                }
                                if (!point_selected_surf_[i] || points_near.size() < options::NUM_MATCH_POINTS)
                                    continue;
                                // 如果近邻点可以形成平面，则计算point-plane距离残差 esti_plane(pabcd, points_near, 0.1f)
                                if (point_selected_surf_[i]) {
//                        auto temp = point_world.getVector4fMap();
//                        temp[3] = 1.0;
//                        float pd2 = plane_coef_[i].dot(temp); // 点到面距离 (|AB|*n)/|n|
                                    float pd2 = pabcd(0) * point_world.x + pabcd(1) * point_world.y +
                                                pabcd(2) * point_world.z +
                                                pabcd(3);
                                    bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                                    if (valid_corr) {
                                        point_selected_surf_[i] = true;
                                        normvec->points[i].x = pabcd(0); // hr: save normvec
                                        normvec->points[i].y = pabcd(1);
                                        normvec->points[i].z = pabcd(2);
                                        normvec->points[i].intensity = pd2;
                                        residuals_[i] = pd2; // 更新残差
                                    }
                                }
                            }
                            effect_feat_num_ = 0;
//                corr_pts_.resize(cur_pts); // 存点坐标及对应的残差
//                corr_norm_.resize(cur_pts); // 存平面单位法向量
                            laserCloudOri->resize(cur_pts);
                            corr_normvect->reserve(cur_pts);
                            for (int i = 0; i < cur_pts; i++) {
                                // todo 新增残差 <= 2.0约束
                                if (point_selected_surf_[i] && (residuals_[i] <= 2.0)) {
                                    laserCloudOri->points[effect_feat_num_] = scan_down_body_->points[i];
                                    corr_normvect->points[effect_feat_num_] = normvec->points[i];
//                        corr_norm_[effect_feat_num_] = plane_coef_[i];
//                        corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
//                        corr_pts_[effect_feat_num_][3] = residuals_[i];
                                    total_residual_ += residuals_[i];
                                    effect_feat_num_++; // 记录有效点（特征点）的数量（Effective Points）
                                }
                            }
//                corr_pts_.resize(effect_feat_num_);
//                corr_norm_.resize(effect_feat_num_);

                            res_mean_last = total_residual_ / effect_feat_num_;
                            if (effect_feat_num_ < 1) {
                                LOG(WARNING) << "No Effective Points!";
                                continue; // todo :return? or continue or break
                            }
                            // debug:
                            cout << "[ mapping ]: Effective feature num: " << effect_feat_num_ << " res_mean_last "
                                 << res_mean_last
                                 << endl;
//                printf("[ LIO ]: time: fov_check and readd: %0.6f match: %0.6f solve: %0.6f  "
//                       "ICP: %0.6f  map incre: %0.6f total: %0.6f icp: %0.6f construct H: %0.6f.\n",
//                       t1 - t0, aver_time_match, aver_time_solve, t3 - t2, t5 - t3, aver_time_consu,
//                       aver_time_icp, aver_time_const_H_time);
                            match_time += omp_get_wtime() - match_start; // 匹配结束
#endif

//                            solve_start = omp_get_wtime(); // 迭代求解开始
                            auto solve_start = std::chrono::high_resolution_clock::now();
                            /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
                            // TODO: 0215更新IESKF迭代求解过程 在迭代过程中引入平面和点的不确定性
                            MatrixXd Hsub(effect_feat_num_, 6); // note: 其他列为0
                            VectorXd meas_vec(effect_feat_num_);
#ifdef USE_VOXEL_OCTREE
                            VectorXd R_inv(effect_feat_num_);
                            MatrixXd Hsub_T_R_inv(6, effect_feat_num_);
#endif
                            // 取状态中的外参
                            const M3F off_R = state.rot_end.cast<float>();
                            const V3F off_t = state.pos_end.cast<float>();
                            for (int i = 0; i < effect_feat_num_; i++) {
                                const PointType &laser_p = laserCloudOri->points[i];
                                V3D point_this(laser_p.x, laser_p.y, laser_p.z);
#ifdef USE_VOXEL_OCTREE
//                                point_this += lidar_T_wrt_IMU; // NOTE：配置参数中雷达和imu的相对旋转为单位阵
                                point_this = lidar_R_wrt_IMU * point_this + lidar_T_wrt_IMU;
                                M3D cov;
                                if (calib_laser) {
                                    calcBodyCov(point_this, ranging_cov, CALIB_ANGLE_COV, cov);
                                } else {
                                    calcBodyCov(point_this, ranging_cov, angle_cov, cov);
                                }
                                cov = state.rot_end * cov * state.rot_end.transpose();
                                M3D point_crossmat;
                                point_crossmat << SKEW_SYM_MATRIX(point_this);
                                const PointType &norm_p = corr_normvect->points[i];
                                V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);
                                V3D point_world = state.rot_end * point_this + state.pos_end;
                                Eigen::Matrix<double, 1, 6> J_nq;
                                J_nq.block<1, 3>(0, 0) = point_world - ptpl_list[i].center; // 论文中公式(13)
                                J_nq.block<1, 3>(0, 3) = -ptpl_list[i].normal;
                                double sigma_l = J_nq * ptpl_list[i].plane_cov * J_nq.transpose(); // 公式(11) 噪声部分
                                R_inv(i) = 1.0 / (sigma_l + norm_vec.transpose() * cov * norm_vec);
                                /*** calculate the Measuremnt Jacobian matrix H ***/
                                V3D A(point_crossmat * state.rot_end.transpose() * norm_vec); // 公式(50)
                                Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;
                                Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i),
                                        A[2] * R_inv(i), norm_p.x * R_inv(i), norm_p.y * R_inv(i),
                                        norm_p.z * R_inv(i); // TODO:新公式 引入平面概率模型后的Hsub_T

#else
                                point_this += lidar_T_wrt_IMU; // NOTE：配置参数中雷达和imu的相对旋转为单位阵
                                M3D point_crossmat;
                                point_crossmat << SKEW_SYM_MATRIX(point_this); // 斜对称矩阵

                                /*** get the normal vector of closest surface/corner ***/
                                const PointType &norm_p = corr_normvect->points[i];
                                V3D norm_vec(norm_p.x, norm_p.y, norm_p.z);

                                /*** calculate the Measuremnt Jacobian matrix H ***/
                                V3D A(point_crossmat * state.rot_end.transpose() * norm_vec); // 公式(50)
                                Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z; // H矩阵 公式(50)
#endif
                                /*** Measuremnt: distance to the closest surface/corner ***/
                                meas_vec(i) = -norm_p.intensity;
                            }
                            // TODO: 更新时间
//                            solve_const_H_time += omp_get_wtime() - solve_start;

                            MatrixXd K(DIM_STATE, effect_feat_num_);

                            EKF_stop_flg_ = false;
                            flg_EKF_converged_ = false;

                            /*** Iterative Kalman Filter Update ***/
                            if (!flg_EKF_inited_) {
                                cout << "||||||||||Initiallizing LiDar||||||||||" << endl;
                                /*** only run in initialization period ***/
                                MatrixXd H_init(MD(9, DIM_STATE)::Zero());
                                MatrixXd z_init(VD(9)::Zero());
                                H_init.block<3, 3>(0, 0) = M3D::Identity();
                                H_init.block<3, 3>(3, 3) = M3D::Identity();
                                H_init.block<3, 3>(6, 15) = M3D::Identity();
                                z_init.block<3, 1>(0, 0) = -Log(state.rot_end);
                                z_init.block<3, 1>(0, 0) = -state.pos_end;
                                auto H_init_T = H_init.transpose();
                                auto &&K_init = state.cov * H_init_T * (H_init * state.cov * H_init_T +
                                                                        0.0001 *
                                                                        MD(9, 9)::Identity()).inverse(); // 公式(54)初始化
                                solution = K_init * z_init;
                                state.resetpose(); // R：单位阵；p,v:Zero3d
                                EKF_stop_flg_ = true;
                            } else {
                                auto &&Hsub_T = Hsub.transpose();
#ifndef USE_VOXEL_OCTREE
                                auto &&HTz = Hsub_T * meas_vec;
                                H_T_H.block<6, 6>(0, 0) = Hsub_T * Hsub;
                                MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state.cov /
                                                                           LASER_POINT_COV).inverse()).inverse(); // 新卡尔曼增益公式前半部分
                                G.block<DIM_STATE, 6>(0, 0) = K_1.block<DIM_STATE, 6>(0, 0) * H_T_H.block<6, 6>(0, 0);
                                auto vec = state_propagat - state; // error
                                solution = K_1.block<DIM_STATE, 6>(0, 0) * HTz + vec -
                                           G.block<DIM_STATE, 6>(0, 0) * vec.block<6, 1>(0, 0); // 公式(65)
#else
                                H_T_H.block<6, 6>(0, 0) = Hsub_T_R_inv * Hsub;
                                // TODO: / LASER_POINT_COV
                                MD(DIM_STATE, DIM_STATE) &&K_1 = (H_T_H + (state.cov /
                                                                           LASER_POINT_COV).inverse()).inverse(); // 新卡尔曼增益公式前半部分
                                K = K_1.block<DIM_STATE, 6>(0, 0) * Hsub_T_R_inv;
                                auto vec = state_propagat - state; // error
                                solution = K * meas_vec + vec - K * Hsub * vec.block<6, 1>(0, 0);
#endif
                                state += solution;
                                rot_add = solution.block<3, 1>(0, 0);
                                t_add = solution.block<3, 1>(3, 0);

                                if ((rot_add.norm() * 57.3 < 0.01) &&
                                    (t_add.norm() * 100 < 0.015)) { // TODO: threshold EKF收敛
                                    flg_EKF_converged_ = true;
                                }

                                deltaR = rot_add.norm() * 57.3; // 弧度 to 角度
                                deltaT = t_add.norm() * 100; // m to cm
                            }
                            euler_cur_ = RotMtoEuler(state.rot_end);

                            /*** Rematch Judgement ***/
                            nearest_search_en = false;
                            if (flg_EKF_converged_ ||
                                ((rematch_num == 0) && (iterCount == (options::NUM_MAX_ITERATIONS - 2)))) {
                                nearest_search_en = true;
                                rematch_num++;
                            }
                            /*** Convergence Judgements and Covariance Update ***/
                            if (!EKF_stop_flg_ &&
                                (rematch_num >= 2 || (iterCount == options::NUM_MAX_ITERATIONS - 1))) {
                                if (flg_EKF_inited_) {
                                    /*** Covariance Update ***/
#ifdef USE_VOXEL_OCTREE
                                    G.setZero();
                                    G.block<DIM_STATE, 6>(0, 0) = K * Hsub;
#endif
                                    state.cov = (I_STATE - G) * state.cov;
                                    total_distance_ += (state.pos_end - position_last_).norm();
                                    position_last_ = state.pos_end;
                                    geoQuat = tf::createQuaternionMsgFromRollPitchYaw
                                            (euler_cur_(0), euler_cur_(1), euler_cur_(2));

                                    VD(DIM_STATE) K_sum = K.rowwise().sum(); // Matrix<double, ((18)), 1>
                                    VD(DIM_STATE) P_diag = state.cov.diagonal();
                                    // cout<<"K: "<<K_sum.transpose()<<endl;
                                    // cout<<"P: "<<P_diag.transpose()<<endl;
                                    // cout<<"position: "<<state.pos_end.transpose()<<" total distance: "<<total_distance<<endl;
                                }
                                EKF_stop_flg_ = true;
                            }
                            // TODO: 更新时间
//                            solve_time += omp_get_wtime() - solve_start;
                            auto solve_end = std::chrono::high_resolution_clock::now();
                            solve_time += std::chrono::duration_cast<std::chrono::duration<double>>(
                                    solve_end - solve_start)
                                                  .count() *
                                          1000;

                            if (EKF_stop_flg_) break;
                        }
                    }
                },
                "lidar IEKF");
#endif

        // TODO:保存tum格式的轨迹数据
//        SaveTrajTUM(LidarMeasures_.lidar_beg_time_, state.rot_end, state.pos_end);
        /******* Publish odometry *******/
        if (pub_odom_aft_mapped_) {
            euler_cur_ = RotMtoEuler(state.rot_end);
            geoQuat = tf::createQuaternionMsgFromRollPitchYaw(euler_cur_(0), euler_cur_(1), euler_cur_(2));
            PublishOdometry(pub_odom_aft_mapped_);
        }
#ifndef USE_VOXEL_OCTREE
        /* update local map */
        t3 = omp_get_wtime();
        Timer::Evaluate([&, this]() { MapIncremental(); }, "    Incremental Mapping");
        t5 = omp_get_wtime();
        LOG(INFO) << "[ mapping ]: In num: " << scan_undistort_->points.size() << " downsamp " << cur_pts
                  << " Map grid num: " << ivox_->NumValidGrids() << " effect num : " << effect_feat_num_;
#else
        /*** add the  points to the voxel map ***/
        auto map_incremental_start = std::chrono::high_resolution_clock::now();
        Timer::Evaluate([&, this]() {
            MapIncremental(crossmat_list, body_var);
        }, "Incremental Mapping");

        auto map_incremental_end = std::chrono::high_resolution_clock::now();
        map_incremental_time = std::chrono::duration_cast<std::chrono::duration<double>>(
                map_incremental_end - map_incremental_start).count() * 1000;
#endif

        total_time =
                t_downsample + match_time + solve_time + map_incremental_time + undistort_time + calc_point_cov_time +
                t_visual;

        /******* Publish points *******/
        PointCloudType::Ptr laserCloudFullRes(dense_map_en ? scan_undistort_ : scan_down_world_);
        int size = laserCloudFullRes->points.size();
        PointCloudType::Ptr laserCloudWorld(new PointCloudType(size, 1));

        for (int i = 0; i < size; i++) {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i], &laserCloudWorld->points[i]);
        }
        *pcl_wait_pub_ = *laserCloudWorld;

        if (scan_pub_en_ || pcd_save_en_) {
            PublishFrameWorld();
        }
//        if (scan_pub_en_ && scan_body_pub_en_) {
//            PublishFrameBody(pub_laser_cloud_body_);
//        }
        if (scan_pub_en_ && scan_effect_pub_en_) {
            PublishFrameEffectWorld(pub_laser_cloud_effect_world_);
        }
        // publish or save map pcd
        if (path_pub_en_ || path_save_en_) {
            PublishPath(pub_path_);
        }
#ifdef USE_VOXEL_OCTREE
        if (publish_voxel_map) {
            pubVoxelMap(voxel_map, publish_max_voxel_layer, voxel_map_pub);
        }
#endif
        // publish_visual_world_map(pubVisualCloud);
        // publish_map(pubLaserCloudMap);
        // Debug variables
        frame_num_++;
#ifndef USE_VOXEL_OCTREE
        aver_time_consu = aver_time_consu * (frame_num_ - 1) / frame_num_ + (t5 - t0) / frame_num_;
        aver_time_icp = aver_time_icp * (frame_num_ - 1) / frame_num_ + (t_update_end - t_update_start_) / frame_num_;
        aver_time_match = aver_time_match * (frame_num_ - 1) / frame_num_ + (match_time) / frame_num_;
#ifdef USE_IKFOM
        aver_time_solve = aver_time_solve * (frame_num - 1)/frame_num + (solve_time + solve_H_time)/frame_num;
        aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1)/frame_num + solve_time / frame_num;
#else
        aver_time_solve = aver_time_solve * (frame_num_ - 1) / frame_num_ + (solve_time) / frame_num_;
        aver_time_const_H_time =
                aver_time_const_H_time * (frame_num_ - 1) / frame_num_ + solve_const_H_time / frame_num_;
        //cout << "construct H:" << aver_time_const_H_time << std::endl;
#endif
        printf("[ mapping ]: time: fov_check and readd: %0.6f match: %0.6f solve: %0.6f  ICP: %0.6f  map incre: %0.6f total: %0.6f icp: %0.6f construct H: %0.6f \n",
               t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu,
               aver_time_icp, aver_time_const_H_time);
#else
        // TODO: 0217新增记录点数以及各部分时间
        mean_raw_points =
                mean_raw_points * (frame_num_ - 1) / frame_num_ + (double) (scan_undistort_->size()) / frame_num_;
        mean_ds_points =
                mean_ds_points * (frame_num_ - 1) / frame_num_ + (double) (scan_down_body_->size()) / frame_num_;
        mean_effect_points =
                mean_effect_points * (frame_num_ - 1) / frame_num_ + (double) effect_feat_num_ / frame_num_;

        undistort_time_mean = undistort_time_mean * (frame_num_ - 1) / frame_num_ + (undistort_time) / frame_num_;
        down_sample_time_mean = down_sample_time_mean * (frame_num_ - 1) / frame_num_ + (t_downsample) / frame_num_;
        calc_cov_time_mean = calc_cov_time_mean * (frame_num_ - 1) / frame_num_ + (calc_point_cov_time) / frame_num_;
        scan_match_time_mean = scan_match_time_mean * (frame_num_ - 1) / frame_num_ + (match_time) / frame_num_;
        ekf_solve_time_mean = ekf_solve_time_mean * (frame_num_ - 1) / frame_num_ + (solve_time) / frame_num_;
        map_update_time_mean =
                map_update_time_mean * (frame_num_ - 1) / frame_num_ + (map_incremental_time) / frame_num_;
        aver_time_consu = aver_time_consu * (frame_num_ - 1) / frame_num_ + (total_time) / frame_num_;

//        time_log_counter++;
        printf("[ mapping ]: time: average undistort: %0.6f average down sample: %0.6f average calc cov: %0.6f  average scan match: %0.6f  average solve: %0.6f average map incremental: %0.6f average total %0.6f \n",
               undistort_time_mean, down_sample_time_mean, calc_cov_time_mean, scan_match_time_mean,
               ekf_solve_time_mean, map_update_time_mean, aver_time_consu);
#endif
        scanIdx++;


        if (lidar_en) {
            euler_cur_ = RotMtoEuler(state.rot_end);
#ifdef USE_IKFOM
            fout_out << setw(20) << LidarMeasures.last_update_time - first_lidar_time << " " << euler_cur.transpose()*57.3 << " " << state_point.pos.transpose() << " " << state_point.vel.transpose() \
            <<" "<<state_point.bg.transpose()<<" "<<state_point.ba.transpose()<<" "<<state_point.grav<<" "<<feats_undistort->points.size()<<endl;
#else
            fout_out << setw(20) << LidarMeasures_.last_update_time_ - first_lidar_time_ << " "
                     << euler_cur_.transpose() * 57.3 << " " << state.pos_end.transpose() << " "
                     << state.vel_end.transpose() \
 << " " << state.bias_g.transpose() << " " << state.bias_a.transpose() << " " << state.gravity.transpose() << " "
                     << scan_undistort_->points.size() << endl;
#endif
        }
        // dump_lio_state_to_log(fp);
    }

    /**
    * @brief 回调函数：处理标准pcl点云
    * @param msg(sensor_msgs::PointCloud2)
    * @return
    */
    void LaserMapping::StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg) {
        mtx_buffer_.lock();
        Timer::Evaluate(
                [&, this]() {
                    scan_count_++;
                    if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                        LOG(ERROR) << "lidar loop back, clear buffer";
                        lidar_buffer_.clear();
                    }

                    PointCloudType::Ptr ptr(new PointCloudType());
                    preprocess_->Process(msg, ptr);
                    lidar_buffer_.push_back(ptr);
                    time_buffer_.push_back(msg->header.stamp.toSec());
                    last_timestamp_lidar_ = msg->header.stamp.toSec();
                },
                "Preprocess (Standard)");
        mtx_buffer_.unlock();
    }

    /**
    * @brief 回调函数：处理livox自定义点云
    * @param msg(ivox_ros_driver::CustomMsg)
    * @return
    */
    void LaserMapping::LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
        mtx_buffer_.lock();
        // 调用func函数并且返回所用时间
        Timer::Evaluate(
                [&, this]() {
                    scan_count_++;
                    if (msg->header.stamp.toSec() < last_timestamp_lidar_) {
                        LOG(WARNING) << "lidar loop back, clear buffer";
                        lidar_buffer_.clear();
                    }
                    // if time_sync_en = false && last_timestamp_imu_ - last_timestamp_lidar_ > 10.0 && imu_buffer_不空 && lidar_buffer不空
//                    // TODO:补充手动给imu和lidar时间差的处理
                    if (!time_sync_en_ && abs(last_timestamp_imu_ - last_timestamp_lidar_) > 10.0 &&
                        !imu_buffer_.empty() && !lidar_buffer_.empty()) {
                        LOG(INFO) << "IMU and LiDAR not Synced, IMU time: " << last_timestamp_imu_
                                  << ", lidar header time: " << last_timestamp_lidar_;
                    }
//
//                    if (time_sync_en_ && !timediff_set_flg_ && abs(last_timestamp_lidar_ - last_timestamp_imu_) > 1 &&
//                        !imu_buffer_.empty()) {
//                        timediff_set_flg_ = true;
//                        timediff_lidar_wrt_imu_ = last_timestamp_lidar_ + 0.1 - last_timestamp_imu_;
//                        LOG(INFO) << "Self sync IMU and LiDAR, time diff is " << timediff_lidar_wrt_imu_;
//                    }

                    printf("[ INFO ]: get point cloud at time: %.6f.\n", msg->header.stamp.toSec());
                    PointCloudType::Ptr ptr(new PointCloudType());
                    preprocess_->Process(msg, ptr); // 对点云进行预处理（过滤无效点）
                    lidar_buffer_.emplace_back(ptr); // 将预处理后的雷达测量和时间戳存入缓存
                    time_buffer_.emplace_back(last_timestamp_lidar_);
                    last_timestamp_lidar_ = msg->header.stamp.toSec();
                },
                "Preprocess (Livox)");

        mtx_buffer_.unlock();
        sig_buffer_.notify_all();
    }

    /**
    * @brief 回调函数：处理imu数据
    * @param msg_in(sensor_msgs::Imu)
    * @return
    */
    void LaserMapping::IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in) {
        publish_count_++;
        sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

//        // todo: 12.13 fast-livo关闭在线时间同步
//        if (abs(timediff_lidar_wrt_imu_) > 0.1 && time_sync_en_) {
//            msg->header.stamp = ros::Time().fromSec(
//                    timediff_lidar_wrt_imu_ + msg_in->header.stamp.toSec()); // imu时间统一转换为lidar时间
//        }

        double timestamp = msg->header.stamp.toSec();

        mtx_buffer_.lock();
        if (timestamp < last_timestamp_imu_) {
            LOG(WARNING) << "imu loop back, clear buffer";
            imu_buffer_.clear();
            flg_reset_ = true;
        }

        last_timestamp_imu_ = timestamp;
        imu_buffer_.emplace_back(msg);
//        cout << "got imu: " << timestamp << " imu size " << imu_buffer_.size() << endl;
        mtx_buffer_.unlock();
        sig_buffer_.notify_all();
    }

    /**
    * @brief 回调函数：处理img数据
    * @param msg_in(sensor_msgs::ImageConstPtr)
    * @return
    */
    void LaserMapping::IMGCallBack(const sensor_msgs::ImageConstPtr &msg) {
        if (first_img_time < 0 && time_buffer_.size() > 0) {
            first_img_time = msg->header.stamp.toSec() - time_buffer_.front();
        }
        if (!img_en) return;
        printf("[ INFO ]: get img at time: %.6f.\n", msg->header.stamp.toSec());
        if (msg->header.stamp.toSec() < last_timestamp_img_) {
            ROS_ERROR("img loop back, clear buffer");
            img_buffer_.clear();
            img_time_buffer_.clear();
        }
        mtx_buffer_.lock();
        img_buffer_.push_back(getImageFromMsg(msg)); // cv::Mat
        img_time_buffer_.push_back(msg->header.stamp.toSec());
        last_timestamp_img_ = msg->header.stamp.toSec();
        // cv::imshow("img", img);
        // cv::waitKey(1);
        // cout<<"last_timestamp_img:::"<<last_timestamp_img<<endl;
        mtx_buffer_.unlock();
        sig_buffer_.notify_all();
    }

    bool LaserMapping::SyncPackages(LidarMeasureGroup &meas) {
        // todo:12.9 新增img_buffer_是否为空判定
        if (lidar_buffer_.empty() && img_buffer_.empty()) { // has lidar topic or img topic?
            return false;
        }
//        ROS_INFO("In sync");
        if (meas.is_lidar_end) { // If meas.is_lidar_end==true, means it just after scan end, clear all buffer in meas.
            meas.measures.clear(); // imu and img measurements
            meas.is_lidar_end = false;
        }
        /*** push a lidar scan ***/
        // 这里的lidar_end_time_很重要，由于fast_lio中的主要贡献之一是back-propagation
        // 最后的结果是将一帧点云的所有点都校正到这一帧扫描结束的位置，该位置对应的时间就是lidar_end_time_
        if (!lidar_pushed_) { // If not in lidar scan, need to generate new meas
            if (lidar_buffer_.empty()) return false;
            meas.lidar_ = lidar_buffer_.front();

            // 计算扫描结束时间lidar_end_time_
            if (meas.lidar_->points.size() <= 1) {
                LOG(WARNING) << "Too few input point cloud!";
                mtx_buffer_.lock();
                if (img_buffer_.size() > 0) { // temp method, ignore img topic when no lidar points, keep sync
                    lidar_buffer_.pop_front();
                    img_buffer_.pop_front();
                }
                mtx_buffer_.unlock();
                sig_buffer_.notify_all();
                // ROS_ERROR("out sync");
                return false;
            }
            // sort by sample timestamp; small to big
            sort(meas.lidar_->points.begin(), meas.lidar_->points.end(), time_list);
            meas.lidar_beg_time_ = time_buffer_.front(); // generate lidar_beg_time 雷达开始时间
            scan_num_++;
            lidar_end_time_ = meas.lidar_beg_time_ +
                              meas.lidar_->points.back().curvature / double(1000); // calc lidar scan end time 雷达扫描结束时间
            lidar_pushed_ = true;
        }
        if (img_buffer_.empty()) { // no img topic, means only has lidar topic
            // imu message needs to be larger than lidar_end_time, keep complete propagate.
            /*** push imu_ data, and pop from imu_ buffer ***/
            if (last_timestamp_imu_ < lidar_end_time_ + 0.02) return false;
            struct MeasureGroup m; // standard method to keep imu message.
            double imu_time = imu_buffer_.front()->header.stamp.toSec();
            m.imu_.clear();
            mtx_buffer_.lock();
            // hr: make sure m.imu_end_time > lidar_end_time
            while ((!imu_buffer_.empty()) && (imu_time < lidar_end_time_)) {
                imu_time = imu_buffer_.front()->header.stamp.toSec();
                if (imu_time > lidar_end_time_) break;
                m.imu_.push_back(imu_buffer_.front());
                imu_buffer_.pop_front();
            }
            lidar_buffer_.pop_front();
            time_buffer_.pop_front();
            mtx_buffer_.unlock();
            sig_buffer_.notify_all();
            lidar_pushed_ = false; // sync one whole lidar scan.
            meas.is_lidar_end = true; // process lidar topic, so timestamp should be lidar scan end.
            meas.measures.push_back(m);
            return true;
        }
        struct MeasureGroup m;
        // has img topic, but img topic timestamp larger than lidar end time, process lidar topic.
        if (img_time_buffer_.front() > lidar_end_time_) {
            if (last_timestamp_imu_ < lidar_end_time_ + 0.02) return false;
            double imu_time = imu_buffer_.front()->header.stamp.toSec();
            m.imu_.clear();
            mtx_buffer_.lock();
            while ((!imu_buffer_.empty() && (imu_time < lidar_end_time_))) {
                imu_time = imu_buffer_.front()->header.stamp.toSec();
                if (imu_time > lidar_end_time_) break;
                m.imu_.push_back(imu_buffer_.front());
                imu_buffer_.pop_front();
            }
            lidar_buffer_.pop_front();
            time_buffer_.pop_front();
            mtx_buffer_.unlock();
            sig_buffer_.notify_all();
            lidar_pushed_ = false; // sync one whole lidar scan.
            meas.is_lidar_end = true; // process lidar topic, so timestamp should be lidar scan end.
            meas.measures.push_back(m);
        } else { // img topic timestamp smaller than lidar end time <=
            // process img topic, record timestamp
            double img_start_time = img_time_buffer_.front();
            if (last_timestamp_imu_ < img_start_time) return false;
            double imu_time = imu_buffer_.front()->header.stamp.toSec();
            m.imu_.clear();
            m.img_offset_time = img_start_time -
                                meas.lidar_beg_time_; // record img offset time, it shoule be the Kalman update timestamp.
            m.img = img_buffer_.front();
            mtx_buffer_.lock();
            // hr: make sure m.imu_end_time > img_start_time
            while ((!imu_buffer_.empty() && (imu_time < img_start_time))) {
                imu_time = imu_buffer_.front()->header.stamp.toSec();
                if (imu_time > img_start_time) break;
                m.imu_.push_back(imu_buffer_.front());
                imu_buffer_.pop_front();
            }
            img_buffer_.pop_front();
            img_time_buffer_.pop_front();
            mtx_buffer_.unlock();
            sig_buffer_.notify_all();
            meas.is_lidar_end = false; // has img topic in lidar scan, so flag "is_lidar_end=false"
            meas.measures.push_back(m);
        }
        return true;
    }

//    void LaserMapping::PrintState(const state_ikfom &s) {
//        LOG(INFO) << "state r: " << s.rot.coeffs().transpose() << ", t: " << s.pos.transpose()
//                  << ", off r: " << s.offset_R_L_I.coeffs().transpose() << ", t: " << s.offset_T_L_I.transpose();
//    }

    void LaserMapping::MapIncremental() {
        PointVector points_to_add;
        PointVector point_no_need_downsample;

        int cur_pts = scan_down_body_->size();
        points_to_add.reserve(cur_pts);
        point_no_need_downsample.reserve(cur_pts);

        std::vector<size_t> index(cur_pts);
        for (size_t i = 0; i < cur_pts; ++i) {
            index[i] = i;
        }

        // 点云层面进行多线程操作
        std::for_each(std::execution::unseq, index.begin(), index.end(), [&](const size_t &i) {
            /* transform to world frame */
            PointBodyToWorld(&(scan_down_body_->points[i]), &(scan_down_world_->points[i]));

            /* decide if need add to map */
            PointType &point_world = scan_down_world_->points[i];
            // 之前进行了最近邻搜索，如果该点可以找到近邻点，需要进行降采样测量，避免单个体素内有过多点
            if (!nearest_points_[i].empty() && flg_EKF_inited_) {
                const PointVector &points_near = nearest_points_[i];

                // filter_size_map: 0.5，该点对应的体素中心坐标
                // 计算该点所属体素的key值，也就是该体素的中心值
                Eigen::Vector3f center =
                        ((point_world.getVector3fMap() / filter_size_map_min_).array().floor() + 0.5) *
                        filter_size_map_min_;

                Eigen::Vector3f dis_2_center = points_near[0].getVector3fMap() - center; // 第一个近邻点(离当前点最近的一个点)到体素中心坐标的距离

                // To avoid too many points accumulating in one voxel, we leave out
                // unnecessary point insertions via a VoxelGrid-like filter in the
                // same way as FastLIO2. Since we have already computed the
                // voxel indices of the nearest neighbors, we will not insert the
                // current point if any of its neighbors is closer to the center
                // of the voxel grid than itself.
                // 如果近邻点比该点更接近对应的体素中心，则不再插入该点

                // 如果它的近邻点离体素中心比较远，则将该点加入到point_no_need_downsample
                // 意思是当前点将被加入到体素地图中
                if (fabs(dis_2_center.x()) > 0.5 * filter_size_map_min_ &&
                    fabs(dis_2_center.y()) > 0.5 * filter_size_map_min_ &&
                    fabs(dis_2_center.z()) > 0.5 * filter_size_map_min_) {
                    point_no_need_downsample.emplace_back(point_world);
                    return;
                }

                bool need_add = true;
                float dist = calc_dist(point_world.getVector3fMap(), center); // 计算当前点与体素中心的欧氏距离
                if (points_near.size() >= options::NUM_MATCH_POINTS) {
                    // 遍历5个近邻点，检查一下是否有近邻点比当前点更接近体素中心，如果有的话，则不插入该点
                    for (int readd_i = 0; readd_i < options::NUM_MATCH_POINTS; readd_i++) {
                        if (calc_dist(points_near[readd_i].getVector3fMap(), center) < dist + 1e-6) {
                            need_add = false;
                            break;
                        }
                    }
                }
                // 如果没有近邻点比当前点更接近体素中心，则添加该点
                // 或者近邻点的个数不足5个，说明地图中该区域的点云比较稀疏，需要将该点插入到地图
                if (need_add) {
                    points_to_add.emplace_back(point_world);
                }
            } else { // 当前点找不到近邻点，则需要开辟新的体素，直接插入该点，不需要降采样了
                points_to_add.emplace_back(point_world);
            }
        });

        Timer::Evaluate(
                [&, this]() {
                    ivox_->AddPoints(points_to_add);
                    ivox_->AddPoints(point_no_need_downsample);
                },
                "    IVox Add Points");
    }

    void LaserMapping::MapIncremental(std::vector<M3D> crossmat_list, std::vector<M3D> body_var) {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZINormal>);
        transformLidar(state, p_imu_, scan_down_body_, world_lidar);
        std::vector<pointWithCov> pv_list;
//#ifdef MP_EN
//        omp_set_num_threads(MP_PROC_NUM); // 4
//#pragma omp parallel
//#endif
//#pragma omp for ordered
        for (size_t i = 0; i < world_lidar->size(); ++i) {
            pointWithCov pv;
//#pragma omp ordered
            pv.point << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
            M3D point_crossmat = crossmat_list[i];
            M3D cov = body_var[i];
            cov = state.rot_end * cov * state.rot_end.transpose() +
                  state.rot_end * (-point_crossmat) * state.cov.block<3, 3>(0, 0) * (-point_crossmat.transpose()) *
                  state.rot_end.transpose() + state.cov.block<3, 3>(3, 3); // 公式3
            pv.cov = cov;
            pv_list.push_back(pv);
        }
        // var_contrast:return (x.cov.diagonal().norm() < y.cov.diagonal().norm());
        std::sort(pv_list.begin(), pv_list.end(), var_contrast);
        Timer::Evaluate(
                [&, this]() {
                    updateVoxelMap(pv_list, max_voxel_size, max_layer, layer_size, max_points_size, max_points_size,
                                   min_eigen_value, voxel_map);
                },
                "update Voxel Map");
    }

/**
 * Lidar point cloud registration
 * will be called by the eskf custom observation model
 * compute point-to-plane residual here
 * @param s kf state
 * @param ekfom_data H matrix
 */
#ifdef USE_IKFOM
    void LaserMapping::ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
        int cnt_pts = scan_down_body_->size(); // scan_down_body_：down sampled scan in body(lidar)

        // 配置索引值,用于后面的for_each并行搜索
        std::vector<size_t> index(cnt_pts);
        for (size_t i = 0; i < index.size(); ++i) {
            index[i] = i;
        }

        // 计算残差
        Timer::Evaluate(
                [&, this]() {
                    // s.offset_R_L_I: 激光雷达坐标系到IMU坐标系的旋转变换
                    // s.rot: 经过IMU前向传播的姿态，IMU坐标系到世界坐标系的旋转变换
                    // 这里得到了将点从激光雷达坐标系转换到世界坐标系的R_wl和t_wl
                    auto R_wl = (s.rot * s.offset_R_L_I).cast<float>();
                    auto t_wl = (s.rot * s.offset_T_L_I + s.pos).cast<float>();

                    /** closest surface search and residual computation **/
                    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                        PointType &point_body = scan_down_body_->points[i]; // 取引用
                        PointType &point_world = scan_down_world_->points[i];

                        /* transform to world frame */
                        common::V3F p_body = point_body.getVector3fMap(); // Eigen::Vector3f x,y,z
                        point_world.getVector3fMap() = R_wl * p_body + t_wl; // to world frame
                        point_world.intensity = point_body.intensity;

                        auto &points_near = nearest_points_[i];
                        if (ekfom_data.converge) {
                            /** Find the closest surfaces in the map **/
                            // 使用ivox_获取地图坐标系中与point_world最近的5个点（N=options::NUM_MATCH_POINTS 5）
                            ivox_->GetClosestPoint(point_world, points_near, options::NUM_MATCH_POINTS);
                            // 设置标志位，表明是否找到足够的接近点(3个)
                            point_selected_surf_[i] = points_near.size() >= options::MIN_NUM_MATCH_POINTS;
                            // 判断近邻点是否形成平面（找到最近的点>=5个，说明可以形成平面），平面法向量存储于plane_coef[i]
                            if (point_selected_surf_[i]) {
                                // 平面拟合并且获取对应的单位平面法向量plane_coef[i](4*1)
                                point_selected_surf_[i] =
                                        common::esti_plane(plane_coef_[i], points_near, options::ESTI_PLANE_THRESHOLD);
                                // ESTI_PLANE_THRESHOLD:0.1
                            }
                        }

                        // 如果近邻点可以形成平面，则计算point-plane距离残差
                        if (point_selected_surf_[i]) {
                            auto temp = point_world.getVector4fMap();
                            temp[3] = 1.0;
                            float pd2 = plane_coef_[i].dot(temp); // 点到面距离 (|AB|*n)/|n|

                            bool valid_corr = p_body.norm() > 81 * pd2 * pd2;
                            if (valid_corr) {
                                point_selected_surf_[i] = true;
                                residuals_[i] = pd2; // 更新残差
                            }
                        }
                    });
                },
                "    ObsModel (Lidar Match)");

        effect_feat_num_ = 0;

        corr_pts_.resize(cnt_pts); // 存点坐标及对应的残差
        corr_norm_.resize(cnt_pts); // 存平面单位法向量
        for (int i = 0; i < cnt_pts; i++) {
            if (point_selected_surf_[i]) {
                corr_norm_[effect_feat_num_] = plane_coef_[i];
                corr_pts_[effect_feat_num_] = scan_down_body_->points[i].getVector4fMap();
                corr_pts_[effect_feat_num_][3] = residuals_[i];

                effect_feat_num_++; // 记录有效点（特征点）的数量（Effective Points）
            }
        }
        corr_pts_.resize(effect_feat_num_);
        corr_norm_.resize(effect_feat_num_);

        if (effect_feat_num_ < 1) {
            ekfom_data.valid = false;
            LOG(WARNING) << "No Effective Points!";
            return;
        }

        // 计算雅可比
        Timer::Evaluate(
                [&, this]() {
                    /*** Computation of Measurement Jacobian matrix H and measurements vector ***/
                    // 初始化H矩阵，行数: 特征点数  列数:12
                    ekfom_data.h_x = Eigen::MatrixXd::Zero(effect_feat_num_, 12);  // 23
                    ekfom_data.h.resize(effect_feat_num_);

                    index.resize(effect_feat_num_);
                    // 取状态中的外参
                    const common::M3F off_R = s.offset_R_L_I.toRotationMatrix().cast<float>();
                    const common::V3F off_t = s.offset_T_L_I.cast<float>();
                    const common::M3F Rt = s.rot.toRotationMatrix().transpose().cast<float>();

                    // 对应论文中公式14、12、13
                    std::for_each(std::execution::par_unseq, index.begin(), index.end(), [&](const size_t &i) {
                        common::V3F point_this_be = corr_pts_[i].head<3>(); // 提取前3个元素，即点云坐标
                        common::M3F
                        point_be_crossmat = SKEW_SYM_MATRIX(point_this_be); // vector转成反对称矩阵，显然后面的求导是使用右扰动模型
                        common::V3F point_this = off_R * point_this_be + off_t; // imu系下的点
                        common::M3F
                        point_crossmat = SKEW_SYM_MATRIX(point_this); // 转反对称矩阵

                        /*** get the normal vector of closest surface/corner ***/
                        common::V3F norm_vec = corr_norm_[i].head<3>(); // 平面特征的法向量

                        /*** calculate the Measurement Jacobian matrix H ***/
                        common::V3F C(Rt * norm_vec);
                        // A = [Pa]^ * R^{T} * u
                        // A^{T} = -u^{T}*R*[Pa]^
                        common::V3F A(point_crossmat *C);

                        if (extrinsic_est_en_) {
                            common::V3F
                            B(point_be_crossmat * off_R.transpose() * C);
                            ekfom_data.h_x.block<1, 12>(i, 0)
                                    << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], B[0],
                                    B[1], B[2], C[0], C[1], C[2];
                        } else {
                            // 这是对误差状态求导, 推导过程参见: R2LIVE 论文附录 只优化位置position和旋转rotation
                            // 点到平面距离对位置误差状态求导: norm_vec[0], norm_vec[1], norm_vec[2]
                            // 点到平面距离对姿态误差状态求导: A[0], A[1], A[2]
                            // 因为要取列向量，但是下标访问的是行向量，所以求导的时候直接求了A
                            // 实际上，点到平面距离对姿态误差状态求导是A^{T}
                            ekfom_data.h_x.block<1, 12>(i, 0)
                                    << norm_vec[0], norm_vec[1], norm_vec[2], A[0], A[1], A[2], 0.0,
                                    0.0, 0.0, 0.0, 0.0, 0.0;
                        }

                        /*** Measurement: distance to the closest surface/corner ***/
                        // 存储点面距离，即残差值
                        ekfom_data.h(i) = -corr_pts_[i][3];
                    });
                },
                "    ObsModel (IEKF Build Jacobian)");
    }
#endif

/////////////////////////////////////  debug save / show /////////////////////////////////////////////////////

    void LaserMapping::PublishPath(const ros::Publisher pub_path) {
        SetPosestamp(msg_body_pose_);
        msg_body_pose_.header.stamp = ros::Time().fromSec(lidar_end_time_);
        msg_body_pose_.header.frame_id = "camera_init";

        /*** if path is too large, the rvis will crash ***/
        path_.poses.push_back(msg_body_pose_);
        if (run_in_offline_ == false) {
            pub_path.publish(path_);
        }
    }

    void LaserMapping::PublishOdometry(const ros::Publisher &pub_odom_aft_mapped) {
        odom_aft_mapped_.header.frame_id = "camera_init";
        odom_aft_mapped_.child_frame_id = "aft_mapped";
        odom_aft_mapped_.header.stamp = ros::Time::now();  // ros::Time().fromSec(lidar_end_time_);
        SetPosestamp(odom_aft_mapped_.pose);

        static tf::TransformBroadcaster br;
        tf::Transform transform;
        tf::Quaternion q;
        transform.setOrigin(tf::Vector3(state.pos_end(0), state.pos_end(1), state.pos_end(2)));
        q.setW(geoQuat.w);
        q.setX(geoQuat.x);
        q.setY(geoQuat.y);
        q.setZ(geoQuat.z);
        transform.setRotation(q);
        br.sendTransform(tf::StampedTransform(transform, odom_aft_mapped_.header.stamp, "camera_init", "aft_mapped"));

        pub_odom_aft_mapped.publish(odom_aft_mapped_);
    }

    void LaserMapping::publish_odom_euler(const ros::Publisher &pubEuler) {
        odomeuler.header.frame_id = "camera_init";
        odomeuler.header.stamp = ros::Time().fromSec(lidar_end_time_);// ros::Time().fromSec(lidar_end_time);

        SetPosestamp(odom_aft_mapped_.pose);

        Eigen::Quaterniond q;
        q.w() = odom_aft_mapped_.pose.pose.orientation.w;
        q.x() = odom_aft_mapped_.pose.pose.orientation.x;
        q.y() = odom_aft_mapped_.pose.pose.orientation.y;
        q.z() = odom_aft_mapped_.pose.pose.orientation.z;
        Eigen::Matrix3d R1 = q.toRotationMatrix();
        Eigen::Vector3d ea = R1.eulerAngles(2, 1, 0); // Z-Y-X
        odomeuler.yaw = ea[0];
        odomeuler.pitch = ea[1];
        odomeuler.row = ea[2];
        std::cout << "yaw pitch roll = " << ea.transpose() << std::endl;

        pubEuler.publish(odomeuler);
    }

    void LaserMapping::PublishFrameWorld() {
        if (!(run_in_offline_ == false && scan_pub_en_) && !pcd_save_en_) {
            return;
        }
        uint size = pcl_wait_pub_->points.size();

        if (!run_in_offline_ && scan_pub_en_) {
            sensor_msgs::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*pcl_wait_pub_, laserCloudmsg);
            laserCloudmsg.header.stamp = ros::Time::now(); //.fromSec(last_timestamp_lidar);
            laserCloudmsg.header.frame_id = "camera_init";
            pub_laser_cloud_world_.publish(laserCloudmsg);
            publish_count_ -= options::PUBFRAME_PERIOD;
        }
        /**************** save map ****************/
        /* 1. make sure you have enough memories
        /* 2. noted that pcd save will influence the real-time performences **/
        if (pcd_save_en_) {
            *pcl_wait_save_ += *pcl_wait_pub_;

            static int scan_wait_num = 0;
            scan_wait_num++;
            if (pcl_wait_save_->size() > 0 && pcd_save_interval_ > 0 && scan_wait_num >= pcd_save_interval_) {
                pcd_index_++;
                std::string all_points_dir(
                        std::string(std::string(ROOT_DIR) + "PCD/scans_") + std::to_string(pcd_index_) +
                        std::string(".pcd"));
                pcl::PCDWriter pcd_writer;
                LOG(INFO) << "current scan saved to /PCD/" << all_points_dir;
                pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
                pcl_wait_save_->clear();
                scan_wait_num = 0;
            }
        }
    }

    void LaserMapping::PublishFrameBody(const ros::Publisher &pub_laser_cloud_body) {
        int size = scan_undistort_->points.size();
        PointCloudType::Ptr laser_cloud_imu_body(new PointCloudType(size, 1));

        for (int i = 0; i < size; i++) {
            PointBodyLidarToIMU(&scan_undistort_->points[i], &laser_cloud_imu_body->points[i]);
        }

        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laser_cloud_imu_body, laserCloudmsg);
        laserCloudmsg.header.stamp = ros::Time().fromSec(lidar_end_time_);
        laserCloudmsg.header.frame_id = "body";
        pub_laser_cloud_body.publish(laserCloudmsg);
        publish_count_ -= options::PUBFRAME_PERIOD;
    }

    void LaserMapping::PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world) {
        PointCloudType::Ptr laserCloudWorld(new PointCloudType(effect_feat_num_, 1));
        for (int i = 0; i < effect_feat_num_; i++) {
            RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
        }
        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
        laserCloudFullRes3.header.frame_id = "camera_init";
        pubLaserCloudEffect.publish(laserCloudFullRes3);
        publish_count_ -= options::PUBFRAME_PERIOD;
    }

    void LaserMapping::publish_frame_world_rgb(const ros::Publisher &pubLaserCloudFullRes,
                                               lidar_selection::LidarSelectorPtr lidar_selector) {
        uint size = pcl_wait_pub_->points.size();
        PointCloudXYZRGB::Ptr laserCloudWorldRGB(new PointCloudXYZRGB(size, 1));
        if (img_en) {
            laserCloudWorldRGB->clear();
            for (int i = 0; i < size; i++) {
                PointTypeRGB pointRGB;
                pointRGB.x = pcl_wait_pub_->points[i].x;
                pointRGB.y = pcl_wait_pub_->points[i].y;
                pointRGB.z = pcl_wait_pub_->points[i].z;
                V3D p_w(pcl_wait_pub_->points[i].x, pcl_wait_pub_->points[i].y, pcl_wait_pub_->points[i].z);
                V2D pc(lidar_selector->new_frame_->w2c(p_w));
                if (lidar_selector->new_frame_->cam_->isInFrame(pc.cast<int>(), 0)) {
                    // cv::Mat img_cur = lidar_selector->new_frame_->img();
                    cv::Mat img_rgb = lidar_selector->img_rgb;
                    V3F pixel = lidar_selector->getpixel(img_rgb, pc);
                    pointRGB.r = pixel[2]; // rgb信息
                    pointRGB.g = pixel[1];
                    pointRGB.b = pixel[0];
                    laserCloudWorldRGB->push_back(pointRGB);
                }

            }
        }

        if (1)//if(publish_count >= PUBFRAME_PERIOD)
        {
            sensor_msgs::PointCloud2 laserCloudmsg;
            if (img_en) {
                cout << "RGB pointcloud size: " << laserCloudWorldRGB->size() << endl;
                pcl::toROSMsg(*laserCloudWorldRGB, laserCloudmsg);
            } else {
                pcl::toROSMsg(*pcl_wait_pub_, laserCloudmsg);
            }
            laserCloudmsg.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
            laserCloudmsg.header.frame_id = "camera_init";
            pubLaserCloudFullRes.publish(laserCloudmsg);
            publish_count -= options::PUBFRAME_PERIOD; // publish_count以imu的发布为准 PUBFRAME_PERIOD：20
            // pcl_wait_pub->clear();
        }
    }

    void LaserMapping::publish_visual_world_sub_map(const ros::Publisher &pubSubVisualCloud) {
        PointCloudType::Ptr laserCloudFullRes(sub_map_cur_frame_point);
        int size = laserCloudFullRes->points.size();
        if (size == 0) return;

        PointCloudType::Ptr sub_pcl_visual_wait_pub(new PointCloudType());
        *sub_pcl_visual_wait_pub = *laserCloudFullRes;
        if (1)//if(publish_count >= PUBFRAME_PERIOD)
        {
            sensor_msgs::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*sub_pcl_visual_wait_pub, laserCloudmsg);
            laserCloudmsg.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
            laserCloudmsg.header.frame_id = "camera_init";
            pubSubVisualCloud.publish(laserCloudmsg);
            publish_count -= options::PUBFRAME_PERIOD;
            // pcl_wait_pub->clear();
        }
    }

    void LaserMapping::publish_frame_world(const ros::Publisher &pubLaserCloudFullRes,
                                           const int point_skip) {
        PointCloudType::Ptr laserCloudFullRes(dense_map_en ? scan_undistort_ : scan_down_body_);
        int size = laserCloudFullRes->points.size();
        PointCloudType::Ptr laserCloudWorld(new PointCloudType(size, 1));
        for (int i = 0; i < size; i++) {
            RGBpointBodyToWorld(&laserCloudFullRes->points[i],
                                &laserCloudWorld->points[i]);
        }
        PointCloudType::Ptr laserCloudWorldPub(new PointCloudType);
        for (int i = 0; i < size; i += point_skip) {
            laserCloudWorldPub->points.push_back(laserCloudWorld->points[i]);
        }
        sensor_msgs::PointCloud2 laserCloudmsg;
        pcl::toROSMsg(*laserCloudWorldPub, laserCloudmsg);
        laserCloudmsg.header.stamp =
                ros::Time::now(); //.fromSec(last_timestamp_lidar);
        laserCloudmsg.header.frame_id = "camera_init";
        pubLaserCloudFullRes.publish(laserCloudmsg);
    }

    void LaserMapping::publish_effect(const ros::Publisher &pubLaserCloudEffect) {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr effect_cloud_world(
                new pcl::PointCloud<pcl::PointXYZRGB>);
        PointCloudType::Ptr laserCloudWorld(new PointCloudType(effect_feat_num_, 1));
        for (int i = 0; i < effect_feat_num_; i++) {
            RGBpointBodyToWorld(&laserCloudOri->points[i], &laserCloudWorld->points[i]);
            pcl::PointXYZRGB pi;
            pi.x = laserCloudWorld->points[i].x;
            pi.y = laserCloudWorld->points[i].y;
            pi.z = laserCloudWorld->points[i].z;
            float v = laserCloudWorld->points[i].intensity / 100;
            v = 1.0 - v;
            uint8_t r, g, b;
            mapJet(v, 0, 1, r, g, b);
            pi.r = r;
            pi.g = g;
            pi.b = b;
            effect_cloud_world->points.push_back(pi);
        }

        sensor_msgs::PointCloud2 laserCloudFullRes3;
        pcl::toROSMsg(*laserCloudWorld, laserCloudFullRes3);
        laserCloudFullRes3.header.stamp =
                ros::Time::now(); //.fromSec(last_timestamp_lidar);
        laserCloudFullRes3.header.frame_id = "camera_init";
        pubLaserCloudEffect.publish(laserCloudFullRes3);
    }

    /** 保存轨迹到txt,格式：TUM **/
    void LaserMapping::Savetrajectory(const std::string &traj_file) {
        std::ofstream ofs;
        ofs.open(traj_file, std::ios::out);
        if (!ofs.is_open()) {
            LOG(ERROR) << "Failed to open traj_file: " << traj_file;
            return;
        }

        ofs << "#timestamp x y z q_x q_y q_z q_w" << std::endl;
        for (const auto &p: path_.poses) {
            ofs << std::fixed << std::setprecision(6) << p.header.stamp.toSec() << " " << std::setprecision(15)
                << p.pose.position.x << " " << p.pose.position.y << " " << p.pose.position.z << " "
                << p.pose.orientation.x << " " << p.pose.orientation.y << " " << p.pose.orientation.z << " "
                << p.pose.orientation.w << std::endl;
        }

        ofs.close();
    }

///////////////////////////  private method /////////////////////////////////////////////////////////////////////
    template<typename T>
    void LaserMapping::SetPosestamp(T &out) {
#ifdef USE_IKFOM
        out.pose.position.x = state_point_.pos(0);
        out.pose.position.y = state_point_.pos(1);
        out.pose.position.z = state_point_.pos(2);
        out.pose.orientation.x = state_point_.rot.coeffs()[0];
        out.pose.orientation.y = state_point_.rot.coeffs()[1];
        out.pose.orientation.z = state_point_.rot.coeffs()[2];
        out.pose.orientation.w = state_point_.rot.coeffs()[3];
#else
        out.pose.position.x = state.pos_end(0);
        out.pose.position.y = state.pos_end(1);
        out.pose.position.z = state.pos_end(2);
        out.pose.orientation.x = geoQuat.x;
        out.pose.orientation.y = geoQuat.y;
        out.pose.orientation.z = geoQuat.z;
        out.pose.orientation.w = geoQuat.w;
#endif
    }

    void LaserMapping::PointBodyToWorld(const PointType *pi, PointType *const po) {
        V3D p_body(pi->x, pi->y, pi->z);
#ifdef USE_IKFOM
        V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                             state_point_.pos);
#else
        V3D p_global(state.rot_end * (p_body + lidar_T_wrt_IMU) + state.pos_end);
#endif
        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
    }

    void LaserMapping::PointBodyToWorld(const V3F &pi, PointType *const po) {
        V3D p_body(pi.x(), pi.y(), pi.z());
#ifdef USE_IKFOM
        V3D p_global(state_point_.rot * (state_point_.offset_R_L_I * p_body + state_point_.offset_T_L_I) +
                             state_point_.pos);
#else
        V3D p_global(state.rot_end * (p_body + lidar_T_wrt_IMU) + state.pos_end);
#endif

        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = std::abs(po->z);
    }

    // todo:
    void LaserMapping::PointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
        V3D p_body_lidar(pi->x, pi->y, pi->z);
        V3D p_body_imu(p_body_lidar + lidar_T_wrt_IMU);

        po->x = p_body_imu(0);
        po->y = p_body_imu(1);
        po->z = p_body_imu(2);
        po->intensity = pi->intensity;
    }

    void LaserMapping::RGBpointBodyToWorld(const PointType *const pi, PointType *const po) {
        V3D p_body(pi->x, pi->y, pi->z);
#ifdef USE_IKFOM
        //state_ikfom transfer_state = kf.get_x();
        V3D p_global(state_point.rot * (state_point.offset_R_L_I*p_body + state_point.offset_T_L_I) + state_point.pos);
#else
        V3D p_global(state.rot_end * (p_body + lidar_T_wrt_IMU) + state.pos_end); // lidar -> imu -> world
#endif
        po->x = p_global(0);
        po->y = p_global(1);
        po->z = p_global(2);
        po->intensity = pi->intensity;
        float intensity = pi->intensity;
        intensity = intensity - floor(intensity);
        int reflection_map = intensity * 10000;
    }

    void LaserMapping::publish_visual_world_map(const ros::Publisher &pubVisualCloud) {
        PointCloudType::Ptr laserCloudFullRes(sub_map_cur_frame_point);
        int size = laserCloudFullRes->points.size();
        if (size == 0) return;
        // PointCloudXYZI::Ptr laserCloudWorld( new PointCloudXYZI(size, 1));

        // for (int i = 0; i < size; i++)
        // {
        //     RGBpointBodyToWorld(&laserCloudFullRes->points[i], \
    //                         &laserCloudWorld->points[i]);
        // }
        // mtx_buffer_pointcloud.lock();
        PointCloudType::Ptr pcl_visual_wait_pub(new PointCloudType());
        *pcl_visual_wait_pub = *laserCloudFullRes;
        if (1)//if(publish_count >= PUBFRAME_PERIOD)
        {
            sensor_msgs::PointCloud2 laserCloudmsg;
            pcl::toROSMsg(*pcl_visual_wait_pub, laserCloudmsg);
            laserCloudmsg.header.stamp = ros::Time::now();//.fromSec(last_timestamp_lidar);
            laserCloudmsg.header.frame_id = "camera_init";
            pubVisualCloud.publish(laserCloudmsg);
            publish_count -= options::PUBFRAME_PERIOD;
            // pcl_wait_pub->clear();
        }
    }

    void LaserMapping::Finish() {
        /**************** save map ****************/
        /* 1. make sure you have enough memories
        /* 2. pcd save will largely influence the real-time performences **/
        if (pcl_wait_save_->size() > 0 && pcd_save_en_) {
            std::string file_name = std::string("scans.pcd");
            std::string all_points_dir(std::string(std::string(ROOT_DIR) + "PCD/") + file_name);
            pcl::PCDWriter pcd_writer;
            LOG(INFO) << "current scan saved to /PCD/" << file_name;
            pcd_writer.writeBinary(all_points_dir, *pcl_wait_save_);
        }

        LOG(INFO) << "finish done";
    }

// TODO:0403 将计算世界坐标系下点的协方差模块化
    void LaserMapping::calcPointcov(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &world_lidar_, CloudPtr scan_undistort,
                                    std::vector<pointWithCov> pv_list) {
        for (size_t i = 0; i < world_lidar_->size(); ++i) {
            pointWithCov pv;
            pv.point << world_lidar_->points[i].x, world_lidar_->points[i].y, world_lidar_->points[i].z;
            V3D point_this(scan_undistort->points[i].x, scan_undistort->points[i].y,
                           scan_undistort->points[i].z);
            // if z=0, error will occur in calcBodyCov. To be solved
            if (point_this[2] == 0)
                point_this[2] = 0.001;

            // TODO:0301坐标系问题
            point_this = lidar_R_wrt_IMU * point_this + lidar_T_wrt_IMU;
            M3D point_crossmat; // 斜对角矩阵
            point_crossmat << SKEW_SYM_MATRIX(point_this);
            M3D cov;
            calcBodyCov(point_this, ranging_cov, angle_cov, cov);
            // TODO:按照论文中公式（3）修改cov计算
            cov = state.rot_end * cov * state.rot_end.transpose() +
                  state.rot_end * (-point_crossmat) * state.cov.block<3, 3>(0, 0) *
                  (-point_crossmat).transpose() * state.rot_end.transpose() +
                  state.cov.block<3, 3>(3, 3);
            pv.cov = cov;
            pv_list.push_back(pv);
            // todo:sigma_pv没用到
//            Eigen::Vector3d sigma_pv = pv.cov.diagonal();
//            sigma_pv[0] = sqrt(sigma_pv[0]);
//            sigma_pv[1] = sqrt(sigma_pv[1]);
//            sigma_pv[2] = sqrt(sigma_pv[2]);
        }
    }
}  // namespace faster_lio