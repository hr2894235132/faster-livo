#ifndef FASTER_LIO_LASER_MAPPING_H
#define FASTER_LIO_LASER_MAPPING_H

#include <omp.h>
#include <mutex>
#include <math.h>
#include <thread>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <so3_math.h>
#include <Eigen/Core>
#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <pcl/filters/voxel_grid.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <condition_variable>
#include <thread>

#include "imu_processing.hpp"
#include <image_transport/image_transport.h>
#include "ivox3d/ivox3d.h"
#include "options.h"
#include "pointcloud_preprocess.h"
#include <faster_lio/Euler.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vikit/camera_loader.h>
#include "lidar_selection.h"

namespace faster_lio {

    class LaserMapping {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

#ifdef IVOX_NODE_TYPE_PHC
        using IVoxType = IVox<3, IVoxNodeType::PHC, PointType>;
#else
        using IVoxType = IVox<3, IVoxNodeType::DEFAULT, PointType>;
#endif

        LaserMapping();

        ~LaserMapping() {
            scan_down_body_ = nullptr;
            scan_undistort_ = nullptr;
            scan_down_world_ = nullptr;
            LOG(INFO) << "laser mapping deconstruct";
        }

        /// init with ros
        bool InitROS(ros::NodeHandle &nh);

        /// init without ros
        bool InitWithoutROS(const std::string &config_yaml);

        void Run();

        // callbacks of lidar and imu
        void StandardPCLCallBack(const sensor_msgs::PointCloud2::ConstPtr &msg);

        void LivoxPCLCallBack(const livox_ros_driver::CustomMsg::ConstPtr &msg);

        void IMUCallBack(const sensor_msgs::Imu::ConstPtr &msg_in);

        void IMGCallBack(const sensor_msgs::ImageConstPtr &msg);

        // sync lidar with imu ，camera
        bool SyncPackages(LidarMeasureGroup &meas);
#ifdef USE_IKFOM
        /// interface of mtk, customized obseravtion model
        void ObsModel(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data);
#endif

        ////////////////////////////// debug save / show ////////////////////////////////////////////////////////////////
        void PublishPath(const ros::Publisher pub_path);

        void PublishOdometry(const ros::Publisher &pub_odom_aft_mapped);

        void publish_odom_euler(const ros::Publisher &pubEuler);

        void PublishFrameWorld();

        void PublishFrameBody(const ros::Publisher &pub_laser_cloud_body);

        void PublishFrameEffectWorld(const ros::Publisher &pub_laser_cloud_effect_world);

        void publish_frame_world_rgb(const ros::Publisher &pubLaserCloudFullRes, lidar_selection::LidarSelectorPtr lidar_selector);

        void publish_visual_world_sub_map(const ros::Publisher &pubSubVisualCloud);

        void publish_visual_world_map(const ros::Publisher &pubVisualCloud);

        void Savetrajectory(const std::string &traj_file);

        void Finish();

    private:
        template<typename T>
        void SetPosestamp(T &out);

        void PointBodyToWorld(PointType const *pi, PointType *const po);

        void PointBodyToWorld(const V3F &pi, PointType *const po);

        void PointBodyLidarToIMU(PointType const *const pi, PointType *const po);

        void RGBpointBodyToWorld(PointType const *const pi, PointType *const po);

        void MapIncremental();

        void SubAndPubToROS(ros::NodeHandle &nh);

        bool LoadParams(ros::NodeHandle &nh);

        bool LoadParamsFromYAML(const std::string &yaml);

//        void PrintState(const state_ikfom &s);

        void InitLidarStatic();

        void InitLidarSelection();

        // todo:12.8 新增函数 img_msg -> cv::Mat
        cv::Mat getImageFromMsg(const sensor_msgs::ImageConstPtr &img_msg) {
            cv::Mat img;
            img = cv_bridge::toCvShare(img_msg, "bgr8")->image;
            return img;
        }

    private:
        /// modules
        IVoxType::Options ivox_options_;
        std::shared_ptr<IVoxType> ivox_ = nullptr;                    // localmap in ivox
        std::shared_ptr<PointCloudPreprocess> preprocess_ = nullptr;  // point cloud preprocess
        std::shared_ptr<ImuProcess> p_imu_ = nullptr;                 // imu process
        // TODO: 编译器可能无法区分这是一个成员函数声明还是一个成员变量声明，产生歧义。
//    lidar_selection::LidarSelectorPtr lidar_selector(new lidar_selection::LidarSelector(grid_size, new SparseMap));
        lidar_selection::LidarSelectorPtr lidar_selector{new lidar_selection::LidarSelector(grid_size, new SparseMap)};


        /// params
        /// local map related
        float det_range_ = 300.0f;
        double cube_len_ = 0;
        double filter_size_map_min_ = 0, fov_deg = 0;
        bool localmap_initialized_ = false;
        std::vector<double> extrinT_{3, 0.0};  // lidar-imu translation
        std::vector<double> extrinR_{9, 0.0};  // lidar-imu rotation
        std::vector<double> cameraextrinT{3, 0.0}; // lidar-camera translation
        std::vector<double> cameraextrinR{9, 0.0}; // lidar-camer rotation
        std::string map_file_path_;
        bool ncc_en;
        int dense_map_en = 1;
        int img_en = 1;
        int lidar_en = 1;
        int debug = 0;
        bool fast_lio_is_ready = false;
        int grid_size, patch_size;
        int MIN_IMG_COUNT = 0;
        double outlier_threshold, ncc_thre;
        double LASER_POINT_COV, IMG_POINT_COV, cam_fx, cam_fy, cam_cx, cam_cy;
        double gyr_cov_scale = 0, acc_cov_scale = 0;
        double total_residual_;
        double res_mean_last = 0.05;
        double total_distance_ = 0;

        /// point clouds data
        CloudPtr scan_undistort_{new PointCloudType()};   // scan after undistortion
        CloudPtr scan_down_body_{new PointCloudType()};   // downsampled scan in body
        CloudPtr scan_down_world_{new PointCloudType()};  // downsampled scan in world
        CloudPtr pcl_wait_pub_{new PointCloudType()};
        CloudPtr sub_map_cur_frame_point{new PointCloudType()};


        std::vector<PointVector> nearest_points_;         // nearest points of current scan
        VV4F corr_pts_;                           // inlier pts
        VV4F corr_norm_;                          // inlier plane norms
        PointCloudType::Ptr laserCloudOri{new PointCloudType()};
        PointCloudType::Ptr corr_normvect{new PointCloudType()};
        pcl::VoxelGrid<PointType> voxel_scan_;            // voxel filter for current scan
        pcl::VoxelGrid<PointType> downSizeFilterSurf;
        pcl::VoxelGrid<PointType> downSizeFilterMap;
        std::vector<float> residuals_;                    // point-to-plane residuals
        std::vector<bool> point_selected_surf_;           // selected points
        vector<vector<int>> pointSearchInd_surf_;
        VV4F plane_coef_;                         // plane coeffs
        PointCloudType::Ptr normvec{new PointCloudType()};


        /// ros pub and sub stuffs
        ros::Subscriber sub_pcl_;
        ros::Subscriber sub_imu_;
        ros::Subscriber sub_img_;
        ros::Publisher pub_laser_cloud_world_;
        ros::Publisher pub_laser_cloud_body_;
        ros::Publisher pub_laser_cloud_effect_world_;
        ros::Publisher pub_odom_aft_mapped_;
        ros::Publisher pub_path_;
        ros::Publisher pub_Euler_;
        image_transport::Publisher img_pub_;
        ros::Publisher pubLaserCloudFullRes;
        ros::Publisher pubLaserCloudFullResRgb;
        ros::Publisher pubVisualCloud;
        ros::Publisher pubSubVisualCloud;
        ros::Publisher pubLaserCloudEffect;
        ros::Publisher pubLaserCloudMap;

        std::mutex mtx_buffer_;
        condition_variable sig_buffer_;
        std::deque<double> time_buffer_;
        std::deque<PointCloudType::Ptr> lidar_buffer_;
        std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer_;
        std::deque<cv::Mat> img_buffer_;
        std::deque<double> img_time_buffer_;
        nav_msgs::Odometry odom_aft_mapped_;
        faster_lio::Euler odomeuler;

        /// options
        bool time_sync_en_ = false;
        double timediff_lidar_wrt_imu_ = 0.0;
        double last_timestamp_lidar_ = 0;
        double lidar_end_time_ = 0;
        double last_timestamp_imu_ = -1.0;
        double first_lidar_time_ = 0.0;
        bool lidar_pushed_ = false;
        double first_img_time = -1.0;
        double last_timestamp_img_ = -1.0;
        bool flg_reset_;


        /// statistics and flags ///
        int scan_count_ = 0;
        int publish_count_ = 0;
        bool flg_first_scan_ = true;
        bool flg_EKF_inited_ = false;
        bool flg_EKF_converged_, EKF_stop_flg_ = false;
        int pcd_index_ = 0;
        double lidar_mean_scantime_ = 0.0;
        int scan_num_ = 0;
        bool timediff_set_flg_ = false;
        int effect_feat_num_ = 0, frame_num_ = 0;
        int publish_count = 0, iterCount = 0;
        double match_time = 0, solve_time = 0, solve_const_H_time = 0;

        ///////////////////////// EKF inputs and output ///////////////////////////////////////////////////////
        //    common::MeasureGroup measures_;                    // sync IMU and lidar scan
        LidarMeasureGroup LidarMeasures_;           // estimator inputs and output;
#ifdef USE_IKFOM
        esekfom::esekf<state_ikfom, 12, input_ikfom> kf_;  // esekf
        state_ikfom state_point_;                          // ekf current state
        vect3 pos_lidar_;                                  // lidar position after eskf update
#else
        StatesGroup state;
#endif

        V3D lidar_T_wrt_IMU;
        M3D lidar_R_wrt_IMU;
        V3D euler_cur_ = V3D::Zero();      // rotation in euler angles
        V3D position_last_ = V3D::Zero();
        bool extrinsic_est_en_ = true;

        /////////////////////////  debug show / save /////////////////////////////////////////////////////////
        bool run_in_offline_ = false;
        bool path_pub_en_ = true;
        bool scan_pub_en_ = false;
        bool dense_pub_en_ = false;
        bool scan_body_pub_en_ = false;
        bool scan_effect_pub_en_ = false;
        bool pcd_save_en_ = false;
        bool runtime_pos_log_ = true;
        int pcd_save_interval_ = -1;
        bool path_save_en_ = false;
        std::string dataset_;

        PointCloudType::Ptr pcl_wait_save_{new PointCloudType()};  // debug save
        nav_msgs::Path path_;
        geometry_msgs::PoseStamped msg_body_pose_;
        geometry_msgs::Quaternion geoQuat;

        /*** variables definition ***/
#ifndef USE_IKFOM
        VD(DIM_STATE) solution; // 18*1
        MD(DIM_STATE, DIM_STATE) G, H_T_H, I_STATE; // 18*18
        V3D rot_add, t_add;
        StatesGroup state_propagat;
        PointType pointOri, pointSel, coeff;
#endif
        double deltaT, deltaR, aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_solve = 0, aver_time_const_H_time = 0;

    };

}  // namespace faster_lio

#endif  // FASTER_LIO_LASER_MAPPING_H