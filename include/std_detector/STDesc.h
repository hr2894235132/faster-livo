//
// Created by hr on 23-4-14.
//

#ifndef SRC_STDESC_H
#define SRC_STDESC_H

#include "omp.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <fstream>
#include <mutex>
#include <pcl/common/io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sstream>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define HASH_P 116101
#define MAX_N 10000000000
#define MAX_FRAME_N 20000

std::vector<double> key_frame_times;
/* 参数信息 */
typedef struct ConfigSetting {
    /* for point cloud pre-process*/
    int stop_skip_enable_ = 0;
    double ds_size_ = 0.5; // 累积点云参数，增加点云密度
    int maximum_corner_num_ = 30; // 限制key point提取的数量
    /* for key points*/
    double plane_merge_normal_thre_{}; // 平面合并（平面生长）法向量阈值
    double plane_merge_dis_thre_{}; // 平面合并（平面生长）距离阈值（论文提到，但是代码中没用到）
    double plane_detection_thre_ = 0.01; // 平面判定条件
    double voxel_size_ = 1.0; // 体素尺寸
    int voxel_init_num_ = 10; // 超过这个数量再进行平面判定
    double proj_image_resolution_ = 0.5; // 投影分辨率
    double proj_dis_min_ = 0.2; // 投影距离（点面距离）阈值
    double proj_dis_max_ = 5;
    double corner_thre_ = 10; // 梯度值阈值
    /* for STD */
    int descriptor_near_num_ = 10; // 找最近的num个点去形成描述子
    double descriptor_min_len_ = 1; // 描述子边长范围
    double descriptor_max_len_ = 10; //
    double non_max_suppression_radius_ = 3.0; // k近邻搜索：按范围搜索参数
    double std_side_resolution_ = 0.2; // 边长的尺度分辨率
    /* for place recognition*/
    int skip_near_num_ = 50; // 匹配描述子帧id相差数目
    int candidate_num_ = 50; // 候选帧的数量
    int sub_frame_num_ = 10; // 子帧数目
    double rough_dis_threshold_ = 0.03; // 描述子配对时边长距离阈值(TODO 怎么定的？？？)
    double vertex_diff_threshold_ = 0.7; // 描述子配对时额外信息（intensity）阈值
    double icp_threshold_ = 0.5; // 判断最优匹配时分数阈值
    double normal_threshold_ = 0.1; // 几何验证时法向量方向差阈值
    double dis_threshold_ = 0.3; // 几何验证时距离差阈值
    double historyKeyframeSearchTimeDiff;
} ConfigSetting;

/* 三角描述子（STD）的结构 */
typedef struct STDesc {
    // STDesc的边长，arranged from short to long
    Eigen::Vector3d side_length_;
    // projection angle between vertices
    Eigen::Vector3d angle_;
    // center of STDesc
    Eigen::Vector3d center_;
    // 帧id
    unsigned int frame_id_;
    // 三个顶点
    Eigen::Vector3d vertex_A_;
    Eigen::Vector3d vertex_B_;
    Eigen::Vector3d vertex_C_;
    // 每个顶点带有的其他一些信息，例如，intensity
    Eigen::Vector3d vertex_attached_;
    double std_time;
} STDesc;

/* 特征点提取部分的平面结构 */
typedef struct Plane {
    // 平面中点点云
    pcl::PointXYZINormal p_center_;
    // 平面的中点
    Eigen::Vector3d center_;
    // 平面法向量
    Eigen::Vector3d normal_;
    // 平面协方差
    Eigen::Matrix3d covariance_;
    // 平面中点的个数
    int points_size_ = 0;
    // 平面判断标志位
    bool is_plane_ = false;
    // 无用变量
    float radius_ = 0;
    float min_eigen_value_ = 1;
    float intercept_ = 0;
//    int id_ = 0;
//    int sub_plane_num_ = 0;
} Plane;

/* 描述子匹配对列表 */
typedef struct STDMatchList {
    // 描述子匹配对
    std::vector<std::pair<STDesc, STDesc>> match_list_;
    // 描述子帧id对
    std::pair<int, int> match_id_;
    // no use
//    double mean_dis_;
} STDMatchList;

/* 哈希键值 */
class VOXEL_LOC {
public:
    int64_t x, y, z;

    explicit VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_LOC &other) const {
        return (x == other.x && y == other.y && z == other.z);
    }
};

/* for down sample function */
struct M_POINT {
    float xyz[3]{};
    float intensity{};
    int count = 0;
};

/* Hash value */
template<>
struct std::hash<VOXEL_LOC> {
    int64_t operator()(const VOXEL_LOC &s) const {
        using std::hash;
        using std::size_t;
        return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x); // 空间哈希函数
    }
};

/* 存描述子的哈希表对应的哈希键值 */
class STDesc_LOC {
public:
    int64_t x, y, z, a, b, c;

    explicit STDesc_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0, int64_t va = 0, int64_t vb = 0, int64_t vc = 0)
            : x(vx), y(vy), z(vz), a(va), b(vb), c(vc) {}

    bool operator==(const STDesc_LOC &other) const {
        // use three attributes
        return (x == other.x && y == other.y && z == other.z);
        // use six attributes
        // return (x == other.x && y == other.y && z == other.z && a == other.a &&
        //         b == other.b && c == other.c);
    }
};

/* Hash value */
template<>
struct std::hash<STDesc_LOC> {
    int64_t operator()(const STDesc_LOC &s) const {
        using std::hash;
        using std::size_t;
        return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x); // 空间哈希函数
//        return ((((((((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x)) * HASH_P) % MAX_N + s.a) * HASH_P) %
//                 MAX_N + s.b) * HASH_P) % MAX_N + s.c;
    }
};

/* 平面检测部分构建的八叉树结构 */
class OctoTree {
public:
    ConfigSetting config_setting_;
    std::vector<Eigen::Vector3d> voxel_points_;
    Plane *plane_ptr_;

    int layer_;
    int octo_state_; // 0 is end of tree, 1 is not
    bool is_project_ = false; // 特征点投影标志位
    std::vector<Eigen::Vector3d> proj_normal_vec_;

    // 平面生长部分的变量：check 6 direction: x,y,z,-x,-y,-z
    bool is_check_connect_[6]{};
    bool connect_[6]{};
    OctoTree *connect_tree_[6]{};
    OctoTree *leaves_[8]{};
//    int max_layer_;
//    bool indoor_mode_;
//    int merge_num_ = 0;
//    bool is_publish_ = false;
//    double voxel_center_[3]; // x, y, z
//    std::vector<int> layer_point_size_;
//    float quater_length_;
//    float planer_threshold_;
//    int max_plane_update_threshold_; // 点的数量大于这个阈值时，执行init_plane()
//    int update_size_threshold_;
//    int all_points_num_;
//    int new_points_num_;
//    int max_points_size_;
//    int max_cov_points_size_;
    bool init_octo_;
//    bool update_cov_enable_;
//    bool update_enable_;

    explicit OctoTree(const ConfigSetting &config_setting) : config_setting_(config_setting) {
        voxel_points_.clear();
        octo_state_ = 0;
        layer_ = 0;
        init_octo_ = false;
        for (auto &leave: leaves_) {
            leave = nullptr;
        }
        for (int i = 0; i < 6; i++) {
            is_check_connect_[i] = false;
            connect_[i] = false;
            connect_tree_[i] = nullptr;
        }
        plane_ptr_ = new Plane;
    }

    void init_plane();

    void init_octo_tree();
};

void down_sampling_voxel(pcl::PointCloud<pcl::PointXYZI> &pl_feat, double voxel_size);

void load_pose_with_time(const std::string &pose_file,
                         std::vector<std::pair<Eigen::Vector3d, Eigen::Matrix3d>> &poses_ve,
                         std::vector<double> &times_vec);

void read_parameters(ros::NodeHandle &nh, ConfigSetting &config_setting);

double time_inc(std::chrono::_V2::system_clock::time_point &t_end,
                std::chrono::_V2::system_clock::time_point &t_begin);

pcl::PointXYZI vec2point(const Eigen::Vector3d &vec);

Eigen::Vector3d point2vec(const pcl::PointXYZI &pi);

void publish_std_pairs(const std::vector<std::pair<STDesc, STDesc>> &match_std_pairs,
                       const ros::Publisher &std_publisher);

bool attach_greater_sort(std::pair<double, int> a, std::pair<double, int> b);

/* PlaneSolver结构体，用来自定义ceres的cost function（自定义仿函数） */
struct PlaneSolver {
    // 构造函数，传递用于计算残差的测量值
    PlaneSolver(Eigen::Vector3d curr_point_, Eigen::Vector3d curr_normal_, Eigen::Vector3d target_point_,
                Eigen::Vector3d target_normal_) : curr_point(std::move(curr_point_)),
                                                  curr_normal(std::move(curr_normal_)),
                                                  target_point(std::move(target_point_)),
                                                  target_normal(std::move(target_normal_)) {};

    // 重载()操作符，用于残差计算
    template<typename T>
    bool operator()(const T *q, const T *t, T *residual) const {
        Eigen::Quaternion<T> q_w_curr{q[3], q[0], q[1], q[2]}; // current -> target坐标系下 current -> world坐标系下
        Eigen::Matrix<T, 3, 1> t_w_curr{t[0], t[1], t[2]};
        Eigen::Matrix<T, 3, 1> cp{T(curr_point.x()), T(curr_point.y()), T(curr_point.z())};
        Eigen::Matrix<T, 3, 1> point_w;
        point_w = q_w_curr * cp + t_w_curr;
        Eigen::Matrix<T, 3, 1> point_target{T(target_point.x()), T(target_point.y()), T(target_point.z())};
        Eigen::Matrix<T, 3, 1> norm{T(target_normal.x()), T(target_normal.y()), T(target_normal.z())};
        residual[0] = norm.template dot(point_w - point_target); // 残差（点面距离）
        return true;
    }

    // 工厂模式函数
    static ceres::CostFunction *
    Create(const Eigen::Vector3d &curr_point_, const Eigen::Vector3d &curr_normal_, Eigen::Vector3d target_point_,
           Eigen::Vector3d target_normal_) {
        /* ceres::AutoDiffCostFunction<CostFunctor, int residualDim, int paramDim>(CostFunctor* functor); */
        return (new ceres::AutoDiffCostFunction<PlaneSolver, 1, 4, 3>(
                new PlaneSolver(curr_point_, curr_normal_, std::move(target_point_), std::move(target_normal_))));
    }

    Eigen::Vector3d curr_point;
    Eigen::Vector3d curr_normal;
    Eigen::Vector3d target_point;
    Eigen::Vector3d target_normal;
};

class STDescManager {
public:
    STDescManager() = default;

    ConfigSetting config_setting_;
    unsigned int current_frame_id_{};

    explicit STDescManager(ConfigSetting &config_setting) : config_setting_(config_setting) {
        current_frame_id_ = 0;
    }

    // hash table,保存所有描述子
    std::unordered_map<STDesc_LOC, std::vector<STDesc>> data_base_;
    // 保存所有的 key clouds
    std::vector<pcl::PointCloud<pcl::PointXYZI>::Ptr> key_cloud_vec_;
    // 保存所有的corner points(关键点)
    std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> corner_cloud_vec_;
    // 保存关键帧的所有平面中的点
    std::vector<pcl::PointCloud<pcl::PointXYZINormal>::Ptr> plane_cloud_vec_;
    // 保存关键帧对应的时间
    std::vector<double> key_cloud_times_vec_;
    /* Four main processing functions */

    // generate STDescs from a point cloud
    void GenerateSTDescs(pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud, std::vector<STDesc> &stds_vec);

    // 搜索结果 <candidate_id, plane icp score>. <-1, x> for no loop
    void SearchLoop(const std::vector<STDesc> &stds_vec, std::pair<int, double> &loop_result,
                    std::pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform,
                    std::vector<std::pair<STDesc, STDesc>> &loop_std_pair);

    // 把描述子加到哈希表(database)中
    void AddSTDescs(const std::vector<STDesc> &stds_vec);

    // Geometrical optimization by plane-to-plane icp
    void PlaneGeomrtricIcp(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                           const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                           std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform) const;

private:
    /* Following are sub-processing functions */

    // voxelization and plane detection
    void init_voxel_map(const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                        std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map) const;

    // build connection for planes
    void build_connection(std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) const;

    // acquire planes from voxel_map
    static void getPlane(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                  pcl::PointCloud<pcl::PointXYZINormal>::Ptr &plane_cloud);

    // extract corner points from pre-build voxel map and clouds
    void corner_extractor(std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map,
                          const pcl::PointCloud<pcl::PointXYZI>::Ptr &input_cloud,
                          pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points);

    void extract_corner(const Eigen::Vector3d &proj_center,
                        const Eigen::Vector3d& proj_normal,
                        const std::vector<Eigen::Vector3d>& proj_points,
                        pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points) const;

    // non-maximum suppression, to control the number of corners
    void non_maxi_suppression(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points) const;

    // build STDescs from corner points.
    void build_stdesc(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &corner_points,
                      std::vector<STDesc> &stds_vec) const;

    // Select a specified number of candidate frames according to the number of STDesc rough matches
    void candidate_selector(const std::vector<STDesc> &stds_vec, std::vector<STDMatchList> &candidate_matcher_vec);

    // Get the best candidate frame by geometry check
    void candidate_verify(const STDMatchList &candidate_matcher, double &verify_score,
                          std::pair<Eigen::Vector3d, Eigen::Matrix3d> &relative_pose,
                          std::vector<std::pair<STDesc, STDesc>> &sucess_match_vec);

    // Get the transform between a matched std pair
    static void triangle_solver(std::pair<STDesc, STDesc> &std_pair, Eigen::Vector3d &t, Eigen::Matrix3d &rot);

    // Geometrical verification by plane-to-plane icp threshold
    double plane_geometric_verify(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &source_cloud,
                                  const pcl::PointCloud<pcl::PointXYZINormal>::Ptr &target_cloud,
                                  const std::pair<Eigen::Vector3d, Eigen::Matrix3d> &transform) const;
};

#endif //SRC_STDESC_H
