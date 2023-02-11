//
// Created by hr on 23-2-8.
//

#ifndef VOXEL_MAP_UTIL_HPP
#define VOXEL_MAP_UTIL_HPP

#include "../common_lib.h"
#include "omp.h"
//#include "imu_processing.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <execution>
#include <openssl/md5.h>
#include <pcl/common/io.h>
#include <rosbag/bag.h>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <utility>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#define HASH_P 116101 // large primes
#define MAX_N 10000000000 // the size of hash map

static int plane_id = 0;

// 点到面对匹配结构
typedef struct pTpl {
    Eigen::Vector3d point; // point coordinates
    Eigen::Vector3d normal; // plane normal
    Eigen::Vector3d center; // center of plane
    Eigen::Matrix<double, 6, 6> plane_cov; // cov of plane
//    double d;
    int layer; // layer of octo_tree
} pTpl;

// 带有协方差信息的3D点云
typedef struct pointWithCov {
    Eigen::Vector3d point;
    Eigen::Vector3d point_world;
    Eigen::Matrix3d cov;
} pointWithCov;

// plane
typedef struct Plane {
    Eigen::Vector3d center;
    Eigen::Vector3d normal;
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 6, 6> plane_cov;
    float radius;
    float min_eigen_value = 1;
    float mid_eigen_value = 1;
    float max_eigen_value = 1;
//    float d = 0;
    int points_size = 0;
    bool is_plane = false;
    bool is_init = false;
    // is_update and last_update_points_size are only for publish plane
    bool is_update = false;
    int last_update_points_size = 0;
    bool update_enable = true;
} Plane;

class VOXEL_LOC {
public:
    int64_t x, y, z;

    explicit VOXEL_LOC(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_LOC &other) const {
        return (x == other.x && y == other.y && z == other.z);
    }
//    bool operator<(const VOXEL_LOC &other) const{}
};

// hash value
namespace std {
    template<>
    struct hash<VOXEL_LOC> {
        int64_t operator()(const VOXEL_LOC &s) const {
            using std::hash;
            using std::size_t;
            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:
            //   return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^ (hash<int64_t>()(s.z) << 1);
            return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x); // 空间哈希函数
        }
    };
}

class OctoTree {
public:
    std::vector<pointWithCov> temp_points_; // all points in an octo tree
    std::vector<pointWithCov> new_points_;  // new points in an octo tree
    Plane *plane_ptr_;
    int max_layer_;
//    bool indoor_mode_;
    int layer_;
    int octo_state_; // 0 is end of tree, 1 is not
    OctoTree *leaves_[8];
    double voxel_center_[3]; // x, y, z
    std::vector<int> layer_point_size_; // todo:???
    float quater_length_;
    float planer_threshold_;
    int max_plane_update_threshold_; // 点的数量大于这个阈值时，执行init_plane()
    int update_size_threshold_;
    int all_points_num_;
    int new_points_num_;
    int max_points_size_;
    int max_cov_points_size_;
    bool init_octo_;
    bool update_cov_enable_;
    bool update_enable_;

    OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
             int max_point_size, int max_cov_points_size, float planer_threshold)
            : max_layer_(max_layer), layer_(layer),
              layer_point_size_(std::move(layer_point_size)), max_points_size_(max_point_size),
              max_cov_points_size_(max_cov_points_size),
              planer_threshold_(planer_threshold) {
        temp_points_.clear();
        octo_state_ = 0;
        new_points_num_ = 0;
        all_points_num_ = 0;
        // when new points num > 5, do an update
        update_size_threshold_ = 5;
        init_octo_ = false;
        update_enable_ = true;
        update_cov_enable_ = true;
//        max_plane_update_threshold_ = layer_point_size_[layer_];
        for (auto &leave: leaves_) {
            leave = nullptr;
        }
        plane_ptr_ = new Plane;
    }

    /* check is plane , calc plane parameters including plane covariance */
    void init_plane(const std::vector<pointWithCov> &points, Plane *plane) {
        /* init parameters of plane */
        plane->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
        plane->covariance = Eigen::Matrix3d::Zero();
        plane->center = Eigen::Vector3d::Zero();
        plane->normal = Eigen::Vector3d::Zero();
        plane->points_size = points.size();
        plane->radius = 0;
        for (auto pv: points) {
            plane->covariance += pv.point * pv.point.transpose();
            plane->center += pv.point;
        }
        plane->center = plane->center / plane->points_size;
        // voxel_map: formula(4) A
        plane->covariance = plane->covariance / plane->points_size - plane->center * plane->center.transpose();
        Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance);
        Eigen::Matrix3cd evecs = es.eigenvectors(); // 复矩阵
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real(); // 转换为real类型（MSSQL浮点数据类型，4字节，小数点后7位）
        Eigen::Matrix3f::Index evalsMin, evalsMax;

    }

    void init_octo_tree() {
        if (temp_points_.size() > max_plane_update_threshold_) {
            init_plane(temp_points_, plane_ptr_);
            if (plane_ptr_->is_plane) {
                octo_state_ = 0; // end of tree 不再划分
                if (temp_points_.size() > max_cov_points_size_) {
                    update_cov_enable_ = false;
                }
                if (temp_points_.size() > max_points_size_) {
                    update_enable_ = false;
                }
            } else {
                octo_state_ = 1;
                cut_octo_tree(); //继续细分
            }
            init_octo_ = true;
            new_points_num_ = 0;
        }
    }

    void cut_octo_tree() {

    }

};

void transformLidar(const faster_lio::StatesGroup &state, const shared_ptr<faster_lio::ImuProcess> &p_imu,
                    const PointCloudType::Ptr &input_cloud, pcl::PointCloud<pcl::PointXYZI>::Ptr &trans_cloud) {
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
        PointType p_c = input_cloud->points[i];
        Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
        p = state.rot_end * p + state.pos_end;
        pcl::PointXYZI pi;
        pi.x = p[0];
        pi.y = p[1];
        pi.z = p[2];
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

void calcBodyCov(Eigen::Vector3d &pb, const double range_inc, const double degree_inc, Eigen::Matrix3d &cov) {

}

// TODO:0211 voxel_zie & planner_threshold float -> double
void buildVoxelMap(const std::vector<pointWithCov> &input_points, const double voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size, const int max_points_size, const int max_cov_points_size,
                   const double planer_threshold, std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
    uint pl_size = input_points.size();
    /* save points into Octo_tree */
    for (uint i = 0; i < pl_size; ++i) {
        // TODO:0210 p_v 从const 改为 const&；
        const pointWithCov &p_v = input_points[i];
        double loc_xyz[3];
        for (int j = 0; j < 3; ++j) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0; // todo: 0.1和-0.1取整后会在一个voxel内，故减1
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
        auto iter = feat_map.find(position);
        // 对于一个新增的点，首先计算索引key，查找此 key 是否已经存在,若存在，则向对应体素里新增点；若不存在，则先创建新OctoTree再插入点
        if (iter != feat_map.end()) {
            feat_map[position]->temp_points_.push_back(p_v);
            feat_map[position]->new_points_num_++;
        } else {
            auto *octoTree = new OctoTree(max_layer, 0, layer_point_size, max_points_size, max_cov_points_size,
                                          planer_threshold);
            feat_map[position] = octoTree;
            feat_map[position]->quater_length_ = voxel_size / 4;
            feat_map[position]->voxel_center_[0] =
                    (0.5 + (double) position.x) * voxel_size; // TODO:0210 将int64_t强转为double
            feat_map[position]->voxel_center_[1] =
                    (0.5 + (double) position.y) * voxel_size; // voxel center coordinate in the world frame
            feat_map[position]->voxel_center_[2] = (0.5 + (double) position.z) * voxel_size;
            feat_map[position]->temp_points_.push_back(p_v);
            feat_map[position]->new_points_num_++;
            feat_map[position]->layer_point_size_ = layer_point_size;
        }
    }
    for (auto &iter: feat_map) {
        iter.second->init_octo_tree();
    }
}

#endif //VOXEL_MAP_UTIL_HPP
