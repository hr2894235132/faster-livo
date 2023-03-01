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
#include <memory>
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
    double d;
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
    Eigen::Vector3d y_normal;
    Eigen::Vector3d x_normal;
    Eigen::Matrix3d covariance;
    Eigen::Matrix<double, 6, 6> plane_cov;
    float radius = 0;
    float min_eigen_value = 1;
    float mid_eigen_value = 1;
    float max_eigen_value = 1;
    float d = 0;
    int points_size = 0;
    bool is_plane = false;
    bool is_init = false;
    int id;
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
              layer_point_size_(layer_point_size), max_points_size_(max_point_size),
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
        max_plane_update_threshold_ = layer_point_size_[layer_];
        // TODO: 0213 修改waring，使用range循环方式
        for (auto &leave: leaves_) {
            leave = nullptr;
        }
        plane_ptr_ = new Plane;
    }

    /* check is plane , calc plane parameters including plane covariance */
    // TODO: 0213修改waring，将init_plane()改为const成员函数
    void init_plane(const std::vector<pointWithCov> &points, Plane *plane) const {
        /* init parameters of plane */
        plane->plane_cov = Eigen::Matrix<double, 6, 6>::Zero();
        plane->covariance = Eigen::Matrix3d::Zero();
        plane->center = Eigen::Vector3d::Zero();
        plane->normal = Eigen::Vector3d::Zero();
        plane->points_size = points.size();
        plane->radius = 0;
        // TODO: 0213 修改waring，使用range循环方式
        for (auto pv: points) {
            plane->covariance += pv.point * pv.point.transpose();
            plane->center += pv.point;
        }
        plane->center = plane->center / plane->points_size;
        // voxel_map: formula(4) A
        plane->covariance = plane->covariance / plane->points_size - plane->center * plane->center.transpose();
        Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance); // the point covariance matrix
        Eigen::Matrix3cd evecs = es.eigenvectors(); // 复矩阵
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real(); // 转换为real类型（MSSQL浮点数据类型，4字节，小数点后7位）
        Eigen::Matrix3f::Index evalsMin, evalsMax; // 最小最大特征值的位置
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        int evalsMid = 3 - evalsMin - evalsMax;
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
        Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
        // plane covariance calculation
        Eigen::Matrix3d J_Q;
        J_Q << 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size; // 公式7
        if (evalsReal(evalsMin) < planer_threshold_) {
            std::vector<int> index(points.size());
            std::vector<Eigen::Matrix<double, 6, 6>> temp_matrix(points.size());
            // TODO: 0213修改range循环格式
            for (const auto &point: points) {
                Eigen::Matrix<double, 6, 3> J;
                Eigen::Matrix3d F;
                for (int m = 0; m < 3; ++m) {
                    if (m != (int) evalsMin) {
                        Eigen::Matrix<double, 1, 3> F_m = (point.point - plane->center).transpose() /
                                                          ((plane->points_size) *
                                                           (evalsReal(evalsMin) - evalsReal(m))) *
                                                          (evecs.real().col(m) *
                                                           evecs.real().col(evalsMin).transpose() +
                                                           evecs.real().col(evalsMin) *
                                                           evecs.real().col(m).transpose()); // 公式7
                        F.row(m) = F_m;
                    } else {
                        Eigen::Matrix<double, 1, 3> F_m;
                        F_m << 0, 0, 0; // 公式7
                        F.row(m) = F_m;
                    }
                }
                J.block<3, 3>(0, 0) = evecs.real() * F; // 公式7
                J.block<3, 3>(3, 0) = J_Q; // 公式7
                plane->plane_cov += J * point.cov * J.transpose(); // 公式8
            }
            // hr: the plane's normal vector is the eigenvector associated with the minimum eigenvalue of A(plane->covariance)
            plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
            plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
            plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
            // TODO: 0213修改waring,强转为float
            plane->min_eigen_value = (float) evalsReal(evalsMin);
            plane->mid_eigen_value = (float) evalsReal(evalsMid);
            plane->max_eigen_value = (float) evalsReal(evalsMax);
            plane->radius = sqrt((float) evalsReal(evalsMax));
            plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                         plane->normal(2) * plane->center(2)); // 计算点面匹配残差时的第二部分
            plane->is_plane = true;
            // TODO: 0213修改waring,条件中有重复分支
            if (plane->last_update_points_size == 0 || plane->points_size - plane->last_update_points_size > 100) {
                plane->last_update_points_size = plane->points_size;
                plane->is_update = true;
            }
//            else if (plane->points_size - plane->last_update_points_size > 100) {
//                plane->last_update_points_size = plane->points_size;
//                plane->is_update = true;
//            }

            if (!plane->is_init) {
                plane->id = plane_id;
                plane_id++;
                plane->is_init = true;
            }

        } else {
            if (!plane->is_init) {
                plane->id = plane_id;
                plane_id++;
                plane->is_init = true;
            }
            if (plane->last_update_points_size == 0 || plane->points_size - plane->last_update_points_size > 100) {
                plane->last_update_points_size = plane->points_size;
                plane->is_update = true;
            }
//            if (plane->last_update_points_size == 0) {
//                plane->last_update_points_size = plane->points_size;
//                plane->is_update = true;
//            } else if (plane->points_size - plane->last_update_points_size > 100) {
//                plane->last_update_points_size = plane->points_size;
//                plane->is_update = true;
//            }
            plane->is_plane = false;
            plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin), evecs.real()(2, evalsMin);
            plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid), evecs.real()(2, evalsMid);
            plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax), evecs.real()(2, evalsMax);
            // TODO: 0213修改waring,强转为float
            plane->min_eigen_value = (float) evalsReal(evalsMin);
            plane->mid_eigen_value = (float) evalsReal(evalsMid);
            plane->max_eigen_value = (float) evalsReal(evalsMax);
            plane->radius = sqrt((float) evalsReal(evalsMax));
            plane->d = -(plane->normal(0) * plane->center(0) + plane->normal(1) * plane->center(1) +
                         plane->normal(2) * plane->center(2));
        }
    }

    // only update plane normal, center and radius with new points
    // TODO：并没有论文中所说的判断新旧法向量的过程

    /*  If the new normal vector and the previously converged normal vector continue to appear a relatively
     * large difference, we assume that this area of the map has changed and needs to be reconstructe */
    void update_plane(const std::vector<pointWithCov> &points, Plane *plane) {
        Eigen::Matrix3d old_covariance = plane->covariance;
        Eigen::Vector3d old_center = plane->center;

        Eigen::Matrix3d sum_ppt =
                (plane->covariance + plane->center * plane->center.transpose()) * plane->points_size; // 还原
        Eigen::Vector3d sum_p = plane->center * plane->points_size;
        for (const auto &point: points) {
            Eigen::Vector3d pv = point.point;
            sum_ppt += pv * pv.transpose();
            sum_p += pv;
        }
        plane->points_size = plane->points_size + points.size();
        plane->center = sum_p / plane->points_size;
        plane->covariance = sum_ppt / plane->points_size - plane->center * plane->center.transpose();
        Eigen::EigenSolver<Eigen::Matrix3d> es(plane->covariance); // the point covariance matrix
        Eigen::Matrix3cd evecs = es.eigenvectors(); // 复矩阵
        Eigen::Vector3cd evals = es.eigenvalues();
        Eigen::Vector3d evalsReal;
        evalsReal = evals.real(); // 转换为real类型（MSSQL浮点数据类型，4字节，小数点后7位）
        Eigen::Matrix3f::Index evalsMin, evalsMax; // 最小最大特征值的位置
        evalsReal.rowwise().sum().minCoeff(&evalsMin);
        evalsReal.rowwise().sum().maxCoeff(&evalsMax);
        int evalsMid = 3 - evalsMin - evalsMax;
        Eigen::Vector3d evecMin = evecs.real().col(evalsMin);
        Eigen::Vector3d evecMid = evecs.real().col(evalsMid);
        Eigen::Vector3d evecMax = evecs.real().col(evalsMax);
        if (evalsReal(evalsMin) < planer_threshold_) {
            plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
                    evecs.real()(2, evalsMin);
            plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
                    evecs.real()(2, evalsMid);
            plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
                    evecs.real()(2, evalsMax);
            plane->min_eigen_value = evalsReal(evalsMin);
            plane->mid_eigen_value = evalsReal(evalsMid);
            plane->max_eigen_value = evalsReal(evalsMax);
            plane->radius = sqrt(evalsReal(evalsMax));
            plane->d = -(plane->normal(0) * plane->center(0) +
                         plane->normal(1) * plane->center(1) +
                         plane->normal(2) * plane->center(2));

            plane->is_plane = true;
            plane->is_update = true;
        } else {
            plane->normal << evecs.real()(0, evalsMin), evecs.real()(1, evalsMin),
                    evecs.real()(2, evalsMin);
            plane->y_normal << evecs.real()(0, evalsMid), evecs.real()(1, evalsMid),
                    evecs.real()(2, evalsMid);
            plane->x_normal << evecs.real()(0, evalsMax), evecs.real()(1, evalsMax),
                    evecs.real()(2, evalsMax);
            plane->min_eigen_value = evalsReal(evalsMin);
            plane->mid_eigen_value = evalsReal(evalsMid);
            plane->max_eigen_value = evalsReal(evalsMax);
            plane->radius = sqrt(evalsReal(evalsMax));
            plane->d = -(plane->normal(0) * plane->center(0) +
                         plane->normal(1) * plane->center(1) +
                         plane->normal(2) * plane->center(2));
            plane->is_plane = false;
            plane->is_update = true;
        }
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
        if (layer_ >= max_layer_) {
            octo_state_ = 0;
            return;
        }
        // 细分八个格子，并把点存入相应的格子
        // TODO: 0213 修改waring，使用range循环方式
//        for (size_t i = 0; i < temp_points_.size(); ++i)
        for (auto &temp_point: temp_points_) {
            int xyz[3] = {0, 0, 0};
            if (temp_point.point[0] > voxel_center_[0]) xyz[0] = 1;
            if (temp_point.point[1] > voxel_center_[1]) xyz[1] = 1;
            if (temp_point.point[2] > voxel_center_[2]) xyz[2] = 1;

            int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2]; // 根据坐标得到octree格子的索引值
            // 更新细分后的体素块中心坐标和四分之一格子长度
            if (leaves_[leafnum] == nullptr) {
/*                OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
                        int max_point_size, int max_cov_points_size, float planer_threshold) */
                leaves_[leafnum] = new OctoTree(max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
                                                max_cov_points_size_, planer_threshold_);
                leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2.0 * xyz[0] - 1) * quater_length_;
                leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2.0 * xyz[1] - 1) * quater_length_;
                leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2.0 * xyz[2] - 1) * quater_length_;
                leaves_[leafnum]->quater_length_ = quater_length_ / 2;
            }
            leaves_[leafnum]->temp_points_.push_back(temp_point);
            leaves_[leafnum]->new_points_num_++;
        }
        // TODO: 0213 修改waring，使用range循环方式
        // 判断细分后的每个格子是否需要继续划分
        for (auto &leave: leaves_) {
            if (leave != nullptr) {
                if (leave->temp_points_.size() > leave->max_plane_update_threshold_) {
                    init_plane(leave->temp_points_, leave->plane_ptr_);
                    if (leave->plane_ptr_->is_plane) {
                        leave->octo_state_ = 0;
                    } else {
                        leave->octo_state_ = 1;
                        leave->cut_octo_tree();
                    }
                    leave->init_octo_ = true;
                    leave->new_points_num_ = 0;
                }
            }
        }
    }

    void UpdateOctoTree(const pointWithCov &pv) {
        if (!init_octo_) {
            new_points_num_++;
            all_points_num_++;
            temp_points_.push_back(pv);
            if (temp_points_.size() > max_plane_update_threshold_) init_octo_tree();
        } else {
            if (plane_ptr_->is_plane) {
                if (update_enable_) {
                    new_points_num_++;
                    all_points_num_++;
                    if (update_cov_enable_) {
                        temp_points_.push_back(pv);
                    } else {
                        new_points_.push_back(pv);
                    }
                    if (new_points_num_ > update_size_threshold_) {
                        if (update_cov_enable_) {
                            init_plane(temp_points_, plane_ptr_);
                        }
                        new_points_num_ = 0;
                    }
                    if (all_points_num_ >= max_cov_points_size_) {
                        update_cov_enable_ = false;
                        std::vector<pointWithCov>().swap(temp_points_); // discard all historical points
                    }
                    if (all_points_num_ >= max_points_size_) {
                        update_enable_ = false;
                        plane_ptr_->update_enable = false;
                        std::vector<pointWithCov>().swap(new_points_); // discard all historical points
                    }
                } else {
                    return;
                }
            } else {
                if (layer_ < max_layer_) {
                    if (!temp_points_.empty()) {
                        std::vector<pointWithCov>().swap(temp_points_);
                    }
                    if (!new_points_.empty()) {
                        std::vector<pointWithCov>().swap(new_points_);
                    }
                    int xyz[3] = {0, 0, 0};
                    if (pv.point[0] > voxel_center_[0]) xyz[0] = 1;
                    if (pv.point[1] > voxel_center_[1]) xyz[1] = 1;
                    if (pv.point[2] > voxel_center_[2]) xyz[2] = 1;
                    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2]; // 根据坐标得到octree格子的索引值
                    if (leaves_[leafnum] != nullptr) {
                        leaves_[leafnum]->UpdateOctoTree(pv);
                    } else {
                        leaves_[leafnum] = new OctoTree(max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
                                                        max_cov_points_size_, planer_threshold_);
                        leaves_[leafnum]->layer_point_size_ = layer_point_size_;
                        leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2.0 * xyz[0] - 1) * quater_length_;
                        leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2.0 * xyz[1] - 1) * quater_length_;
                        leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2.0 * xyz[2] - 1) * quater_length_;
                        leaves_[leafnum]->quater_length_ = quater_length_ / 2;
                        leaves_[leafnum]->UpdateOctoTree(pv);
                    }
                } else {
                    if (update_enable_) {
                        new_points_num_++;
                        all_points_num_++;
                        if (update_cov_enable_) {
                            temp_points_.push_back(pv);
                        } else {
                            new_points_.push_back(pv);
                        }
                        if (new_points_num_ > update_size_threshold_) {
                            if (update_cov_enable_) {
                                init_plane(temp_points_, plane_ptr_);
                            } else {
                                update_plane(new_points_, plane_ptr_);
                                new_points_.clear();
                            }
                            new_points_num_ = 0;
                        }
                        if (all_points_num_ >= max_cov_points_size_) {
                            update_cov_enable_ = false;
                            std::vector<pointWithCov>().swap(temp_points_);
                        }
                        if (all_points_num_ >= max_points_size_) {
                            update_enable_ = false;
                            plane_ptr_->update_enable = false;
                            std::vector<pointWithCov>().swap(new_points_);
                        }
                    }
                }
            }
        }
    }

};

void transformLidar(const faster_lio::StatesGroup &state, const shared_ptr<faster_lio::ImuProcess> &p_imu,
                    const PointCloudType::Ptr &input_cloud, pcl::PointCloud<pcl::PointXYZINormal>::Ptr &trans_cloud) {
    trans_cloud->clear();
    for (size_t i = 0; i < input_cloud->size(); i++) {
        PointType p_c = input_cloud->points[i];
        Eigen::Vector3d p(p_c.x, p_c.y, p_c.z);
        // TODO:0301先转换为imu系
        p = p_imu->Lidar_R_wrt_IMU_ * p + p_imu->Lidar_T_wrt_IMU_;

        p = state.rot_end * p + state.pos_end;
        pcl::PointXYZINormal pi;
        pi.x = p[0];
        pi.y = p[1];
        pi.z = p[2];
        pi.intensity = p_c.intensity;
        trans_cloud->points.push_back(pi);
    }
}

/* 计算去畸变后点的协方差（body系）*/
//calcBodyCov(point_this, ranging_cov, angle_cov, cov);
void calcBodyCov(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &cov) {
    double range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]); // d_i: the depth measurement
    double range_var = range_inc * range_inc; // 距离误差的方差
    Eigen::Matrix2d direction_var;
    direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2); // 方向误差的方差
    Eigen::Vector3d direction(pb);
    // normalize: 把自身的各元素除以它的范数
    direction.normalize(); // w_i: the measured bearing direction（方向向量）
    Eigen::Matrix3d direction_hat;
    // the skew-symmetric matrix of w_i
    direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
    /* N: an orthonormal basis of the tangent plane at w_i */
    Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
    base_vector1.normalize();
    Eigen::Vector3d base_vector2 = base_vector1.cross(direction); // 叉乘
    base_vector2.normalize();
    Eigen::Matrix<double, 3, 2> N; // 公式1
    N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);

    Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
    cov = direction * range_var * direction.transpose() + A * direction_var * A.transpose(); // 公式1
}

// TODO:0211 voxel_zie & planner_threshold float -> double but double对性能影响太大
void buildVoxelMap(const std::vector<pointWithCov> &input_points, const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size, const int max_points_size, const int max_cov_points_size,
                   const float planer_threshold, std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
    uint pl_size = input_points.size();
    /* save points into Octo_tree */
    for (uint i = 0; i < pl_size; ++i) {
        // TODO:0210 p_v 从const 改为 const&；
        const pointWithCov &p_v = input_points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; ++j) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0; // todo: 0.1和-0.1取整后会在一个voxel内，故减1
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
        auto iter = feat_map.find(position);
        // 对于一个新增的点，首先计算索引key，查找此 key 是否已经存在,若存在，则向对应体素里新增点；若不存在，则先创建新OctoTree再插入点
        if (iter != feat_map.end()) {
            cout << "build map: find it !!!!!!!!" << endl;
            feat_map[position]->temp_points_.push_back(p_v);
            feat_map[position]->new_points_num_++;
        } else {
            cout << "bbbbbbbbbbbbb" << endl;
            auto *octoTree = new OctoTree(max_layer, 0, layer_point_size, max_points_size, max_cov_points_size,
                                          planer_threshold);
            cout << "aaaaaaaaaaaaaa" << endl;
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

void updateVoxelMap(const std::vector<pointWithCov> &input_points, const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size, const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold, std::unordered_map<VOXEL_LOC, OctoTree *> &feat_map) {
    uint plsize = input_points.size();
    for (uint i = 0; i < plsize; ++i) {
        const pointWithCov p_v = input_points[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; ++j) {
            loc_xyz[j] = p_v.point[j] / voxel_size;
//            cout << loc_xyz[j] << " " << endl;
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
        auto iter = feat_map.find(position);
        if (iter != feat_map.end()) {
//            cout << "kkkkkkkkkkkkkkkk" << endl;
            /* when the new points are added to an existing voxel, the parameters and the uncertainty of the plane
             * in the voxel should be updated */
            feat_map[position]->UpdateOctoTree(p_v);
//            cout << "jjjjjjjjjjjjj" << endl;
        } else {
//            cout << "fffffffffff" << endl;
            /* When the new points lie in an unpopulated voxel, it will construct the voxel */
            OctoTree *octo_tree = new OctoTree(max_layer, 0, layer_point_size, max_points_size, max_cov_points_size,
                                               planer_threshold);
//            std::shared_ptr<OctoTree> octo_tree = std::make_shared<OctoTree>(max_layer, 0, layer_point_size,
//                                                                          max_points_size, max_cov_points_size,
//                                                                          planer_threshold);
//            cout << "ggggggggggg" << endl;
            feat_map[position] = octo_tree;
//            cout << "gggggggggggg" << endl;
            feat_map[position]->quater_length_ = voxel_size / 4;
//            cout << "ggggggggggggg" << endl;
            feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
//            cout << "gggggggggggggg" << endl;
            feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
//            cout << "ggggggggggggggg" << endl;
            feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
//            cout << "gggggggggggggggg" << endl;
            feat_map[position]->UpdateOctoTree(p_v);
//            cout << "hhhhhhhhhhhhhhhh" << endl;
        }
    }
}

void build_single_residual(const pointWithCov &pv, const OctoTree *current_octo, const int current_layer,
                           const int max_layer, const double sigma_num, bool &is_success, double &prob,
                           pTpl &single_ptpl) {
    double radius_k = 3;
    Eigen::Vector3d p_w = pv.point_world;
    if (current_octo->plane_ptr_->is_plane) {
        Plane &plane = *current_octo->plane_ptr_;
        Eigen::Vector3d p_world_to_center = p_w - plane.center;
        double proj_x = p_world_to_center.dot(plane.x_normal);
        double proj_y = p_world_to_center.dot(plane.y_normal);
        float dist_to_plane = fabs(
                plane.normal[0] * p_w(0) + plane.normal[1] * p_w(1) + plane.normal[2] * p_w(2) + plane.d);
        float dist_to_center = (plane.center(0) - p_w(0)) * (plane.center(0) - p_w(0)) +
                               (plane.center(1) - p_w(1)) * (plane.center(1) - p_w(1)) +
                               (plane.center(2) - p_w(2)) * (plane.center(2) - p_w(2));
        float range_dist = sqrt(dist_to_center - dist_to_plane * dist_to_plane);
        // plane.radius：sqrt(evalsReal(evalsMax))
        if (range_dist <= radius_k * plane.radius) { // todo: why????
            Eigen::Matrix<double, 1, 6> J_nq;
            J_nq.block<1, 3>(0, 0) = p_w - plane.center; // 公式11
            J_nq.block<1, 3>(0, 3) = -plane.normal; // 公式11
            double sigma_l = J_nq * plane.plane_cov * J_nq.transpose();
            sigma_l += plane.normal.transpose() * pv.cov * plane.normal; // 公式10
            // 3sigma判定
            if (dist_to_plane < sigma_num * sqrt(sigma_l)) {
                is_success = true;
                double this_prob =
                        1.0 / (sqrt(sigma_l)) * exp(-0.5 * dist_to_plane * dist_to_plane / sigma_l); // 概率密度函数
                //  if a point matches more than one plane based on the 3σ criterion, the plane with the highest probability will be matched
                if (this_prob > prob) {
                    prob = this_prob;
                    single_ptpl.point = pv.point;
                    single_ptpl.plane_cov = plane.plane_cov;
                    single_ptpl.normal = plane.normal;
                    single_ptpl.center = plane.center;
                    single_ptpl.d = plane.d;
                    single_ptpl.layer = current_layer;
                }
                return;
            } else {
                return;
            }
        } else {
            return;
        }
    } else {
        if (current_layer < max_layer) {
            // TODO: 0214 修改waring，使用range循环方式
            for (auto leave: current_octo->leaves_) {
                if (leave != nullptr) {
                    OctoTree *leaf_octo = leave;
                    build_single_residual(pv, leaf_octo, current_layer + 1, max_layer, sigma_num, is_success, prob,
                                          single_ptpl);
                }
            }
            return;
        } else {
            return;
        }
    }
}

void BuildResidualListOMP(const unordered_map<VOXEL_LOC, OctoTree *> &voxel_map, const double voxel_size,
                          const double sigma_num, const int max_layer, const std::vector<pointWithCov> &pv_list,
                          std::vector<pTpl> &ptpl_list, std::vector<Eigen::Vector3d> &non_match) {
    std::mutex mylock;
    ptpl_list.clear();
    cout << "size!!!!" << pv_list.size() << endl;
    std::vector<pTpl> all_ptpl_list(pv_list.size());
    std::vector<bool> useful_ptpl(pv_list.size());
    std::vector<size_t> index(pv_list.size());
    for (size_t i = 0; i < index.size(); ++i) {
        index[i] = i;
        useful_ptpl[i] = false;
    }
#ifdef MP_EN
    omp_set_num_threads(MP_PROC_NUM); // 4
#pragma omp parallel for
#endif
    for (int i = 0; i < index.size(); ++i) {
        pointWithCov pv = pv_list[i];
        float loc_xyz[3];
        for (int j = 0; j < 3; ++j) {
            loc_xyz[j] = pv.point_world[j] / voxel_size;
            if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0;
        }
        VOXEL_LOC position((int64_t) loc_xyz[0], (int64_t) loc_xyz[1], (int64_t) loc_xyz[2]);
        auto iter = voxel_map.find(position);
        if (iter != voxel_map.end()) {
            OctoTree *current_octo = iter->second;
            pTpl single_ptpl;
            bool is_success = false;
            double prob = 0;
            build_single_residual(pv, current_octo, 0, max_layer, sigma_num, is_success, prob, single_ptpl);
            if (!is_success) {
                VOXEL_LOC near_position = position;
                if (loc_xyz[0] >
                    (current_octo->voxel_center_[0] + current_octo->quater_length_)) {
                    near_position.x = near_position.x + 1;
                } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
                                         current_octo->quater_length_)) {
                    near_position.x = near_position.x - 1;
                }
                if (loc_xyz[1] >
                    (current_octo->voxel_center_[1] + current_octo->quater_length_)) {
                    near_position.y = near_position.y + 1;
                } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
                                         current_octo->quater_length_)) {
                    near_position.y = near_position.y - 1;
                }
                if (loc_xyz[2] >
                    (current_octo->voxel_center_[2] + current_octo->quater_length_)) {
                    near_position.z = near_position.z + 1;
                } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
                                         current_octo->quater_length_)) {
                    near_position.z = near_position.z - 1;
                }
                auto iter_near = voxel_map.find(near_position);
                if (iter_near != voxel_map.end()) {
                    build_single_residual(pv, iter_near->second, 0, max_layer, sigma_num,
                                          is_success, prob, single_ptpl);
                }
            }
            if (is_success) {
                mylock.lock();
                useful_ptpl[i] = true;
                all_ptpl_list[i] = single_ptpl;
                mylock.unlock();
            } else {
                mylock.lock();
                useful_ptpl[i] = false;
                mylock.unlock();
            }
        }
    }
    for (size_t i = 0; i < useful_ptpl.size(); ++i) {
        if (useful_ptpl[i]) ptpl_list.push_back(all_ptpl_list[i]); // 如果有效匹配，存入ptpl_list，以备后面计算残差
    }
}

void GetUpdatePlane(const OctoTree *current_octo, const int pub_max_voxel_layer, std::vector<Plane> &plane_list) {
    if (current_octo->layer_ > pub_max_voxel_layer) {
        return;
    }
    if (current_octo->plane_ptr_->is_plane) {
        plane_list.push_back(*current_octo->plane_ptr_);
    }
    if (current_octo->layer_ < current_octo->max_layer_) {
        if (!current_octo->plane_ptr_->is_plane) {
            for (auto leave: current_octo->leaves_) {
                if (leave != nullptr) {
                    GetUpdatePlane(leave, pub_max_voxel_layer, plane_list);
                }
            }
        }
    }
}

void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g, uint8_t &b) {
    r = 255;
    g = 255;
    b = 255;

    if (v < vmin) {
        v = vmin;
    }

    if (v > vmax) {
        v = vmax;
    }

    double dr, dg, db;

    if (v < 0.1242) {
        db = 0.504 + ((1. - 0.504) / 0.1242) * v;
        dg = dr = 0.;
    } else if (v < 0.3747) {
        db = 1.;
        dr = 0.;
        dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
    } else if (v < 0.6253) {
        db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
        dg = 1.;
        dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
    } else if (v < 0.8758) {
        db = 0.;
        dr = 1.;
        dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
    } else {
        db = 0.;
        dg = 0.;
        dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
    }

    r = (uint8_t) (255 * dr);
    g = (uint8_t) (255 * dg);
    b = (uint8_t) (255 * db);
}

void CalcVectQuation(const Eigen::Vector3d &x_vec, const Eigen::Vector3d &y_vec, const Eigen::Vector3d &z_vec,
                     geometry_msgs::Quaternion &q) {
    Eigen::Matrix3d rot;
    rot << x_vec(0), x_vec(1), x_vec(2), y_vec(0), y_vec(1), y_vec(2), z_vec(0), z_vec(1), z_vec(2);
    Eigen::Matrix3d rotation = rot.transpose();
    Eigen::Quaterniond eq(rotation);
    q.w = eq.w();
    q.x = eq.x();
    q.y = eq.y();
    q.z = eq.z();
}

void pubSinglePlane(visualization_msgs::MarkerArray &plane_pub, const std::string &plane_ns, const Plane &single_plane,
                    const float alpha, const Eigen::Vector3d &rgb) {
    visualization_msgs::Marker plane;
    plane.header.frame_id = "camera_init";
    plane.header.stamp = ros::Time();
    plane.ns = plane_ns;
    plane.id = single_plane.id;
    plane.type = visualization_msgs::Marker::CYLINDER;
    plane.action = visualization_msgs::Marker::ADD;
    plane.pose.position.x = single_plane.center[0];
    plane.pose.position.y = single_plane.center[1];
    plane.pose.position.z = single_plane.center[2];
    geometry_msgs::Quaternion q;
    CalcVectQuation(single_plane.x_normal, single_plane.y_normal, single_plane.normal, q);
    plane.pose.orientation = q;
    plane.scale.x = 3 * sqrt(single_plane.max_eigen_value);
    plane.scale.y = 3 * sqrt(single_plane.mid_eigen_value);
    plane.scale.z = 2 * sqrt(single_plane.min_eigen_value);
    plane.color.a = alpha;
    plane.color.r = rgb(0);
    plane.color.g = rgb(1);
    plane.color.b = rgb(2);
    plane.lifetime = ros::Duration();
    plane_pub.markers.push_back(plane);
}

void pubVoxelMap(const std::unordered_map<VOXEL_LOC, OctoTree *> &voxel_map, const int pub_max_voxel_layer,
                 const ros::Publisher &plane_map_pub) {
    double max_trace = 0.25;
    double pow_num = 0.2;
    ros::Rate loop(500);
    float use_alpha = 0.8;
    visualization_msgs::MarkerArray voxel_plane;
    voxel_plane.markers.reserve(1000000);
    std::vector<Plane> pub_plane_list;
    for (const auto &iter: voxel_map) {
        GetUpdatePlane(iter.second, pub_max_voxel_layer, pub_plane_list); // 得到pub_plane_list
    }
    for (auto &p_list: pub_plane_list) {
        faster_lio::V3D plane_cov = p_list.plane_cov.block<3, 3>(0, 0).diagonal();
        double trace = plane_cov.sum();
        if (trace >= max_trace) {
            trace = max_trace;
        }
        trace = trace * (1.0 / max_trace);
        trace = pow(trace, pow_num);
        uint8_t r, g, b;
        mapJet(trace, 0, 1, r, g, b);
        Eigen::Vector3d plane_rgb(r / 256.0, g / 256.0, b / 256.0);
        float alpha;
        if (p_list.is_plane) {
            alpha = use_alpha;
        } else {
            alpha = 0;
        }
        pubSinglePlane(voxel_plane, "plane", p_list, alpha, plane_rgb);
    }
    plane_map_pub.publish(voxel_plane);
    loop.sleep();
}


#endif //VOXEL_MAP_UTIL_HPP
