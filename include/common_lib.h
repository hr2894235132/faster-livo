#ifndef COMMON_LIB_H
#define COMMON_LIB_H

#include <so3_math.h>
#include <eigen_conversions/eigen_msg.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <ros/ros.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/array.hpp>
#include <unsupported/Eigen/ArpackSupport>
#include <sophus/se3.h>
#include <boost/shared_ptr.hpp>
#include <unordered_map>

#include "faster_lio/Pose6D.h"
#include <faster_lio//States.h>
#include "options.h"
#include "so3_math.h"

using PointType = pcl::PointXYZINormal;
using PointCloudType = pcl::PointCloud<PointType>; // 即fast-lio中的PointCloudXYZI
using CloudPtr = PointCloudType::Ptr;
using PointVector = std::vector<PointType, Eigen::aligned_allocator<PointType>>;
using PointTypeRGB = pcl::PointXYZRGB;
using PointCloudXYZRGB = pcl::PointCloud<PointTypeRGB>;

#define VEC_FROM_ARRAY(v)        v[0],v[1],v[2]
#define MAT_FROM_ARRAY(v)        v[0],v[1],v[2],v[3],v[4],v[5],v[6],v[7],v[8]
#define MD(a, b)  Matrix<double, (a), (b)>
#define VD(a)    Matrix<double, (a), 1>
#define MF(a, b)  Matrix<float, (a), (b)>
#define VF(a)    Matrix<float, (a), 1>
#define DIM_STATE (18)      // Dimension of states (Let Dim(SO(3)) = 3)
#define INIT_COV   (0.001)

#define HASH_P 116101
#define MAX_N 10000000000
//#define PUBFRAME_PERIOD (20)

using namespace std;
using namespace Eigen;

namespace faster_lio {

    constexpr double G_m_s2 = 9.81;  // Gravity const in GuangDong/China

    template<typename S>
    inline Eigen::Matrix<S, 3, 1> VecFromArray(const std::vector<double> &v) {
        return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
    }

    template<typename S>
    inline Eigen::Matrix<S, 3, 1> VecFromArray(const boost::array<S, 3> &v) {
        return Eigen::Matrix<S, 3, 1>(v[0], v[1], v[2]);
    }

    template<typename S>
    inline Eigen::Matrix<S, 3, 3> MatFromArray(const std::vector<double> &v) {
        Eigen::Matrix<S, 3, 3> m;
        m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
        return m;
    }

    template<typename S>
    inline Eigen::Matrix<S, 3, 3> MatFromArray(const boost::array<S, 9> &v) {
        Eigen::Matrix<S, 3, 3> m;
        m << v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8];
        return m;
    }

    inline std::string DEBUG_FILE_DIR(const std::string &name) { return std::string(ROOT_DIR) + "Log/" + name; }

    using Pose6D = faster_lio::Pose6D;
    using V2D = Eigen::Vector2d;
    using V3D = Eigen::Vector3d;
    using V4D = Eigen::Vector4d;
    using V5D = Eigen::Matrix<double, 5, 1>;
    using M3D = Eigen::Matrix3d;
    using M4D = Eigen::Matrix4d;
    using V3F = Eigen::Vector3f;
    using V4F = Eigen::Vector4f;
    using V5F = Eigen::Matrix<float, 5, 1>;
    using M3F = Eigen::Matrix3f;
    using M4F = Eigen::Matrix4f;

    using VV3D = std::vector<V3D, Eigen::aligned_allocator<V3D>>;
    using VV3F = std::vector<V3F, Eigen::aligned_allocator<V3F>>;
    using VV4F = std::vector<V4F, Eigen::aligned_allocator<V4F>>; // vector<Eigen::vector4f>
    using VV4D = std::vector<V4D, Eigen::aligned_allocator<V4D>>;
    using VV5F = std::vector<V5F, Eigen::aligned_allocator<V5F>>;
    using VV5D = std::vector<V5D, Eigen::aligned_allocator<V5D>>;

    const M3D Eye3d = M3D::Identity();
    const M3F Eye3f = M3F::Identity();
    const V3D Zero3d(0, 0, 0);
    const V3F Zero3f(0, 0, 0);

    // HR:
    namespace lidar_selection {
        class Point;

        typedef std::shared_ptr<Point> PointPtr;

        class VOXEL_POINTS {
        public:
            std::vector<PointPtr> voxel_points;
            int count;
            // bool is_visited;

            VOXEL_POINTS(int num) : count(num) {}
        };

        class Warp {
        public:
            Eigen::Matrix2d A_cur_ref;
            int search_level;
            // bool is_visited;

            Warp(int level, Eigen::Matrix2d warp_matrix) : search_level(level), A_cur_ref(warp_matrix) {}
        };
    }

/// sync imu and img measurements
    struct MeasureGroup {
        MeasureGroup() { img_offset_time = 0.0; };

        double img_offset_time;
        deque<sensor_msgs::Imu::ConstPtr> imu_;
        cv::Mat img;
    };

    // HR:lidar measurements
    struct LidarMeasureGroup {
        double lidar_beg_time_;
        double last_update_time_;
        double lidar_end_time_ = 0;
        PointCloudType::Ptr lidar_ = nullptr;
        deque<struct MeasureGroup> measures; // imu and img measurements
        bool is_lidar_end;
        int lidar_scan_index_now;

        LidarMeasureGroup() {
            lidar_beg_time_ = 0.0;
            is_lidar_end = false;
            this->lidar_.reset(new PointCloudType());
            deque<struct MeasureGroup>().swap(this->measures);
            lidar_scan_index_now = 0;
            last_update_time_ = 0.0;
        };

        void debug_show() {
            int i = 0;
            ROS_WARN("Lidar selector debug:");
            cout << "last_update_time:" << setprecision(20) << this->last_update_time_ << endl;
            cout << "lidar_beg_time:" << setprecision(20) << this->lidar_beg_time_ << endl;
            for (auto it = this->measures.begin(); it != this->measures.end(); ++it, ++i) {
                cout << "In " << i << " measures: ";
                for (auto it_meas = it->imu_.begin(); it_meas != it->imu_.end(); ++it_meas) {
                    cout << setprecision(20) << (*it_meas)->header.stamp.toSec() - this->lidar_beg_time_
                              << " ";
                }
                cout << "img_time:" << setprecision(20) << it->img_offset_time << endl;
            }
            cout << "is_lidar_end:" << this->is_lidar_end << "lidar_end_time:"
                      << this->lidar_->points.back().curvature / double(1000) << endl;
            cout << "lidar_.points.size(): " << this->lidar_->points.size() << endl << endl;
        };
    };

    // HR:视觉稀疏地图
    struct SparseMap {
        vector<V3D> points;
        vector<float *> patch;
        vector<float> values;
        vector<cv::Mat> imgs;
        vector<M3D> R_ref;
        vector<V3D> P_ref;
        vector<V3D> xyz_ref;
        vector<V2D> px;
        M3D Rcl;
        V3D Pcl;

        SparseMap() {
            this->points.clear();
            this->patch.clear();
            this->values.clear();
            this->imgs.clear();
            this->R_ref.clear();
            this->P_ref.clear();
            this->px.clear();
            this->xyz_ref.clear();
            this->Rcl = M3D::Identity();
            this->Pcl = Zero3d;
        };

        void set_camera2lidar(vector<double> &R, vector<double> &P) {
            this->Rcl << MAT_FROM_ARRAY(R);
            this->Pcl << VEC_FROM_ARRAY(P);
        };

        void reset() {
            this->points.clear();
            this->patch.clear();
            this->values.clear();
            this->imgs.clear();
            this->R_ref.clear();
            this->P_ref.clear();
            this->px.clear();
            this->xyz_ref.clear();
        }
    };

    namespace lidar_selection {
        // TODO：稀疏子地图类？
        struct SubSparseMap {
            vector<float> align_errors;
            vector<float> propa_errors;
            vector<float> errors;
            vector<int> index;
            vector<float *> patch;
            vector<float *> patch_with_border;
            vector <V2D> px_cur;
            vector <V2D> propa_px_cur;
            vector<int> search_levels;
            vector <PointPtr> voxel_points;

            SubSparseMap() {
                this->align_errors.clear();
                this->propa_errors.clear();
                this->search_levels.clear();
                this->errors.clear();
                this->index.clear();
                this->patch.clear();
                this->patch_with_border.clear();
                this->px_cur.clear(); //Feature Alignment
                this->propa_px_cur.clear();
                this->voxel_points.clear();
            };

            void reset() {
                vector<float>().swap(this->align_errors);
                this->align_errors.reserve(500);

                vector<float>().swap(this->propa_errors);
                this->propa_errors.reserve(500);

                vector<int>().swap(this->search_levels);
                this->search_levels.reserve(500);

                vector<float>().swap(this->errors);
                this->errors.reserve(500);

                vector<int>().swap(this->index);
                this->index.reserve(500);

                vector<float *>().swap(this->patch);
                this->patch.reserve(500);

                vector<float *>().swap(this->patch_with_border);
                this->patch_with_border.reserve(500);

                vector<V2D>().swap(this->px_cur);  //Feature Alignment
                this->px_cur.reserve(500);

                vector<V2D>().swap(this->propa_px_cur);
                this->propa_px_cur.reserve(500);

                this->voxel_points.clear();
                // vector<PointPtr>().swap(this->voxel_points);
                this->voxel_points.reserve(500);
            }
        };
    }
    typedef boost::shared_ptr<SparseMap> SparseMapPtr;

    // HR:
    struct StatesGroup {
        StatesGroup() {
            this->rot_end = M3D::Identity();
            this->pos_end = Zero3d;
            this->vel_end = Zero3d;
            this->bias_g = Zero3d;
            this->bias_a = Zero3d;
            this->gravity = Zero3d;
            this->cov = Matrix<double, DIM_STATE, DIM_STATE>::Identity() * INIT_COV;
        };

        StatesGroup(const StatesGroup &b) {
            this->rot_end = b.rot_end;
            this->pos_end = b.pos_end;
            this->vel_end = b.vel_end;
            this->bias_g = b.bias_g;
            this->bias_a = b.bias_a;
            this->gravity = b.gravity;
            this->cov = b.cov;
        };

        StatesGroup &operator=(const StatesGroup &b) {
            this->rot_end = b.rot_end;
            this->pos_end = b.pos_end;
            this->vel_end = b.vel_end;
            this->bias_g = b.bias_g;
            this->bias_a = b.bias_a;
            this->gravity = b.gravity;
            this->cov = b.cov;
            return *this;
        };

        StatesGroup operator+(const Matrix<double, DIM_STATE, 1> &state_add) {
            StatesGroup a;
            a.rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
            a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
            a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
            a.bias_g = this->bias_g + state_add.block<3, 1>(9, 0);
            a.bias_a = this->bias_a + state_add.block<3, 1>(12, 0);
            a.gravity = this->gravity + state_add.block<3, 1>(15, 0);
            a.cov = this->cov;
            return a;
        };

        StatesGroup &operator+=(const Matrix<double, DIM_STATE, 1> &state_add) {
            this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
            this->pos_end += state_add.block<3, 1>(3, 0);
            this->vel_end += state_add.block<3, 1>(6, 0);
            this->bias_g += state_add.block<3, 1>(9, 0);
            this->bias_a += state_add.block<3, 1>(12, 0);
            this->gravity += state_add.block<3, 1>(15, 0);
            return *this;
        };

        Matrix<double, DIM_STATE, 1> operator-(const StatesGroup &b) {
            Matrix<double, DIM_STATE, 1> a;
            M3D rotd(b.rot_end.transpose() * this->rot_end);
            a.block<3, 1>(0, 0) = Log(rotd);
            a.block<3, 1>(3, 0) = this->pos_end - b.pos_end;
            a.block<3, 1>(6, 0) = this->vel_end - b.vel_end;
            a.block<3, 1>(9, 0) = this->bias_g - b.bias_g;
            a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
            a.block<3, 1>(15, 0) = this->gravity - b.gravity;
            return a;
        };

        void resetpose() {
            this->rot_end = M3D::Identity();
            this->pos_end = Zero3d;
            this->vel_end = Zero3d;
        }

        M3D rot_end;      // the estimated attitude (rotation matrix) at the end lidar point
        V3D pos_end;      // the estimated position at the end lidar point (world frame)
        V3D vel_end;      // the estimated velocity at the end lidar point (world frame)
        V3D bias_g;       // gyroscope bias
        V3D bias_a;       // accelerator bias
        V3D gravity;      // the estimated gravity acceleration
        Matrix<double, DIM_STATE, DIM_STATE> cov;     // states covariance
    };

    template<typename T>
    T rad2deg(const T &radians) {
        return radians * 180.0 / M_PI;
    }

    template<typename T>
    T deg2rad(const T &degrees) {
        return degrees * M_PI / 180.0;
    }

/**
 * set a pose 6d from ekf status
 * @tparam T
 * @param t
 * @param a
 * @param g
 * @param v
 * @param p
 * @param R
 * @return
 */
    template<typename T>
    Pose6D set_pose6d(const double t, const Eigen::Matrix<T, 3, 1> &a, const Eigen::Matrix<T, 3, 1> &g,
                      const Eigen::Matrix<T, 3, 1> &v, const Eigen::Matrix<T, 3, 1> &p,
                      const Eigen::Matrix<T, 3, 3> &R) {
        Pose6D rot_kp;
        rot_kp.offset_time = t;
        for (int i = 0; i < 3; i++) {
            rot_kp.acc[i] = a(i);
            rot_kp.gyr[i] = g(i);
            rot_kp.vel[i] = v(i);
            rot_kp.pos[i] = p(i);
            for (int j = 0; j < 3; j++) rot_kp.rot[i * 3 + j] = R(i, j);
        }
        return rot_kp;
    }

/* comment
plane equation: Ax + By + Cz + D = 0
convert to: A/D*x + B/D*y + C/D*z = -1
solve: A0*x0 = b0
where A0_i = [x_i, y_i, z_i], x0 = [A/D, B/D, C/D]^T, b0 = [-1, ..., -1]^T
normvec_:  normalized x0
*/
/**
 * 计算一组点的法线
 * @tparam T
 * @param normvec
 * @param point
 * @param threshold
 * @param point_num
 * @return
 */
    template<typename T>
    bool esti_normvector(Eigen::Matrix<T, 3, 1> &normvec, const PointVector &point, const T &threshold,
                         const int &point_num) {
        Eigen::MatrixXf A(point_num, 3);
        Eigen::MatrixXf b(point_num, 1);
        b.setOnes();
        b *= -1.0f;

        for (int j = 0; j < point_num; j++) {
            A(j, 0) = point[j].x;
            A(j, 1) = point[j].y;
            A(j, 2) = point[j].z;
        }
        normvec = A.colPivHouseholderQr().solve(b);

        for (int j = 0; j < point_num; j++) {
            if (fabs(normvec(0) * point[j].x + normvec(1) * point[j].y + normvec(2) * point[j].z + 1.0f) > threshold) {
                return false;
            }
        }

        normvec.normalize();
        return true;
    }

/**
 * squared distance
 * @param p1
 * @param p2
 * @return
 */
    inline float calc_dist(const PointType &p1, const PointType &p2) {
        return (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
    }

    inline float calc_dist(const Eigen::Vector3f &p1, const Eigen::Vector3f &p2) { return (p1 - p2).squaredNorm(); }

/**
 * estimate a plane
 * @tparam T
 * @param pca_result
 * @param point
 * @param threshold
 * @return
 */
    template<typename T>
    inline bool esti_plane(Eigen::Matrix<T, 4, 1> &pca_result, const PointVector &point, const T &threshold = 0.1f) {
        if (point.size() < options::MIN_NUM_MATCH_POINTS) {
            return false;
        }

        Eigen::Matrix<T, 3, 1> normvec;

        // 用5个点进行拟合
        if (point.size() == options::NUM_MATCH_POINTS) {
            Eigen::Matrix<T, options::NUM_MATCH_POINTS, 3> A;
            Eigen::Matrix<T, options::NUM_MATCH_POINTS, 1> b;

            A.setZero();
            b.setOnes();
            b *= -1.0f;

            for (int j = 0; j < options::NUM_MATCH_POINTS; j++) {
                A(j, 0) = point[j].x;
                A(j, 1) = point[j].y;
                A(j, 2) = point[j].z;
            }

            normvec = A.colPivHouseholderQr().solve(b);
        } else {
            Eigen::MatrixXd A(point.size(), 3);
            Eigen::VectorXd b(point.size(), 1);

            A.setZero();
            b.setOnes();
            b *= -1.0f;

            for (int j = 0; j < point.size(); j++) {
                A(j, 0) = point[j].x;
                A(j, 1) = point[j].y;
                A(j, 2) = point[j].z;
            }

            Eigen::MatrixXd n = A.colPivHouseholderQr().solve(b);
            normvec(0, 0) = n(0, 0);
            normvec(1, 0) = n(1, 0);
            normvec(2, 0) = n(2, 0);
        }

        T n = normvec.norm();
        pca_result(0) = normvec(0) / n;
        pca_result(1) = normvec(1) / n;
        pca_result(2) = normvec(2) / n;
        pca_result(3) = 1.0 / n;

        for (const auto &p: point) {
            Eigen::Matrix<T, 4, 1> temp = p.getVector4fMap();
            temp[3] = 1.0;
            if (fabs(pca_result.dot(temp)) > threshold) {
                return false;
            }
        }
        return true;
    }

}  // namespace faster_lio::common
// HR:Key of hash table
class VOXEL_KEY {
public:
    int64_t x;
    int64_t y;
    int64_t z;

    VOXEL_KEY(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

    bool operator==(const VOXEL_KEY &other) const {
        return (x == other.x && y == other.y && z == other.z);
    }

    bool operator<(const VOXEL_KEY &p) const {
        if (x < p.x) return true;
        if (x > p.x) return false;
        if (y < p.y) return true;
        if (y > p.y) return false;
        if (z < p.z) return true;
        if (z > p.z) return false;
    }
};

// HR:Hash value
namespace std{
    template<>
    struct hash<VOXEL_KEY>{
        size_t operator() (const VOXEL_KEY &s) const
        {
            using std::size_t;
            using std::hash;

            // Compute individual hash values for first,
            // second and third and combine them using XOR
            // and bit shifting:
            //   return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1) ^ (hash<int64_t>()(s.z) << 1);
            /* 以体素的空间坐标v（是整数坐标，与物理坐标的关系是乘以体素的边长）作为 key，对v的三个维度各乘以一个很大的整数再做异或位运算，
             * 得到的结果再对哈希表的容量N取余，得到体素的 index —— id_v。实际应用中，我们是先有点再去找体素，因此需要先把点的坐标
             * 除以体素边长再取整算出 */
            return (((hash<int64_t>()(s.z)*HASH_P)%MAX_N + hash<int64_t>()(s.y))*HASH_P)%MAX_N + hash<int64_t>()(s.x);
        }
    };
}

#endif

