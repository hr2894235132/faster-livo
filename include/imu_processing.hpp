#ifndef FASTER_LIO_IMU_PROCESSING_H
#define FASTER_LIO_IMU_PROCESSING_H

#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <cmath>
#include <deque>
#include <fstream>
#include <ros/ros.h>

#include <faster_lio/States.h>
#include "common_lib.h"
#include "so3_math.h"
#include "utils.h"
#include "tools.hpp"

#ifdef USE_IKFOM
#include "use-ikfom.hpp"
#endif

namespace faster_lio {

    /// *************Preconfiguration
    // todo: 12.12 20 -> 200
    constexpr int MAX_INI_COUNT = 20;

    bool time_list(const PointType &x, const PointType &y) { return (x.curvature < y.curvature); };

/// IMU Process and undistortion
    class ImuProcess {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        ImuProcess();

        ~ImuProcess();

        void Reset();

        void SetExtrinsic(const V3D &transl, const M3D &rot);

        void SetGyrCovScale(const V3D &scaler);

        void SetAccCovScale(const V3D &scaler);

        void SetGyrBiasCov(const V3D &b_g);

        void SetAccBiasCov(const V3D &b_a);

#ifdef USE_IKFOM
        void Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                     PointCloudType::Ptr pcl_un_);
#else

        void Process(const LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudType::Ptr cur_pcl_un_);

        void Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat, PointCloudType::Ptr cur_pcl_un_);

        void UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudType &pcl_out);

#endif

        std::ofstream fout_imu_;
        Eigen::Matrix<double, 12, 12> Q_;
        V3D cov_acc_;
        V3D cov_gyr_;
        V3D cov_acc_scale_;
        V3D cov_gyr_scale_;
        V3D cov_bias_gyr_;
        V3D cov_bias_acc_;
        double first_lidar_time;

    private:
#ifdef USE_IKFOM
        void IMUInit(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state, int &N);

        void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                          PointCloudType &pcl_out);
#else

        void IMUInit(const MeasureGroup &meas, StatesGroup &state, int &N);

        void Forward(const MeasureGroup &meas, StatesGroup &state_inout, double pcl_beg_time, double end_time);

        void Backward(const LidarMeasureGroup &lidar_meas, StatesGroup &state_inout, PointCloudType &pcl_out);

#endif

        PointCloudType::Ptr cur_pcl_un_;
        sensor_msgs::ImuConstPtr last_imu_;
        std::deque<sensor_msgs::ImuConstPtr> v_imu_;
        std::vector<Pose6D> IMUpose_;
        std::vector<M3D> v_rot_pcl_;
        M3D Lidar_R_wrt_IMU_;
        V3D Lidar_T_wrt_IMU_;
        V3D mean_acc_;
        V3D mean_gyr_;
        V3D angvel_last_;
        V3D acc_s_last_;
        V3D last_acc;
        V3D last_ang;
        double last_lidar_end_time_ = 0;
        int init_iter_num_ = 1;
        bool b_first_frame_ = true;
        bool imu_need_init_ = true;
    };

    ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true) {
        init_iter_num_ = 1;
#ifdef USE_IKFOM
        Q_ = process_noise_cov();
#endif
        cov_acc_ = V3D(0.1, 0.1, 0.1);
        cov_gyr_ = V3D(0.1, 0.1, 0.1);
        cov_bias_gyr_ = V3D(0.00003, 0.00003, 0.00003);
        cov_bias_acc_ = V3D(0.01, 0.01, 0.01);
        mean_acc_ = V3D(0, 0, -1.0);
        mean_gyr_ = V3D(0, 0, 0);
        angvel_last_ = Zero3d;
        Lidar_T_wrt_IMU_ = Zero3d;
        Lidar_R_wrt_IMU_ = Eye3d;
        last_imu_.reset(new sensor_msgs::Imu());
    }

    ImuProcess::~ImuProcess() {}

    void ImuProcess::Reset() {
        ROS_WARN("Reset ImuProcess");
        mean_acc_ = V3D(0, 0, -1.0);
        mean_gyr_ = V3D(0, 0, 0);
        angvel_last_ = Zero3d;
        imu_need_init_ = true;
        init_iter_num_ = 1;
        v_imu_.clear();
        IMUpose_.clear();
        last_imu_.reset(new sensor_msgs::Imu());
        cur_pcl_un_.reset(new PointCloudType());
    }

    void ImuProcess::SetExtrinsic(const V3D &transl, const M3D &rot) {
        Lidar_T_wrt_IMU_ = transl;
        Lidar_R_wrt_IMU_ = rot;
    }

    void ImuProcess::SetGyrCovScale(const V3D &scaler) { cov_gyr_scale_ = scaler; }

    void ImuProcess::SetAccCovScale(const V3D &scaler) { cov_acc_scale_ = scaler; }

    void ImuProcess::SetGyrBiasCov(const V3D &b_g) { cov_bias_gyr_ = b_g; }

    void ImuProcess::SetAccBiasCov(const V3D &b_a) { cov_bias_acc_ = b_a; }

#ifdef USE_IKFOM
    void ImuProcess::IMUInit(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                             int &N) {
        /** 1. initializing the gravity_, gyro bias, acc and gyro covariance
         ** 2. normalize the acceleration measurenments to unit gravity_ **/

        V3D cur_acc, cur_gyr;

        if (b_first_frame_) {
            Reset();
            N = 1;
            b_first_frame_ = false;
            const auto &imu_acc = meas.imu_.front()->linear_acceleration;
            const auto &gyr_acc = meas.imu_.front()->angular_velocity;
            mean_acc_ << imu_acc.x, imu_acc.y, imu_acc.z;
            mean_gyr_ << gyr_acc.x, gyr_acc.y, gyr_acc.z;
        }

        for (const auto &imu: meas.imu_) {
            const auto &imu_acc = imu->linear_acceleration;
            const auto &gyr_acc = imu->angular_velocity;
            cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
            cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

            mean_acc_ += (cur_acc - mean_acc_) / N;
            mean_gyr_ += (cur_gyr - mean_gyr_) / N;

            cov_acc_ =
                    cov_acc_ * (N - 1.0) / N +
                    (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
            cov_gyr_ =
                    cov_gyr_ * (N - 1.0) / N +
                    (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);

            N++;
        }
        state_ikfom init_state = kf_state.get_x();
        init_state.grav = S2(-mean_acc_ / mean_acc_.norm() * G_m_s2);

        init_state.bg = mean_gyr_;
        init_state.offset_T_L_I = Lidar_T_wrt_IMU_;
        init_state.offset_R_L_I = Lidar_R_wrt_IMU_;
        kf_state.change_x(init_state); // 更改状态x的值

        esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
        init_P.setIdentity();
        init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
        init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
        init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = 0.0001;
        init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = 0.001;
        init_P(21, 21) = init_P(22, 22) = 0.00001;
        kf_state.change_P(init_P);
        last_imu_ = meas.imu_.back();
    }
#else

    void ImuProcess::IMUInit(const MeasureGroup &meas, StatesGroup &state_inout, int &N) {
        /** 1. initializing the gravity_, gyro bias, acc and gyro covariance
        ** 2. normalize the acceleration measurenments to unit gravity_ **/
        ROS_INFO("IMU Initializing: %.1f %%", double(N) / MAX_INI_COUNT * 100);
        V3D cur_acc, cur_gyr;

        if (b_first_frame_) {
            Reset();
            N = 1;
            b_first_frame_ = false;
            const auto &imu_acc = meas.imu_.front()->linear_acceleration;
            const auto &gyr_acc = meas.imu_.front()->angular_velocity;
            mean_acc_ << imu_acc.x, imu_acc.y, imu_acc.z;
            mean_gyr_ << gyr_acc.x, gyr_acc.y, gyr_acc.z;
        }

        for (const auto &imu: meas.imu_) {
            const auto &imu_acc = imu->linear_acceleration;
            const auto &gyr_acc = imu->angular_velocity;
            cur_acc << imu_acc.x, imu_acc.y, imu_acc.z;
            cur_gyr << gyr_acc.x, gyr_acc.y, gyr_acc.z;

            mean_acc_ += (cur_acc - mean_acc_) / N;
            mean_gyr_ += (cur_gyr - mean_gyr_) / N;

            cov_acc_ = cov_acc_ * (N - 1.0) / N +
                       (cur_acc - mean_acc_).cwiseProduct(cur_acc - mean_acc_) * (N - 1.0) / (N * N);
            cov_gyr_ = cov_gyr_ * (N - 1.0) / N +
                       (cur_gyr - mean_gyr_).cwiseProduct(cur_gyr - mean_gyr_) * (N - 1.0) / (N * N);
            // cout<<"acc norm: "<<cur_acc.norm()<<" "<<mean_acc.norm()<<endl;
            N++;
        }
        state_inout.gravity = -mean_acc_ / mean_acc_.norm() * G_m_s2;
        state_inout.rot_end = Eye3d; // Exp(mean_acc.cross(V3D(0, 0, -1 / scale_gravity)));
        state_inout.bias_g = mean_gyr_;

        last_imu_ = meas.imu_.back();
    }

#endif

#ifdef USE_IKFOM
    void
    ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                             PointCloudType &pcl_out) {
        /*** add the imu_ of the last frame-tail to the of current frame-head ***/
        auto v_imu = meas.imu_;
        v_imu.push_front(last_imu_);
        const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
        const double &imu_end_time = v_imu.back()->header.stamp.toSec();
        const double &pcl_beg_time = meas.lidar_bag_time_;
        const double &pcl_end_time = meas.lidar_end_time_;

        /*** sort point clouds by offset time ***/
        pcl_out = *(meas.lidar_);
        sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);

        /*** Initialize IMU pose ***/
        state_ikfom imu_state = kf_state.get_x();
        IMUpose_.clear();
        IMUpose_.push_back(set_pose6d(0.0, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                      imu_state.rot.toRotationMatrix()));

        /*** forward propagation at each imu_ point ***/
        V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
        M3D R_imu;

        double dt = 0;

        input_ikfom in;
        for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
            auto &&head = *(it_imu);
            auto &&tail = *(it_imu + 1);

            if (tail->header.stamp.toSec() < last_lidar_end_time_) {
                continue;
            }

            angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                    0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                    0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
            acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                    0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                    0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

            acc_avr = acc_avr * G_m_s2 / mean_acc_.norm();  // - state_inout.ba;

            if (head->header.stamp.toSec() < last_lidar_end_time_) {
                dt = tail->header.stamp.toSec() - last_lidar_end_time_;
            } else {
                dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
            }

            in.acc = acc_avr;
            in.gyro = angvel_avr;
            Q_.block<3, 3>(0, 0).diagonal() = cov_gyr_;
            Q_.block<3, 3>(3, 3).diagonal() = cov_acc_;
            Q_.block<3, 3>(6, 6).diagonal() = cov_bias_gyr_;
            Q_.block<3, 3>(9, 9).diagonal() = cov_bias_acc_;
            kf_state.predict(dt, Q_, in);

            /* save the poses at each IMU measurements */
            imu_state = kf_state.get_x();
            angvel_last_ = angvel_avr - imu_state.bg;
            acc_s_last_ = imu_state.rot * (acc_avr - imu_state.ba); // todo:世界坐标系下？
            for (int i = 0; i < 3; i++) {
                acc_s_last_[i] += imu_state.grav[i];
            }

            double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
            IMUpose_.emplace_back(set_pose6d(offs_t, acc_s_last_, angvel_last_, imu_state.vel, imu_state.pos,
                                             imu_state.rot.toRotationMatrix()));
        }

        /*** calculated the pos and attitude prediction at the frame-end ***/
        double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
        dt = note * (pcl_end_time - imu_end_time);
        kf_state.predict(dt, Q_, in);

        imu_state = kf_state.get_x();
        last_imu_ = meas.imu_.back();
        last_lidar_end_time_ = pcl_end_time;

        /*** undistort each lidar point (backward propagation) ***/
        if (pcl_out.points.empty()) {
            return;
        }
        auto it_pcl = pcl_out.points.end() - 1;
        for (auto it_kp = IMUpose_.end() - 1; it_kp != IMUpose_.begin(); it_kp--) {
            auto head = it_kp - 1;
            auto tail = it_kp;
            R_imu = MatFromArray(head->rot);
            vel_imu = VecFromArray(head->vel);
            pos_imu = VecFromArray(head->pos);
            acc_imu = VecFromArray(tail->acc);
            angvel_avr = VecFromArray(tail->gyr);

            for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
                dt = it_pcl->curvature / double(1000) - head->offset_time;

                /* Transform to the 'end' frame, using only the rotation
                 * Note: Compensation direction is INVERSE of Frame's moving direction
                 * So if we want to compensate a point at timestamp-i to the frame-e
                 * p_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
                M3D R_i(R_imu * Exp(angvel_avr, dt));

                V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
                // T_ei:点所在时刻IMU在世界坐标系下的位置 - end时刻IMU在世界坐标系下的位置
                V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
                V3D p_compensate =
                        imu_state.offset_R_L_I.conjugate() *
                        (imu_state.rot.conjugate() *
                         (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) -
                         imu_state.offset_T_L_I);  // not accurate!

                // save Undistorted points and their rotation
                it_pcl->x = p_compensate(0);
                it_pcl->y = p_compensate(1);
                it_pcl->z = p_compensate(2);

                if (it_pcl == pcl_out.points.begin()) {
                    break;
                }
            }
        }
    }
#else

    /* 前向传播 */
    void ImuProcess::Forward(const MeasureGroup &meas, StatesGroup &state_inout,
                             double pcl_beg_time, double end_time) {
        /*** add the imu of the last frame-tail to the of current frame-head ***/
        /*** 将上一帧尾部的imu添加到当前帧头部的imu ***/
        auto v_imu = meas.imu_;
        v_imu.push_front(last_imu_);
        const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
        const double &imu_end_time = v_imu.back()->header.stamp.toSec();
// cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
//          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;
        /*** Initialize IMU pose ***/
        if (IMUpose_.empty()) {
            IMUpose_.push_back(set_pose6d(0.0, acc_s_last_, angvel_last_, state_inout.vel_end, state_inout.pos_end,
                                          state_inout.rot_end));
        }

        /*** forward propagation at each imu point ***/
        V3D angvel_avr = angvel_last_, acc_avr, acc_imu = acc_s_last_, vel_imu(state_inout.vel_end), pos_imu(
                state_inout.pos_end);
        M3D R_imu(state_inout.rot_end);
        MD(DIM_STATE, DIM_STATE) F_x, cov_w;
        double dt = 0;
        for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
            auto &&head = *(it_imu);
            auto &&tail = *(it_imu + 1);
            if (tail->header.stamp.toSec() < last_lidar_end_time_) continue;

            angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                    0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                    0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
            acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                    0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                    0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);
            // 原始测量的中值作为更新
            last_acc = acc_avr;
            last_ang = angvel_avr;
            // #ifdef DEBUG_PRINT
            fout_imu_ << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose()
                      << " " << acc_avr.transpose() << endl;
            // #endif
            angvel_avr -= state_inout.bias_g;
            acc_avr = acc_avr * G_m_s2 / mean_acc_.norm() - state_inout.bias_a;
            if (head->header.stamp.toSec() < last_lidar_end_time_) {
                dt = tail->header.stamp.toSec() - last_lidar_end_time_;
            } else {
                dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
            }
            /* covariance propagation */
            M3D acc_avr_skew;
            M3D Exp_f = Exp(angvel_avr, dt);
            acc_avr_skew << SKEW_SYM_MATRIX(acc_avr);

            F_x.setIdentity();
            cov_w.setZero();

            F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
            F_x.block<3, 3>(0, 9) = -Eye3d * dt;
            // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
            F_x.block<3, 3>(3, 6) = Eye3d * dt;
            F_x.block<3, 3>(6, 0) = -R_imu * acc_avr_skew * dt;
            F_x.block<3, 3>(6, 12) = -R_imu * dt;
            F_x.block<3, 3>(6, 15) = Eye3d * dt;

            cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr_ * dt * dt;
            cov_w.block<3, 3>(6, 6) = R_imu * cov_acc_.asDiagonal() * R_imu.transpose() * dt * dt;
            cov_w.block<3, 3>(9, 9).diagonal() = cov_bias_gyr_ * dt * dt; // bias gyro covariance
            cov_w.block<3, 3>(12, 12).diagonal() = cov_bias_acc_ * dt * dt; // bias acc covariance

            state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

            /* propogation of IMU attitude */
            R_imu = R_imu * Exp_f;

            /* Specific acceleration (global frame) of IMU */
            acc_imu = R_imu * acc_avr + state_inout.gravity;

            /* propogation of IMU position */
            pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

            /* velocity of IMU */
            vel_imu = vel_imu + acc_imu * dt;
            /* save the poses at each IMU measurements */
            angvel_last_ = angvel_avr;
            acc_s_last_ = acc_imu;
            double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
            IMUpose_.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
        }
        /*** calculated the pos and attitude prediction at the frame-end ***/
        double note = end_time > imu_end_time ? 1.0 : -1.0;
        dt = note * (end_time - imu_end_time);
        state_inout.vel_end = vel_imu + note * acc_imu * dt;
        state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
        state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;

        last_imu_ = v_imu.back();
        last_lidar_end_time_ = end_time;
        // auto pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lid_offset_to_IMU;
        // auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

#ifdef DEBUG_PRINT
        cout<<"[ IMU Process ]: vel "<<state_inout.vel_end.transpose()<<" pos "<<state_inout.pos_end.transpose()<<" ba"<<state_inout.bias_a.transpose()<<" bg "<<state_inout.bias_g.transpose()<<endl;
        cout<<"propagated cov: "<<state_inout.cov.diagonal().transpose()<<endl;
#endif
    }

    /* 反向传播 */
    void ImuProcess::Backward(const LidarMeasureGroup &lidar_meas, StatesGroup &state_inout,
                              PointCloudType &pcl_out) {
        /*** undistort each lidar point (backward propagation) ***/
        M3D R_imu;
        V3D acc_imu, angvel_avr, vel_imu, pos_imu;
        double dt;
        auto pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lidar_T_wrt_IMU_; // world frame
        auto it_pcl = pcl_out.points.end() - 1;
        for (auto it_kp = IMUpose_.end(); it_kp != IMUpose_.begin(); it_kp--) {
            auto head = it_kp - 1;
            auto tail = it_kp;
            // 拿到前一帧的imu数据
            R_imu << MAT_FROM_ARRAY(head->rot);
            acc_imu << VEC_FROM_ARRAY(head->acc);
            vel_imu << VEC_FROM_ARRAY(head->vel);
            pos_imu << VEC_FROM_ARRAY(head->pos);
            angvel_avr << VEC_FROM_ARRAY(head->gyr);
            for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
                dt = it_pcl->curvature / double(1000) - head->offset_time;

                /* Transform to the 'end' frame, using only the rotation
                 * Note: Compensation direction is INVERSE of Frame's moving direction
                 * So if we want to compensate a point at timestamp-i to the frame-e
                 * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
                M3D R_i(R_imu * Exp(angvel_avr, dt));
                // T_ei:点所在时刻IMU在世界坐标系下的位置 - end时刻IMU在世界坐标系下的位置
                V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lidar_T_wrt_IMU_ - pos_liD_e);

                V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
                V3D P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);
                /// save Undistorted points and their rotation
                it_pcl->x = P_compensate[0];
                it_pcl->y = P_compensate[1];
                it_pcl->z = P_compensate[2];

                if (it_pcl == pcl_out.begin()) break;
            }
        }
    }

#endif

#ifdef USE_IKFOM
    void ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                             PointCloudType::Ptr cur_pcl_un_) {
        if (meas.imu_.empty()) {
            return;
        }

        ROS_ASSERT(meas.lidar_ != nullptr);

        if (imu_need_init_) {
            /// The very first lidar frame
            IMUInit(meas, kf_state, init_iter_num_);

            imu_need_init_ = true;

            last_imu_ = meas.imu_.back();

            state_ikfom imu_state = kf_state.get_x();

//            //TODO:重力对齐
//            Eigen::Vector3d g_b = imu_state.grav;
////                      cout << "11111111" << g_b << endl;
//            Eigen::Matrix3d R_wb = Tools::g2R(g_b);
////                      R_wb = Eigen::Quaterniond::FromTwoVectors(g_b.normalized(), g_w).toRotationMatrix();
//            double yaw = Tools::R2ypr(R_wb).x();
//            R_wb = Tools::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R_wb;
//            Eigen::Vector3d g_alig = R_wb * g_b;
//            imu_state.grav = S2(g_alig);
////                    std::cout << "g_aligment: " << g_alig[0]  << ","<< g_alig[1] << "," << g_alig[2] << std::endl;
//
//            imu_state.vel = R_wb * imu_state.vel;
//            imu_state.pos = R_wb * imu_state.pos;
//            imu_state.rot = R_wb * imu_state.rot;
//
//            std::cout << "g0     " << g_alig.transpose() << std::endl;
//            std::cout << "my R0  "  << Tools::R2ypr(R_wb).transpose() << std::endl;

            // 累计20帧
            if (init_iter_num_ > MAX_INI_COUNT) {
                cov_acc_ *= pow(G_m_s2 / mean_acc_.norm(), 2);
                imu_need_init_ = false;

                cov_acc_ = cov_acc_scale_;
                cov_gyr_ = cov_gyr_scale_;
                LOG(INFO) << "IMU Initial Done";
                fout_imu_.open(DEBUG_FILE_DIR("imu_.txt"), std::ios::out);
            }
            return;
        }

        Timer::Evaluate([&, this]() { UndistortPcl(meas, kf_state, *cur_pcl_un_); }, "Undistort Pcl");

    }
#else

    // 使用拆分实现的前向传播和反向传播进行点云去畸变
    void ImuProcess::Process(const LidarMeasureGroup &lidar_meas, StatesGroup &stat,
                             PointCloudType::Ptr cur_pcl_un_) {
        double t1, t2;
        t1 = omp_get_wtime();
        ROS_ASSERT(lidar_meas.lidar_ != nullptr);
        MeasureGroup meas = lidar_meas.measures.back();
        if (imu_need_init_) {
            if (meas.imu_.empty()) return;
            IMUInit(meas, stat, init_iter_num_);
            imu_need_init_ = true;
            last_imu_ = meas.imu_.back();

            // 累积足够帧再处理
            if (init_iter_num_ > MAX_INI_COUNT) {
                cov_acc_ *= pow(G_m_s2 / mean_acc_.norm(), 2);
                imu_need_init_ = false;
                ROS_INFO(
                        "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f", \
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc_.norm(), cov_acc_scale_[0],
                        cov_acc_scale_[1],
                        cov_acc_scale_[2], cov_acc_[0], cov_acc_[1], cov_acc_[2], cov_gyr_[0], cov_gyr_[1],
                        cov_gyr_[2]);
                cov_acc_ = cov_acc_.cwiseProduct(cov_acc_scale_);
                cov_gyr_ = cov_gyr_.cwiseProduct(cov_gyr_scale_);

                ROS_INFO(
                        "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f", \
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc_.norm(), cov_bias_gyr_[0], cov_bias_gyr_[1],
                        cov_bias_gyr_[2], cov_acc_[0], cov_acc_[1], cov_acc_[2], cov_gyr_[0], cov_gyr_[1], cov_gyr_[2]);
                fout_imu_.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
            }
            return;
        }
        /// Undistort points： the first point is assummed as the base frame
        /// Compensate lidar points with IMU rotation (with only rotation now)
        if (lidar_meas.is_lidar_end) {
            // sort point clouds by offset time
            *cur_pcl_un_ = *(lidar_meas.lidar_);
            sort(cur_pcl_un_->points.begin(), cur_pcl_un_->points.end(), time_list);
            const double &pcl_beg_time = lidar_meas.lidar_beg_time_;
            const double &pcl_end_time = pcl_beg_time + lidar_meas.lidar_->points.back().curvature / double(1000);
            Forward(meas, stat, pcl_beg_time, pcl_end_time);
            // cout<<"[ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
    //        <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;
            // cout<<"Time:";
            // for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
            //   cout<<it->offset_time<<" ";
            // }
            // cout<<endl<<"size:"<<IMUpose.size()<<endl;

            Backward(lidar_meas, stat, *cur_pcl_un_);
            last_lidar_end_time_ = pcl_end_time;
            IMUpose_.clear();
            Timer::Evaluate([&, this]() { Forward(meas, stat, pcl_beg_time, pcl_end_time); }, "Forward");
            Timer::Evaluate([&, this]() { Backward(lidar_meas, stat, *cur_pcl_un_); }, "Backward");
        } else {
            const double &pcl_beg_time = lidar_meas.lidar_beg_time_;
            const double &img_end_time = pcl_beg_time + meas.img_offset_time;
            // todo : 12.13 如果一帧雷达扫描没有结束，用图像的最后一帧时间作为雷达的最后一帧时间
            Forward(meas, stat, pcl_beg_time, img_end_time);
        }

        // {
        //   static ros::Publisher pub_UndistortPcl =
        //       nh.advertise<sensor_msgs::PointCloud2>("/livox_undistort", 100);
        //   sensor_msgs::PointCloud2 pcl_out_msg;
        //   pcl::toROSMsg(*cur_pcl_un_, pcl_out_msg);
        //   pcl_out_msg.header.stamp = ros::Time().fromSec(meas.lidar_beg_time);
        //   pcl_out_msg.header.frame_id = "/livox";
        //   pub_UndistortPcl.publish(pcl_out_msg);
        // }

        t2 = omp_get_wtime();

        // cout<<"[ IMU Process ]: Time: "<<t2 - t1<<endl;
    }

    void ImuProcess::UndistortPcl(LidarMeasureGroup &lidar_meas, StatesGroup &state_inout,
                                  PointCloudType &pcl_out) {
        /*** add the imu of the last frame-tail to the of current frame-head ***/
        MeasureGroup meas;
        meas = lidar_meas.measures.back();
        // cout<<"meas.imu.size: "<<meas.imu.size()<<endl;
        auto v_imu = meas.imu_;
        v_imu.push_front(last_imu_); // 将上一帧最后尾部的imu添加到当前帧头部的imu
        const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
        const double &imu_end_time = v_imu.back()->header.stamp.toSec();
        const double pcl_beg_time = MAX(lidar_meas.lidar_beg_time_, lidar_meas.last_update_time_); // todo :???
        // const double &pcl_beg_time = meas.lidar_beg_time;

        /*** sort point clouds by offset time ***/
        pcl_out.clear();
        auto pcl_it = lidar_meas.lidar_->points.begin() + lidar_meas.lidar_scan_index_now; // 保证pcl_out的顺序
        auto pcl_it_end = lidar_meas.lidar_->points.end();
        const double pcl_end_time = lidar_meas.is_lidar_end ?
                                    lidar_meas.lidar_beg_time_ +
                                    lidar_meas.lidar_->points.back().curvature / double(1000) :
                                    lidar_meas.lidar_beg_time_ + lidar_meas.measures.back().img_offset_time;
        const double pcl_offset_time = lidar_meas.is_lidar_end ?
                                       (pcl_end_time - lidar_meas.lidar_beg_time_) * double(1000) :
                                       0.0;
        while (pcl_it != pcl_it_end && pcl_it->curvature <= pcl_offset_time) {
            pcl_out.push_back(*pcl_it);
            pcl_it++;
            lidar_meas.lidar_scan_index_now++;
        }
        // cout<<"pcl_offset_time:  "<<pcl_offset_time<<"pcl_it->curvature:  "<<pcl_it->curvature<<endl;
        // cout<<"lidar_meas.lidar_scan_index_now:"<<lidar_meas.lidar_scan_index_now<<endl;
        lidar_meas.last_update_time_ = pcl_end_time;
        if (lidar_meas.is_lidar_end) {
            lidar_meas.lidar_scan_index_now = 0;
        }
        // sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
        // lidar_meas.debug_show();
        // cout<<"UndistortPcl [ IMU Process ]: Process lidar from "<<pcl_beg_time<<" to "<<pcl_end_time<<", " \
  //          <<meas.imu.size()<<" imu msgs from "<<imu_beg_time<<" to "<<imu_end_time<<endl;
        // cout<<"v_imu.size: "<<v_imu.size()<<endl;
        /*** Initialize IMU pose ***/
        IMUpose_.clear();
        // IMUpose.push_back(set_pose6d(0.0, Zero3d, Zero3d, state.vel_end, state.pos_end, state.rot_end));
        IMUpose_.push_back(
                set_pose6d(0.0, acc_s_last_, angvel_last_, state_inout.vel_end, state_inout.pos_end,
                           state_inout.rot_end));

        /*** forward propagation at each imu point ***/
        V3D acc_imu(acc_s_last_), angvel_avr(angvel_last_), acc_avr, vel_imu(state_inout.vel_end), pos_imu(
                state_inout.pos_end);
        M3D R_imu(state_inout.rot_end);
        MD(DIM_STATE, DIM_STATE) F_x, cov_w;

        double dt = 0;
        for (auto it_imu = v_imu.begin(); it_imu != v_imu.end() - 1; it_imu++) {
            auto &&head = *(it_imu);
            auto &&tail = *(it_imu + 1);

            if (tail->header.stamp.toSec() < last_lidar_end_time_) continue;

            angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
                    0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
                    0.5 * (head->angular_velocity.z + tail->angular_velocity.z);

            // angvel_avr<<tail->angular_velocity.x, tail->angular_velocity.y, tail->angular_velocity.z;

            acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
                    0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
                    0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

            // #ifdef DEBUG_PRINT
            fout_imu_ << setw(10) << head->header.stamp.toSec() - first_lidar_time << " " << angvel_avr.transpose()
                     << " "
                     << acc_avr.transpose() << endl;
            // #endif

            angvel_avr -= state_inout.bias_g;
            acc_avr = acc_avr * G_m_s2 / mean_acc_.norm() - state_inout.bias_a;

            if (head->header.stamp.toSec() < last_lidar_end_time_) {
                dt = tail->header.stamp.toSec() - last_lidar_end_time_;
            } else {
                dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
            }

            /* covariance propagation */
            M3D acc_avr_skew;
            M3D Exp_f = Exp(angvel_avr, dt);
            acc_avr_skew << SKEW_SYM_MATRIX(acc_avr);

            F_x.setIdentity();
            cov_w.setZero();

            F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
            F_x.block<3, 3>(0, 9) = -Eye3d * dt;
            // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
            F_x.block<3, 3>(3, 6) = Eye3d * dt;
            F_x.block<3, 3>(6, 0) = -R_imu * acc_avr_skew * dt;
            F_x.block<3, 3>(6, 12) = -R_imu * dt;
            F_x.block<3, 3>(6, 15) = Eye3d * dt;

            cov_w.block<3, 3>(0, 0).diagonal() = cov_gyr_ * dt * dt;
            cov_w.block<3, 3>(6, 6) = R_imu * cov_acc_.asDiagonal() * R_imu.transpose() * dt * dt;
            cov_w.block<3, 3>(9, 9).diagonal() = cov_bias_gyr_ * dt * dt; // bias gyro covariance
            cov_w.block<3, 3>(12, 12).diagonal() = cov_bias_acc_ * dt * dt; // bias acc covariance

            state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

            /* propogation of IMU attitude */
            R_imu = R_imu * Exp_f;

            /* Specific acceleration (global frame) of IMU */
            acc_imu = R_imu * acc_avr + state_inout.gravity;

            /* propogation of IMU */
            pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

            /* velocity of IMU */
            vel_imu = vel_imu + acc_imu * dt;

            /* save the poses at each IMU measurements */
            angvel_last_ = angvel_avr;
            acc_s_last_ = acc_imu;
            double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
            // cout<<setw(20)<<"offset_t: "<<offs_t<<"tail->header.stamp.toSec(): "<<tail->header.stamp.toSec()<<endl;
            IMUpose_.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
        }

        /*** calculated the pos and attitude prediction at the frame-end ***/
        /*** 计算帧末的位置和姿态预测 ***/
        if (imu_end_time > pcl_beg_time) {
            double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
            dt = note * (pcl_end_time - imu_end_time);
            state_inout.vel_end = vel_imu + note * acc_imu * dt;
            state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
            state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
        } else {
            double note = pcl_end_time > pcl_beg_time ? 1.0 : -1.0;
            dt = note * (pcl_end_time - pcl_beg_time);
            state_inout.vel_end = vel_imu + note * acc_imu * dt;
            state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
            state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;
        }

        last_imu_ = v_imu.back();
        last_lidar_end_time_ = pcl_end_time;

        auto pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lidar_T_wrt_IMU_;
        // auto R_liD_e   = state_inout.rot_end * Lidar_R_to_IMU;

        // cout<<"[ IMU Process ]: vel "<<state_inout.vel_end.transpose()<<" pos "<<state_inout.pos_end.transpose()<<" ba"<<state_inout.bias_a.transpose()<<" bg "<<state_inout.bias_g.transpose()<<endl;
        // cout<<"propagated cov: "<<state_inout.cov.diagonal().transpose()<<endl;

        //   cout<<"UndistortPcl Time:";
        //   for (auto it = IMUpose.begin(); it != IMUpose.end(); ++it) {
        //     cout<<it->offset_time<<" ";
        //   }
        //   cout<<endl<<"UndistortPcl size:"<<IMUpose.size()<<endl;
        //   cout<<"Undistorted pcl_out.size: "<<pcl_out.size()
        //          <<"lidar_meas.size: "<<lidar_meas.lidar->points.size()<<endl;
        if (pcl_out.points.size() < 1) return;
        /*** undistort each lidar point (backward propagation) ***/
        auto it_pcl = pcl_out.points.end() - 1;
        for (auto it_kp = IMUpose_.end() - 1; it_kp != IMUpose_.begin(); it_kp--) {
            auto head = it_kp - 1;
            auto tail = it_kp;
            R_imu << MAT_FROM_ARRAY(head->rot);
            acc_imu << VEC_FROM_ARRAY(head->acc);
            // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
            vel_imu << VEC_FROM_ARRAY(head->vel);
            pos_imu << VEC_FROM_ARRAY(head->pos);
            angvel_avr << VEC_FROM_ARRAY(head->gyr);

            for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
                dt = it_pcl->curvature / double(1000) - head->offset_time;

                /* Transform to the 'end' frame, using only the rotation
                 * Note: Compensation direction is INVERSE of Frame's moving direction
                 * So if we want to compensate a point at timestamp-i to the frame-e
                 * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
                M3D R_i(R_imu * Exp(angvel_avr, dt));
                V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lidar_T_wrt_IMU_ - pos_liD_e);

                V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
                V3D P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);

                /// save Undistorted points and their rotation
                it_pcl->x = P_compensate(0);
                it_pcl->y = P_compensate(1);
                it_pcl->z = P_compensate(2);

                if (it_pcl == pcl_out.points.begin()) break;
            }
        }
    }

    void ImuProcess::Process2(LidarMeasureGroup &lidar_meas, StatesGroup &stat,
                              PointCloudType::Ptr cur_pcl_un_) {
        double t1, t2, t3;
        t1 = omp_get_wtime();
        ROS_ASSERT(lidar_meas.lidar_ != nullptr);
        MeasureGroup meas = lidar_meas.measures.back();

        if (imu_need_init_) {
            if (meas.imu_.empty()) { return; };
            /// The very first lidar frame
            IMUInit(meas, stat, init_iter_num_);

            imu_need_init_ = true;

            last_imu_ = meas.imu_.back();

            if (init_iter_num_ > MAX_INI_COUNT) {
                cov_acc_ *= pow(G_m_s2 / mean_acc_.norm(), 2);
                imu_need_init_ = false;
                ROS_INFO(
                        "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f", \
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc_.norm(), cov_acc_scale_[0], cov_acc_scale_[1],
                        cov_acc_scale_[2], cov_acc_[0], cov_acc_[1], cov_acc_[2], cov_gyr_[0], cov_gyr_[1], cov_gyr_[2]);
                cov_acc_ = cov_acc_.cwiseProduct(cov_acc_scale_); // 两个矩阵各元素相乘
                cov_gyr_ = cov_gyr_.cwiseProduct(cov_gyr_scale_);

                // cov_acc = Eye3d * cov_acc_scale;
                // cov_gyr = Eye3d * cov_gyr_scale;
                // cout<<"mean acc: "<<mean_acc<<" acc measures in word frame:"<<state.rot_end.transpose()*mean_acc<<endl;
                ROS_INFO(
                        "IMU Initials: Gravity: %.4f %.4f %.4f %.4f; state.bias_g: %.4f %.4f %.4f; acc covarience: %.8f %.8f %.8f; gry covarience: %.8f %.8f %.8f", \
               stat.gravity[0], stat.gravity[1], stat.gravity[2], mean_acc_.norm(), cov_bias_gyr_[0], cov_bias_gyr_[1],
                        cov_bias_gyr_[2], cov_acc_[0], cov_acc_[1], cov_acc_[2], cov_gyr_[0], cov_gyr_[1], cov_gyr_[2]);
                fout_imu_.open(DEBUG_FILE_DIR("imu.txt"), ios::out);
            }

            return;
        }
        Timer::Evaluate([&, this]() { UndistortPcl(lidar_meas, stat, *cur_pcl_un_); }, "Undistort Pcl");
    }
}  // namespace faster_lio
#endif

#endif
