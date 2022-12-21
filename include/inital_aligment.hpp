////
//// Created by hr on 22-11-14.
////
//#pragma once
//
//#include <eigen3/Eigen/Dense>
//#include <iostream>
//#include <ros/ros.h>
//#include <map>
//#include "../include/common_lib.h"
//#include "../include/use-ikfom.hpp"
//#include "imu_processing.hpp"
//#include "tools.hpp"
//
//using namespace Eigen;
//using namespace std;
////shared_ptr<ImuProcess> p_intergration(new ImuProcess());
//using namespace faster_lio;
//
//class LidarFrame {
//public:
//    LidarFrame() {};
//
////    LidarFrame(const map<int, vector<pair<int, PointCloudXYZI> > > &_points, double _t) : t{_t}{
////        points = _points;
////    };
////    map<int, vector<pair<int, PointCloudXYZI> > > points;
////    double t;
////    Matrix3d R;
////    Vector3d T;
////    bool is_key_frame;
////    StatesGroup m_state_prior;
//    faster_lio::state_ikfom *state;
////    StatesGroup *p;
////    shared_ptr<ImuProcess> p_intergration(new ImuProcess());
//    faster_lio::ImuProcess *pre_intergraiton;
//
//public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
//};
//
//// TODO:优化重力中ImageFrame的替换
//// TODO:函数参数使用时继续调整
//void RefineGravity(deque<LidarFrame> &all_lidar_frame, Vector3d &g, StatesGroup &state_in,
//                   deque<sensor_msgs::Imu::ConstPtr> &init_imu,
//                   double dt_end_pose) {
//    Vector3d g0 = g.normalized() * common::G_m_s2;
//    Vector3d lx, ly;
//    VectorXd x;
//    //VectorXd x;
//    int all_frame_count = all_lidar_frame.size();
//    int n_state = all_frame_count * 3 + 2; // 限定重力模值，仅优化重力方向，因此重力相关的自由度为2
//
//    MatrixXd A{n_state, n_state};
//    A.setZero();
//    VectorXd b{n_state};
//    b.setZero();
//
//    deque<LidarFrame>::iterator frame_i;
//    deque<LidarFrame>::iterator frame_j;
////    frame_i->second.R =  * frame_i->second.pre_integration->offset_R_L_I;
////    frame_i->second.T =  ;
////    frame_j->second.R =  * frame_j->second.pre_integration->offset_R_L_I;
////    frame_j->second.T =  ;
//    for (int k = 0; k < 4; k++) {
//        MatrixXd lxly(3, 2);
//        lxly = Tools::TangentBasis(g0);
//        int i = 0;
//        for (frame_i = all_lidar_frame.begin(); next(frame_i) != all_lidar_frame.end(); frame_i++, i++) {
//            frame_j = next(frame_i);
//            if (!frame_j->pre_intergraiton) return;
//
//            MatrixXd tmp_A(6, 8);
//            tmp_A.setZero();
//            VectorXd tmp_b(6);
//            tmp_b.setZero();
//
////            double dt = frame_j->second.pre_integration->sum_dt; // 预积分的时间
//            double dt = frame_j->pre_intergraiton->sum_dt;
//            Matrix3d frame_i_R = frame_i->state->rot.toRotationMatrix();
//            Matrix3d frame_j_R = frame_j->state->rot.toRotationMatrix();
//            Vector3d frame_i_T = frame_i->state->pos;
//            Vector3d frame_j_T = frame_j->state->pos;
//
//            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
//            tmp_A.block<3, 2>(0, 6) = frame_i_R.transpose() * dt * dt / 2 * Matrix3d::Identity() * lxly;
////            tmp_A.block<3, 1>(0, 8) = frame_i->R.transpose() * (frame_j->T - frame_i->T) / 100.0;
//            tmp_b.block<3, 1>(0, 0) = frame_j->pre_intergraiton->state_inout.pos_end +
//                                      frame_i_R.transpose() * frame_j_R * frame_j->state->offset_T_L_I -
//                                      frame_j->state->offset_T_L_I -
//                                      frame_i_R.transpose() * dt * dt / 2 * g0 -
//                                      frame_i_R.transpose() * (frame_j_T - frame_i_T);
//
//            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
//            tmp_A.block<3, 3>(3, 3) = frame_i_R.transpose() * frame_j_R;
//            tmp_A.block<3, 2>(3, 6) = frame_i_R.transpose() * dt * Matrix3d::Identity() * lxly;
//            tmp_b.block<3, 1>(3, 0) = frame_j->pre_intergraiton->state_inout.vel_end -
//                                      frame_i_R.transpose() * dt * Matrix3d::Identity() * g0;
//
//
//            Matrix<double, 6, 6> cov_inv = Matrix<double, 6, 6>::Zero();
//            //cov.block<6, 6>(0, 0) = IMU_cov[i + 1];
//            //MatrixXd cov_inv = cov.inverse();
//            cov_inv.setIdentity();
//
//            MatrixXd r_A = tmp_A.transpose() * cov_inv * tmp_A;
//            VectorXd r_b = tmp_A.transpose() * cov_inv * tmp_b;
//
//            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
//            b.segment<6>(i * 3) += r_b.head<6>();
//
//            A.bottomRightCorner<2, 2>() += r_A.bottomRightCorner<2, 2>();
//            b.tail<2>() += r_b.tail<2>();
//
//            A.block<6, 2>(i * 3, n_state - 2) += r_A.topRightCorner<6, 2>();
//            A.block<2, 6>(n_state - 2, i * 3) += r_A.bottomLeftCorner<2, 6>();
//        }
//        A = A * 1000.0;
//        b = b * 1000.0;
//        x = A.ldlt().solve(b);
//        VectorXd dg = x.segment<2>(n_state - 2);
//        g0 = (g0 + lxly * dg).normalized() * common::G_m_s2;
//        //double s = x(n_state - 1);
//    }
//    g = g0;
//}
//
