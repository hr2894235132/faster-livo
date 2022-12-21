//
// Created by hr on 22-12-5.
//
// This file is part of SVO - Semi-direct Visual Odometry.
//
// Copyright (C) 2014 Christian Forster <forster at ifi dot uzh dot ch>
// (Robotics and Perception Group, University of Zurich, Switzerland).
//
// SVO is free software: you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the Free Software
// Foundation, either version 3 of the License, or any later version.
//
// SVO is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef SVO_FEATURE_H_
#define SVO_FEATURE_H_

#include <frame.h>
#include <point.h>
#include <common_lib.h>

namespace faster_lio{
    namespace lidar_selection {

// A salient image region that is tracked across frames.
        struct Feature
        {
            EIGEN_MAKE_ALIGNED_OPERATOR_NEW

            enum FeatureType {
                CORNER, // 角点
                EDGELET // 边缘集
            };
            int id_;
            FeatureType type;     //!< Type can be corner or edgelet. 特征类型，分为角点和边缘集
            Frame* frame;         //!< Pointer to frame in which the feature was detected. 特征类型，分为角点和边缘集
            cv::Mat img;
            vector<cv::Mat> ImgPyr;
            Vector2d px;          //!< Coordinates in pixels on pyramid level 0. 第0层图像的特征点位置
            Vector3d f;           //!< Unit-bearing vector of the feature. 归一化平面坐标
            int level;            //!< Image pyramid level where feature was extracted. 特征点所在金字塔的层数
            PointPtr point;         //!< Pointer to 3D point which corresponds to the feature. 该特征点所关联的3D地图点
            Vector2d grad;        //!< Dominant gradient direction for edglets, normalized. 边缘集的梯度方向，模为1
            float score;
            float error;
            // Vector2d grad_cur_;   //!< edgelete grad direction in cur frame
            Sophus::SE3 T_f_w_;
            float* patch;
            Feature(float* _patch, const Vector2d& _px, const Vector3d& _f, const Sophus::SE3& _T_f_w, const float &_score, int _level) :
                    type(CORNER),
                    px(_px),
                    f(_f),
                    T_f_w_(_T_f_w),
                    level(_level),
                    patch(_patch),
                    score(_score)
            {}
            inline Vector3d pos() const { return T_f_w_.inverse().translation(); }
            ~Feature()
            {
                // printf("The feature %d has been destructed.", id_);
                delete[] patch;
            }
        };

    } // namespace lidar_selection
}
#endif // SVO_FEATURE_H_

