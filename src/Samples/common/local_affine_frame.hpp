#pragma once

#include <common/camera_radial.hpp>
#include <OneACPose/types.hpp>

namespace common {

  struct Feature_LAF_D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    // 2D feature location
    OneACPose::Vec2 x;
    // affine shape around x, aka dx0_dx
    OneACPose::Mat2 M;
    // depth at location x
    double lambda;
    // depth derivatives at location x
    OneACPose::RowVec2 dlambda_dx;

    void as_3D(
      const common::CameraPtr& cam,
      OneACPose::Vec3& Y, OneACPose::Mat32& dY_dx) const {
      // 3D LAF 0 (q0, Q0)
      const auto [q0, dq0_dx0] = cam->q_gradient(x);
      // 3D-projected affine shape, aka q0_dx
      const OneACPose::Mat32 Q0 = dq0_dx0 * M;
      // combine 3D LAF 0 with depth and its derivatives
      Y.noalias() = q0 * lambda;
      dY_dx.noalias() = q0 * dlambda_dx + Q0 * lambda;
    }
  };

}