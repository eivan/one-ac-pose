#pragma once

#include <vector>
#include <OneACPose/types.hpp>

#include "camera.hpp"

namespace common {

  struct Feature_LAF_D {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Feature_LAF_D() = default;
    Feature_LAF_D(
      const OneACPose::Vec2& x_,
      const OneACPose::Mat2& M_,
      const double& lambda_,
      const OneACPose::RowVec2& dlambda_dx_)
      : x(x_), M(M_), lambda(lambda_), dlambda_dx(dlambda_dx_) {}

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
      OneACPose::Vec3& Y,
      OneACPose::Mat32& dY_dx) const;
  };

  using LAFs = std::vector<common::Feature_LAF_D>;

}