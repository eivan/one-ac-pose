#pragma once
#include <Eigen/Dense>

namespace OneACPose {

  using Vec2 = Eigen::Vector2d;
  using Vec3 = Eigen::Vector3d;

  using Mat2 = Eigen::Matrix2d;
  using Mat3 = Eigen::Matrix3d;

  using Mat23 = Eigen::Matrix<double, 2, 3>;
  using Mat32 = Eigen::Matrix<double, 3, 2>;
  using Mat34 = Eigen::Matrix<double, 3, 4>;

  using RowVec2 = Eigen::Matrix<double, 1, 2, Eigen::RowMajor>;
  using RowVec3 = Eigen::Matrix<double, 1, 3, Eigen::RowMajor>;

  using MMat32d = Eigen::Map<OneACPose::Mat32>;
  using MVec3d = Eigen::Map<OneACPose::Vec3>;

  using MMat32dc = Eigen::Map<const OneACPose::Mat32>;
  using MVec3dc = Eigen::Map<const OneACPose::Vec3>;
  
}