#include "numeric.hpp"

using namespace OneACPose;

Mat3 common::LookAt(const Vec3& forward, const Vec3& up)
{
  const Vec3 zc = forward.normalized();
  const Vec3 xc = up.cross(zc).normalized();
  const Vec3 yc = zc.cross(xc).normalized();
  Mat3 R;
  R.row(0) = xc;
  R.row(1) = yc;
  R.row(2) = zc;
  return R;
}

double common::getRotationMagnitude(const Mat3& R2) {
  const Mat3 R1 = Mat3::Identity();
  double cos_theta = (R1.array() * R2.array()).sum() / 3.0;
  cos_theta = std::clamp(cos_theta, -1.0, 1.0);
  //return (std::acos(R2(0, 0)) + std::acos(R2(1, 1)) + std::acos(R2(2, 2))) / 3.0;
  return std::acos(cos_theta);
}

OneACPose::Mat32 common::nullspace(const OneACPose::Vec3& N)
{
  return N.jacobiSvd(Eigen::ComputeFullU).matrixU().topRightCorner<3,2>();
}