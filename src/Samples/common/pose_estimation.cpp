#include "pose_estimation.hpp"

#include <OneACPose/solver_one_ac_depth.hpp>

namespace common {

  void estimatePose_1ACD(
    const CameraPtr& cam0,
    const CameraPtr& cam1,
    const Feature_LAF_D& laf0,
    const Feature_LAF_D& laf1,
    double& scale, OneACPose::Mat3& R, OneACPose::Vec3& t)
  {
    using namespace OneACPose;

    // Project 2D LAF 0 into 3D using depth and depth derivatives
    Vec3 src;
    Mat32 src_diff;
    laf0.as_3D(cam0, src, src_diff);

    // Project 2D LAF 1 into 3D using depth and depth derivatives
    Vec3 dst;
    Mat32 dst_diff;
    laf1.as_3D(cam1, dst, dst_diff);

    // estimate pose using 1AC+D
    OneACPose::OneACD(
      dst, src,
      dst_diff, src_diff,
      scale, R, t);
  }

}