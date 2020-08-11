#pragma once

#include <OneACPose/types.hpp>

#include "camera.hpp"
#include "local_affine_frame.hpp"


namespace common {
  
  void estimatePose_1ACD(
    const CameraPtr& cam0,
    const CameraPtr& cam1,
    const Feature_LAF_D& laf0,
    const Feature_LAF_D& laf1,
    double& scale, OneACPose::Mat3& R, OneACPose::Vec3& t);
  
}