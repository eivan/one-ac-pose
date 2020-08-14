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
  
  bool estimatePose_1ACD_GCRANSAC(
    const OneACPose::Mat3& intrinsics_src, const OneACPose::Mat3& intrinsics_dst,
    const common::LAFs& LAFs_left, const common::LAFs& LAFs_right,
    const std::vector<size_t>& im_left_shape,
    const std::vector<size_t>& im_right_shape,
    const size_t& cell_number_in_neighborhood_graph_,
    const int& fps_,
    const double& spatial_coherence_weight_,
    const double& confidence_,
    const double& inlier_outlier_threshold_,
    std::vector<size_t>& inliers,
    OneACPose::Mat3& essential_matrix,
    const bool verbose = true);
  
}