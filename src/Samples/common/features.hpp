#pragma once

#include <string>
#include <common/local_affine_frame.hpp>

namespace common {

  bool extract_and_match_LAFs(
    const std::string& im_left, const std::string& im_right,
    const std::string& depth_left, const std::string& depth_right,
    common::LAFs& LAFs_left, common::LAFs& LAFs_right,
    std::vector<size_t>* im_left_shape = nullptr,
    std::vector<size_t>* im_right_shape = nullptr);

  void display_inliers(
    const std::string& im_left, const std::string& im_right,
    const common::LAFs& LAFs_left, const common::LAFs& LAFs_right,
    const std::vector<size_t>& inliers);

}