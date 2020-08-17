#pragma once

#include <string>
#include <common/local_affine_frame.hpp>

#include <VlFeatExtraction/Extraction.hpp>
#include <VlFeatExtraction/Matching.hpp>

namespace common {

  bool extract_and_match_LAFs(
    const VlFeatExtraction::SiftExtractionOptions& options_f,
    const VlFeatExtraction::SiftMatchingOptions& options_m,
    const std::string& im_left, const std::string& im_right,
    const std::string& depth_left, const std::string& depth_right,
    common::LAFs& LAFs_left, common::LAFs& LAFs_right,
    std::vector<size_t>* im_left_shape = nullptr,
    std::vector<size_t>* im_right_shape = nullptr,
    bool dump_extracted_featues_to_disk = false);

  bool augment_LAFs_with_depth(
    const VlFeatExtraction::FeatureKeypoints& fkp_left,
    const VlFeatExtraction::FeatureKeypoints& fkp_right,
    const VlFeatExtraction::FeatureMatches& matches,
    const std::string& depth_left, const std::string& depth_right,
    common::LAFs& LAFs_left,
    common::LAFs& LAFs_right);

  void display_inliers(
    const std::string& im_left, const std::string& im_right,
    const common::LAFs& LAFs_left, const common::LAFs& LAFs_right,
    const std::vector<size_t>& inliers);

}