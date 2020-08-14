#include <iostream>
#include <fstream>

#include <common/pose_estimation.hpp>
#include <common/features.hpp>

using namespace OneACPose;

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cerr << "Insufficient number of commandline parameters" << std::endl;
    exit(1);
  }

  // ==========================================================================
  // parameters and setting
  // ==========================================================================

  const std::string im_left = std::string(argv[1]) + ".jpg";
  const std::string im_right = std::string(argv[2]) + ".jpg";

  const std::string depth_left = std::string(argv[1]) + ".d.npy";
  const std::string depth_right = std::string(argv[2]) + ".d.npy";

  const std::string intrinsics_left = (argc > 3) ? std::string(argv[3]) : std::string(argv[1]) + ".K";
  const std::string intrinsics_right = (argc > 4) ? std::string(argv[4]) : std::string(argv[2]) + ".K";

  // GCRANSAC parameters and setting
  const double confidence_ = 0.99; // The RANSAC confidence value
  const double inlier_outlier_threshold_ = 3.0; // The used inlier-outlier threshold in GC-RANSAC for essential matrix estimation.
  const double spatial_coherence_weight_ = 0.975; // The weigd_t of the spatial coherence term in the graph-cut energy minimization.
  const size_t cell_number_in_neighborhood_graph_ = 8; // The number of cells along each axis in the neighborhood graph.
  const int fps_ = -1; // The required FPS limit. If it is set to -1, the algorithm will not be interrupted before finishing.

  // ==========================================================================
  // extract LAFs + Depth and match the features
  // ==========================================================================

  common::LAFs LAFs_left;
  common::LAFs LAFs_right;
  std::vector<size_t> im_left_shape;
  std::vector<size_t> im_right_shape;

  bool success_features = extract_and_match_LAFs(
    im_left, im_right,
    depth_left, depth_right,
    LAFs_left, LAFs_right,
    &im_left_shape, &im_right_shape);

  if (!success_features) {
    std::cerr << "Failed feature extraction and/or matching." << std::endl;
    return EXIT_FAILURE;
  }

  // ==========================================================================
  // load intrinsic camera matrices
  // ==========================================================================

  Mat3 intrinsics_src; {
    std::ifstream file_intrinsics_left(intrinsics_left);
    std::copy(std::istream_iterator<double>(file_intrinsics_left), std::istream_iterator<double>(), intrinsics_src.data());
  }

  Mat3 intrinsics_dst; {
    std::ifstream file_intrinsics_right(intrinsics_right);
    std::copy(std::istream_iterator<double>(file_intrinsics_right), std::istream_iterator<double>(), intrinsics_dst.data());
  }

  // ==========================================================================
  // perform ropust relative pose estimation using 1AC+D and GCRANSAC
  // ==========================================================================
  std::vector<size_t> inliers;
  Mat3 essential_matrix;

  bool success_gcransac = estimatePose_1ACD_GCRANSAC(
    intrinsics_src, intrinsics_dst,
    LAFs_left, LAFs_right,
    im_left_shape, im_right_shape,
    cell_number_in_neighborhood_graph_, fps_, spatial_coherence_weight_, confidence_, inlier_outlier_threshold_,
    inliers, essential_matrix);

  if (!success_features) {
    std::cerr << "Failed to perform robust estimation on the matched features." << std::endl;
    return EXIT_FAILURE;
  }

  // Display matches using OpenCV
  display_inliers(
    im_left, im_right,
    LAFs_left, LAFs_right,
    inliers);
}