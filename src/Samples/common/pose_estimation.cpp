#include "pose_estimation.hpp"

#include <thread>

#include <OneACPose/solver_one_ac_depth.hpp>

#include <opencv2/opencv.hpp>

#include <graph-cut-ransac/src/pygcransac/include/essential_estimator.h>
#include <graph-cut-ransac/src/pygcransac/include/fundamental_estimator.h>
#include <graph-cut-ransac/src/pygcransac/include/progressive_napsac_sampler.h>

#include <common/camera_radial.hpp>

#include "solver_relative_pose_one_affine_and_relative_depth_minimal_svd.hpp"

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
  const bool verbose)
{
  auto cam_source = std::make_shared<common::Camera_Radial>();
  auto cam_destination = std::make_shared<common::Camera_Radial>();
  cam_source->set_params({ intrinsics_src(0,0), intrinsics_src(1,1), intrinsics_src(0,2), intrinsics_src(1,2), 0.0, 0.0, 0.0 });
  cam_destination->set_params({ intrinsics_dst(0,0), intrinsics_dst(1,1), intrinsics_dst(0,2), intrinsics_dst(1,2), 0.0, 0.0, 0.0 });

  // Normalize the point coordinate by the intrinsic matrices
  const int stride = 2 * 2 + 2 * 6;
  cv::Mat points(LAFs_right.size(), 4, CV_64F);
  cv::Mat normalized_points(LAFs_right.size(), stride, CV_64F);

  double* point_ptr = reinterpret_cast<double*>(normalized_points.data);

  for (int i = 0; i < LAFs_right.size(); ++i) {

    // points:
    points.at<double>(i, 0) = LAFs_left[i].x.x();
    points.at<double>(i, 1) = LAFs_left[i].x.y();
    points.at<double>(i, 2) = LAFs_right[i].x.x();
    points.at<double>(i, 3) = LAFs_right[i].x.y();

    // normalized:
    OneACPose::Vec3 src, dst;
    OneACPose::Mat32 src_diff, dst_diff;
    LAFs_left[i].as_3D(cam_source, src, src_diff);
    LAFs_right[i].as_3D(cam_destination, dst, dst_diff);
    Eigen::Map<OneACPose::Vec2>(point_ptr + 0).noalias() = src.head<2>();
    Eigen::Map<OneACPose::Vec2>(point_ptr + 2).noalias() = dst.head<2>();
    Eigen::Map<OneACPose::Mat32>(point_ptr + 4 + 0).noalias() = src_diff;
    Eigen::Map<OneACPose::Mat32>(point_ptr + 4 + 6).noalias() = dst_diff;

    point_ptr += stride;
  }

  // Normalize the threshold by the average of the focal lengths
  const double normalized_threshold =
    inlier_outlier_threshold_ / ((intrinsics_src(0, 0) + intrinsics_src(1, 1) +
      intrinsics_dst(0, 0) + intrinsics_dst(1, 1)) / 4.0);

  using namespace gcransac;

  // Initialize the neighborhood used in Graph-cut RANSAC and, perhaps,
  // in the sampler if NAPSAC or Progressive-NAPSAC sampling is applied.
  std::chrono::time_point<std::chrono::system_clock> start, end; // Variables for time measurement
  start = std::chrono::system_clock::now(); // The starting time of the neighborhood calculation

  neighborhood::GridNeighborhoodGraph neighborhood(&points,
    im_left_shape[1] / static_cast<double>(cell_number_in_neighborhood_graph_),
    im_left_shape[0] / static_cast<double>(cell_number_in_neighborhood_graph_),
    im_right_shape[1] / static_cast<double>(cell_number_in_neighborhood_graph_),
    im_right_shape[0] / static_cast<double>(cell_number_in_neighborhood_graph_),
    cell_number_in_neighborhood_graph_);
  std::cout << "neighborhood.getNeighborNumber() = " << neighborhood.getNeighborNumber() << std::endl;
  end = std::chrono::system_clock::now(); // The end time of the neighborhood calculation
  std::chrono::duration<double> elapsed_seconds = end - start; // The elapsed time in seconds
  printf("Neighborhood calculation time = %f secs\n", elapsed_seconds.count());

  // Checking if the neighborhood graph is initialized successfully.
  if (!neighborhood.isInitialized())
  {
    fprintf(stderr, "The neighborhood graph is not initialized successfully.\n");
    return false;
  }
  
  // The 1AC+D estimator for essential matrix fitting
  typedef gcransac::estimator::EssentialMatrixEstimator<gcransac::estimator::solver::PoseOneAffineAndRelativeDepthSolverMinimalSVD, // The solver used for fitting a model to a minimal sample
    gcransac::estimator::solver::FundamentalMatrixEightPointSolver> // The solver used for fitting a model to a non-minimal sample
    OneACD_EssentialMatrixEstimator;

  // Apply Graph-cut RANSAC
  OneACD_EssentialMatrixEstimator estimator(intrinsics_src, intrinsics_dst);

  EssentialMatrix model;

  // Initialize the samplers
  sampler::UniformSampler main_sampler(&points);
  sampler::UniformSampler local_optimization_sampler(&points); // The local optimization sampler is used inside the local optimization

                                                               // Checking if the samplers are initialized successfully.
  if (!main_sampler.isInitialized() ||
    !local_optimization_sampler.isInitialized())
  {
    fprintf(stderr, "One of the samplers is not initialized successfully.\n");
    return false;
  }

  GCRANSAC<OneACD_EssentialMatrixEstimator, neighborhood::GridNeighborhoodGraph> gcransac;
  gcransac.setFPS(fps_); // Set the desired FPS (-1 means no limit)
  gcransac.settings.threshold = normalized_threshold; // The inlier-outlier threshold
  gcransac.settings.spatial_coherence_weight = spatial_coherence_weight_; // The weight of the spatial coherence term
  gcransac.settings.confidence = confidence_; // The required confidence in the results
  gcransac.settings.max_local_optimization_number = 50; // The maximum number of local optimizations
  gcransac.settings.max_iteration_number = 5000; // The maximum number of iterations
  gcransac.settings.min_iteration_number = 50; // The minimum number of iterations
  gcransac.settings.neighborhood_sphere_radius = cell_number_in_neighborhood_graph_; // The radius of the neighborhood ball
  gcransac.settings.core_number = std::thread::hardware_concurrency(); // The number of parallel processes

  // ==========================================================================
  // estimate relative pose using GCRANSAC
  // ==========================================================================

  // Start GC-RANSAC
  gcransac.run(normalized_points,
    estimator,
    &main_sampler,
    &local_optimization_sampler,
    &neighborhood,
    model);

  // Get the statistics of the results
  const utils::RANSACStatistics& statistics = gcransac.getRansacStatistics();

  // Print the statistics
  if (verbose) {
    printf("Elapsed time = %f secs\n", statistics.processing_time);
    printf("Inlier number = %d\n", static_cast<int>(statistics.inliers.size()));
    printf("Applied number of local optimizations = %d\n", static_cast<int>(statistics.local_optimization_number));
    printf("Applied number of graph-cuts = %d\n", static_cast<int>(statistics.graph_cut_number));
    printf("Number of iterations = %d\n\n", static_cast<int>(statistics.iteration_number));
  }

  // Pass results
  inliers = statistics.inliers;
  essential_matrix.noalias() = model.descriptor.block<3, 3>(0, 0);

  return true;
}

}