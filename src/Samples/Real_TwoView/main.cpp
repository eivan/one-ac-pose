#include <iostream>
#include <time.h>
#include <chrono>

#include <common/numeric.hpp>
#include <common/camera_radial.hpp>
#include <common/local_affine_frame.hpp>
#include <common/pose_estimation.hpp>

#include <cnpy/cnpy.h>

#include <opencv2/opencv.hpp>

#include <VlFeatExtraction/Extraction.hpp>
#include <VlFeatExtraction/Matching.hpp>

#include <graph-cut-ransac/src/pygcransac/include/essential_estimator.h>
#include <graph-cut-ransac/src/pygcransac/include/fundamental_estimator.h>
#include <graph-cut-ransac/src/pygcransac/include/progressive_napsac_sampler.h>
#include "solver_relative_pose_one_affine_and_relative_depth_minimal_svd.hpp"
#include <thread>

using namespace OneACPose;
using namespace VlFeatExtraction;

bool extract_and_match_LAFs(
  const std::string& im_left, const std::string& im_right,
  const std::string& depth_left, const std::string& depth_right,
  common::LAFs& LAFs_left, common::LAFs& LAFs_right,
  std::vector<size_t>* im_left_shape = nullptr,
  std::vector<size_t>* im_right_shape = nullptr);

// The default estimator for essential matrix fitting
typedef gcransac::estimator::EssentialMatrixEstimator<gcransac::estimator::solver::PoseOneAffineAndRelativeDepthSolverMinimalSVD, // The solver used for fitting a model to a minimal sample
  gcransac::estimator::solver::FundamentalMatrixEightPointSolver> // The solver used for fitting a model to a non-minimal sample
  OneACD_EssentialMatrixEstimator;

bool estimatePose_1ACD_GCRANSAC(
  const Mat3& intrinsics_src, const Mat3& intrinsics_dst,
  const common::LAFs& LAFs_left, const common::LAFs& LAFs_right,
  const std::vector<size_t>& im_left_shape,
  const std::vector<size_t>& im_right_shape,
  const size_t& cell_number_in_neighborhood_graph_,
  const int& fps_,
  const double& spatial_coherence_weight_,
  const double& confidence_,
  const double& inlier_outlier_threshold_,
  std::vector<size_t>& inliers,
  Mat3& essential_matrix,
  const bool verbose = true);

void display_inliers(
  const std::string& im_left, const std::string& im_right,
  const common::LAFs& LAFs_left, const common::LAFs& LAFs_right,
  const std::vector<size_t>& inliers);

int main(int argc, char** argv)
{
  if (argc != 3) {
    std::cerr << "Insufficient number of commandline parameters" << std::endl;
    exit(1);
  }

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

double sample_bilinear(const cv::Mat& img, float x, float y)
{
  assert(!img.empty());
  assert(img.channels() == 3);

  const int x_i = (int)x;
  const int y_i = (int)y;

  const int x0 = cv::borderInterpolate(x_i, img.cols, cv::BORDER_REFLECT_101);
  const int x1 = cv::borderInterpolate(x_i + 1, img.cols, cv::BORDER_REFLECT_101);
  const int y0 = cv::borderInterpolate(y_i, img.rows, cv::BORDER_REFLECT_101);
  const int y1 = cv::borderInterpolate(y_i + 1, img.rows, cv::BORDER_REFLECT_101);

  const double a = x - (double)x_i;
  const double c = y - (double)y_i;

  return (img.at<double>(y0, x0) * (1 - a) + img.at<double>(y0, x1) * a) * (1 - c)
    + (img.at<double>(y1, x0) * (1 - a) + img.at<double>(y1, x1) * a) * c;
}

bool extract_and_match_LAFs(
  const std::string& im_left, const std::string& im_right,
  const std::string& depth_left, const std::string& depth_right,
  common::LAFs& LAFs_left, common::LAFs& LAFs_right,
  std::vector<size_t>* im_left_shape, std::vector<size_t>* im_right_shape)
{
  try {
    cnpy::NpyArray arr1 = cnpy::npy_load(im_left + ".lafs");
    common::Feature_LAF_D* data1 = arr1.data<common::Feature_LAF_D>();
    LAFs_left.insert(LAFs_left.end(), data1, data1 + arr1.shape[0]);

    cnpy::NpyArray arr2 = cnpy::npy_load(im_right + ".lafs");
    common::Feature_LAF_D* data2 = arr2.data<common::Feature_LAF_D>();
    LAFs_right.insert(LAFs_right.end(), data2, data2 + arr2.shape[0]);

    cnpy::NpyArray arr1d = cnpy::npy_load(depth_left);
    cnpy::NpyArray arr2d = cnpy::npy_load(depth_right);
    if (im_left_shape) {
      *im_left_shape = arr1d.shape;
    }
    if (im_right_shape) {
      *im_right_shape = arr2d.shape;
    }

    return true;
  }
  catch (const std::runtime_error& error) {
    std::cout << error.what() << std::endl;
  }

  FeatureKeypoints fkp_left, fkp_right;
  FeatureDescriptors dsc_left, dsc_right;

  SiftExtractionOptions options_f;
  SiftMatchingOptions options_m;

  options_f.estimate_affine_shape = true;
  options_f.domain_size_pooling = false;

  // Feature extraction from a path-to-an-image
  auto extract_by_filename = [](
    const std::string& filename,
    const SiftExtractionOptions& options,
    FeatureKeypoints* keypoints,
    FeatureDescriptors* descriptors) {

      // Read Image
      cv::Mat im = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

      if (im.empty()) {
        std::cerr << "Failed to read image " << filename << std::endl;
        return false;
      }

      cv::Mat imF;
      im.convertTo(imF, CV_32F, 1.0 / 255.0);

      // Perform extraction
      return extract(imF.ptr<float>(), imF.cols, imF.rows, options, keypoints, descriptors);
  };

  std::cout << "Extracting features (" << im_left << ")" << std::endl;
  bool success1 = extract_by_filename(im_left, options_f, &fkp_left, &dsc_left);

  std::cout << "Extracting features (" << im_right << ")" << std::endl;
  bool success2 = extract_by_filename(im_right, options_f, &fkp_right, &dsc_right);

  if (success1 && success2) {

    FeatureMatches matches;

    std::cout << "Matching features" << std::endl;
    MatchSiftFeaturesCPU(options_m, dsc_left, dsc_right, &matches);

    std::cout << "Number of matches: " << matches.size() << std::endl;

    LAFs_left.reserve(matches.size());
    LAFs_right.reserve(matches.size());

    cnpy::NpyArray arr1 = cnpy::npy_load(depth_left);
    cnpy::NpyArray arr2 = cnpy::npy_load(depth_right);
    // arr1.shape ~ rows, cols ~ height, width
    cv::Mat depth1(arr1.shape[0], arr1.shape[1], CV_64F, arr1.data<double>());
    cv::Mat depth2(arr2.shape[0], arr2.shape[1], CV_64F, arr2.data<double>());

    if (im_left_shape) {
      *im_left_shape = arr1.shape;
    }

    if (im_right_shape) {
      *im_right_shape = arr2.shape;
    }

    for (int i = 0; i < matches.size(); ++i) {
      double lambda;
      RowVec2 dlambda_dx;

      float step_size = 0.85;

      // Left image and depth
      const int match_l = matches[i].point2D_idx1;
      const auto& f_l = fkp_left[match_l];
      lambda =
        sample_bilinear(depth1, f_l.x, f_l.y);
      dlambda_dx[0] =
        sample_bilinear(depth1, f_l.x + f_l.a11 * step_size, f_l.y + f_l.a21 * step_size) -
        sample_bilinear(depth1, f_l.x - f_l.a11 * step_size, f_l.y - f_l.a21 * step_size);
      dlambda_dx[1] =
        sample_bilinear(depth1, f_l.x + f_l.a12 * step_size, f_l.y + f_l.a22 * step_size) -
        sample_bilinear(depth1, f_l.x - f_l.a12 * step_size, f_l.y - f_l.a22 * step_size);
      LAFs_left.emplace_back(
        (Vec2() << f_l.x, f_l.y).finished(),
        (Mat2() << f_l.a11, f_l.a12, f_l.a21, f_l.a22).finished(),
        lambda, dlambda_dx);

      // Right image and depth
      const int match_r = matches[i].point2D_idx2;
      const auto& f_r = fkp_right[match_r];
      lambda =
        sample_bilinear(depth2, f_r.x, f_r.y);
      dlambda_dx[0] =
        sample_bilinear(depth2, f_r.x + f_r.a11 * step_size, f_r.y + f_r.a21 * step_size) -
        sample_bilinear(depth2, f_r.x - f_r.a11 * step_size, f_r.y - f_r.a21 * step_size);
      dlambda_dx[1] =
        sample_bilinear(depth2, f_r.x + f_r.a12 * step_size, f_r.y + f_r.a22 * step_size) -
        sample_bilinear(depth2, f_r.x - f_r.a12 * step_size, f_r.y - f_r.a22 * step_size);
      LAFs_right.emplace_back(
        (Vec2() << f_r.x, f_r.y).finished(),
        (Mat2() << f_r.a11, f_r.a12, f_r.a21, f_r.a22).finished(),
        lambda, dlambda_dx);

    }

    cnpy::npy_save<common::Feature_LAF_D>(im_left + ".lafs", LAFs_left);
    cnpy::npy_save<common::Feature_LAF_D>(im_right + ".lafs", LAFs_right);

    return true;
  }
  else {
    std::cerr << "failed to extract features" << std::endl;
    return false;
  }
}

bool estimatePose_1ACD_GCRANSAC(
  const Mat3& intrinsics_src, const Mat3& intrinsics_dst,
  const common::LAFs& LAFs_left, const common::LAFs& LAFs_right,
  const std::vector<size_t>& im_left_shape,
  const std::vector<size_t>& im_right_shape,
  const size_t& cell_number_in_neighborhood_graph_,
  const int& fps_,
  const double& spatial_coherence_weight_,
  const double& confidence_,
  const double& inlier_outlier_threshold_,
  std::vector<size_t>& inliers,
  Mat3& essential_matrix,
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

void display_inliers(
  const std::string& im_left, const std::string& im_right,
  const common::LAFs& LAFs_left, const common::LAFs& LAFs_right,
  const std::vector<size_t>& inliers)
{
  // Keypoints
  std::vector<cv::KeyPoint> keypoints_left;
  std::vector<cv::KeyPoint> keypoints_right;

  auto convertToCV = [](std::vector<cv::KeyPoint>& keypointsCV, const common::LAFs& keypoints) {
    keypointsCV.clear();
    keypointsCV.reserve(keypoints.size());
    for (const auto& kp : keypoints) {
      keypointsCV.emplace_back((float)kp.x.x(), (float)kp.x.y(), 0.f);
    }
  };

  convertToCV(keypoints_left, LAFs_left);
  convertToCV(keypoints_right, LAFs_right);

  // Matches
  std::vector<cv::DMatch> matches_to_draw;
  matches_to_draw.reserve(inliers.size());
  for (unsigned int i = 0; i < inliers.size(); i++) {
    matches_to_draw.emplace_back(inliers[i], inliers[i], 0.f);
  }

  // Draw matches/inliers
  auto cv_image_left = cv::imread(im_left);
  auto cv_image_right = cv::imread(im_right);

  cv::Mat output = cv::Mat::zeros(std::max(cv_image_left.rows, cv_image_right.rows), cv_image_left.cols + cv_image_right.cols, cv_image_left.type());
  cv::drawMatches(cv_image_left, keypoints_left, cv_image_right, keypoints_right, matches_to_draw, output);

  cv::imshow("GCRANSAC result", output);
  cv::waitKey(0);
}