#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>

#include <common/pose_estimation.hpp>
#include <common/features.hpp>
#include <opencv2/core.hpp>

namespace py = pybind11;

py::tuple estimatePose_1ACD_GCRANSAC(
  py::array_t<double>  LAFs_left,
  py::array_t<double>  LAFs_right,
  py::array_t<double>  intrinsics_src,
  py::array_t<double>  intrinsics_dst,
  py::array_t<size_t>  im_left_shape,
  py::array_t<size_t>  im_right_shape,
  size_t cell_number_in_neighborhood_graph,
  int fps,
  double spatial_coherence_weight,
  double confidence,
  double inlier_outlier_threshold) {

  // ==========================================================================
  // LAFs
  // ==========================================================================
  py::buffer_info buf_LAFs_left = LAFs_left.request();
  py::buffer_info buf_LAFs_right = LAFs_right.request();
  const size_t num_features_left = buf_LAFs_left.shape[0];
  const size_t num_features_right = buf_LAFs_right.shape[0];
  const size_t num_matches = num_features_left;
  if (num_features_left != num_features_right) {
    throw std::invalid_argument("LAFs_left does not have the same number of rows as LAFs_right");
  }
  if (num_features_left < 7) {
    throw std::invalid_argument("LAFs_left and LAFs_right should have at least 7 rows");
  }
  if (buf_LAFs_left.shape[1] != 9 || buf_LAFs_right.shape[1] != 9) {
    throw std::invalid_argument("LAFs_left and LAFs_right both should have 9 columns");
  }
  // (params)
  common::LAFs LAFs_left_;
  common::LAFs LAFs_right_;
  auto ptr_LAFs_left = static_cast<common::Feature_LAF_D*>(buf_LAFs_left.ptr);
  auto ptr_LAFs_right = static_cast<common::Feature_LAF_D*>(buf_LAFs_right.ptr);
  LAFs_left_.assign(ptr_LAFs_left, ptr_LAFs_left + num_matches);
  LAFs_right_.assign(ptr_LAFs_right, ptr_LAFs_right + num_matches);

  // ==========================================================================
  // intrinsics
  // ==========================================================================
  py::buffer_info buf_intrinsics_src = intrinsics_src.request();
  py::buffer_info buf_intrinsics_dst = intrinsics_dst.request();
  if (buf_intrinsics_src.shape[0] != 3 && buf_intrinsics_src.shape[1] != 3) {
    throw std::invalid_argument("intrinsics_src should be a 3-by-3 array");
  }
  if (buf_intrinsics_dst.shape[0] != 3 && buf_intrinsics_dst.shape[1] != 3) {
    throw std::invalid_argument("intrinsics_dst should be a 3-by-3 array");
  }

  // ==========================================================================
  // shapes
  // ==========================================================================
  py::buffer_info buf_im_left_shape = im_left_shape.request();
  py::buffer_info buf_im_right_shape = im_right_shape.request();
  if (buf_im_left_shape.shape[0] != 2 && buf_im_left_shape.shape[1] != 1) {
    throw std::invalid_argument("im_left_shape should be a 2-by-1 array");
  }
  if (buf_im_right_shape.shape[0] != 2 && buf_im_right_shape.shape[1] != 1) {
    throw std::invalid_argument("im_right_shape should be a 2-by-1 array");
  }
  // (params)
  std::vector<size_t> im_left_shape_;
  std::vector<size_t> im_right_shape_;
  auto ptr_im_left_shape = static_cast<size_t*>(buf_im_left_shape.ptr);
  auto ptr_im_right_shape = static_cast<size_t*>(buf_im_right_shape.ptr);
  im_left_shape_.assign(ptr_im_left_shape, ptr_im_left_shape + 2);
  im_right_shape_.assign(ptr_im_right_shape, ptr_im_right_shape + 2);

  // ==========================================================================
  // call
  // ==========================================================================
  std::vector<size_t> inliers;
  OneACPose::Mat3 essential_matrix;

  bool success = common::estimatePose_1ACD_GCRANSAC(
    Eigen::Map<OneACPose::Mat3>(static_cast<double*>(buf_intrinsics_src.ptr)),
    Eigen::Map<OneACPose::Mat3>(static_cast<double*>(buf_intrinsics_dst.ptr)),
    LAFs_left_, LAFs_right_,
    im_left_shape_, im_right_shape_,
    cell_number_in_neighborhood_graph,
    fps,
    spatial_coherence_weight,
    confidence,
    inlier_outlier_threshold,
    inliers,
    essential_matrix,
    false
  );

  // ==========================================================================
  // return
  // ==========================================================================
  py::array_t<bool> inliers_ = py::array_t<bool>(num_matches);

  if (!success || inliers.size() == 0) {
    // return
    return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), inliers_);
  }
  else {
    // inliers
    py::buffer_info buf_inliers_ = inliers_.request();
    auto ptr_buf_inliers_ = static_cast<bool*>(buf_inliers_.ptr);
    for (size_t i = 0; i < num_matches; i++) {
      ptr_buf_inliers_[i] = false;
    }
    for (const auto& inlier : inliers) {
      ptr_buf_inliers_[inlier] = true;
    }

    // essential matrix
    py::array_t<double> essential_matrix_ = py::array_t<double>({ 3,3 });
    py::buffer_info buf_essential_matrix_ = essential_matrix_.request();
    auto ptr_essential_matrix_ = static_cast<double*>(buf_essential_matrix_.ptr);
    
    essential_matrix.transposeInPlace(); // rows -> columns
    for (size_t i = 0; i < 9; i++) {
      ptr_essential_matrix_[i] = essential_matrix.data()[i];
    }

    // return
    return py::make_tuple(essential_matrix_, inliers_);
  }
}

py::tuple extract_LAFs(
  py::array_t<float> image_float,
  int max_num_features = 8192,
  int first_octave = -1,
  int num_octaves = 4,
  int octave_resolution = 3,
  bool domain_size_pooling = false
) {
  // TODO: size checks

  VlFeatExtraction::SiftExtractionOptions options_f;
  options_f.estimate_affine_shape = true;
  options_f.max_num_features = max_num_features;
  options_f.first_octave = first_octave;
  options_f.num_octaves = num_octaves;
  options_f.octave_resolution = octave_resolution;
  options_f.domain_size_pooling = domain_size_pooling;

  py::buffer_info buf_image_float = image_float.request();

  VlFeatExtraction::FeatureKeypoints keypoints;
  VlFeatExtraction::FeatureDescriptors descriptors;

  bool success = VlFeatExtraction::extract(
    static_cast<float*>(buf_image_float.ptr),
    buf_image_float.shape[1],
    buf_image_float.shape[0],
    options_f,
    &keypoints, &descriptors);

  if (!success || keypoints.size() == 0) {
    return py::make_tuple(pybind11::cast<pybind11::none>(Py_None), pybind11::cast<pybind11::none>(Py_None));
  }
  else {
    // keypoints
    py::array_t<float> keypoints_ = py::array_t<float>({ keypoints.size(), 6 });
    auto buf_keypoints_ = keypoints_.request();
    auto ptr_keypoints_src = static_cast<float*>(&(keypoints[0].x));
    auto ptr_keypoints_dst = static_cast<float*>(buf_keypoints_.ptr);
    std::copy(ptr_keypoints_src, ptr_keypoints_src + 6 * keypoints.size(), ptr_keypoints_dst);

    // descriptors
    py::array_t<uint8_t> descriptors_ = py::array_t<uint8_t>({ (size_t)descriptors.rows(), (size_t)descriptors.cols() });
    auto buf_descriptors_ = descriptors_.request();
    auto ptr_descriptors_src = descriptors.data();
    auto ptr_descriptors_dst = static_cast<uint8_t*>(buf_descriptors_.ptr);
    std::copy(ptr_descriptors_src, ptr_descriptors_src + descriptors.rows() * descriptors.cols(), ptr_descriptors_dst);

    return py::make_tuple(keypoints_, descriptors_);
  }
}

py::tuple match_LAFs(
  py::array_t<uint8_t> descriptors_left,
  py::array_t<uint8_t> descriptors_right,
  bool sort_matches_by_score = true
) {
  // TODO: size checks

  VlFeatExtraction::FeatureMatches matches;
  VlFeatExtraction::SiftMatchingOptions options_m;
  VlFeatExtraction::FeatureDescriptors descriptors_left_;
  VlFeatExtraction::FeatureDescriptors descriptors_right_;

  {
    auto buf_descriptors_left = descriptors_left.request();
    descriptors_left_.resize(buf_descriptors_left.shape[0], buf_descriptors_left.shape[1]);
    auto ptr_descriptors_left = static_cast<uint8_t*>(buf_descriptors_left.ptr);
    std::copy(ptr_descriptors_left, ptr_descriptors_left + descriptors_left_.size(), descriptors_left_.data());
  }
  {
    auto buf_descriptors_right = descriptors_right.request();
    descriptors_right_.resize(buf_descriptors_right.shape[0], buf_descriptors_right.shape[1]);
    auto ptr_descriptors_right = static_cast<uint8_t*>(buf_descriptors_right.ptr);
    std::copy(ptr_descriptors_right, ptr_descriptors_right + descriptors_right_.size(), descriptors_right_.data());
  }

  VlFeatExtraction::MatchSiftFeaturesCPU(
    options_m,
    descriptors_left_,
    descriptors_right_,
    &matches,
    sort_matches_by_score);

  const size_t num_matches = matches.size();

  // first component of tuple: py::array_t<int> of matching indices
  py::array_t<int> matches_ = py::array_t<int>({ num_matches, 2 });
  auto buf_matches_ = matches_.request();
  auto ptr_matches_ = static_cast<int*>(buf_matches_.ptr);

  // second component of tuple: py::array_t<float> of scores
  py::array_t<float> scores_ = py::array_t<float>(num_matches);
  auto buf_scores_ = scores_.request();
  auto ptr_scores_ = static_cast<int*>(buf_scores_.ptr);

  for (size_t i = 0; i < num_matches; ++i) {
    ptr_matches_[2 * i + 0] = matches[i].point2D_idx1;
    ptr_matches_[2 * i + 1] = matches[i].point2D_idx2;
    ptr_scores_[i] = matches[i].score;
  }

  return py::make_tuple(matches_, scores_);
}

double sample_bilinear(const float* img, size_t rows, size_t cols, float x, float y)
{
  assert(!img.empty());
  assert(img.channels() == 3);

  const int x_i = (int)x;
  const int y_i = (int)y;

  const int x0 = cv::borderInterpolate(x_i, cols, cv::BORDER_REFLECT_101);
  const int x1 = cv::borderInterpolate(x_i + 1, cols, cv::BORDER_REFLECT_101);
  const int y0 = cv::borderInterpolate(y_i, rows, cv::BORDER_REFLECT_101);
  const int y1 = cv::borderInterpolate(y_i + 1, rows, cv::BORDER_REFLECT_101);

  const float a = x - (float)x_i;
  const float c = y - (float)y_i;

  auto at = [&img, &cols](int y, int x) {
    return img[y * cols + x];
  };

  return (at(y0,x0) * (1.f - a) + at(y0,x1) * a) * (1.f - c)
    + (at(y1,x0) * (1.f - a) + at(y1,x1) * a) * c;
}

py::array_t<double> augment_LAFs_with_depth(
  py::array_t<float> keypoints,
  py::array_t<float> depth_image
) {
  // TODO: size checks

  auto buf_keypoints = keypoints.request();
  auto ptr_keypoints = static_cast<VlFeatExtraction::FeatureKeypoint*>(buf_keypoints.ptr);
  const auto num_keypoints = buf_keypoints.shape[0];

  auto buf_depth_image = depth_image.request();
  auto ptr_depth_image = static_cast<float*>(buf_depth_image.ptr);
  const size_t rows = buf_depth_image.shape[0];
  const size_t cols = buf_depth_image.shape[1];

  py::array_t<double> LAFS_with_depth = py::array_t<double>({ num_keypoints, 9 });
  auto buf_LAFS_with_depth = LAFS_with_depth.request();
  auto ptr_LAFS_with_depth = static_cast<double*>(buf_LAFS_with_depth.ptr);

  for (size_t i = 0; i < num_keypoints; ++i) {
    double lambda;
    OneACPose::RowVec2 dlambda_dx;

    const float step_size = 0.9;

    // Left image and depth
    const auto& f_l = ptr_keypoints[i];
    lambda =
      sample_bilinear(ptr_depth_image, rows, cols, f_l.x, f_l.y);
    dlambda_dx[0] =
      sample_bilinear(ptr_depth_image, rows, cols, f_l.x + f_l.a11 * step_size, f_l.y + f_l.a21 * step_size) -
      sample_bilinear(ptr_depth_image, rows, cols, f_l.x - f_l.a11 * step_size, f_l.y - f_l.a21 * step_size);
    dlambda_dx[1] =
      sample_bilinear(ptr_depth_image, rows, cols, f_l.x + f_l.a12 * step_size, f_l.y + f_l.a22 * step_size) -
      sample_bilinear(ptr_depth_image, rows, cols, f_l.x - f_l.a12 * step_size, f_l.y - f_l.a22 * step_size);

    ptr_LAFS_with_depth[9 * i + 0] = f_l.x;
    ptr_LAFS_with_depth[9 * i + 1] = f_l.y;
    ptr_LAFS_with_depth[9 * i + 2] = f_l.a11;
    ptr_LAFS_with_depth[9 * i + 3] = f_l.a12;
    ptr_LAFS_with_depth[9 * i + 4] = f_l.a21;
    ptr_LAFS_with_depth[9 * i + 5] = f_l.a22;
    ptr_LAFS_with_depth[9 * i + 6] = lambda;
    ptr_LAFS_with_depth[9 * i + 7] = dlambda_dx[0];
    ptr_LAFS_with_depth[9 * i + 8] = dlambda_dx[1];
  }

  return LAFS_with_depth;
}

PYBIND11_PLUGIN(pyoneacpose) {

  py::module m("pygcransac", R"doc(
        Python module
        -----------------------
        .. currentmodule:: pyoneacpose
        .. autosummary::
           :toctree: _generate
           
           estimatePose_1ACD_GCRANSAC,
           extract_LAFs,
           match_LAFs,
           augment_LAFs_with_depth,

    )doc");

  // estimation

  m.def("estimatePose_1ACD_GCRANSAC", &estimatePose_1ACD_GCRANSAC, R"doc(some doc)doc",
    py::arg("LAFs_left"),
    py::arg("LAFs_right"),
    py::arg("intrinsics_src"),
    py::arg("intrinsics_dst"),
    py::arg("im_left_shape"),
    py::arg("im_right_shape"),
    py::arg("cell_number_in_neighborhood_graph") = 8,
    py::arg("fps") = -1,
    py::arg("spatial_coherence_weight") = 0.975,
    py::arg("confidence") = 0.99,
    py::arg("inlier_outlier_threshold") = 3.0
  );

  // features
  m.def("extract_LAFs", &extract_LAFs, R"doc(some doc)doc",
    py::arg("image_float"),
    py::arg("max_num_features") = 8192,
    py::arg("first_octave") = -1,
    py::arg("num_octaves") = 4,
    py::arg("octave_resolution") = 3,
    py::arg("domain_size_pooling") = false
  );

  m.def("match_LAFs", &match_LAFs, R"doc(some doc)doc",
    py::arg("descriptors_left"),
    py::arg("descriptors_right"),
    py::arg("sort_matches_by_score") = true
  );

  m.def("augment_LAFs_with_depth", &augment_LAFs_with_depth, R"doc(some doc)doc",
    py::arg("keypoints"),
    py::arg("depth_image")
  );

  return m.ptr();
}
