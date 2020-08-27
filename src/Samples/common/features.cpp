#include "features.hpp"

#include <opencv2/opencv.hpp>


#include <cnpy/cnpy.h>

using namespace OneACPose;
using namespace VlFeatExtraction;

namespace common {

  bool extract_and_match_LAFs(
    const VlFeatExtraction::SiftExtractionOptions& options_f,
    const VlFeatExtraction::SiftMatchingOptions& options_m,
    const std::string& im_left, const std::string& im_right,
    const std::string& depth_left, const std::string& depth_right,
    common::LAFs& LAFs_left, common::LAFs& LAFs_right,
    std::vector<size_t>* im_left_shape, std::vector<size_t>* im_right_shape,
    bool dump_extracted_featues_to_disk)
  {
    if (dump_extracted_featues_to_disk) {
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
        LAFs_left.clear();
        LAFs_right.clear();
      }
    }

    FeatureKeypoints fkp_left, fkp_right;
    FeatureDescriptors dsc_left, dsc_right;

    // Feature extraction from a path-to-an-image
    auto extract_by_filename = [](
      const std::string& filename,
      const SiftExtractionOptions& options,
      FeatureKeypoints* keypoints,
      FeatureDescriptors* descriptors,
      std::vector<size_t>* im_shape) {

        // Read Image
        cv::Mat im = cv::imread(filename, 0);

        if (im.empty()) {
          std::cerr << "Failed to read image " << filename << std::endl;
          return false;
        }

        cv::Mat imF;
        im.convertTo(imF, CV_32F, 1.0 / 255.0);

        if (im_shape) {
          im_shape->resize(2);
          im_shape->at(0) = imF.rows;
          im_shape->at(1) = imF.cols;
        }

        // Perform extraction
        return extract(imF.ptr<float>(), imF.cols, imF.rows, options, keypoints, descriptors);
    };

    std::cout << "Extracting features (" << im_left << ")" << std::endl;
    bool success1 = extract_by_filename(im_left, options_f, &fkp_left, &dsc_left, im_left_shape);

    std::cout << "Extracting features (" << im_right << ")" << std::endl;
    bool success2 = extract_by_filename(im_right, options_f, &fkp_right, &dsc_right, im_right_shape);

    if (success1 && success2) {

      // Compute matches
      FeatureMatches matches;

      std::cout << "Matching features" << std::endl;
      MatchSiftFeaturesCPU(options_m, dsc_left, dsc_right, &matches);
      std::cout << "Number of matches: " << matches.size() << std::endl;

      // Extract depth for feature locations
      bool success_depth_augmentation = augment_LAFs_with_depth(
        fkp_left, fkp_right, matches,
        depth_left, depth_right,
        LAFs_left, LAFs_right);

      if (!success_depth_augmentation) {
        std::cerr << "Failed to extract depth values for feature locations" << std::endl;
        return false;
      }

      if (dump_extracted_featues_to_disk) {
        cnpy::npy_save<common::Feature_LAF_D>(im_left + ".lafs", LAFs_left);
        cnpy::npy_save<common::Feature_LAF_D>(im_right + ".lafs", LAFs_right);
      }

      return true;
    }
    else {
      std::cerr << "failed to extract features" << std::endl;
      return false;
    }
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

  bool augment_LAFs_with_depth(
    const VlFeatExtraction::FeatureKeypoints& fkp_left,
    const VlFeatExtraction::FeatureKeypoints& fkp_right,
    const VlFeatExtraction::FeatureMatches& matches,
    const std::string& depth_left, const std::string& depth_right,
    common::LAFs& LAFs_left,
    common::LAFs& LAFs_right)
  {
    try {
      cnpy::NpyArray arr1 = cnpy::npy_load(depth_left);
      cnpy::NpyArray arr2 = cnpy::npy_load(depth_right);

      // arr1.shape ~ rows, cols ~ height, width
      cv::Mat depth1(arr1.shape[0], arr1.shape[1], CV_64F, arr1.data<double>());
      cv::Mat depth2(arr2.shape[0], arr2.shape[1], CV_64F, arr2.data<double>());

      LAFs_left.reserve(matches.size());
      LAFs_right.reserve(matches.size());

      for (int i = 0; i < matches.size(); ++i) {
        double lambda;
        RowVec2 dlambda_dx;

        float step_size = 0.9;

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
    }
    catch (const std::runtime_error& error) {
      std::cerr << error.what() << std::endl;
      return false;
    }
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

}