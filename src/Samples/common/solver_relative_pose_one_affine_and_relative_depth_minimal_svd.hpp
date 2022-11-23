#pragma once

//#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <complex>

#include <opencv2/core.hpp>
//#include "pose.h"
//#include "pose_utils.h"
//#include "solver_engine.h"
//#include "GaussJordan.hpp"
#include <graph-cut-ransac/src/pygcransac/include/solver_engine.h>
//#include <glog/logging.h>

#include <OneACPose/solver_one_ac_depth.hpp>
#include <Samples/common/numeric.hpp>

namespace gcransac
{
  namespace estimator
  {
    namespace solver
    {
      inline Eigen::Matrix3d getEssentialMatrixFromRelativePose(
        const Eigen::Matrix3d& R_dst_src, const Eigen::Vector3d& t_dst_src) {

        // The cross product matrix of the translation vector
        Eigen::Matrix3d cross_prod_t_dst_src;
        cross_prod_t_dst_src << 0, -t_dst_src(2), t_dst_src(1), t_dst_src(2), 0, -t_dst_src(0),
          -t_dst_src(1), t_dst_src(0), 0;

        return cross_prod_t_dst_src * R_dst_src;
      }

      class PoseOneAffineAndRelativeDepthSolverMinimalSVD : public SolverEngine
      {
      protected:
        Eigen::Matrix3d intrinsics_source_inverse,
          intrinsics_destination_inverse;

      public:
        void initialize(Eigen::Matrix3d intrinsics_source_,
          Eigen::Matrix3d intrinsics_destination_);

        // The minimum number of points required for the estimation
        static constexpr size_t sampleSize()
        {
          return 1;
        }

        static constexpr const char* name()
        {
          return "Minimal SVD Solver";
        }

        // Determines if there is a chance of returning multiple models
        // the function 'estimateModel' is applied.
        static constexpr bool returnMultipleModels()
        {
          return maximumSolutions() > 1;
        }

        static constexpr size_t maximumSolutions()
        {
          return 1;
        }

        // Estimate the model parameters from the given point sample
        // using weighted fitting if possible.
        inline bool estimateModel(
          const cv::Mat& data_, // The set of data points
          const size_t* sample_, // The sample used for the estimation
          size_t sample_number_, // The size of the sample
          std::vector<Model>& models_, // The estimated model parameters
          const double* weights_ = nullptr) const; // The weight for each point

      };

      void PoseOneAffineAndRelativeDepthSolverMinimalSVD::initialize(
        Eigen::Matrix3d intrinsics_source_,
        Eigen::Matrix3d intrinsics_destination_)
      {
        intrinsics_source_inverse = intrinsics_source_.inverse();
        intrinsics_destination_inverse = intrinsics_destination_.inverse();
      }

      inline bool PoseOneAffineAndRelativeDepthSolverMinimalSVD::estimateModel(
        const cv::Mat& data_, // The set of data points
        const size_t* sample_, // The sample used for the estimation
        size_t sample_number_, // The size of the sample
        std::vector<Model>& models_, // The estimated model parameters
        const double* weights_) const // The weight for each point
      {
        const size_t N = sample_number_;
        if (N < sampleSize()) {
          //LOG(INFO)
          std::cerr
            << "You must supply at least 1 correspondences for the linear 1 point + depth essential matrix algorithm." << std::endl;
          return false;
        }

        // Step 1. Check if there enough correspondences with valid depths (i.e. >0)
        const double* data_ptr = reinterpret_cast<double*>(data_.data);
        const size_t cols = data_.cols;
        std::vector<size_t> good_point_indices;
        good_point_indices.reserve(N);

        for (size_t i = 0; i < N; i++)
        {
          size_t idx;
          if (sample_ == nullptr)
            idx = i * cols;
          else
            idx = sample_[i] * cols;

          /*if (*(data_ptr + idx + 4) < std::numeric_limits<double>::epsilon() ||
            *(data_ptr + idx + 5) < std::numeric_limits<double>::epsilon())
            continue;*/

            // TODO: azt is meg lehetne csinálni (túlhatározott esetben), hogy ha a mélységderiváltak rosszak, akkor azok az egyenletek kimaradnak projectionConstraint-ből
          good_point_indices.push_back(idx);
        }

        if (good_point_indices.size() < sampleSize()) {
          //LOG(INFO)
          //	<< "You must supply at least 1 correspondences for the linear 1 point + depth essential matrix algorithm.";
          return false;
        }

        OneACPose::Vec3 src, dst;
        OneACPose::Mat32 src_diff, dst_diff;
        {
          const double* point_ptr = data_ptr + good_point_indices[0];

          using namespace OneACPose;
          src = MVec3dc(point_ptr); point_ptr += MVec3dc::SizeAtCompileTime;
          dst = MVec3dc(point_ptr); point_ptr += MVec3dc::SizeAtCompileTime;
          src_diff = MMat32dc(point_ptr); point_ptr += MMat32dc::SizeAtCompileTime;
          dst_diff = MMat32dc(point_ptr); point_ptr += MMat32dc::SizeAtCompileTime;
        }

        // Step 3. Compute rotation, scaling and translation
        Eigen::Matrix3d R;
        Eigen::Vector3d t;
        double scale;

        // estimate pose using 1AC+D
        OneACPose::OneACD(
          dst, src,
          dst_diff, src_diff,
          scale, R, t);

        Eigen::Matrix3d essential_matrix;

        if (R.hasNaN() || t.hasNaN())
          return false;

        //essential_matrix.noalias() = common::cross_product(t) * R;
        essential_matrix.noalias() = getEssentialMatrixFromRelativePose(R, t);

        if (R.hasNaN() || t.hasNaN())
          return false;

        Eigen::Vector3d scales;
        scales << 1, scale, 0;

        Model model;
        model.descriptor = Eigen::Matrix<double, 3, 3>();
        model.descriptor.block<3, 3>(0, 0) = essential_matrix;
        /*model.descriptor.block<3, 3>(0, 3) = R;
        model.descriptor.block<3, 1>(0, 6) = t;
        model.descriptor.block<3, 1>(0, 7) = scales;*/

        models_.reserve(1);
        models_.emplace_back(model);

        return true;
      }
    }
  }
}

