#pragma once
#include "solver_one_ac_depth.hpp"

namespace OneACPose {

  void OneACD(
    const Vec3& dst, const Vec3& src, 
    const Mat32& dst_diff, const Mat32& src_diff, 
    double& scale, Mat3& R, Vec3& t)
  {
    // build a covariance matrix using the derivatives
    // then decompose it using SVD
    const Eigen::Matrix3d sigma = dst_diff * src_diff.transpose();
    auto svd = sigma.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto& sv = svd.singularValues();

    // construct a rotation using the left and right singular vectors
    // also compute the relative scale using the singular values
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) {
      R.noalias() = svd.matrixU() * Eigen::Vector3d(1, 1, -1).asDiagonal() * svd.matrixV().transpose();
      scale = (sv[0] + sv[1] - sv[2]) / src_diff.squaredNorm();
    }
    else {
      R.noalias() = svd.matrixU() * svd.matrixV().transpose();
      scale = (sv[0] + sv[1] + sv[2]) / src_diff.squaredNorm();
    }

    // finally, compute translation, knowing R and scale
    t.noalias() = dst - scale * R * src;
  }
  
  void OneACD_Umeyama(
    const Vec3& dst, const Vec3& src, 
    const Mat32& dst_diff, const Mat32& src_diff, 
    double& scale, Mat3& R, Vec3& t)
  {
    // build a covariance matrix using the derivatives
    // then decompose it using SVD
    const Eigen::Matrix3d sigma = dst_diff * src_diff.transpose();
    auto svd = sigma.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
    auto& sv = svd.singularValues();

    // construct a rotation using the left and right singular vectors
    // also compute the relative scale using the singular values
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) {
      R.noalias() = svd.matrixU() * Eigen::Vector3d(1, 1, -1).asDiagonal() * svd.matrixV().transpose();
      scale = (sv[0] + sv[1] - sv[2]) / src_diff.squaredNorm();
    }
    else {
      R.noalias() = svd.matrixU() * svd.matrixV().transpose();
      scale = (sv[0] + sv[1] + sv[2]) / src_diff.squaredNorm();
    }

    // finally, compute translation, knowing R and scale
    t.noalias() = dst - scale * R * src;
  }

}