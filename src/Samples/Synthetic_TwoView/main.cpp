#include <iostream>
#include <time.h>

#include <OneACPose/solver_one_ac_depth.hpp>

#include <common/camera_radial.hpp>
#include <common/numeric.hpp>
#include <common/local_affine_frame.hpp>

using namespace OneACPose;

void estimatePose_1ACD(
  const common::CameraPtr& cam0,
  const common::CameraPtr& cam1,
  const common::Feature_LAF_D& laf0,
  const common::Feature_LAF_D& laf1,
  double& scale, OneACPose::Mat3& R, OneACPose::Vec3& t);

void compute_synthetic_LAF(
  const Mat34& P, const common::CameraPtr& cam,
  const Vec3& X, const Mat32& dX_dx,
  common::Feature_LAF_D& laf);

int main()
{
  srand(time(0));

  // ==========================================================================
  // initialize two poses
  // ==========================================================================
  const Vec3 target{ Vec3::Random() };
  //const Vec3 target{ Vec3::Zero() };
  const double distance_from_target = 5;

  // init pose 0
  const Vec3 C0{ Vec3::UnitX() * distance_from_target };
  const Mat3 R0{ common::LookAt(target - C0) };
  const Mat34 P0{ (Mat34() << R0, -R0 * C0).finished() };

  // init pose 1
  const Vec3 C1{ Vec3::UnitZ() * distance_from_target };
  const Mat3 R1{ common::LookAt(target - C1) };
  const Mat34 P1{ (Mat34() << R1, -R1 * C1).finished() };

  // ==========================================================================
  // initialize two cameras
  // ==========================================================================
  common::CameraPtr cam0 = std::make_shared<common::Camera_Radial>();
  cam0->set_params({ 1000.0, 1000.0, 500.0, 500.0, 0.0, 0.0, 0.0 });

  common::CameraPtr cam1 = std::make_shared<common::Camera_Radial>();
  cam1->set_params({ 1000.0, 1000.0, 500.0, 500.0, 0.0, 0.0, 0.0 });

  // ==========================================================================
  // initialize 3D structure
  // ==========================================================================
  // surface point
  const Vec3 X{ Vec3::Random() };
  // surface normal at X
  const Vec3 N{ Vec3::Random().normalized() };
  // local frame of surface around X (perpendicular to N)
  const Mat32 dX_dx{ common::nullspace(N) };

  // ==========================================================================
  // compute synthetic LAFs with depths, by projecting X and dX_dx onto the image plane
  // ==========================================================================
  common::Feature_LAF_D laf0;
  compute_synthetic_LAF(P0, cam0, X, dX_dx, laf0);

  common::Feature_LAF_D laf1;
  compute_synthetic_LAF(P1, cam1, X, dX_dx, laf1);

  // ==========================================================================
  // perform estimation
  // ==========================================================================

  double scale;
  Mat3 R;
  Vec3 t;

  estimatePose_1ACD(
    cam0, cam1, // calibrated cameras
    laf0, // LAF 0: 2d location, 2D shape, depth and depth derivatives
    laf1, // LAF 1, 2d location, 2D shape, depth and depth derivatives
    scale, R, t);

  // ==========================================================================
  // measure errors wrt ground truth
  // ==========================================================================
  std::cout << "R_gt:\t" << (R1 * R0.transpose()) << std::endl;
  std::cout << "R_est:\t" << R << std::endl;

  std::cout << "t_est:\t" << (t).transpose() << std::endl;
  std::cout << "t_gt:\t" << (R1 * (C0 - C1)).transpose() << std::endl;

  std::cout << "scale:\t" << scale << std::endl;
}

void compute_synthetic_LAF(
  const Mat34& P, const common::CameraPtr& cam,
  const Vec3& X, const Mat32& dX_dx,
  common::Feature_LAF_D& laf)
{
  const Vec3 Y = P * X.homogeneous();
  const Mat32 dY_dx = P.topLeftCorner<3, 3>() * dX_dx;

  Mat23 dx_dY;
  std::tie(laf.x.noalias(), dx_dY.noalias()) = cam->p_gradient(Y);
  // affine shape around x, aka dx0_dx
  laf.M.noalias() = dx_dY * dY_dx;

  RowVec3 dlambda_dY;
  std::tie(laf.lambda, dlambda_dY.noalias()) = cam->depth_gradient(Y);
  // aka dlambda_dx
  laf.dlambda_dx.noalias() = dlambda_dY * dY_dx;
}

void estimatePose_1ACD(
  const common::CameraPtr& cam0,
  const common::CameraPtr& cam1,
  const common::Feature_LAF_D& laf0,
  const common::Feature_LAF_D& laf1,
  double& scale, OneACPose::Mat3& R, OneACPose::Vec3& t)
{
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