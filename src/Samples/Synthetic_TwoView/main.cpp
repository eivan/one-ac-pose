#include <iostream>

#include <OneACPose/solver_one_ac_depth.hpp>

#include <common/camera_radial.hpp>
#include <common/numeric.hpp>
#include <time.h>

using namespace OneACPose;

int main()
{
  // ==========================================================================
  // initialize two poses
  // ==========================================================================
  const Vec3 target{ Vec3::Zero() };
  const double distance_from_target = 5;

  const Vec3 C0{ Vec3::UnitX() * distance_from_target };
  const Mat3 R0{ common::LookAt(target - C0) };
  const Mat34 P0{ (Mat34() << R0, -R0 * C0).finished() };

  const Vec3 C1{ Vec3::UnitZ() * distance_from_target };
  const Mat3 R1{ common::LookAt(target - C1) };
  const Mat34 P1{ (Mat34() << R1, -R1 * C1).finished() };

  std::shared_ptr<common::Camera> cam0 = std::make_shared<common::Camera_Radial>();
  cam0->set_params({ 1000.0, 1000.0, 500.0, 500.0, 0.0, 0.0, 0.0 });

  std::shared_ptr<common::Camera> cam1 = std::make_shared<common::Camera_Radial>();
  cam1->set_params({ 1000.0, 1000.0, 500.0, 500.0, 0.0, 0.0, 0.0 });

  // ==========================================================================
  // initialize 3D structure
  // ==========================================================================
  srand(time(0));
  const Vec3 X{ Vec3::Random() }; // surface point
  const Vec3 N{ Vec3::Random().normalized() }; // surface normal at X
  const Mat32 dX_dx{ common::nullspace(N) }; // local frame of surface around X (perpendicular to NÖ

  // ==========================================================================
  // project feats with gradients
  // ==========================================================================

  // Transform X into camera coordinate systems
  const Vec3 Y0 = P0 * X.homogeneous();
  const Vec3 Y1 = P1 * X.homogeneous();
  // differentiate
  const Mat3 dY0_dX = P0.topLeftCorner<3, 3>();
  const Mat3 dY1_dX = P1.topLeftCorner<3, 3>();
  const Mat32 dY0_dx = dY0_dX * dX_dx;
  const Mat32 dY1_dx = dY1_dX * dX_dx;

  // LAF 0 (x0, M0)
  const auto [x0, dx0_dY0] = cam0->p_gradient(Y0);
  const Mat2 M0 = dx0_dY0 * dY0_dx; // affine shape around x0, aka dx0_dx

  // LAF 1 (x1, M1)
  const auto [x1, dx1_dY1] = cam0->p_gradient(Y1);
  const Mat2 M1 = dx1_dY1 * dY1_dx; // affine shape around x1, aka dx1_dx

  // ==========================================================================
  // compute synthetic depths and its gradients
  // ==========================================================================
  const auto [lambda0, dlambda0_dY0] = cam0->depth_gradient(Y0);
  const RowVec2 dlambda0_dx = dlambda0_dY0 * dY0_dx; // aka dlambda0_dx

  const auto [lambda1, dlambda1_dY1] = cam1->depth_gradient(Y1);
  const RowVec2 dlambda1_dx = dlambda1_dY1 * dY1_dx; // aka dlambda1_dx

  // ==========================================================================
  // perform estimation
  // ==========================================================================
  
  // 3D LAF 0 (q0, Q0)
  const auto [q0, dq0_dx0] = cam0->q_gradient(x0);
  const Mat32 Q0 = dq0_dx0 * M0; // 3D-projected affine shape, aka q0_dx

  // 3D LAF 1 (q1, Q1)
  const auto [q1, dq1_dx1] = cam1->q_gradient(x1);
  const Mat32 Q1 = dq1_dx1 * M1; // 3D-projected affine shape, aka q1_dx

  // combine 3D LAF 0 with depth and its derivatives
  const Vec3 src = q0 * lambda0;
  const Mat32 src_diff = q0 * dlambda0_dx + Q0 * lambda0; // aka dsrc_dx

  // combine 3D LAF 1 with depth and its derivatives
  const Vec3 dst = q1 * lambda1;
  const Mat32 dst_diff = q1 * dlambda1_dx + Q1 * lambda1; // aka dsrc_dx

  // estimate pose using 1AC+D
  double scale;
  Mat3 R;
  Vec3 t;
  OneACPose::OneACD(
    dst, src,
    dst_diff, src_diff,
    scale, R, t);

  // ==========================================================================
  // measure errors wrt ground truth
  // ==========================================================================
  std::cout << "R_est:\t" << R << std::endl;
  std::cout << "R_gt:\t" << (R1 * R0.transpose()) << std::endl;

  std::cout << "t_est:\t" << (t).transpose() << std::endl;
  std::cout << "t_gt:\t" << (C1 - C0).transpose() << std::endl;

  std::cout << "scale:\t" << scale << std::endl;
}