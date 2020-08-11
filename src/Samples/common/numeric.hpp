#pragma once

#include <Eigen/Dense>
#include <OneACPose/types.hpp>

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

namespace common
{
  inline double R2D(double radian)
  {
    return radian / M_PI * 180;
  }

  template <typename T>
  inline T Square(const T& x)
  {
    return x * x;
  }

  template <typename T>
  inline T Cube(const T& x)
  {
    return x * x * x;
  }

  OneACPose::Mat3 LookAt(const OneACPose::Vec3& forward, const OneACPose::Vec3& up = OneACPose::Vec3::UnitY());

  double getRotationMagnitude(const OneACPose::Mat3& R2);

  inline OneACPose::Mat3 cross_product(const OneACPose::Vec3& v)
  {
    OneACPose::Mat3 result; result <<
      0, -v(2), v(1),
      v(2), 0, -v(0),
      -v(1), v(0), 0;
    return result;
  }

  template<typename TMat>
  inline double frobenius_normSq(const TMat& A)
  {
    return A.array().abs2().sum();
  }

  template<typename TMat>
  inline double frobenius_norm(const TMat& A)
  {
    return std::sqrt(frobenius_normSq(A));
  }

  template<typename TMat>
  inline double matrix_error(const TMat& m1, const TMat& m2)
  {
    auto m1n = m1 / FrobeniusNorm(m1);
    auto m2n = m2 / FrobeniusNorm(m2);
    return std::min(FrobeniusNorm(m1n - m2n), FrobeniusNorm(m1n + m2n));
  }

  OneACPose::Mat32 nullspace(const OneACPose::Vec3& N);
}