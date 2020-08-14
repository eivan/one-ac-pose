#pragma once

#include "camera.hpp"

namespace common {

  // Pinhole camera with 3 radial distortion parameters
  class Camera_Radial : public Camera {

  public:
    Camera_Radial();

    OneACPose::Vec2 p(const OneACPose::Vec3& X) const final override;
    std::pair<OneACPose::Vec2, OneACPose::Mat23> p_gradient(const OneACPose::Vec3& X) const final override;

    OneACPose::Vec3 q(const OneACPose::Vec2& x) const final override;
    std::pair<OneACPose::Vec3, OneACPose::Mat32> q_gradient(const OneACPose::Vec2& x) const final override;

    double depth(const OneACPose::Vec3& X) const final override;
    std::pair<double, OneACPose::RowVec3> depth_gradient(const OneACPose::Vec3& X) const final override;

    void set_params(const std::vector<double>& params) final override;

    std::vector<double> get_params() const final override;

  private:
    inline double& fx();
    inline double& fy();
    inline double& cx();
    inline double& cy();

    inline const double& fx() const;
    inline const double& fy() const;
    inline const double& cx() const;
    inline const double& cy() const;

    inline double& k1();
    inline double& k2();
    inline double& k3();

    inline const double& k1() const;
    inline const double& k2() const;
    inline const double& k3() const;

    template <typename V>
    inline V distortion_formula(const V& r2) const;

    template <typename V>
    inline Eigen::Matrix<V, 2, 1> add_disto_cam(const Eigen::Matrix<V, 2, 1>& p) const;

    struct DistortionFunctor {
      const Camera_Radial& mCamera;

      explicit DistortionFunctor(const Camera_Radial& cam);

      template <typename T>
      T operator()(const T& r2) const {
        return r2 * Square(mCamera.distortion_formula(r2));
      }

      DEFINE_REAL_DERIVATIVE;
    };

    template <typename V>
    inline Eigen::Matrix<V, 2, 1> remove_disto_cam(const Eigen::Matrix<V, 2, 1>& p) const;

    template <typename V>
    inline Eigen::Matrix<V, 2, 1> p_pinhole(const Eigen::Matrix<V, 2, 1>& p) const;

    template <typename V>
    inline Eigen::Matrix<V, 2, 1> q_pinhole(const Eigen::Matrix<V, 2, 1>& p) const;

    template <typename T>
    inline Eigen::Matrix<T, 2, 1> p_radial(const Eigen::Matrix<T, 3, 1>& X) const;
    DIFFERENTIATE(p_radial, 3, 2);

    template <typename T>
    inline Eigen::Matrix<T, 3, 1> q_radial(const Eigen::Matrix<T, 2, 1>& x) const;
    DIFFERENTIATE(q_radial, 2, 3);

    template <typename T>
    inline T depth_radial(const Eigen::Matrix<T, 3, 1>& X) const;
    DIFFERENTIATE_VECTOR_SCALAR(depth_radial, 3);

  private:
    OneACPose::Mat3 mK;
    OneACPose::Vec3 mDistortionParameters;

  };

  inline double& common::Camera_Radial::fx() { return mK(0, 0); }

  inline double& common::Camera_Radial::fy() { return mK(1, 1); }

  inline double& common::Camera_Radial::cx() { return mK(0, 2); }

  inline double& common::Camera_Radial::cy() { return mK(1, 2); }

  inline const double& common::Camera_Radial::fx() const { return mK(0, 0); }

  inline const double& common::Camera_Radial::fy() const { return mK(1, 1); }

  inline const double& common::Camera_Radial::cx() const { return mK(0, 2); }

  inline const double& common::Camera_Radial::cy() const { return mK(1, 2); }

  inline double& common::Camera_Radial::k1() { return mDistortionParameters[0]; }

  inline double& common::Camera_Radial::k2() { return mDistortionParameters[1]; }

  inline double& common::Camera_Radial::k3() { return mDistortionParameters[2]; }

  inline const double& common::Camera_Radial::k1() const { return mDistortionParameters[0]; }

  inline const double& common::Camera_Radial::k2() const { return mDistortionParameters[1]; }

  inline const double& common::Camera_Radial::k3() const { return mDistortionParameters[2]; }

  template<typename V>
  inline V Camera_Radial::distortion_formula(const V& r2) const {
    return 1. + r2 * (k1() + r2 * (k2() + r2 * k3()));
  }

  template<typename V>
  inline Eigen::Matrix<V, 2, 1> Camera_Radial::add_disto_cam(const Eigen::Matrix<V, 2, 1>& p) const {
    const auto r2 = p.squaredNorm();
    return p * distortion_formula(r2);
  }

  template<typename V>
  inline Eigen::Matrix<V, 2, 1> Camera_Radial::remove_disto_cam(const Eigen::Matrix<V, 2, 1>& p) const {
    const V r2 = p.squaredNorm();

    if (r2 == 0.0) {
      return p;
    }
    else {
      DistortionFunctor distoFunctor(*this);

      V radius = common::sqrt(bisection(distoFunctor, r2) / r2);
      return radius * p;
    }
  }

  template<typename V>
  inline Eigen::Matrix<V, 2, 1> Camera_Radial::p_pinhole(const Eigen::Matrix<V, 2, 1>& p) const {
    return { fx() * p(0) + cx(), fy() * p(1) + cy() };
  }

  template<typename V>
  inline Eigen::Matrix<V, 2, 1> Camera_Radial::q_pinhole(const Eigen::Matrix<V, 2, 1>& p) const {
    return { (p(0) - cx()) / fx(), (p(1) - cy()) / fy() };
  }

  template<typename T>
  inline Eigen::Matrix<T, 2, 1> Camera_Radial::p_radial(const Eigen::Matrix<T, 3, 1>& X) const {
    Eigen::Matrix<T, 2, 1> p = X.template head<2>() / X(2);
    return p_pinhole(add_disto_cam(p));
  }

  template<typename T>
  inline Eigen::Matrix<T, 3, 1> Camera_Radial::q_radial(const Eigen::Matrix<T, 2, 1>& x) const {
    auto p2 = remove_disto_cam(q_pinhole(x));
    //return Eigen::Matrix<T, 3, 1>(p2(0), p2(1), T(1.0)).normalized();
    return p2.homogeneous();
  }

  template<typename T>
  inline T Camera_Radial::depth_radial(const Eigen::Matrix<T, 3, 1>& X) const
  {
    // NOTE: needs to be X = depth * q_radial
    // and as there's .normalized() in q_radial, here the norm should be returned
    //return X.norm();
    return X(2);
  }

}