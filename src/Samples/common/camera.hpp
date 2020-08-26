#pragma once

#include <memory>

#include <OneACPose/types.hpp>
#include "dual_number.hpp"

namespace common {

  // Camera base class
  class Camera {
  public:
    // The camera to image projection funciton, returns the value.
    virtual OneACPose::Vec2 p(const OneACPose::Vec3& X) const = 0;

    // The gradient camera to image projection funciton, returns the value and the gradient as a pair.
    virtual std::pair<OneACPose::Vec2, OneACPose::Mat23> p_gradient(const OneACPose::Vec3& X) const = 0;

    // The image to camera projection function, returns the value.
    virtual OneACPose::Vec3 q(const OneACPose::Vec2& x) const = 0;

    // The gradient image to camera projection function, returns the value and the gradient as a pair.
    virtual std::pair<OneACPose::Vec3, OneACPose::Mat32> q_gradient(const OneACPose::Vec2& x) const = 0;

    // TODO
    virtual double depth(const OneACPose::Vec3& X) const = 0;

    // TODO
    virtual std::pair<double, OneACPose::RowVec3> depth_gradient(const OneACPose::Vec3& X) const = 0;

    // Sets camera parameters.
    virtual void set_params(const std::vector<double>& params) = 0;

    // Returns camera parameters.
    virtual std::vector<double> get_params() const = 0;

    void LoadParams(const std::string_view filename);
  };

  using CameraPtr = std::shared_ptr<common::Camera>;

}