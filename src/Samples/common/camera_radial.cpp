// MIT License
// 
// Copyright(c) 2018 Ivan Eichhardt
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "camera_radial.hpp"
#include <string_view>
#include <fstream>
#include <iterator>

using namespace OneACPose;

common::Camera_Radial::Camera_Radial() : Camera(), mK(Mat3::Identity()) {}

Vec2 common::Camera_Radial::p(const Vec3& X) const {
  return p_radial(X);
}

std::pair<Vec2, Mat23> common::Camera_Radial::p_gradient(const Vec3& X) const {
  return p_radial_gradient(X);
}

Vec3 common::Camera_Radial::q(const Vec2& x) const {
  return q_radial(x);
}

std::pair<Vec3, Mat32> common::Camera_Radial::q_gradient(const Vec2& x) const {
  return q_radial_gradient(x);
}

double common::Camera_Radial::depth(const OneACPose::Vec3& X) const
{
  return depth_radial(X);
}

std::pair<double, OneACPose::RowVec3> common::Camera_Radial::depth_gradient(const OneACPose::Vec3& X) const
{
  return depth_radial_gradient(X);
}

void common::Camera_Radial::set_params(const std::vector<double>& params) {
  assert(params.size() == 7);
  fx() = params[0]; fy() = params[1];
  cx() = params[2]; cy() = params[3];
  for (int i = 0; i < 3; ++i) {
    mDistortionParameters(i) = params[4 + i];
  }
}

std::vector<double> common::Camera_Radial::get_params() const {
  return {
    fx(), fy(), cx(), cy(),
    mDistortionParameters(0), mDistortionParameters(1), mDistortionParameters(2)
  };
}

inline common::Camera_Radial::DistortionFunctor::DistortionFunctor(const Camera_Radial& cam) : mCamera(cam) {}